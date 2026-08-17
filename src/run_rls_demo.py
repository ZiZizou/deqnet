"""Streaming RLS adaptive-filtering demo.

System identification: recover an unknown plant w_o in R^d from a stream of
noisy linear observations  d_t = w_o^T x_t + nu_t, with four contenders:

  1. digital_lms   : w <- w + mu * e * x                       (1st-order, digital)
  2. digital_rls   : standard RLS recursion with forgetting lambda
  3. fabric_lms    : continuous-time gradient flow dw/dt = mu * e * x,
                     integrated by forward Euler at fine dt          (1st-order, fabric)
  4. fabric_rls    : per sample:
                       R <- lambda * R + outer(x, x)              # write phase (digital, v1)
                       p <- lambda * p + d_t * x                  # observation, NOT error
                       w = LinearSolveLayer(p, R)                 # settle phase (the fabric)

RLS recursion note: the exponentially-weighted least-squares solution is
  w_t = R_t^{-1} p_t
where  R_t = sum_{k<=t} lambda^{t-k} x_k x_k^T  and  p_t = sum_{k<=t} lambda^{t-k} d_k x_k.
Note the *observation* d_k enters p, not the error e_k.  The README's earlier
p <- decay*p + e*x sketch was wrong; this is the correct form.

Phase 1 (Learned Robust IR-RLS) extends the four contenders with two
robust variants whose per-sample influence v_t is gated by a learned
robust weighter.  See ``utils/learned_robust.py`` and the
``LEARNED_RLS_ISTA_PLAN.md`` file.

Precision policy (plan decision 0b / 8):
  * Training the robust weighter runs in float32 on CPU (solver tol
    1e-5, max_iter ~ 50).  The grad plumbing is exercised in float64
    in ``tests/test_learned_robust.py`` (Phase 0 gate).
  * Evaluation (the Monte-Carlo comparison) runs in float64; the
    frozen weighter is ``.double()``-cast for that run.
  * Parity / gradcheck gates run in float64 (solver tol 1e-10) to
    isolate algorithmic error from precision error.

Metrics:
  - Ensemble-mean ||w_t - w_o||^2 vs t (log scale), all 4 contenders on one plot
  - Discrepancy: ||w_fabric - w_digital_RLS|| / ||w_digital_RLS|| vs t (should be 1e-4..1e-6)
  - Settling budget: iterations per sample for LinearSolveLayer
  - Tracking: time-varying w_o with process noise, steady-state misadjustment vs lambda
  - Ljung overlay: fabric-LMS (ODE dw/dt = mu*e*x) and digital
    LMS at decreasing step sizes mu*dt converge onto the continuous trajectory.

Results are saved to ./results/rls_demo/ (and ./results/rls_demo/robust/ for Phase 1).
"""
import os
import sys
import argparse
import json
import time
from datetime import datetime
import numpy as np
import torch


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from utils.circuit_block import LinearSolveLayer, EquilibriumSolve


# Phase B defaults: log-space init for the Cauchy weighter.
# These reproduce the legacy KIMI-corrected softplus init
#   c     = softplus(-2.25) + 1e-3   ~ 0.1012
#   alpha = softplus(-2.0)           ~ 0.1269
# in the new parameterization (Phase B1: c=exp(log_c), alpha=exp(log_alpha)).
import math as _math
import torch.nn.functional as _F
_DEFAULT_LOG_C_INIT = float(_math.log(
    _F.softplus(torch.tensor(-2.25)).item() + 1e-3))
_DEFAULT_LOG_ALPHA_INIT = float(_math.log(
    _F.softplus(torch.tensor(-2.0)).item()))
_DEFAULT_C_INIT = float(_F.softplus(torch.tensor(-2.25)).item() + 1e-3)
_DEFAULT_ALPHA_INIT = float(_F.softplus(torch.tensor(-2.0)).item())


def _resolve_device(use_gpu, gpu_id):
    """Resolve --gpu / --gpu_id flags to a torch.device.

    use_gpu=False            -> CPU
    use_gpu=True, gpu_id>=0  -> cuda:<gpu_id> (raise if CUDA unavailable)
    """
    if not use_gpu:
        return torch.device('cpu')
    if not torch.cuda.is_available():
        raise RuntimeError("--gpu set but CUDA is not available")
    return torch.device(f'cuda:{gpu_id}')


# ----------------------------------------------------------------------------
# 1. Contenders
# ----------------------------------------------------------------------------

class DigitalLMS:
    """w <- w + mu * e * x, scalar mu."""
    def __init__(self, d, mu=0.05, w_init=None, device='cpu'):
        self.d = d
        self.mu = mu
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.w = (torch.zeros(d, device=self.device) if w_init is None
                  else w_init.clone().to(self.device))

    def step(self, x_t, d_t):
        e = d_t - self.w @ x_t
        self.w = self.w + self.mu * e * x_t
        return self.w.clone()


class DigitalRLS:
    """Standard RLS with forgetting factor lambda.  Maintains P = R^{-1}
    for O(d^2) updates instead of O(d^3) explicit solve per step."""
    def __init__(self, d, lam=0.99, delta=1.0, w_init=None, device='cpu'):
        self.d = d
        self.lam = lam
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.w = (torch.zeros(d, device=self.device) if w_init is None
                  else w_init.clone().to(self.device))
        self.P = (1.0 / delta) * torch.eye(d, device=self.device)

    def step(self, x_t, d_t):
        # Standard RLS: k = P x / (lam + x^T P x); w += k * e; P = (P - k x^T P) / lam
        Px = self.P @ x_t
        denom = self.lam + x_t @ Px
        k = Px / denom
        e = d_t - self.w @ x_t
        self.w = self.w + k * e
        self.P = (self.P - torch.outer(k, Px)) / self.lam
        return self.w.clone()


class FabricLMS:
    """Continuous-time LMS: dw/dt = mu * e(t) * x(t).

    Integrated by forward Euler at fine dt.  When mu*dt -> 0, this converges
    onto the same trajectory as DigitalLMS with the same mu*dt — the Ljung
    experiment demonstrates this directly."""
    def __init__(self, d, mu=0.05, dt=0.1, w_init=None, device='cpu'):
        self.d = d
        self.mu = mu
        self.dt = dt
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.w = (torch.zeros(d, device=self.device) if w_init is None
                  else w_init.clone().to(self.device))

    def step(self, x_t, d_t):
        # Hold x and d constant over the inner Euler substep.
        e = d_t - self.w @ x_t
        self.w = self.w + self.dt * self.mu * e * x_t
        return self.w.clone()


class FabricRLS:
    """Per-sample:
         R <- lam * R + outer(x, x)
         p <- lam * p + d_t * x
         w = LinearSolveLayer(p, R)            # analog settle

    beta for the fixed-point map is chosen adaptively as
    2 / (lambda_min(R) + lambda_max(R)) (Chebyshev step) so that the
    solver converges in O(1) Anderson iterations regardless of how large
    R has grown through accumulation.  R is symmetric PSD so we use the
    eigvalsh branch.
    """
    def __init__(self, d, lam=0.99, R0=None, w_init=None,
                 max_iter=100, tol=1e-8, beta='chebyshev', device='cpu',
                 log_every=0, log_name=''):
        self.d = d
        self.lam = lam
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        # Match dtype to R0 if provided; otherwise default.
        if R0 is not None:
            self._init_dtype = R0.dtype
        elif w_init is not None:
            self._init_dtype = w_init.dtype
        else:
            self._init_dtype = torch.get_default_dtype()
        self.R = R0.to(self.device) if R0 is not None else torch.eye(d, device=self.device, dtype=self._init_dtype)
        self.w = (torch.zeros(d, device=self.device, dtype=self._init_dtype) if w_init is None
                  else w_init.clone().to(self.device))
        self.p = torch.zeros(d, device=self.device, dtype=self._init_dtype)
        self.max_iter = max_iter
        self.tol = tol
        self.beta_mode = beta
        self.last_iters = []
        self.log_every = log_every
        self.log_name = log_name
        self._step_count = 0

    def _chebyshev_beta(self, R):
        eigs = torch.linalg.eigvalsh(R)
        lam_min = eigs[0].item()
        lam_max = eigs[-1].item()
        if lam_max <= 0:
            return 1.0
        return float(2.0 / (lam_min + lam_max))

    def step(self, x_t, d_t):
        self.R = self.lam * self.R + torch.outer(x_t, x_t)
        self.p = self.lam * self.p + d_t * x_t
        if self.beta_mode == 'chebyshev':
            beta = self._chebyshev_beta(self.R)
        else:
            beta = float(self.beta_mode)
        layer = LinearSolveLayer(max_iter=self.max_iter, tol=self.tol, beta=beta)
        # Solve w* = R^{-1} p via analog settle.
        w_star = layer(self.p.unsqueeze(0), self.R).squeeze(0)
        # Track iterations from the last EquilibriumSolve.
        from utils.circuit_block import EquilibriumSolve
        n_iters = -1
        if EquilibriumSolve.last_info is not None:
            n_iters = EquilibriumSolve.last_info.get('n_iter', -1)
            self.last_iters.append(n_iters)
        self.w = w_star
        self._step_count += 1
        if self.log_every and (self._step_count % self.log_every == 0
                               or self._step_count == 1):
            print(f"    [{self.log_name}] sample {self._step_count}"
                  f"  LinearSolveLayer iters={n_iters}", flush=True)
        return self.w.clone()


class FabricBatchRLS:
    """Block/batch least-squares via a single fabric equilibrium solve.

    Given a block of observations X and d, accumulate the (regularized,
    uniformly-weighted) normal equations R w = p, then compute w* by
    invoking LinearSolveLayer exactly once.  The accumulated system
    corresponds to the least-squares problem

        w* = argmin  weight * ||X w - d||^2  +  w^T R0 w

    whose normal equations are  (weight * X^T X + R0) w = weight * X^T d.
    The scalar ``weight`` (default 1.0) is a uniform per-sample weight; it
    is not a forgetting factor and does not change the unweighted
    solution when set to 1.  R0 is the positive-definite regularizer
    (default identity).
    """
    def __init__(self, d, R0=None, weight=1.0, max_iter=200, tol=1e-10,
                 beta='chebyshev', device='cpu'):
        self.d = d
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.R = (R0.to(self.device) if R0 is not None
                  else torch.eye(d, device=self.device))
        self.p = torch.zeros(d, device=self.device)
        self.weight = float(weight)
        self.max_iter = max_iter
        self.tol = tol
        self.beta_mode = beta

    def _chebyshev_beta(self, R):
        eigs = torch.linalg.eigvalsh(R)
        lam_min = eigs[0].item()
        lam_max = eigs[-1].item()
        if lam_max <= 0:
            return 1.0
        return float(2.0 / (lam_min + lam_max))

    def accumulate(self, X, d):
        """Accumulate one block: X is (T, d), d is (T,)."""
        X = X.to(self.device)
        d = d.to(self.device)
        R_new = self.R + self.weight * (X.t() @ X)
        # Symmetrize to remove fp-rounding asymmetry before any further
        # eigendecomposition, beta computation, or solve.
        self.R = 0.5 * (R_new + R_new.t())
        self.p = self.p + self.weight * (X.t() @ d)

    def _solve_beta(self):
        if self.beta_mode == 'chebyshev':
            return self._chebyshev_beta(self.R)
        return float(self.beta_mode)

    def solve(self):
        """One settle: w* = R^{-1} p via LinearSolveLayer."""
        beta = self._solve_beta()
        layer = LinearSolveLayer(max_iter=self.max_iter, tol=self.tol, beta=beta)
        w_star = layer(self.p.unsqueeze(0), self.R).squeeze(0)
        info = EquilibriumSolve.last_info
        return w_star, info


def batch_lstsq_reference(X, d, R0, weight=1.0):
    """Direct reference for the uniformly-weighted regularized
    least-squares problem:

        w* = argmin  weight * ||X w - d||^2  +  w^T R0 w
           = lstsq([sqrt(weight) * X; L], [sqrt(weight) * d; 0]).solution

    where R0 = L^T L.  For scalar identity regularization R0 = delta I,
    L = sqrt(delta) * I.  Returns w_ref on the same device as X.  When
    ``weight=1`` this matches the unweighted reference documented in the
    spec.
    """
    weight = float(weight)
    device = X.device
    n = X.shape[1]
    R0 = R0.to(device)
    sw = float(np.sqrt(max(weight, 0.0)))
    eigs, Q = torch.linalg.eigh(R0)
    L = Q @ torch.diag(torch.sqrt(eigs.clamp_min(0.0))) @ Q.t()
    A = torch.cat([sw * X, L], dim=0)
    rhs = torch.cat([sw * d, torch.zeros(n, device=device, dtype=d.dtype)])
    sol, *_ = torch.linalg.lstsq(A, rhs)
    return sol


def batch_experiment_metrics(w_fabric, R, p, X, d, R0, w_o=None, weight=1.0):
    """Compare fabric and reference solutions for the same weighted
    regularized least-squares problem.  Returns a dict of metrics plus the
    reference solution.  ``weight`` defaults to 1.0 for the unweighted
    problem; the metric definitions use the same weight the fabric solve
    accumulated, so a fabric/reference mismatch can only come from solver
    inaccuracy.
    """
    weight = float(weight)
    w_ref = batch_lstsq_reference(X, d, R0, weight=weight)
    abs_err = (w_fabric - w_ref).abs().max().item()
    rel_err = ((w_fabric - w_ref).norm() / w_ref.norm().clamp_min(1e-12)).item()
    normal_res = (R @ w_fabric - p).norm().item()
    # The objective actually being minimized by both fabric and reference
    # (no arbitrary 0.5 scaling).
    obj = (weight * (X @ w_fabric - d).pow(2).sum()
           + w_fabric @ R0 @ w_fabric).item()
    out = {
        'abs_err': abs_err,
        'rel_err': rel_err,
        'normal_eq_residual': normal_res,
        'regularized_objective': obj,
        'w_ref': w_ref.detach().cpu(),
        'w_fabric': w_fabric.detach().cpu(),
    }
    if w_o is not None:
        out['plant_error'] = (w_fabric - w_o).pow(2).sum().item()
    return out
# ----------------------------------------------------------------------------
# 2. Stream generators
# ----------------------------------------------------------------------------

def _make_noise(T, sigma, mode='gaussian', p_burst=0.02, kappa=20.0,
                generator=None, dtype=torch.float64, device='cpu',
                return_burst=False):
    """Per-sample additive noise nu_t.

    mode='gaussian':
        nu_t = sigma * randn(T)                       # byte-identical to the
        legacy ``make_stream`` noise (default).
    mode='impulsive':
        nu_t = sigma * (randn(T) + kappa * burst_t * randn(T))
        burst_t = (rand(T) < p_burst)               # Bernoulli gate
        # Equivalent: nu_t = sigma * (1 + kappa * burst_t) * randn(T) when the
        # burst is on, sigma * randn(T) otherwise.  The two-form factor
        # (randn + kappa * burst * randn) keeps the noise scale *per sample*
        # interpretable: nominal |nu| ~ sigma, burst |nu| ~ sigma * sqrt(1 + kappa^2).

    ``generator`` is the torch.Generator used to draw the randn and rand
    tensors; passing an explicit generator keeps the noise deterministic
    for a given seed (plan decision 6: backward-compatible defaults).

    ``return_burst`` (Phase 0 oracle-oos-headroom): if True, return
    ``(nu, burst)`` where ``burst`` is the (T,) 0/1 indicator of which
    samples were drawn as bursts (1 = burst, 0 = clean).  This is the
    TRUE burst mask -- the simulation's Bernoulli gate, already computed
    here -- which the oracle weighter needs for perfect down-weighting.
    In gaussian mode there are no bursts, so ``burst`` is an all-zero
    tensor (a clean block).  Default False keeps the return type
    backward-compatible.
    """
    if mode == 'gaussian':
        nu = sigma * torch.randn(T, generator=generator, dtype=dtype, device=device)
        if return_burst:
            burst = torch.zeros(T, dtype=dtype, device=device)
            return nu, burst
        return nu
    if mode == 'impulsive':
        z1 = torch.randn(T, generator=generator, dtype=dtype, device=device)
        z2 = torch.randn(T, generator=generator, dtype=dtype, device=device)
        # Use a separate rand stream for the burst gate so the impulsive
        # noise is reproducible from the same seed (gate 5 checks this).
        burst = (torch.rand(T, generator=generator, dtype=dtype, device=device) < p_burst).to(dtype)
        nu = sigma * (z1 + kappa * burst * z2)
        if return_burst:
            return nu, burst
        return nu
    raise ValueError(f"unknown noise mode: {mode!r}")


def make_stream(w_o, T, sigma, mode='iid', noise='gaussian', p_burst=0.02,
                kappa=20.0, seed=0, dtype=torch.float64, device='cpu'):
    """Generate (x_t, d_t) stream of length T.

    mode='iid':   x_t ~ N(0, I)
    mode='ar1':   x_t = rho * x_{t-1} + sqrt(1-rho^2) * eps_t, eps_t ~ N(0, I)

    noise='gaussian':  nu_t = sigma * randn(T)                   legacy / default
    noise='impulsive': nu_t = sigma * (randn + kappa * burst * randn)
    with burst_t = (rand < p_burst).  All backward-compatible defaults.
    """
    device = torch.device(device) if not isinstance(device, torch.device) else device
    g = torch.Generator(device=device).manual_seed(seed)
    d = w_o.shape[0]
    x = torch.zeros(T, d, dtype=dtype, device=device)
    if mode == 'iid':
        x = torch.randn(T, d, generator=g, dtype=dtype, device=device)
    elif mode == 'ar1':
        rho = 0.9
        eps = torch.randn(T, d, generator=g, dtype=dtype, device=device)
        x[0] = eps[0]
        for t in range(1, T):
            x[t] = rho * x[t - 1] + np.sqrt(1 - rho ** 2) * eps[t]
    else:
        raise ValueError(f"unknown mode: {mode}")
    nu = _make_noise(T, sigma, mode=noise, p_burst=p_burst, kappa=kappa,
                     generator=g, dtype=dtype, device=device)
    d_obs = x @ w_o + nu
    return x, d_obs


def make_tracking_stream(w_o0, T, sigma, process_std, seed=0, dtype=torch.float64, device='cpu'):
    """Stream with time-varying plant: w_o(t+1) = w_o(t) + q_t, q_t ~ N(0, process_std^2 * I)."""
    device = torch.device(device) if not isinstance(device, torch.device) else device
    g = torch.Generator(device=device).manual_seed(seed)
    d = w_o0.shape[0]
    x = torch.randn(T, d, generator=g, dtype=dtype, device=device)
    w_o = torch.zeros(T, d, dtype=dtype, device=device)
    w_o[0] = w_o0
    for t in range(1, T):
        w_o[t] = w_o[t - 1] + process_std * torch.randn(d, generator=g, dtype=dtype, device=device)
    nu = sigma * torch.randn(T, generator=g, dtype=dtype, device=device)
    d_obs = (x * w_o).sum(dim=-1) + nu
    return x, d_obs, w_o


# ----------------------------------------------------------------------------
# 3. Single-trial runner
# ----------------------------------------------------------------------------

def run_trial(contender, x, d_obs, w_o):
    """Returns (W[T,d]) and (extra_info, dict).  `contender` is an
    already-constructed contender instance.  `w_o` may be shape (d,) for
    static plant or (T, d) for tracking."""
    T = x.shape[0]
    d = x.shape[1]
    W = torch.zeros(T, d, dtype=x.dtype, device=x.device)
    for t in range(T):
        W[t] = contender.step(x[t], d_obs[t])
    extras = {}
    if hasattr(contender, 'last_iters'):
        extras['last_iters'] = list(contender.last_iters)
    return W, extras


# ----------------------------------------------------------------------------
# 4. Monte Carlo
# ----------------------------------------------------------------------------

def monte_carlo(contender_factory, w_o, T, sigma, n_trials,
                 mode='iid', noise='gaussian', p_burst=0.02, kappa=20.0,
                 seed_base=0, dtype=torch.float64, device='cpu',
                 name='', log_every=1, quiet=False):
    """`contender_factory` is a zero-arg callable returning a fresh contender
    instance.  Returns (W_mean[T,d], W_runs[n_trials, T, d], per-trial extras list).

    `name` and `log_every` control progress logging: prints every `log_every`
    trials with the elapsed wall time.  Set `quiet=True` to suppress all logging
    (used when called from inside another logged loop).

    ``noise``/``p_burst``/``kappa`` are forwarded to ``make_stream`` /
    ``_make_noise`` (plan decision 6: backwards-compatible defaults)."""
    device = torch.device(device) if not isinstance(device, torch.device) else device
    d = w_o.shape[0]
    W_runs = torch.zeros(n_trials, T, d, dtype=dtype, device=device)
    extras_all = []
    if not quiet:
        print(f"  [{name}] {n_trials} trials, T={T}, mode={mode}, noise={noise}"
              f" (p_burst={p_burst}, kappa={kappa})", flush=True)
    loop_start = time.time()
    for trial in range(n_trials):
        x, d_obs = make_stream(w_o, T, sigma, mode=mode, noise=noise,
                               p_burst=p_burst, kappa=kappa,
                               seed=seed_base + trial, dtype=dtype, device=device)
        contender = contender_factory()
        W, extras = run_trial(contender, x, d_obs, w_o)
        W_runs[trial] = W
        extras_all.append(extras)
        if not quiet and (trial + 1) % log_every == 0:
            elapsed = time.time() - loop_start
            avg = elapsed / (trial + 1)
            eta = avg * (n_trials - trial - 1)
            print(f"    trial {trial + 1}/{n_trials}"
                  f"  elapsed={elapsed:.1f}s  avg={avg:.2f}s/trial  eta={eta:.1f}s",
                  flush=True)
    W_mean = W_runs.mean(dim=0)
    if not quiet:
        total = time.time() - loop_start
        print(f"  [{name}] done: {n_trials} trials in {total:.2f}s"
              f" ({total / max(n_trials, 1):.2f}s/trial)", flush=True)
    return W_mean, W_runs, extras_all


# ----------------------------------------------------------------------------
# 5. Metrics
# ----------------------------------------------------------------------------

def w_err_sq(W, w_o):
    """Squared error ||w_t - w_o||^2 over time, shape (T,)."""
    diff = W - w_o.unsqueeze(0)
    return (diff * diff).sum(dim=-1)


def relative_discrepancy(W_a, W_b):
    """||w_a - w_b|| / ||w_b|| over time, shape (T,).  Uses norm of b
    as denominator so it's meaningful even when w_b -> 0."""
    denom = W_b.norm(dim=-1).clamp_min(1e-12)
    return (W_a - W_b).norm(dim=-1) / denom


def steady_state_misadjustment(W, w_o_track, tail_frac=0.5):
    """Mean ||w_t - w_o(t)||^2 over the last `tail_frac` of the stream.
    Used for the tracking experiment."""
    T = W.shape[0]
    tail = int(T * tail_frac)
    err = w_err_sq(W[tail:], w_o_track[tail:])
    return err.mean().item()


# ----------------------------------------------------------------------------
# 5b. Training-hardening helpers (Phase A4, D3).
# ----------------------------------------------------------------------------


def summarize_settles(settle_log, *, max_iter=None, settle_info_log=None):
    """Summarize settle counts, cap hits, and optional residuals."""
    residuals = [float(info['final_residual'])
                 for info in (settle_info_log or [])
                 if isinstance(info, dict) and info.get('final_residual') is not None]
    arr = [s for s in settle_log if isinstance(s, int) and s >= 0]
    cap_hits = sum(1 for s in arr if max_iter is not None and s >= max_iter)
    return {
        'n_settles': len(settle_log),
        'mean_n_iter': float(sum(arr) / len(arr)) if arr else None,
        'max_n_iter': int(max(arr)) if arr else None,
        'max_final_residual': max(residuals) if residuals else None,
        'cap_hit_count': int(cap_hits),
        'cap_hit_fraction': float(cap_hits) / len(arr) if arr else 0.0,
    }


def paired_ci(diff, *, alpha=0.05):
    """Paired 95% (default) confidence interval on per-block MSE diffs
    (Phase D3).  Returns ``(mean, ci95)`` where ``ci95`` is half-width.

    ``diff`` is a 1-D tensor / array of paired per-block MSE differences
    ``mse_a - mse_b``.  Uses the normal approximation; for non-Gaussian
    robustness, replace with a bootstrap CI.
    """
    import math
    arr = np.asarray(diff, dtype=np.float64)
    n = arr.size
    if n == 0:
        return float('nan'), float('nan')
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    z = 1.96 if abs(alpha - 0.05) < 1e-9 else 1.96  # 95% normal
    ci = z * std / math.sqrt(n)
    return mean, ci


# ----------------------------------------------------------------------------
# 6. Plotting
# ----------------------------------------------------------------------------

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def plot_learning_curves(out_dir, results, w_o, suptitle):
    """results: dict[name -> (W_mean, W_runs)].  Plots ensemble-mean
    ||w_t - w_o||^2 vs t on log scale."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    T = next(iter(results.values()))[0].shape[0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {'digital_lms': 'C0', 'digital_rls': 'C1',
              'fabric_lms': 'C2', 'fabric_rls': 'C3'}
    styles = {'digital_lms': '-', 'digital_rls': '-',
              'fabric_lms': '--', 'fabric_rls': '--'}
    for name, (W_mean, _) in results.items():
        err = w_err_sq(W_mean, w_o).cpu().numpy()
        ax.semilogy(np.arange(T), err,
                    label=name, color=colors.get(name, 'k'),
                    linestyle=styles.get(name, '-'), linewidth=1.5)
    ax.set_xlabel('sample t')
    ax.set_ylabel(r'$\|w_t - w_o\|^2$')
    ax.set_title(suptitle)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'learning_curves.png'), dpi=140)
    plt.close(fig)


def plot_discrepancy(out_dir, W_fabric_rls_mean, W_digital_rls_mean):
    """Discrepancy ||w_fabric - w_digital|| / ||w_digital|| vs t."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    T = W_fabric_rls_mean.shape[0]
    disc = relative_discrepancy(W_fabric_rls_mean, W_digital_rls_mean).cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(np.arange(T), disc, color='C3', linewidth=1.5)
    ax.set_xlabel('sample t')
    ax.set_ylabel(r'$\|w_\mathrm{fabric} - w_\mathrm{digital\,RLS}\| / \|w_\mathrm{digital\,RLS}\|$')
    ax.set_title('Fabric-RLS vs Digital-RLS discrepancy')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'discrepancy.png'), dpi=140)
    plt.close(fig)


def plot_settling_budget(out_dir, iters_list, label):
    """Histogram of iterations per sample for LinearSolveLayer."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    iters = np.array(iters_list)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(iters, bins=np.arange(iters.min(), iters.max() + 2) - 0.5,
            color='C3', edgecolor='black', alpha=0.7)
    ax.set_xlabel('iterations per sample')
    ax.set_ylabel('count')
    ax.set_title(f'{label}  (mean={iters.mean():.2f}, max={iters.max()}, median={np.median(iters):.0f})')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'settling_budget.png'), dpi=140)
    plt.close(fig)


def plot_misadjust_vs_lambda(out_dir, lambdas, misadj_rls, misadj_fabric):
    """Steady-state misadjustment vs lambda for digital RLS and fabric RLS."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(lambdas, misadj_rls, 'o-', color='C1', label='digital RLS')
    ax.plot(lambdas, misadj_fabric, 's--', color='C3', label='fabric RLS')
    ax.set_xlabel(r'forgetting factor $\lambda$')
    ax.set_ylabel('steady-state misadjustment')
    ax.set_title('Tracking experiment: misadjustment vs $\\lambda$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'misadjust_vs_lambda.png'), dpi=140)
    plt.close(fig)


def plot_ljung_overlay(out_dir, w_o, x, d_obs, mu_values, dt_values):
    """Three panels, one per (mu*dt) value: fabric-LMS (continuous flow at
    fine dt) vs digital-LMS (discrete at coarse dt) trajectories.

    For each panel we also overlay a "continuous reference" path: digital
    LMS at very small step size (proxy for the ODE flow)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(mu_values)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
    T = x.shape[0]
    # Continuous reference: digital LMS at very small dt (proxy for ODE flow).
    mu_ref = 0.05
    dt_ref = 0.001
    n_substeps = max(1, int(round(1.0 / dt_ref)))  # 1000 substeps per "sample"
    w_ref = torch.zeros_like(w_o)
    err_ref = np.zeros(T)
    for t in range(T):
        for _ in range(n_substeps):
            e = d_obs[t] - w_ref @ x[t]
            w_ref = w_ref + dt_ref * mu_ref * e * x[t]
        err_ref[t] = ((w_ref - w_o) ** 2).sum().item()
    for i, (mu, dt) in enumerate(zip(mu_values, dt_values)):
        ax = axes[i]
        # Digital LMS with step mu*dt at the sample rate.
        w_d = torch.zeros_like(w_o)
        err_d = np.zeros(T)
        for t in range(T):
            e = d_obs[t] - w_d @ x[t]
            w_d = w_d + mu * dt * e * x[t]
            err_d[t] = ((w_d - w_o) ** 2).sum().item()
        # Fabric LMS (continuous-time ODE): integrate by forward Euler at fine sub-dt.
        n_sub = max(1, int(round(dt / 0.001)))
        dt_inner = dt / n_sub
        w_f = torch.zeros_like(w_o)
        err_f = np.zeros(T)
        for t in range(T):
            for _ in range(n_sub):
                e = d_obs[t] - w_f @ x[t]
                w_f = w_f + dt_inner * mu * e * x[t]
            err_f[t] = ((w_f - w_o) ** 2).sum().item()
        ax.semilogy(np.arange(T), err_d, '-', color='C0', label=f'digital LMS ($\\mu\\Delta t$={mu*dt:.3f})')
        ax.semilogy(np.arange(T), err_f, '--', color='C2', label=f'fabric LMS (ODE, dt={dt})')
        ax.semilogy(np.arange(T), err_ref, ':', color='gray', label='reference ($\\Delta t$=0.001)')
        ax.set_xlabel('sample t')
        if i == 0:
            ax.set_ylabel(r'$\|w_t - w_o\|^2$')
        ax.set_title(f'$\\mu$={mu}, $\\Delta t$={dt}')
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=0.3)
    fig.suptitle('Ljung: digital LMS -> continuous LMS as $\\mu\\Delta t \\to 0$')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'ljung_overlay.png'), dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 7. Main experiment flow
# ----------------------------------------------------------------------------

def run_batch_experiment(args, out_dir, w_o, device, dtype, seed):
    """Single-block batch least-squares experiment.

    Accumulate one block of observations into the regularized normal
    equations, solve once via LinearSolveLayer, compare to a direct
    torch.linalg.lstsq reference on the same regularized objective,
    and save metrics + solution snapshots.
    """
    print(f"\n=== Batch experiment: T={args.T_batch}, d={args.d}, "
          f"sigma={args.sigma}, regularization delta={args.batch_delta}, "
          f"weight={args.batch_weight}, n_trials={args.n_trials} ===",
          flush=True)
    fabric_ws = []
    ref_ws = []
    metrics_all = []
    n_calls = 0
    for trial in range(args.n_trials):
        torch.manual_seed(seed + trial)
        if device.type == 'cuda':
            torch.cuda.manual_seed(seed + trial)
        g = torch.Generator(device=device).manual_seed(seed + trial)
        x = torch.randn(args.T_batch, args.d, generator=g, dtype=dtype, device=device)
        nu = args.sigma * torch.randn(args.T_batch, generator=g, dtype=dtype, device=device)
        d_obs = x @ w_o + nu

        R0 = args.batch_delta * torch.eye(args.d, device=device, dtype=dtype)
        solver = FabricBatchRLS(d=args.d, R0=R0, weight=args.batch_weight,
                                max_iter=args.linear_max_iter, tol=args.linear_tol,
                                beta=args.linear_beta, device=device)
        solver.accumulate(x, d_obs)
        # Accumulate() already symmetrizes self.R; use it directly.

        # Snapshot EquilibriumSolve.last_info before the solve, then verify
        # it was replaced by a new info dict after exactly one solve call.
        info_before = EquilibriumSolve.last_info
        w_fabric, info = solver.solve()
        n_calls += 1
        if info is None:
            raise RuntimeError("LinearSolveLayer did not produce an info dict")
        if info_before is not None and info_before is EquilibriumSolve.last_info:
            raise RuntimeError(
                "FabricBatchRLS.solve() did not update EquilibriumSolve.last_info"
            )

        m = batch_experiment_metrics(w_fabric, solver.R, solver.p, x, d_obs, R0,
                                     w_o=w_o, weight=args.batch_weight)
        m['trial'] = trial
        m['settle_n_iter'] = info.get('n_iter', -1)
        m['settle_final_residual'] = info.get('final_residual', float('nan'))
        m['settle_converged'] = bool(info.get('converged', False))
        metrics_all.append(m)
        fabric_ws.append(w_fabric.detach().cpu())
        ref_ws.append(m['w_ref'])
        print(f"  trial {trial + 1}/{args.n_trials}"
              f"  settle_iters={m['settle_n_iter']}"
              f"  rel_err={m['rel_err']:.3e}"
              f"  normal_eq_res={m['normal_eq_residual']:.3e}"
              f"  plant_err={m.get('plant_error', float('nan')):.3e}",
              flush=True)

    fabric_stack = torch.stack(fabric_ws, dim=0)
    ref_stack = torch.stack(ref_ws, dim=0)
    summary = {
        'd': args.d,
        'T_block': args.T_batch,
        'sigma': args.sigma,
        'n_trials': args.n_trials,
        'batch_delta': args.batch_delta,
        'batch_weight': args.batch_weight,
        'seed': seed,
        'device': str(device),
        'solver': {
            'method': 'anderson',
            'max_iter': args.linear_max_iter,
            'tol': args.linear_tol,
            'beta': args.linear_beta,
        },
        'aggregate': {
            'rel_err_max': float(max(m['rel_err'] for m in metrics_all)),
            'rel_err_mean': float(np.mean([m['rel_err'] for m in metrics_all])),
            'abs_err_max': float(max(m['abs_err'] for m in metrics_all)),
            'normal_eq_residual_max': float(max(m['normal_eq_residual'] for m in metrics_all)),
            'plant_error_mean': float(np.mean([m.get('plant_error', float('nan'))
                                               for m in metrics_all])),
            'settle_n_iter_mean': float(np.mean([m['settle_n_iter']
                                                 for m in metrics_all])),
            'settle_n_iter_max': int(max(m['settle_n_iter']
                                          for m in metrics_all)),
            'all_converged': bool(all(m['settle_converged'] for m in metrics_all)),
            'n_equilibrium_calls': n_calls,
        },
        'trials': [{k: (float(v) if isinstance(v, (np.floating, float))
                        else int(v) if isinstance(v, (np.integer, int))
                        else bool(v) if isinstance(v, (np.bool_, bool))
                        else v)
                    for k, v in m.items()
                    if k not in ('w_ref', 'w_fabric')}
                   for m in metrics_all],
    }
    np.savez(os.path.join(out_dir, 'batch_solutions.npz'),
             w_fabric=fabric_stack.numpy(),
             w_ref=ref_stack.numpy(),
             w_o=w_o.detach().cpu().numpy())
    with open(os.path.join(out_dir, 'batch_metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  [batch] n_equilibrium_calls={n_calls}"
          f"  rel_err_mean={summary['aggregate']['rel_err_mean']:.3e}"
          f"  rel_err_max={summary['aggregate']['rel_err_max']:.3e}"
          f"  settle_iters_mean={summary['aggregate']['settle_n_iter_mean']:.2f}")
    print(f"  Wrote {out_dir}/batch_metrics.json and batch_solutions.npz")
    return summary


def run_experiment(args):
    out_dir = _ensure_dir(args.out_dir)
    torch.set_default_dtype(torch.float64)
    dtype = torch.float64

    # ---- Device selection ----
    device = _resolve_device(args.gpu, args.gpu_id)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)
        gpu_name = torch.cuda.get_device_name(device.index)
        print(f"Device: {device} ({gpu_name})")
    else:
        print(f"Device: {device}")

    # ---- Common plant ----
    g = torch.Generator(device=device).manual_seed(args.seed)
    w_o = torch.randn(args.d, generator=g, dtype=dtype, device=device)
    w_o = w_o / w_o.norm()  # unit-norm plant
    if args.batch_only:
        run_batch_experiment(args, out_dir, w_o, device, dtype, args.seed)
        return

    if args.grid_sweep:
        run_grid_sweep(args, out_dir, w_o, device, dtype, args.seed)
        return

    if args.train_robust_block:
        run_robust_block_experiment(args, out_dir, w_o, device, dtype, args.seed)
        return

    if args.train_robust:
        run_robust_experiment(args, out_dir, w_o, device, dtype, args.seed)
        return


    # ---- 4 contenders ----
    def digital_lms_factory():
        return DigitalLMS(d=args.d, mu=args.mu_lms, device=device)
    def digital_rls_factory():
        return DigitalRLS(d=args.d, lam=args.lam_rls, device=device)
    def fabric_lms_factory():
        return FabricLMS(d=args.d, mu=args.mu_lms, dt=args.dt_fabric_lms, device=device)
    def fabric_rls_factory(log_name=''):
        return FabricRLS(d=args.d, lam=args.lam_rls, R0=torch.eye(args.d, device=device),
                         max_iter=args.linear_max_iter, tol=args.linear_tol,
                         beta=args.linear_beta, device=device,
                         log_every=args.fabric_rls_log_every, log_name=log_name)

    contenders = {
        'digital_lms': digital_lms_factory,
        'digital_rls': digital_rls_factory,
        'fabric_lms': fabric_lms_factory,
        'fabric_rls': fabric_rls_factory,
    }

    # ---- Part 1: iid ----
    print(f"\n=== iid input, T={args.T}, sigma={args.sigma}, "
          f"n_trials={args.n_trials} ===", flush=True)
    results_iid = {}
    extras_iid = {}
    for name, factory in contenders.items():
        def factory_wrapped(f=factory, n=name):
            try:
                return f(log_name=n)
            except TypeError:
                return f()
        t0 = time.time()
        W_mean, _, extras = monte_carlo(factory_wrapped,
                                       w_o, args.T, args.sigma,
                                       args.n_trials, mode='iid',
                                       noise=args.noise, p_burst=args.p_burst,
                                       kappa=args.kappa,
                                       seed_base=args.seed, dtype=dtype, device=device,
                                       name=f'iid/{name}',
                                       log_every=max(1, args.n_trials // 5))
        print(f"  [{name}] total: {time.time() - t0:.2f}s", flush=True)
        results_iid[name] = (W_mean, None)
        extras_iid[name] = extras
    plot_learning_curves(out_dir, results_iid, w_o,
                          f'iid: ensemble-mean $\\|w-w_o\\|^2$ ({args.n_trials} trials)')

    # Discrepancy plot
    W_fabric_rls_mean = results_iid['fabric_rls'][0]
    W_digital_rls_mean = results_iid['digital_rls'][0]
    plot_discrepancy(out_dir, W_fabric_rls_mean, W_digital_rls_mean)

    # Settling budget: collect iterations from fabric_rls
    iters = []
    for extras in extras_iid['fabric_rls']:
        iters.extend(extras.get('last_iters', []))
    plot_settling_budget(out_dir, iters, label='LinearSolveLayer iterations per sample')
    if iters:
        print(f"  settling budget: mean={np.mean(iters):.2f}, max={max(iters)}, "
              f"median={np.median(iters):.0f}, n={len(iters)}", flush=True)

    # ---- Part 2: AR(1) correlated ----
    print(f"\n=== AR(1) input (rho=0.9), T={args.T} ===", flush=True)
    results_ar1 = {}
    for name, factory in contenders.items():
        def factory_wrapped(f=factory, n=name):
            try:
                return f(log_name=n)
            except TypeError:
                return f()
        t0 = time.time()
        W_mean, _, _ = monte_carlo(factory_wrapped,
                                   w_o, args.T, args.sigma,
                                   args.n_trials, mode='ar1',
                                   noise=args.noise, p_burst=args.p_burst,
                                   kappa=args.kappa,
                                   seed_base=args.seed, dtype=dtype, device=device,
                                   name=f'ar1/{name}',
                                   log_every=max(1, args.n_trials // 5))
        print(f"  [{name}] total: {time.time() - t0:.2f}s", flush=True)
        results_ar1[name] = (W_mean, None)
    plot_learning_curves(out_dir, results_ar1, w_o,
                          f'AR(1) $\\rho$=0.9: ensemble-mean $\\|w-w_o\\|^2$')

    # ---- Part 3: Tracking experiment ----
    print(f"\n=== Tracking experiment: time-varying w_o, sweep lambda ===", flush=True)
    lambdas = args.tracking_lambdas
    misadj_rls, misadj_fabric = [], []
    for lam in lambdas:
        def rls_factory():
            return DigitalRLS(d=args.d, lam=lam, device=device)
        def frls_factory():
            return FabricRLS(d=args.d, lam=lam, R0=torch.eye(args.d, device=device),
                             max_iter=args.linear_max_iter, tol=args.linear_tol,
                             beta=args.linear_beta, device=device,
                             log_every=0, log_name='')
        # Single trial per lambda (averaging in tracking would average over
        # the time-varying trajectory, which we want).
        x, d_obs, w_o_track = make_tracking_stream(
            w_o, args.T, args.sigma, args.process_std,
            seed=args.seed, dtype=dtype, device=device)
        t0 = time.time()
        W_rls, _ = run_trial(rls_factory(), x, d_obs, w_o_track)
        dt_d = time.time() - t0
        t0 = time.time()
        W_frls, _ = run_trial(frls_factory(), x, d_obs, w_o_track)
        dt_f = time.time() - t0
        misadj_rls.append(steady_state_misadjustment(W_rls, w_o_track))
        misadj_fabric.append(steady_state_misadjustment(W_frls, w_o_track))
        print(f"  lambda={lam}: digital RLS misadj={misadj_rls[-1]:.4e}"
              f" ({dt_d:.2f}s), fabric RLS misadj={misadj_fabric[-1]:.4e}"
              f" ({dt_f:.2f}s)", flush=True)
    plot_misadjust_vs_lambda(out_dir, lambdas, misadj_rls, misadj_fabric)

    # ---- Part 4: Ljung overlay ----
    print(f"\n=== Ljung overlay: digital LMS -> continuous LMS as mu*dt -> 0 ===")
    x_lj, d_lj = make_stream(w_o, args.T, args.sigma, mode='iid',
                              seed=args.seed, dtype=dtype, device=device)
    mu_values = [0.05, 0.05, 0.05]
    dt_values = [1.0, 0.5, 0.1]  # decreasing step sizes
    plot_ljung_overlay(out_dir, w_o, x_lj, d_lj, mu_values, dt_values)

    # ---- Save metrics ----
    metrics = {
        'd': args.d,
        'T': args.T,
        'sigma': args.sigma,
        'n_trials': args.n_trials,
        'device': str(device),
        'settling': {'mean': float(np.mean(iters)) if iters else None,
                     'max': int(max(iters)) if iters else None,
                     'median': float(np.median(iters)) if iters else None,
                     'n': len(iters)},
        'tracking': {'lambdas': lambdas,
                     'digital_rls_misadj': [float(x) for x in misadj_rls],
                     'fabric_rls_misadj': [float(x) for x in misadj_fabric]},
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nWrote metrics to {out_dir}/metrics.json")
    print(f"Figures in {out_dir}/")


# ----------------------------------------------------------------------------
# 8. Phase 1: Learned Robust IR-RLS
# ----------------------------------------------------------------------------

def train_robust_weighter(d, *, epochs=80, T=128, lam=0.99, sigma=0.01,
                          p_burst=0.02, kappa=20.0, device='cpu',
                          lr=1e-2, truncate_every=32, linear_max_iter=50,
                          linear_tol=1e-5, seed=0, log_every=10,
                          save_path=None, return_history=False,
                          backward_mode='exact'):
    """Train a ``LearnedRobustWeighter`` end-to-end on impulsive noise.

    Per plan decision 7 (and KIMI correction #1):
      * Per-epoch randomized plant w_o (unit norm).
      * T=128 samples, prequential MSE
          L_pre  = sum_t (d_{t+1} - w_t^T x_{t+1})^2
        plus a terminal loss
          L_term = 10 * ||w_T - w_o||^2.
      * Truncation every ``truncate_every`` samples detaches R, p, AND w
        (decision 4, finding D).
      * Adam optimizer; raw_c init = -2.25 (c ~ 0.107, putting the knee
        between nominal sigma ~ 0.01 and burst magnitude ~ kappa * sigma
        ~ 0.2; KIMI correction #1).

    Precision policy (0b / 8): float32 CPU throughout, solver tol 1e-5,
    max_iter ~ 50.

    Returns the trained weighter (in ``eval()`` mode).  When
    ``return_history=True``, also returns a list of dicts with per-epoch
    metrics (loss, terminal-error, c, alpha).
    """
    # Lazy import to avoid the circular dependency (plan decision 9).
    from utils.learned_robust import FabricRobustRLS, LearnedRobustWeighter

    torch.manual_seed(seed)
    weighter = LearnedRobustWeighter(c_init=_DEFAULT_C_INIT, alpha_init=_DEFAULT_ALPHA_INIT).to(device)
    # Precision policy (0b/8): training runs float32 CPU.  Cast the params
    # explicitly -- ``run_experiment`` has already set the default dtype to
    # float64, which would otherwise mint a float64 weighter and silently
    # promote the whole training graph (the stream is float32).
    weighter = weighter.to(torch.float32)
    optimizer = torch.optim.Adam(weighter.parameters(), lr=lr)
    history = []

    for epoch in range(epochs):
        # Random unit-norm plant per epoch; orthogonal adversarial-physics
        # to the burst signal so the weighter learns the *physics* of
        # bursting, not w_o itself.
        w_o = torch.randn(d, device=device, dtype=torch.float32)
        w_o = w_o / w_o.norm()

        # Train in float32 (decision 0b / 8).
        x, d_obs = make_stream(w_o, T, sigma, mode='iid',
                               noise='impulsive', p_burst=p_burst,
                               kappa=kappa, seed=seed + epoch,
                               dtype=torch.float32, device=device)

        R0 = torch.eye(d, device=device, dtype=torch.float32)
        filter = FabricRobustRLS(d=d, weighter=weighter, lam=lam,
                                  R0=R0,
                                  max_iter=linear_max_iter, tol=linear_tol,
                                  beta='chebyshev', device=device,
                                  training=True, log_every=0,
                                  backward_mode=backward_mode)

        loss_preq = torch.zeros((), device=device)
        for t in range(T):
            # Prequential error: w_t (with graph) predicts sample t.
            e_t = d_obs[t] - filter.w @ x[t]
            loss_preq = loss_preq + e_t.pow(2)
            # Update filter state for the next sample.
            filter.step(x[t], d_obs[t])
            # Truncated BPTT: detach R, p, AND w
            # (decision 4; the e_prior path threads the old graph through w).
            if (t + 1) % truncate_every == 0:
                filter.w = filter.w.detach()
                filter.R = filter.R.detach()
                filter.p = filter.p.detach()
        # ``loss_preq`` keeps its grad_fn through the last ``truncate_every``
        # steps only (the most recent chunk).  Earlier chunks are detached
        # at the next ``filter.w.detach()`` boundary because the next
        # ``e_t`` then loses its grad_fn.  The implicit gradient (mode
        # ``backward_mode``, default 'exact' per the 2026-08-13 fix: the
        # phantom VJP underflows to zero through the chained settles at
        # production scale) is cheap enough that the chain through the
        # last chunk is tractable.

        loss_term = 10.0 * (filter.w - w_o).pow(2).sum()
        loss = loss_preq + loss_term

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            c_here = weighter.c.item()
            alpha_here = weighter.alpha.item()
        rec = {
            'epoch': epoch,
            'loss': float(loss.item()),
            'loss_preq': float(loss_preq.item()),
            'loss_term': float(loss_term.item()),
            'c': float(c_here),
            'alpha': float(alpha_here),
        }
        history.append(rec)
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"  [train_robust] epoch {epoch:4d}"
                  f"  loss={rec['loss']:.4e}"
                  f"  preq={rec['loss_preq']:.4e}"
                  f"  term={rec['loss_term']:.4e}"
                  f"  c={c_here:.4f}  alpha={alpha_here:.4f}",
                  flush=True)

    if save_path is not None:
        torch.save(weighter.state_dict(), save_path)
        print(f"  [train_robust] saved weighter to {save_path}")

    weighter.eval()
    if return_history:
        return weighter, history
    return weighter


def _psi_curve(weighter, e_grid):
    """Return (v, psi) over an e-grid; psi(e) = v(e) * e (the influence
    *  prior error, the per-sample pseudo-residual weighting)."""
    with torch.no_grad():
        v = weighter(e_grid)
        psi = v * e_grid
    return v.detach().cpu().numpy(), psi.detach().cpu().numpy()


def plot_robust_learning_curves(out_dir, results, w_o, suptitle):
    """4 contenders under impulsive noise: ensemble-mean ||w_t - w_o||^2."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    T = next(iter(results.values()))[0].shape[0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {'digital_rls': 'C1',
              'fabric_rls': 'C3',
              'digital_robust_rls': 'C4',
              'fabric_robust_rls': 'C5'}
    styles = {'digital_rls': '-',
              'fabric_rls': '--',
              'digital_robust_rls': '-',
              'fabric_robust_rls': '--'}
    for name, (W_mean, _) in results.items():
        err = w_err_sq(W_mean, w_o).cpu().numpy()
        ax.semilogy(np.arange(T), err,
                    label=name, color=colors.get(name, 'k'),
                    linestyle=styles.get(name, '-'), linewidth=1.5)
    ax.set_xlabel('sample t')
    ax.set_ylabel(r'$\|w_t - w_o\|^2$')
    ax.set_title(suptitle)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'robust_learning_curves.png'), dpi=140)
    plt.close(fig)


def plot_robust_discrepancy(out_dir, W_fabric_robust, W_digital_robust):
    """||w_fabric_robust - w_digital_robust|| / ||w_digital_robust|| vs t."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    T = W_fabric_robust.shape[0]
    disc = relative_discrepancy(W_fabric_robust, W_digital_robust).cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(np.arange(T), disc, color='C5', linewidth=1.5)
    ax.set_xlabel('sample t')
    ax.set_ylabel(r'$\|w_{\mathrm{fabric\,robust}} - w_{\mathrm{digital\,robust}}\|/\|\cdot\|$')
    ax.set_title('Fabric-robust vs Digital-robust discrepancy')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'robust_discrepancy.png'), dpi=140)
    plt.close(fig)


def plot_robust_influence(out_dir, e_grid_np, v_init, psi_init, v_trained, psi_trained):
    """Two-panel plot: v(e) and psi(e) = v(e) * e (the influence*error
    pseudo-residual) overlay of init vs trained weighter."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(e_grid_np, v_init, '--', color='C0', label='init v(e)')
    axes[0].plot(e_grid_np, v_trained, '-', color='C3', label='trained v(e)')
    axes[0].set_xlabel('e_prior')
    axes[0].set_ylabel(r'$v(e)$')
    axes[0].set_title('Influence v(e)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(e_grid_np, psi_init, '--', color='C0', label=r'init $\psi(e) = v(e) \cdot e$')
    axes[1].plot(e_grid_np, psi_trained, '-', color='C3', label=r'trained $\psi(e)$')
    axes[1].set_xlabel('e_prior')
    axes[1].set_ylabel(r'$\psi(e) = v(e) \cdot e$')
    axes[1].set_title('Influence-weighted error')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Learned robust weighter (init vs trained)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'robust_influence.png'), dpi=140)
    plt.close(fig)


def run_robust_experiment(args, out_dir, w_o, device, dtype, seed):
    """Phase 1 robust experiment: 4 contenders under impulsive noise
    with a frozen (or freshly trained) weighter.

    Plots:
      * robust_learning_curves.png  -- 4 contenders
      * robust_discrepancy.png      -- fabric-robust vs digital-robust
      * settling_budget.png         -- fabric-robust settle-iter histogram
      * robust_influence.png        -- v(e) and psi(e), init vs trained
    """
    # Lazy import (plan decision 9).
    from utils.learned_robust import (
        FabricRobustRLS, DigitalRobustRLS,
        LearnedRobustWeighter, constant_weighter,
    )

    # ---- Train or load weighter ----
    weighter_train = LearnedRobustWeighter(c_init=_DEFAULT_C_INIT, alpha_init=_DEFAULT_ALPHA_INIT)
    if args.robust_weighter_path and os.path.exists(args.robust_weighter_path):
        sd = torch.load(args.robust_weighter_path, map_location='cpu')
        weighter_train.load_state_dict(sd)
        print(f"  [robust] loaded trained weighter from {args.robust_weighter_path}")
    else:
        print(f"  [robust] training weighter: epochs={args.robust_epochs},"
              f" T={args.robust_T}, sigma={args.sigma}, p_burst={args.p_burst},"
              f" kappa={args.kappa}, lr={args.robust_lr}, truncate={args.truncate},"
              f" backward={args.robust_backward_mode}")
        weighter_train, history = train_robust_weighter(
            d=args.d, epochs=args.robust_epochs, T=args.robust_T,
            lam=args.lam_rls, sigma=args.sigma, p_burst=args.p_burst,
            kappa=args.kappa, device=device, lr=args.robust_lr,
            truncate_every=args.truncate,
            seed=seed, log_every=max(1, args.robust_epochs // 8),
            return_history=True, backward_mode=args.robust_backward_mode)
        # The trainer's solver defaults (max_iter=50, tol=1e-5) are the
        # documented float32 training policy (0b/8); we deliberately do NOT
        # pass args.linear_max_iter/linear_tol here -- those are the float64
        # eval tolerances (1e-8) used by the frozen-weighter contenders below.
        with open(os.path.join(out_dir, 'robust_training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        # Save the trained weighter for reuse.
        save_path = os.path.join(out_dir, 'robust_weighter.pt')
        torch.save(weighter_train.state_dict(), save_path)
        print(f"  [robust] saved trained weighter to {save_path}")

    # Cast frozen weighter to eval dtype (decision 10: train float32, eval float64).
    weighter = weighter_train.to(device).to(dtype)
    weighter.eval()
    const_w = constant_weighter(1.0).to(device).to(dtype)

    # ---- 4 contenders ----
    def digital_rls_factory():
        return DigitalRLS(d=args.d, lam=args.lam_rls, device=device)

    def fabric_rls_factory(log_name=''):
        return FabricRLS(d=args.d, lam=args.lam_rls,
                         R0=torch.eye(args.d, device=device, dtype=dtype),
                         max_iter=args.linear_max_iter, tol=args.linear_tol,
                         beta=args.linear_beta, device=device,
                         log_every=0, log_name=log_name)

    def digital_robust_factory():
        # v==1 (constant weighter) -> byte-identical to DigitalRLS.
        return DigitalRobustRLS(d=args.d, weighter=const_w, lam=args.lam_rls,
                                device=device, training=False)

    def fabric_robust_factory(log_name=''):
        return FabricRobustRLS(d=args.d, weighter=weighter, lam=args.lam_rls,
                                R0=torch.eye(args.d, device=device, dtype=dtype),
                                max_iter=args.linear_max_iter, tol=args.linear_tol,
                                beta=args.linear_beta, device=device,
                                training=False, log_every=0,
                                log_name=log_name)

    contenders = {
        'digital_rls': digital_rls_factory,
        'fabric_rls': fabric_rls_factory,
        'digital_robust_rls': digital_robust_factory,
        'fabric_robust_rls': fabric_robust_factory,
    }

    print(f"\n=== Robust experiment: iid input, noise={args.noise}, "
          f"T={args.T}, sigma={args.sigma}, n_trials={args.n_trials} ===",
          flush=True)
    results = {}
    extras_all = {}
    for name, factory in contenders.items():
        def factory_wrapped(f=factory, n=name):
            try:
                return f(log_name=n)
            except TypeError:
                return f()
        t0 = time.time()
        W_mean, _, extras = monte_carlo(factory_wrapped,
                                       w_o, args.T, args.sigma,
                                       args.n_trials, mode='iid',
                                       noise=args.noise, p_burst=args.p_burst,
                                       kappa=args.kappa,
                                       seed_base=seed, dtype=dtype, device=device,
                                       name=f'robust/{name}',
                                       log_every=max(1, args.n_trials // 5))
        print(f"  [{name}] total: {time.time() - t0:.2f}s", flush=True)
        results[name] = (W_mean, None)
        extras_all[name] = extras

    plot_robust_learning_curves(out_dir, results, w_o,
                                rf'Robust: ensemble-mean $\|w-w_o\|^2$ '
                                f'({args.n_trials} trials, noise={args.noise})')
    plot_robust_discrepancy(out_dir, results['fabric_robust_rls'][0],
                            results['digital_robust_rls'][0])

    iters = []
    for extras in extras_all['fabric_robust_rls']:
        iters.extend(extras.get('last_iters', []))
    if iters:
        plot_settling_budget(out_dir, iters,
                             label='Fabric-robust LinearSolveLayer iterations per sample')
        print(f"  [fabric_robust] settling budget: mean={np.mean(iters):.2f}, "
              f"max={max(iters)}, median={np.median(iters):.0f}, n={len(iters)}",
              flush=True)

    # ---- Influence curve: init vs trained overlay ----
    e_grid = torch.linspace(-0.5, 0.5, 401, device=device, dtype=dtype)
    init_w = LearnedRobustWeighter(c_init=_DEFAULT_C_INIT, alpha_init=_DEFAULT_ALPHA_INIT).to(device).to(dtype)
    init_w.eval()
    v_init, psi_init = _psi_curve(init_w, e_grid)
    v_trained, psi_trained = _psi_curve(weighter, e_grid)
    plot_robust_influence(out_dir, e_grid.cpu().numpy(),
                          v_init, psi_init, v_trained, psi_trained)

    # ---- Save metrics ----
    metrics = {
        'd': args.d,
        'T': args.T,
        'sigma': args.sigma,
        'noise': args.noise,
        'p_burst': args.p_burst,
        'kappa': args.kappa,
        'n_trials': args.n_trials,
        'device': str(device),
        'training': {
            'epochs': args.robust_epochs,
            'T': args.robust_T,
            'truncate_every': args.truncate,
            'lr': args.robust_lr,
        },
        'weighter': {
            'c': float(weighter.c.item()),
            'alpha': float(weighter.alpha.item()),
        },
        'settling': {'mean': float(np.mean(iters)) if iters else None,
                     'max': int(max(iters)) if iters else None,
                     'median': float(np.median(iters)) if iters else None,
                     'n': len(iters)},
    }
    with open(os.path.join(out_dir, 'robust_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  [robust] Wrote metrics to {out_dir}/robust_metrics.json")
    print(f"  [robust] trained weighter: c={metrics['weighter']['c']:.4f}, "
          f"alpha={metrics['weighter']['alpha']:.4f}")
    print(f"  Figures in {out_dir}/")


def parse_args():
    parser = argparse.ArgumentParser(description='Streaming RLS adaptive-filtering demo')
    parser.add_argument('--out_dir', type=str, default='./results/rls_demo')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--d', type=int, default=8)
    parser.add_argument('--T', type=int, default=500)
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--mu_lms', type=float, default=0.05)
    parser.add_argument('--lam_rls', type=float, default=0.99)
    parser.add_argument('--dt_fabric_lms', type=float, default=0.1)
    parser.add_argument('--linear_max_iter', type=int, default=100)
    parser.add_argument('--linear_tol', type=float, default=1e-8)
    parser.add_argument('--linear_beta', type=str, default='chebyshev',
                        help="Beta for LinearSolveLayer: 'chebyshev' for auto "
                             "Chebyshev step, or a float for fixed step.")
    parser.add_argument('--process_std', type=float, default=1e-3)
    parser.add_argument('--tracking_lambdas', type=float, nargs='+',
                        default=[0.90, 0.95, 0.98, 0.99, 0.995, 0.999])
    parser.add_argument('--gpu', action='store_true',
                        help="Use CUDA. Default is CPU. Pairs with --gpu_id to select "
                             "which GPU (default 0).")
    parser.add_argument('--gpu_id', type=int, default=0,
                        help="GPU index when --gpu is set. Ignored otherwise.")
    parser.add_argument('--fabric_rls_log_every', type=int, default=0,
                        help="Log FabricRLS settling-iter count every N samples "
                             "(0 = silent, default). Useful for diagnosing solver cost.")
    parser.add_argument('--batch_only', action='store_true',
                        help="Run only the block/batch least-squares experiment "
                             "(single settle via LinearSolveLayer vs torch.linalg.lstsq). "
                             "Skips the streaming iid/AR(1)/tracking/Ljung experiments.")
    parser.add_argument('--T_batch', type=int, default=512,
                        help="Block size for the batch experiment (number of samples "
                             "accumulated into one normal-equation system).")
    parser.add_argument('--batch_delta', type=float, default=1e-2,
                        help="Identity regularization strength for the batch "
                             "experiment's R0 = delta * I.  Ensures R is SPD and "
                             "matches between fabric solve and the direct "
                             "lstsq reference.")
    parser.add_argument('--batch_weight', type=float, default=1.0,
                        help="Uniform per-sample weight for batch accumulation.  "
                             "1.0 reproduces the unweighted normal equations; "
                             "the solved problem becomes argmin weight*||Xw-d||^2 "
                             "+ w^T R0 w (a uniform-weight least-squares variant, "
                             "not a forgetting factor).")
    # ---- Phase 1: noise plumbing ----
    parser.add_argument('--noise', type=str, default='gaussian',
                        choices=['gaussian', 'impulsive'],
                        help="Noise model for make_stream (plan decision 6). "
                             "'gaussian' is the legacy default; 'impulsive' is "
                             "the contaminated-Gaussian mixture used to train "
                             "and evaluate the robust weighter.")
    parser.add_argument('--p_burst', type=float, default=0.02,
                        help="Burst probability for impulsive noise (default 0.02).")
    parser.add_argument('--kappa', type=float, default=20.0,
                        help="Burst magnitude multiplier for impulsive noise "
                             "(default 20.0; burst draws ~ kappa * sigma * randn).")
    # ---- Phase 1: robust weighter training / eval ----
    parser.add_argument('--train_robust', action='store_true',
                        help="Run the Phase 1 robust experiment (train weighter "
                             "if --robust_weighter_path is not supplied, then "
                             "evaluate 4 contenders under --noise).")
    parser.add_argument('--robust_epochs', type=int, default=80,
                        help="Number of epochs to train the robust weighter "
                             "(plan decision 7; default 80).")
    parser.add_argument('--robust_T', type=int, default=128,
                        help="Stream length per training epoch (default 128).")
    parser.add_argument('--truncate', type=int, default=32,
                        help="Truncated BPTT window: detach R, p, AND w every "
                             "this many samples (plan decision 4; default 32).")
    parser.add_argument('--robust_lr', type=float, default=1e-2,
                        help="Adam learning rate for the robust weighter (default 1e-2).")
    parser.add_argument('--robust_weighter_path', type=str, default=None,
                        help="Optional path to a trained weighter state-dict. "
                             "If supplied and the file exists, skip training and "
                             "use the loaded weighter for evaluation.")
    parser.add_argument('--robust_backward_mode', type=str, default='exact',
                        choices=['phantom', 'exact'],
                        help="Implicit gradient mode for streaming robust "
                             "training. 'exact' (default) is the correct CG "
                             "adjoint; 'phantom' is the cheap Geng et al. "
                             "approximation that underflows to zero through "
                             "chained settles at scale (2026-08-13 fix).")
    # ---- Phase 1.5: block robust IRLS ----
    parser.add_argument('--train_robust_block', action='store_true',
                        help="Run the Phase 1.5 block robust experiment: train weighter "
                             "end-to-end on (X, d_obs) block regression with K outer "
                             "IRLS iterations, then evaluate contenders "
                             "(plain batch LS, learned block IRLS, init-only block IRLS, "
                             "streaming fabric_robust_rls).")
    parser.add_argument('--block_N', type=int, default=512,
                        help="Block size N for the block robust experiment (default 512).")
    parser.add_argument('--block_K', type=int, default=8,
                        help="Number of outer IRLS iterations K (default 8).")
    parser.add_argument('--block_delta', type=float, default=1e-2,
                        help="Identity regularization for the block R0 = delta * I "
                             "(default 1e-2).")
    parser.add_argument('--block_epochs', type=int, default=80,
                        help="Number of epochs to train the block weighter (default 80).")
    parser.add_argument('--block_lr', type=float, default=1e-2,
                        help="Adam learning rate for the block weighter (default 1e-2).")
    parser.add_argument('--block_backward_mode', type=str, default='exact',
                        choices=['phantom', 'exact'],
                        help="Implicit gradient mode for block robust "
                             "training. 'exact' (default) is the correct CG "
                             "adjoint; 'phantom' is the cheap Geng et al. "
                             "approximation that underflows to exactly zero "
                             "through the chained-K settles in float32 at "
                             "production scale, silently freezing training "
                             "(2026-08-13 fix).")
    parser.add_argument('--block_solver_backend', type=str, default='dense',
                        choices=['dense', 'circuit_stamp'],
                        help="Block IRLS backend. 'dense' (default) is the "
                             "legacy path that forms R and p every outer "
                             "iteration and solves R w = p via "
                             "LinearSolveLayer. 'circuit_stamp' uses the "
                             "element-level KCL residual from "
                             "TransformerConductanceBank + LeakageToGround "
                             "and solves f(w) = 0 via "
                             "WeightedGramCircuitSolve (equilibria, not "
                             "direct linear solves).  'circuit_stamp' is "
                             "algebraically identical to 'dense' for the "
                             "weighted-Gram IRLS objective and is the "
                             "circuit-aware backend documented in "
                             "docs/weighted_gram_circuit.md.")
    parser.add_argument('--block_weighter_path', type=str, default=None,
                        help="Optional path to a trained block weighter state-dict. "
                             "If supplied and the file exists, skip training and use "
                             "the loaded weighter for evaluation.")
    parser.add_argument('--block_K_max', type=int, nargs='+', default=None,
                        help="K values for the K-sweep (default 0 1 2 4 8 16).")
    parser.add_argument('--block_N_max', type=int, nargs='+', default=None,
                        help="N values for the N-sweep (default 32 64 128 256 512).")
    # ---- Grid sweep: (c, alpha) family-vs-training decomposition ----
    parser.add_argument('--grid_sweep', action='store_true',
                        help="Run the (c, alpha) fixed-curve grid sweep at the "
                             "block config: dense log-grid over the Cauchy curve "
                             "parameters, compared against plain / trained / "
                             "oracle on the SAME blocks (k_sweep fixed-block "
                             "protocol).  Verdict: TRAINING (grid-best ~ oracle) "
                             "vs FAMILY (grid-best >> oracle).")
    parser.add_argument('--grid_c_lo', type=float, default=0.01,
                        help="Low c for the (c, alpha) grid (log-spaced; "
                             "default 0.01, below the burst scale kappa*sigma).")
    parser.add_argument('--grid_c_hi', type=float, default=0.5,
                        help="High c for the grid (Phase D1: default 0.5; "
                             "the c=1.0 boundary in the legacy grid is past "
                             "any reasonable Cauchy knee).")
    parser.add_argument('--grid_alpha_lo', type=float, default=0.05,
                        help="Low alpha for the grid (log-spaced; default 0.05).")
    parser.add_argument('--grid_alpha_hi', type=float, default=100.0,
                        help="High alpha for the grid (Phase D1: default "
                             "raised to 100; the alpha=20 boundary in the "
                             "legacy grid was an artifact of insufficient "
                             "range).")
    parser.add_argument('--grid_n_alpha', type=int, default=36,
                        help="Number of alpha values in the dense grid "
                             "(Phase D1: default 36; ~900-1600 fixed curves "
                             "is feasible on GPU and likely feasible CPU).")
    parser.add_argument('--grid_n_c', type=int, default=36,
                        help="Number of c values in the dense grid "
                             "(Phase D1: default 36).")
    # ---- Phase A: training-hardening CLI (A1) ----
    parser.add_argument('--block_train_updates', type=int, default=200,
                        help="Number of optimizer updates in the hardened "
                             "block trainer (Phase A1: 200-2000 typical; "
                             "default 200).")
    parser.add_argument('--block_train_blocks_per_update', type=int, default=8,
                        help="Independent blocks per gradient-accumulation "
                             "update (Phase A1 + A3: 8-16 typical; default 8).")
    parser.add_argument('--block_train_dtype', type=str, default='float32',
                        choices=['float32', 'float64'],
                        help="Training dtype (Phase A1: float64 for the "
                             "diagnostic, float32 for production).")
    parser.add_argument('--block_train_linear_tol', type=float, default=1e-6,
                        help="Forward linear-solver tolerance (Phase A1).")
    parser.add_argument('--block_train_linear_max_iter', type=int, default=200,
                        help="Forward linear-solver max iterations "
                             "(Phase A1).")
    parser.add_argument('--block_train_backward_tol', type=float, default=None,
                        help="Backward implicit-solver tolerance (Phase A1). "
                             "Defaults to --block_train_linear_tol.")
    parser.add_argument('--block_train_backward_max_iter', type=int, default=None,
                        help="Backward implicit-solver max iterations "
                             "(Phase A1). Defaults to "
                             "--block_train_linear_max_iter.")
    parser.add_argument('--block_train_solver', type=str, default='deq',
                        choices=['deq', 'direct'],
                        help="Inner solver for the hardened trainer "
                             "(Phase C5 diagnostic: 'direct' uses "
                             "torch.linalg.solve and bypasses the "
                             "equilibrium solver).")
    parser.add_argument('--block_train_seed', type=int, default=0,
                        help="Training seed (Phase A1).")
    # ---- Phase C2: per-coordinate learning rates ----
    parser.add_argument('--block_train_lr_c', type=float, default=3e-3,
                        help="Adam learning rate for log_c (Phase C2).")
    parser.add_argument('--block_train_lr_alpha', type=float, default=3e-3,
                        help="Adam learning rate for log_alpha (Phase C2).")
    # ---- Phase C4: multistart initial condition ----
    parser.add_argument('--block_train_start', type=str, default='A',
                        choices=['A', 'B', 'C', 'D', 'E'],
                        help="Multistart initial condition for the "
                             "hardened trainer (Phase C4). A=(0.10, 0.13) "
                             "default; B=(0.05, 0.13); C=(0.10, 1.0); "
                             "D=(0.18, 1.0); E=(0.18, 5.0).")
    parser.add_argument('--block_train_out_dir', type=str, default=None,
                        help="Output directory for the hardened trainer "
                             "(plan required artifact layout: writes "
                             "config.json, training_history.json, "
                             "solver_stats.json, final_weighter.pt, "
                             "final_weighter_metadata.json).")
    return parser.parse_args()




# ----------------------------------------------------------------------------
# 9. Phase 1.5: Block Robust IRLS (training + experiment)
# ----------------------------------------------------------------------------


def train_robust_weighter_block(d, *, N=512, K=8, delta=1e-2, epochs=80,
                                  sigma=0.01, p_burst=0.02, kappa=20.0,
                                  device='cpu', lr=1e-2, linear_max_iter=50,
                                  linear_tol=1e-5, seed=0, log_every=10,
                                  save_path=None, return_history=False,
                                  backward_mode='exact',
                                  backend='dense'):
    """Train a ``LearnedRobustWeighter`` end-to-end in *block* mode.

    Per plan decision 1 (Phase 1.5):
      * Per-epoch randomized plant w_o (unit norm).
      * Block size N=512, K=8 outer IRLS iterations.
      * Loss supervises against the noiseless block symbols
        ``X @ w_o`` (free in simulation; plan decision 1):
            loss = ||X w^K - X w_o||^2  +  10 * ||w^K - w_o||^2
      * Float32 CPU throughout (decision 0b / 8).
      * K settles with exact implicit backward by default (2026-08-13 fix:
        the phantom VJP is numerically unstable through the chained-K
        settles and underflows to exactly zero in float32 at production
        scale d=16/N=512/K=8, silently freezing training; the exact CG
        adjoint is sane at every size tested).  ``backward_mode='phantom'``
        remains available for comparison via ``--block_backward_mode``.
      * No truncation needed: K is small (default 8) so backprop
        through all K settles is cheap (plan decision 2).

    Returns the trained weighter (in ``eval()`` mode).  When
    ``return_history=True`` also returns a list of dicts with per-epoch
    metrics (loss, plant_error, c, alpha).
    """
    # Lazy import (plan decision 9).
    from utils.learned_robust import (
        block_robust_rls, LearnedRobustWeighter, make_block,
    )

    torch.manual_seed(seed)
    weighter = LearnedRobustWeighter(c_init=_DEFAULT_C_INIT, alpha_init=_DEFAULT_ALPHA_INIT).to(device)
    # Precision policy (0b/8): training runs float32 CPU.  Cast explicitly
    # (see train_robust_weighter): run_experiment's default-dtype float64
    # would otherwise mint a float64 weighter and promote the whole graph.
    weighter = weighter.to(torch.float32)
    optimizer = torch.optim.Adam(weighter.parameters(), lr=lr)
    history = []

    for epoch in range(epochs):
        # Random unit-norm plant per epoch (KIMI audit #1 keeps the
        # weighter focused on the burst physics, not on a particular w_o).
        w_o = torch.randn(d, device=device, dtype=torch.float32)
        w_o = w_o / w_o.norm()

        # Block regressor and noisy observations (float32 training).
        X, d_obs = make_block(w_o, N, sigma, mode='iid',
                              noise='impulsive', p_burst=p_burst,
                              kappa=kappa, seed=seed + epoch,
                              dtype=torch.float32, device=device)

        # Run K+1 settles: 1 plain batch LS + K IRLS outer iterations.
        w_K = block_robust_rls(X, d_obs, weighter, delta=delta, K=K,
                               max_iter=linear_max_iter, tol=linear_tol,
                               backward_mode=backward_mode,
                               backend=backend)

        # Loss: ||X w^K - X w_o||^2 + 10 * ||w^K - w_o||^2.
        loss_data = (X @ w_K - X @ w_o).pow(2).sum()
        loss_term = 10.0 * (w_K - w_o).pow(2).sum()
        loss = loss_data + loss_term

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            c_here = weighter.c.item()
            alpha_here = weighter.alpha.item()
        rec = {
            'epoch': epoch,
            'loss': float(loss.item()),
            'loss_data': float(loss_data.item()),
            'loss_term': float(loss_term.item()),
            'plant_error': float((w_K - w_o).pow(2).sum().item()),
            'data_error': float((X @ w_K - X @ w_o).pow(2).sum().item() / N),
            'c': float(c_here),
            'alpha': float(alpha_here),
        }
        history.append(rec)
        if log_every and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"  [train_block] epoch {epoch:4d}"
                  f"  loss={rec['loss']:.4e}"
                  f"  data={rec['data_error']:.4e}"
                  f"  plant={rec['plant_error']:.4e}"
                  f"  c={c_here:.4f}  alpha={alpha_here:.4f}",
                  flush=True)

    if save_path is not None:
        torch.save(weighter.state_dict(), save_path)
        print(f"  [train_block] saved weighter to {save_path}")

    weighter.eval()
    if return_history:
        return weighter, history
    return weighter


# ----------------------------------------------------------------------------
# 9b. Phase A/C: training-hardened block IRLS trainer.
# ----------------------------------------------------------------------------


# Phase C4 multistart initial conditions (c_init, alpha_init).
MULTISTART_INITS = {
    'A': (0.10, 0.13),    # existing/default (KIMI correction #1)
    'B': (0.05, 0.13),
    'C': (0.10, 1.0),
    'D': (0.18, 1.0),
    'E': (0.18, 5.0),
}


def _print_block_training_header(*, updates, blocks_per_update, K, N, dtype,
                                  solver, linear_tol, linear_max_iter,
                                  backward_tol, backward_max_iter,
                                  backward_mode, lr_c, lr_alpha, start_name):
    """Visible run header (Phase C1)."""
    print("Block IRLS training:")
    print(f"  backward_mode: {backward_mode}")
    print(f"  solver: {solver.upper()}")
    print(f"  dtype: {dtype}")
    print(f"  blocks/update: {blocks_per_update}")
    print(f"  updates: {updates}")
    print(f"  K: {K}, N: {N}")
    print(f"  linear_tol: {linear_tol}, linear_max_iter: {linear_max_iter}")
    print(f"  backward_tol: {backward_tol}, "
          f"backward_max_iter: {backward_max_iter}")
    print(f"  lr_c: {lr_c}, lr_alpha: {lr_alpha}")
    print(f"  start: {start_name}")


def train_robust_weighter_hardened(
        d, *, N=512, K=8, delta=1e-2,
        updates=200, blocks_per_update=8,
        dtype=torch.float32,
        linear_tol=1e-6, linear_max_iter=200,
        backward_tol=None, backward_max_iter=None,
        backward_mode='exact', solver='deq',
        start='A', seed=0,
        lr_c=3e-3, lr_alpha=3e-3,
        sigma=0.01, p_burst=0.02, kappa=20.0,
        device='cpu', log_every=10, save_dir=None,
        cap_hit_warn_threshold=0.01, validation_blocks=32,
        validation_seed=10000):
    """Training-hardened block IRLS trainer (Phases A1-A4 + C2 + C4).

    Implements:
      * gradient accumulation over ``blocks_per_update`` independent
        blocks per optimizer step (Phase A3);
      * per-update history records (Phase A2): update, loss_mean, c,
        alpha, grad_norm_log_c, grad_norm_log_alpha, solver_n_iter_mean,
        solver_residual_max, solver_cap_hit_fraction;
      * solver-quality recording via ``summarize_settles`` (Phase A4):
        if the cap-hit fraction exceeds ``cap_hit_warn_threshold`` on
        float64, a ``UserWarning`` is emitted (production float32 simply
        records the rate without aborting);
      * per-coordinate Adam param groups (Phase C2) so that the two
        learned coordinates can move at different speeds;
      * exact backward only (Phase C1): ``backward_mode='phantom'``
        triggers a loud warning and the run is still executed for
        diagnostic purposes;
      * direct-solve option (Phase C5): ``solver='direct'`` swaps the
        equilibrium settle for ``torch.linalg.solve`` so the diagnostic
        can isolate solver accuracy from training geometry.

    The clean-target loss (Phase assumption 4) is
    ``mse(X w_hat, X w_true)``, NOT noisy ``d`` -- fitting impulses
    directly would be self-defeating.

    Returns the trained weighter in ``eval()`` mode, plus the history.
    """
    # Lazy import (plan decision 9).
    from utils.learned_robust import (
        block_robust_rls, LearnedRobustWeighter, make_block,
    )

    # Resolve start to (c_init, alpha_init).
    if isinstance(start, str):
        if start not in MULTISTART_INITS:
            raise ValueError(
                f"unknown multistart label: {start!r}; "
                f"valid: {sorted(MULTISTART_INITS)}")
        c_init, alpha_init = MULTISTART_INITS[start]
    else:
        c_init, alpha_init = start  # tuple of two floats

    if backward_tol is None:
        backward_tol = linear_tol
    if backward_max_iter is None:
        backward_max_iter = linear_max_iter

    if backward_mode != 'exact':
        import warnings as _w
        _w.warn(
            "WARNING: phantom backward is known to be catastrophically biased "
            "for chained block IRLS and is unsuitable for optimization claims.",
            UserWarning,
        )

    _print_block_training_header(
        updates=updates, blocks_per_update=blocks_per_update, K=K, N=N,
        dtype=str(dtype).replace('torch.', ''),
        solver=solver, linear_tol=linear_tol, linear_max_iter=linear_max_iter,
        backward_tol=backward_tol, backward_max_iter=backward_max_iter,
        backward_mode=backward_mode, lr_c=lr_c, lr_alpha=lr_alpha,
        start_name=(start if isinstance(start, str)
                    else f"({c_init:.4g}, {alpha_init:.4g})"),
    )

    torch.manual_seed(seed)
    weighter = LearnedRobustWeighter(c_init=c_init, alpha_init=alpha_init)
    weighter = weighter.to(device).to(dtype)

    # Per-coordinate Adam param groups (Phase C2).
    optimizer = torch.optim.Adam([
        {'params': [weighter.log_c], 'lr': lr_c},
        {'params': [weighter.log_alpha], 'lr': lr_alpha},
    ], betas=(0.9, 0.999))

    history = []
    validation_history = []
    block_idx = 0
    is_float64 = (dtype == torch.float64)
    validation_blocks_fixed = build_validation_block_set(
        d, N, K, delta, sigma, p_burst, kappa, validation_blocks,
        base_seed=validation_seed, dtype=dtype, device=device)

    for upd in range(updates):
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), device=device, dtype=dtype)
        all_settle_iters = []
        all_settle_info = []

        for q in range(blocks_per_update):
            w_o = torch.randn(d, device=device, dtype=dtype)
            w_o = w_o / w_o.norm()
            X, d_obs = make_block(w_o, N, sigma, dtype=dtype,
                                  mode='iid', noise='impulsive',
                                  p_burst=p_burst, kappa=kappa,
                                  seed=seed + 1009 * (upd + 1) + q,
                                  device=device)
            settle_log = []
            w_hat = block_robust_rls(
                X, d_obs, weighter, delta=delta, K=K,
                max_iter=linear_max_iter, tol=linear_tol,
                backward_mode=backward_mode,
                settle_log=settle_log,
                settle_info_log=all_settle_info,
                backward_tol=backward_tol,
                backward_max_iter=backward_max_iter,
                backend='dense',
                solver=solver,
            )
            clean_target = X @ w_o
            block_loss = ((X @ w_hat - clean_target).pow(2)).mean()
            # Average gradients across blocks; free each block's outer
            # graph after its backward so memory stays bounded.
            (block_loss / blocks_per_update).backward()
            loss_sum = loss_sum + block_loss.detach()
            all_settle_iters.extend(settle_log)
            block_idx += 1

        # Gradient norms (before step).
        g_log_c = weighter.log_c.grad
        g_log_a = weighter.log_alpha.grad
        grad_norm_log_c = float(g_log_c.norm().item()) if g_log_c is not None else 0.0
        grad_norm_log_alpha = float(g_log_a.norm().item()) if g_log_a is not None else 0.0

        optimizer.step()

        stats = summarize_settles(all_settle_iters, max_iter=linear_max_iter,
                                  settle_info_log=all_settle_info)
        cap_hit_frac = stats['cap_hit_fraction'] or 0.0
        if is_float64 and cap_hit_frac > cap_hit_warn_threshold:
            import warnings as _w
            _w.warn(
                f"[train_hardened] float64 cap-hit fraction {cap_hit_frac:.3f} "
                f"exceeds {cap_hit_warn_threshold}; check solver tol/max_iter.",
                UserWarning,
            )

        with torch.no_grad():
            c_here = weighter.c.item()
            alpha_here = weighter.alpha.item()
        rec = {
            'update': upd,
            'loss_mean': float((loss_sum / blocks_per_update).item()),
            'c': c_here,
            'alpha': alpha_here,
            'grad_norm_log_c': grad_norm_log_c,
            'grad_norm_log_alpha': grad_norm_log_alpha,
            'solver_n_iter_mean': stats['mean_n_iter'],
            'solver_n_iter_max': stats['max_n_iter'],
            'solver_cap_hit_fraction': cap_hit_frac,
            'solver_residual_max': stats['max_final_residual'],
            'solver_n_settles': stats['n_settles'],
        }
        history.append(rec)
        if log_every and ((upd % log_every == 0) or upd == updates - 1):
            _, val_mean = validation_block_mse(
                weighter, validation_blocks_fixed, K, delta,
                linear_max_iter=linear_max_iter, linear_tol=linear_tol)
            validation_history.append({
                'update': upd, 'loss_mean': val_mean,
                'n_blocks': len(validation_blocks_fixed),
            })
            print(
                f"  [train_hardened] upd {upd:4d}"
                f"  loss={rec['loss_mean']:.4e}"
                f"  c={c_here:.4f}  alpha={alpha_here:.4f}"
                f"  |g_log_c|={grad_norm_log_c:.2e}"
                f"  |g_log_a|={grad_norm_log_alpha:.2e}"
                f"  solver_iters mean={stats['mean_n_iter']}"
                f"  cap_hit={cap_hit_frac:.3f}",
                flush=True,
            )

    weighter.eval()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        weights_path = os.path.join(save_dir, 'final_weighter.pt')
        metadata_path = os.path.join(save_dir, 'final_weighter_metadata.json')
        torch.save(weighter.state_dict(), weights_path)
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        with open(os.path.join(save_dir, 'validation_history.json'), 'w') as f:
            json.dump(validation_history, f, indent=2)
        cap_fracs = [h['solver_cap_hit_fraction'] for h in history]
        residuals = [h['solver_residual_max'] for h in history
                     if h.get('solver_residual_max') is not None]
        with open(os.path.join(save_dir, 'solver_stats.json'), 'w') as f:
            json.dump({
                'cap_hit_fraction_max': max(cap_fracs) if cap_fracs else None,
                'cap_hit_fraction_mean': (sum(cap_fracs) / len(cap_fracs)) if cap_fracs else None,
                'solver_residual_max': max(residuals) if residuals else None,
                'updates': len(history),
            }, f, indent=2)
        with open(metadata_path, 'w') as f:
            json.dump({
                'weighter_parameterization': 'log_exp_v1',
                'c': float(weighter.c.item()),
                'alpha': float(weighter.alpha.item()),
                'v_max': float(weighter.v_max.item()),
                'start': start,
                'seed': seed,
                'updates': updates,
                'blocks_per_update': blocks_per_update,
                'K': K,
                'N': N,
                'delta': delta,
                'dtype': str(dtype).replace('torch.', ''),
                'solver': solver,
                'backward_mode': backward_mode,
                'linear_tol': linear_tol,
                'linear_max_iter': linear_max_iter,
                'backward_tol': backward_tol,
                'backward_max_iter': backward_max_iter,
                'lr_c': lr_c,
                'lr_alpha': lr_alpha,
            }, f, indent=2)
        print(f"  [train_hardened] saved weighter to {weights_path} + "
              f"metadata to {metadata_path}")
    return weighter, history


def build_validation_block_set(d, N, K, delta, sigma, p_burst, kappa,
                                 n_blocks, *, base_seed=10000, dtype=torch.float64,
                                 device='cpu'):
    """Generate a fixed validation block set (Phase C3).

    Each block uses a deterministic seed ``base_seed + i`` (i in
    ``range(n_blocks)``) and is held fixed across all contenders, so MSEs
    are paired and confidence intervals are well-defined (Phase D3).
    Returns a list of dicts with keys ``X``, ``d``, ``w_o``, ``seed``.
    """
    from utils.learned_robust import make_block
    blocks = []
    for i in range(n_blocks):
        g = torch.Generator(device=device).manual_seed(base_seed + i)
        w_o = torch.randn(d, generator=g, dtype=dtype, device=device)
        w_o = w_o / w_o.norm()
        X, d_obs = make_block(w_o, N, sigma, mode='iid', noise='impulsive',
                              p_burst=p_burst, kappa=kappa,
                              seed=base_seed + i, dtype=dtype, device=device)
        blocks.append({'X': X, 'd': d_obs, 'w_o': w_o, 'seed': base_seed + i})
    return blocks


@torch.no_grad()
def validation_block_mse(weighter, blocks, K, delta, *, linear_max_iter=200,
                           linear_tol=1e-10, backend='dense'):
    """Run ``block_robust_rls`` on each validation block and return the
    per-block MSE list and the mean (Phase C3 helper).

    No backprop; tight converged forward solves; same blocks for every
    contender so that paired CIs can be computed.  Pass ``weighter=None``
    to score plain batch LS (constant v=1, K=0).
    """
    from utils.learned_robust import (
        block_robust_rls, constant_weighter,
    )
    per_block = []
    for blk in blocks:
        if weighter is None:
            wgt = constant_weighter(1.0, dtype=blocks[0]['X'].dtype,
                                    device=blocks[0]['X'].device)
        else:
            wgt = weighter
        w = block_robust_rls(blk['X'], blk['d'], wgt, delta=delta, K=K,
                             max_iter=linear_max_iter, tol=linear_tol,
                             backward_mode='exact', backend=backend)
        mse = float(((blk['X'] @ w - blk['X'] @ blk['w_o']).pow(2)).mean().item())
        per_block.append(mse)
    return per_block, float(np.mean(per_block))


def plot_block_mse_vs_K(out_dir, K_values, mse_plain, mse_learned, mse_init,
                        mse_oracle=None, mse_huber=None, mse_hampel=None):
    """Plot MSE-vs-K for the block contenders (one curve per contender).

    Phase 0 (oracle-oos-headroom): ``mse_oracle`` adds the 4th (oracle)
    curve -- the idealized true-burst-mask ceiling for any curve.
    Classical robust baselines: ``mse_huber`` / ``mse_hampel`` add the
    MAD-adaptive curves.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(K_values, mse_plain, 'o-', color='C1', label='plain batch LS',
                linewidth=1.5)
    ax.semilogy(K_values, mse_learned, 's-', color='C3', label='learned block IRLS',
                linewidth=1.5)
    ax.semilogy(K_values, mse_init, '^--', color='C0', label='init-only block IRLS',
                linewidth=1.5, alpha=0.7)
    if mse_oracle is not None:
        ax.semilogy(K_values, mse_oracle, 'd-', color='C6',
                    label='oracle block IRLS (true burst mask)',
                    linewidth=1.5)
    if mse_huber is not None:
        ax.semilogy(K_values, mse_huber, 'x-', color='C2',
                    label='Huber + MAD (per-block scale)', linewidth=1.5)
    if mse_hampel is not None:
        ax.semilogy(K_values, mse_hampel, '*--', color='C4',
                    label='Hampel + MAD (per-block scale)', linewidth=1.5)
    ax.set_xlabel('K (outer IRLS iterations)')
    ax.set_ylabel(r'MSE $= \|X w^K - X w_o\|^2 / N$')
    ax.set_title('Block robust IRLS: MSE vs K (oracle ceiling)')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'robust_block_mse_vs_K.png'), dpi=140)
    plt.close(fig)


def plot_block_mse_vs_N(out_dir, N_values,
                        mse_plain, mse_init, mse_learned,
                        mse_streaming_plain, mse_streaming_init,
                        mse_streaming_learned,
                        mse_oracle=None, mse_huber=None, mse_hampel=None,
                        oos=None):
    """Plot MSE-vs-N for the 3x2 contender matrix (block vs streaming
    by non-robust / fixed-curve-robust / learned-robust).

    Phase 0 (oracle-oos-headroom) extensions:
      * ``mse_oracle`` adds the oracle batch curve to the in-sample panel.
      * ``mse_huber`` / ``mse_hampel`` add the classical MAD-adaptive
        batch curves to the in-sample panel.
      * ``oos`` (dict with 'plain'/'init'/'learned'/'oracle'/'huber'/
        'hampel' lists) adds a second, out-of-sample panel: fit on a train
        block, score on a fresh test block.  All keys optional; absent
        curves are skipped.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n_panels = 2 if oos is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(8 if n_panels == 1 else 14, 5))
    if n_panels == 1:
        axes = [axes]

    ax = axes[0]
    ax.semilogy(N_values, mse_plain, 'o-', color='C1',
                label='plain batch LS (v=1)', linewidth=1.5)
    ax.semilogy(N_values, mse_init, '^--', color='C0',
                label='init block IRLS (fixed curve)', linewidth=1.5, alpha=0.7)
    ax.semilogy(N_values, mse_learned, 's-', color='C3',
                label='learned block IRLS', linewidth=1.5)
    if mse_oracle is not None:
        ax.semilogy(N_values, mse_oracle, 'd-', color='C6',
                    label='oracle block IRLS (true burst mask)', linewidth=1.5)
    if mse_huber is not None:
        ax.semilogy(N_values, mse_huber, 'x-', color='C2',
                    label='Huber + MAD (per-block scale)', linewidth=1.5)
    if mse_hampel is not None:
        ax.semilogy(N_values, mse_hampel, '*--', color='C4',
                    label='Hampel + MAD (per-block scale)', linewidth=1.5)
    ax.semilogy(N_values, mse_streaming_plain, 'v--', color='C5',
                label='streaming EW-RLS (v=1)', linewidth=1.5, alpha=0.7)
    ax.semilogy(N_values, mse_streaming_init, 'P--', color='C4',
                label='streaming fixed-curve IR-RLS (init)', linewidth=1.5, alpha=0.7)
    ax.semilogy(N_values, mse_streaming_learned, 'D--', color='C2',
                label='streaming learned IR-RLS', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('block size N')
    ax.set_ylabel(r'MSE $= \|X w^K - X w_o\|^2 / N$')
    ax.set_title('In-sample (fit and score on the same block)')
    ax.legend(fontsize=7)
    ax.grid(True, which='both', alpha=0.3)

    if oos is not None:
        ax2 = axes[1]
        def _plot_oos(key, fmt, color, label):
            if oos.get(key) is not None:
                ax2.semilogy(N_values, oos[key], fmt, color=color, label=label,
                             linewidth=1.5)
        _plot_oos('plain', 'o-', 'C1', 'plain batch LS (v=1)')
        _plot_oos('init', '^--', 'C0', 'init block IRLS (fixed curve)')
        _plot_oos('learned', 's-', 'C3', 'learned block IRLS')
        _plot_oos('oracle', 'd-', 'C6', 'oracle block IRLS (true burst mask)')
        _plot_oos('huber', 'x-', 'C2', 'Huber + MAD')
        _plot_oos('hampel', '*--', 'C4', 'Hampel + MAD')
        ax2.set_xlabel('block size N')
        ax2.set_ylabel(r'OOS MSE $= \|X_{test} w - X_{test} w_o\|^2 / N$')
        ax2.set_title('Out-of-sample (fit on train block, score on a fresh block)')
        ax2.legend(fontsize=8)
        ax2.grid(True, which='both', alpha=0.3)
        fig.suptitle('Block robust IRLS: MSE vs N '
                     '(3x2 contender matrix + oracle + OOS)', fontsize=10)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'robust_block_mse_vs_N.png'), dpi=140)
    plt.close(fig)


def plot_block_settle_iter_histogram(out_dir, all_iters, K, N):
    """Histogram of settle iters per (K+1) settle pair across (K, N) sweep."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(all_iters, bins=20, color='C3', alpha=0.8,
            label=f'settles (K={K}, N={N})')
    ax.set_xlabel('LinearSolveLayer iterations per settle')
    ax.set_ylabel('count')
    ax.set_title('Block robust IRLS: settle iterations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'robust_block_settle_iters.png'), dpi=140)
    plt.close(fig)


def _load_or_train_block_weighter(args, out_dir, device, dtype, seed):
    """Shared train-or-load for the block robust weighter.

    Used by both ``run_robust_block_experiment`` and ``run_grid_sweep`` so
    the trained curve is a single source of truth across the two
    diagnostics.

    Returns ``(weighter_train, history, block_dir)``.  ``weighter_train``
    is the float32 training-state weighter in eval mode (NOT yet cast to
    the eval dtype -- callers cast with ``.to(device).to(dtype)`` as
    needed, decision 10).
    """
    from utils.learned_robust import LearnedRobustWeighter

    block_dir = os.path.join(out_dir, 'robust_block')
    _ensure_dir(block_dir)
    weighter_train = LearnedRobustWeighter(c_init=_DEFAULT_C_INIT, alpha_init=_DEFAULT_ALPHA_INIT)
    if args.block_weighter_path and os.path.exists(args.block_weighter_path):
        sd = torch.load(args.block_weighter_path, map_location='cpu')
        weighter_train.load_state_dict(sd)
        print(f"  [block] loaded trained weighter from {args.block_weighter_path}")
        history = []
    else:
        train_dir = args.block_train_out_dir
        if train_dir is None:
            train_dir = os.path.join(
                block_dir, 'training_hardening',
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_seed{args.block_train_seed}")
        train_dtype = torch.float64 if args.block_train_dtype == 'float64' else torch.float32
        weighter_train, history = train_robust_weighter_hardened(
            d=args.d, N=args.block_N, K=args.block_K, delta=args.block_delta,
            updates=args.block_train_updates,
            blocks_per_update=args.block_train_blocks_per_update,
            dtype=train_dtype,
            linear_tol=args.block_train_linear_tol,
            linear_max_iter=args.block_train_linear_max_iter,
            backward_tol=args.block_train_backward_tol,
            backward_max_iter=args.block_train_backward_max_iter,
            backward_mode=args.block_backward_mode,
            solver=args.block_train_solver,
            start=args.block_train_start,
            seed=args.block_train_seed,
            lr_c=args.block_train_lr_c,
            lr_alpha=args.block_train_lr_alpha,
            sigma=args.sigma, p_burst=args.p_burst, kappa=args.kappa,
            device=device, save_dir=train_dir)
        with open(os.path.join(train_dir, 'training_history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        with open(os.path.join(train_dir, 'config.json'), 'w') as f:
            json.dump({
                'seed': args.block_train_seed,
                'dtype': str(train_dtype).replace('torch.', ''),
                'updates': args.block_train_updates,
                'blocks_per_update': args.block_train_blocks_per_update,
                'd': args.d, 'N': args.block_N, 'K': args.block_K,
                'delta': args.block_delta,
                'solver': args.block_train_solver,
                'backward_mode': args.block_backward_mode,
                'linear_tol': args.block_train_linear_tol,
                'linear_max_iter': args.block_train_linear_max_iter,
                'backward_tol': args.block_train_backward_tol,
                'backward_max_iter': args.block_train_backward_max_iter,
            }, f, indent=2)
        print(f"  [block] hardened training artifacts: {train_dir}")
    return weighter_train, history, block_dir


def plot_grid_heatmap(out_dir, c_grid, alpha_grid, mse_grid, title=''):
    """Heatmap of block-IRLS MSE over the (c, alpha) log grid.

    ``mse_grid`` is (n_c, n_alpha); plotted on log axes with a log color
    scale (the MSE ranges many decades across the grid).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(np.log10(mse_grid.clip(min=1e-30)), aspect='auto',
                   origin='lower',
                   extent=[np.log10(alpha_grid[0]), np.log10(alpha_grid[-1]),
                           np.log10(c_grid[0]), np.log10(c_grid[-1])])
    ax.set_xlabel(r'$\log_{10} \alpha$')
    ax.set_ylabel(r'$\log_{10} c$')
    ax.set_title(title or 'Block robust IRLS: MSE over (c, alpha) grid')
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(r'$\log_{10}$ MSE')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'robust_block_grid_heatmap.png'), dpi=140)
    plt.close(fig)


def run_grid_sweep(args, out_dir, w_o, device, dtype, seed):
    """Dense (c, alpha) fixed-curve grid sweep -- the family-vs-training
    decomposition diagnostic.

    For every (c, alpha) on a log grid (``FixedCauchyWeighter``), run
    block IRLS at the production block config (N=args.block_N,
    K=args.block_K) and score the in-sample MSE on blocks held FIXED
    across the whole grid (the k_sweep fixed-block protocol, seed +
    trial*1000).  Also score plain / oracle / trained-learned on the same
    blocks.

    Verdict logic (the decision-relevant decomposition):
      * grid-best ~ oracle  => the Cauchy family CAN express near-oracle
        behavior; the learned curve just failed to find it => TRAINING
        problem (fix: multi-block-per-epoch + log-reparameterization).
      * grid-best >> oracle => no scalar v(e) curve reaches the oracle;
        the family itself is the ceiling => FAMILY problem (skip to the
        richer weighter v(e, |x|) / with memory).
      * trained >> grid-best confirms the curve was under-converged even
        relative to the best hand-placed curve.

    Outputs ``grid_sweep.json`` + ``robust_block_grid_heatmap.png`` in
    ``results/rls_demo/robust_block/``.
    """
    from utils.learned_robust import (
        block_robust_rls, FixedCauchyWeighter, constant_weighter,
        make_block, oracle_weighter,
    )

    block_dir = os.path.join(out_dir, 'robust_block')
    _ensure_dir(block_dir)

    # Trained curve (load or train via the shared helper) + eval cast.
    weighter_train, history, block_dir = _load_or_train_block_weighter(
        args, out_dir, device, dtype, seed)
    weighter = weighter_train.to(device).to(dtype)
    weighter.eval()
    const_w = constant_weighter(1.0, dtype=dtype).to(device)

    # Dense log-spaced grids (logspace over the raw ranges).
    c_grid = np.logspace(np.log10(args.grid_c_lo), np.log10(args.grid_c_hi),
                         args.grid_n_c)
    alpha_grid = np.logspace(np.log10(args.grid_alpha_lo),
                             np.log10(args.grid_alpha_hi), args.grid_n_alpha)
    print(f"\n=== (c, alpha) grid sweep: N={args.block_N}, K={args.block_K}, "
          f"n_c={args.grid_n_c}, n_alpha={args.grid_n_alpha}, "
          f"n_trials={args.n_trials} ===", flush=True)

    # Pre-build one stateless FixedCauchyWeighter per grid point.
    grid_weights = [[FixedCauchyWeighter(c=float(c), alpha=float(a),
                                         dtype=dtype).to(device)
                     for a in alpha_grid] for c in c_grid]

    n_trials = max(1, args.n_trials)
    mse_plain_acc = 0.0
    mse_oracle_acc = 0.0
    mse_learned_acc = 0.0
    mse_grid = np.zeros((args.grid_n_c, args.grid_n_alpha))
    # Per-block MSE arrays (Phase D3): same blocks across all contenders so
    # paired CIs are well-defined.
    per_block_plain = np.zeros(n_trials)
    per_block_oracle = np.zeros(n_trials)
    per_block_learned = np.zeros(n_trials)
    per_block_grid_best = np.zeros(n_trials)

    # Phase D2: disjoint model-selection and report sets.  Grid/dev seeds
    # use a dedicated offset (1_000_000) so the final OOS holdout
    # (seed=200_000_000+) does not collide with any grid evaluation.
    grid_seed_base = 1_000_000

    for trial in range(n_trials):
        trial_seed = grid_seed_base + trial * 1000
        torch.manual_seed(trial_seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(trial_seed)
        w_o_l = torch.randn(args.d, device=device, dtype=dtype)
        w_o_l = w_o_l / w_o_l.norm()
        X, d_obs, _, burst = make_block(
            w_o_l, args.block_N, args.sigma, mode='iid',
            noise=args.noise, p_burst=args.p_burst, kappa=args.kappa,
            seed=trial_seed, dtype=dtype, device=device,
            return_noise=True)
        oracle_w = oracle_weighter(burst, dtype=dtype, device=device)

        def _mse(wgt):
            w = block_robust_rls(X, d_obs, wgt, delta=args.block_delta,
                                 K=args.block_K, max_iter=args.linear_max_iter,
                                 tol=args.linear_tol, backward_mode='phantom',
                                 backend=args.block_solver_backend)
            return float(((X @ w - X @ w_o_l).pow(2)).mean().item())

        mse_p = _mse(const_w)
        mse_o = _mse(oracle_w)
        mse_l = _mse(weighter)
        mse_plain_acc += mse_p
        mse_oracle_acc += mse_o
        mse_learned_acc += mse_l
        per_block_plain[trial] = mse_p
        per_block_oracle[trial] = mse_o
        per_block_learned[trial] = mse_l
        for i in range(args.grid_n_c):
            for j in range(args.grid_n_alpha):
                mse_grid[i, j] += _mse(grid_weights[i][j])
        print(f"  [grid] trial {trial}: plain={mse_p:.4e} "
              f"learned={mse_l:.4e} oracle={mse_o:.4e}", flush=True)

    plain_mse = mse_plain_acc / n_trials
    oracle_mse = mse_oracle_acc / n_trials
    learned_mse = mse_learned_acc / n_trials
    mse_grid /= n_trials
    best_ij = np.unravel_index(np.argmin(mse_grid), mse_grid.shape)
    best_c = float(c_grid[best_ij[0]])
    best_alpha = float(alpha_grid[best_ij[1]])
    grid_best_mse = float(mse_grid[best_ij])
    per_block_grid_best = mse_grid[best_ij[0], best_ij[1]] * np.ones(n_trials)

    # Phase D3: paired confidence intervals on per-block MSE differences.
    diff_learned_oracle = per_block_learned - per_block_oracle
    diff_learned_plain = per_block_learned - per_block_plain
    diff_learned_grid = per_block_learned - per_block_grid_best
    mean_l_o, ci_l_o = paired_ci(diff_learned_oracle)
    mean_l_p, ci_l_p = paired_ci(diff_learned_plain)
    mean_l_g, ci_l_g = paired_ci(diff_learned_grid)

    # Verdict: grid-best within 10% of the plain->oracle span => the family
    # can reach near-oracle => TRAINING problem; otherwise FAMILY ceiling.
    gap_total = plain_mse - oracle_mse
    headroom_ratio_grid = (grid_best_mse - oracle_mse) / max(gap_total, 1e-30)
    headroom_ratio_learned = ((learned_mse - oracle_mse)
                              / max(gap_total, 1e-30))
    verdict = 'TRAINING' if headroom_ratio_grid <= 0.10 else 'FAMILY'
    trained_vs_grid_ratio = learned_mse / max(grid_best_mse, 1e-30)

    print(f"\n=== Grid-sweep verdict (N={args.block_N}, K={args.block_K}) ===",
          flush=True)
    print(f"  plain={plain_mse:.4e}  learned={learned_mse:.4e}  "
          f"oracle={oracle_mse:.4e}", flush=True)
    print(f"  grid-best (c={best_c:.4f}, alpha={best_alpha:.4f}): "
          f"mse={grid_best_mse:.4e}", flush=True)
    print(f"  headroom_ratio_grid={headroom_ratio_grid:.4f} "
          f"(grid-best vs oracle over plain->oracle span)", flush=True)
    print(f"  headroom_ratio_learned={headroom_ratio_learned:.4f} "
          f"(trained weighter vs oracle over plain->oracle span, "
          f"target <= 0.05)", flush=True)
    print(f"  paired CI (learned - oracle): {mean_l_o:+.3e} +- {ci_l_o:.3e}",
          flush=True)
    print(f"  paired CI (learned - plain): {mean_l_p:+.3e} +- {ci_l_p:.3e}",
          flush=True)
    print(f"  paired CI (learned - grid-best): {mean_l_g:+.3e} +- {ci_l_g:.3e}",
          flush=True)
    print(f"  trained/grid-best ratio={trained_vs_grid_ratio:.4f} "
          f"(>1 => trained curve under-converged vs best hand-placed curve)",
          flush=True)
    print(f"  verdict={verdict}", flush=True)

    plot_grid_heatmap(block_dir, c_grid, alpha_grid, mse_grid,
                      title=f'Block robust IRLS: MSE over (c, alpha) grid '
                            f'(N={args.block_N}, K={args.block_K})')

    grid_out = {
        'note': ('(c, alpha) grid sweep: for every fixed-curve point, block '
                 'IRLS in-sample MSE at K=args.block_K on blocks held fixed '
                 'across the grid (k_sweep fixed-block protocol).  Verdict: '
                 'TRAINING if grid-best ~ oracle (family can express it; '
                 'training failed); FAMILY if grid-best >> oracle (scalar '
                 'v(e) ceiling).  Phase D2: grid seeds are disjoint from '
                 'the training and final-OOS seed sets; Phase D3: per-block '
                 'MSE arrays and paired confidence intervals on '
                 '(learned - oracle), (learned - plain), '
                 '(learned - grid-best).'),
        'phase': 'D (extended grid evidence)',
        'seed_set': 'grid/dev',
        'grid_seed_base': grid_seed_base,
        'N': args.block_N,
        'K': args.block_K,
        'delta': args.block_delta,
        'noise': args.noise,
        'sigma': args.sigma,
        'p_burst': args.p_burst,
        'kappa': args.kappa,
        'n_trials': n_trials,
        'c_grid': [float(x) for x in c_grid],
        'alpha_grid': [float(x) for x in alpha_grid],
        'mse_matrix': mse_grid.tolist(),
        'plain_mse': plain_mse,
        'learned_mse': learned_mse,
        'oracle_mse': oracle_mse,
        'grid_best': {'c': best_c, 'alpha': best_alpha, 'mse': grid_best_mse},
        'trained_vs_grid_best_ratio': trained_vs_grid_ratio,
        'headroom_ratio_grid': headroom_ratio_grid,
        'headroom_ratio_learned': headroom_ratio_learned,
        'per_block_mse': {
            'plain': per_block_plain.tolist(),
            'learned': per_block_learned.tolist(),
            'oracle': per_block_oracle.tolist(),
            'grid_best': per_block_grid_best.tolist(),
        },
        'paired_ci': {
            'learned_minus_oracle': {'mean': mean_l_o, 'ci95': ci_l_o},
            'learned_minus_plain': {'mean': mean_l_p, 'ci95': ci_l_p},
            'learned_minus_grid_best': {'mean': mean_l_g, 'ci95': ci_l_g},
        },
        'verdict': verdict,
        'weighter': {'c': float(weighter.c.item()),
                     'alpha': float(weighter.alpha.item())},
    }
    with open(os.path.join(block_dir, 'grid_sweep.json'), 'w') as f:
        json.dump(grid_out, f, indent=2)
    print(f"  [grid] wrote {block_dir}/grid_sweep.json + "
          f"robust_block_grid_heatmap.png")


def run_robust_block_experiment(args, out_dir, w_o, device, dtype, seed):
    """Phase 1.5 block robust experiment: 6 contenders under impulsive noise.

    Contenders (the 3x2 matrix: non-robust / fixed-curve-robust / learned-robust
    crossed with batch-block / per-sample-streaming):

      1. plain batch LS                  (the K=0 limit, v=1, batch)
      2. init-only block IRLS            (weighter at init, K+1 settles, batch)
      3. learned block IRLS              (frozen trained weighter, K+1 settles)
      4. streaming EW-RLS                (v=1, per-sample, classical RLS baseline)
      5. streaming fixed-curve IR-RLS    (init weighter, per-sample)
      6. streaming learned IR-RLS        (trained weighter, per-sample)

    Phase 0 additions (oracle-oos-headroom): an oracle batch contender
    (7) that knows the true burst mask, plus out-of-sample scoring:

      7. oracle block IRLS               (true burst mask -> v=floor on bursts)

    Plots:
      * robust_block_mse_vs_K.png      -- MSE vs K (plain / init / learned /
                                          oracle block)
      * robust_block_mse_vs_N.png      -- MSE vs N (all 6 contenders + oracle,
                                          plus an out-of-sample panel)
      * robust_block_settle_iters.png  -- LinearSolveLayer iter distribution

    The k_sweep holds each trial's block fixed across all K values (seed
    depends on trial only, not K) so the plain column (v=1, K-independent)
    serves as a self-check: it should be ~flat.  The n_sweep varies the
    block per N (seed includes + N_l) because N is the real axis there.

    Plus a ``fixed_curve_vs_learned`` summary in ``block_metrics.json``:
    the best fixed-curve depth (argmin over the init column of the k_sweep)
    and the learned block at that same K — isolates "better curve" from
    "more depth."

    Out-of-sample scoring (Phase 0): for every (K, N) block a fresh test
    block is generated with the SAME plant w_o but a seed offset
    (``_OOS_SEED_OFFSET``), each contender fits on the train block, and the
    fitted w is scored as ``||X_test w - X_test w_o||^2 / N`` against the
    test block.  Stored as ``oos_k_sweep`` / ``oos_n_sweep``.  A headroom
    verdict (CEILING vs HEADROOM) is computed on the in-sample k_sweep at
    K=args.block_K: if the oracle-vs-learned gap is <= 10% of the
    plain-vs-oracle gap, learned is essentially at the ceiling.
    """
    # Lazy import (plan decision 9).
    from utils.learned_robust import (
        block_robust_rls, digital_block_robust_rls, FabricRobustRLS,
        LearnedRobustWeighter, constant_weighter, make_block, oracle_weighter,
        FixedCauchyWeighter, MADRobustWeighter,
    )

    block_dir = os.path.join(out_dir, 'robust_block')
    _ensure_dir(block_dir)

    # ---- Train or load block weighter (shared helper) ----
    weighter_train, history, block_dir = _load_or_train_block_weighter(
        args, out_dir, device, dtype, seed)

    # Cast frozen weighter to eval dtype (decision 10: train float32, eval float64).
    weighter = weighter_train.to(device).to(dtype)
    weighter.eval()
    const_w = constant_weighter(1.0, dtype=dtype).to(device)
    init_w = LearnedRobustWeighter(c_init=_DEFAULT_C_INIT, alpha_init=_DEFAULT_ALPHA_INIT).to(device).to(dtype)
    init_w.eval()
    # Classical robust baselines (MAD-adaptive, per-block scale).
    huber_w = MADRobustWeighter(mode='huber', dtype=dtype).to(device)
    hampel_w = MADRobustWeighter(mode='hampel', dtype=dtype).to(device)

    # ---- Generate blocks for the K-sweep and N-sweep ----
    # K-sweep: fix N at args.block_N, sweep K.
    # N-sweep: fix K at args.block_K, sweep N.
    K_values = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32] if args.block_K_max is None else list(args.block_K_max)
    N_values = [32, 64, 128, 256, 512] if args.block_N_max is None else list(args.block_N_max)

    # Out-of-sample seed offset: the test block shares the plant w_o but is
    # drawn from a seed far from every training/sweep seed, so it is a fresh
    # block (Phase 0 oracle-oos-headroom).
    _OOS_SEED_OFFSET = 1_000_000

    def make_block_fixed(seed_):
        torch.manual_seed(seed_)
        if device.type == 'cuda':
            torch.cuda.manual_seed(seed_)
        w_o_l = torch.randn(args.d, device=device, dtype=dtype)
        w_o_l = w_o_l / w_o_l.norm()
        return w_o_l

    def block_mse(weighter_, X, d_obs, w_o_l, K_l, delta_l):
        """Run block IRLS with K iterations and return (mse, settle_iters).

        ``settle_log`` collects the n_iter of every settle (K+1 of them),
        not just the final one, so the histogram reflects the full block
        settle cost.  Backend is selected by ``args.block_solver_backend``
        (default ``'dense'``; ``'circuit_stamp'`` selects the element-level
        KCL backend from ``utils.circuit_stamp``).
        """
        settle_iters = []
        w = block_robust_rls(X, d_obs, weighter_, delta=delta_l, K=K_l,
                             max_iter=args.linear_max_iter,
                             tol=args.linear_tol,
                             backward_mode='phantom',
                             settle_log=settle_iters,
                             backend=args.block_solver_backend)
        mse = float(((X @ w - X @ w_o_l).pow(2)).mean().item())
        return mse, settle_iters, w

    def oos_block_mse(weighter_, X_tr, d_tr, X_te, w_o_l, K_l, delta_l):
        """Out-of-sample block IRLS: fit on the train block (X_tr, d_tr) at
        depth K_l, then score the fitted w on the FRESH test block X_te as
        ``||X_te @ w - X_te @ w_o_l||^2 / N`` (Phase 0).  Returns the OOS
        MSE (settle iters are not collected here to keep the OOS sweep
        cheap and focused on the ceiling question).
        """
        w = block_robust_rls(X_tr, d_tr, weighter_, delta=delta_l, K=K_l,
                             max_iter=args.linear_max_iter,
                             tol=args.linear_tol,
                             backward_mode='phantom',
                             backend=args.block_solver_backend)
        return float(((X_te @ w - X_te @ w_o_l).pow(2)).mean().item())

    # ---- K-sweep ----
    print(f"\n=== Block robust: K-sweep, N={args.block_N}, "
          f"noise={args.noise}, sigma={args.sigma} ===", flush=True)
    k_sweep_plain = []
    k_sweep_learned = []
    k_sweep_init = []
    k_sweep_oracle = []
    k_sweep_huber = []
    k_sweep_hampel = []
    k_sweep_iters = []
    n_trials = max(1, args.n_trials)
    for K_l in K_values:
        mse_plain_acc = 0.0
        mse_learned_acc = 0.0
        mse_init_acc = 0.0
        mse_oracle_acc = 0.0
        mse_huber_acc = 0.0
        mse_hampel_acc = 0.0
        for trial in range(n_trials):
            # Hold the block fixed across all K values within a trial
            # (seed depends on trial only, not K) so the K-sweep isolates
            # the depth effect from per-K data noise.  The 'plain' column
            # (v=1, K-independent) is the self-check: it should be ~flat.
            w_o_l = make_block_fixed(seed + trial * 1000)
            X, d_obs, nu, burst = make_block(
                w_o_l, args.block_N, args.sigma, mode='iid',
                noise=args.noise, p_burst=args.p_burst,
                kappa=args.kappa, seed=seed + trial * 1000,
                dtype=dtype, device=device, return_noise=True)
            oracle_w = oracle_weighter(burst, dtype=dtype, device=device)
            mse_p, _, _ = block_mse(const_w, X, d_obs, w_o_l, K_l, args.block_delta)
            mse_l, it_l, _ = block_mse(weighter, X, d_obs, w_o_l, K_l, args.block_delta)
            mse_i, _, _ = block_mse(init_w, X, d_obs, w_o_l, K_l, args.block_delta)
            mse_o, _, _ = block_mse(oracle_w, X, d_obs, w_o_l, K_l, args.block_delta)
            mse_h, _, _ = block_mse(huber_w, X, d_obs, w_o_l, K_l, args.block_delta)
            mse_hm, _, _ = block_mse(hampel_w, X, d_obs, w_o_l, K_l, args.block_delta)
            mse_plain_acc += mse_p
            mse_learned_acc += mse_l
            mse_init_acc += mse_i
            mse_oracle_acc += mse_o
            mse_huber_acc += mse_h
            mse_hampel_acc += mse_hm
            k_sweep_iters.extend(it_l)
        k_sweep_plain.append(mse_plain_acc / n_trials)
        k_sweep_learned.append(mse_learned_acc / n_trials)
        k_sweep_init.append(mse_init_acc / n_trials)
        k_sweep_oracle.append(mse_oracle_acc / n_trials)
        k_sweep_huber.append(mse_huber_acc / n_trials)
        k_sweep_hampel.append(mse_hampel_acc / n_trials)
        print(f"  K={K_l:2d}: plain={k_sweep_plain[-1]:.4e}  "
              f"learned={k_sweep_learned[-1]:.4e}  "
              f"init={k_sweep_init[-1]:.4e}  "
              f"oracle={k_sweep_oracle[-1]:.4e}  "
              f"huber={k_sweep_huber[-1]:.4e}  "
              f"hampel={k_sweep_hampel[-1]:.4e}", flush=True)

    plot_block_mse_vs_K(block_dir, K_values, k_sweep_plain, k_sweep_learned,
                         k_sweep_init, k_sweep_oracle, mse_huber=k_sweep_huber,
                         mse_hampel=k_sweep_hampel)

    # ---- N-sweep ----
    print(f"\n=== Block robust: N-sweep, K={args.block_K}, "
          f"noise={args.noise}, sigma={args.sigma} ===", flush=True)
    n_sweep_plain = []
    n_sweep_init = []
    n_sweep_learned = []
    n_sweep_oracle = []
    n_sweep_huber = []
    n_sweep_hampel = []
    n_sweep_streaming_plain = []
    n_sweep_streaming_init = []
    n_sweep_streaming_learned = []

    def streaming_robust_rls_factory(d_l, weighter_l, dtype_l):
        from utils.learned_robust import FabricRobustRLS
        return FabricRobustRLS(d=d_l, weighter=weighter_l, lam=0.99,
                                R0=torch.eye(d_l, device=device, dtype=dtype_l),
                                max_iter=args.linear_max_iter,
                                tol=args.linear_tol,
                                beta=args.linear_beta,
                                device=device, training=False, log_every=0)

    def streaming_mse(weighter_l, X, d_obs, w_o_l, N_l):
        """Run T=N_l weighted-RLS samples with the given weighter; return
        the MSE of the final w against the same scoring as block_mse.

        ``weighter_l`` may be a constant (v=1) weighter for the plain
        EW-RLS baseline, the fixed-curve init weighter for the
        classical robust RLS baseline, or the trained learned weighter.
        """
        f = streaming_robust_rls_factory(args.d, weighter_l, dtype)
        for t in range(N_l):
            f.step(X[t], d_obs[t])
        w_stream = f.w
        return float(((X @ w_stream - X @ w_o_l).pow(2)).mean().item())

    for N_l in N_values:
        mse_plain_acc = 0.0
        mse_init_acc = 0.0
        mse_learned_acc = 0.0
        mse_oracle_acc = 0.0
        mse_huber_acc = 0.0
        mse_hampel_acc = 0.0
        mse_str_plain_acc = 0.0
        mse_str_init_acc = 0.0
        mse_str_learned_acc = 0.0
        for trial in range(n_trials):
            w_o_l = make_block_fixed(seed + trial * 1000 + N_l)
            X, d_obs, nu, burst = make_block(
                w_o_l, N_l, args.sigma, mode='iid',
                noise=args.noise, p_burst=args.p_burst,
                kappa=args.kappa, seed=seed + trial * 1000 + N_l,
                dtype=dtype, device=device, return_noise=True)
            oracle_w = oracle_weighter(burst, dtype=dtype, device=device)
            mse_p, _, _ = block_mse(const_w, X, d_obs, w_o_l, args.block_K, args.block_delta)
            mse_i, _, _ = block_mse(init_w, X, d_obs, w_o_l, args.block_K, args.block_delta)
            mse_l, _, _ = block_mse(weighter, X, d_obs, w_o_l, args.block_K, args.block_delta)
            mse_o, _, _ = block_mse(oracle_w, X, d_obs, w_o_l, args.block_K, args.block_delta)
            mse_h, _, _ = block_mse(huber_w, X, d_obs, w_o_l, args.block_K, args.block_delta)
            mse_hm, _, _ = block_mse(hampel_w, X, d_obs, w_o_l, args.block_K, args.block_delta)
            mse_plain_acc += mse_p
            mse_init_acc += mse_i
            mse_learned_acc += mse_l
            mse_oracle_acc += mse_o
            mse_huber_acc += mse_h
            mse_hampel_acc += mse_hm

            # Three streaming contenders (3x2 matrix, streaming column):
            #   streaming_plain  = classical EW-RLS       (v=1)
            #   streaming_init   = fixed-curve robust RLS (init weighter)
            #   streaming_learned= learned robust RLS    (trained weighter)
            mse_str_plain_acc += streaming_mse(const_w, X, d_obs, w_o_l, N_l)
            mse_str_init_acc += streaming_mse(init_w, X, d_obs, w_o_l, N_l)
            mse_str_learned_acc += streaming_mse(weighter, X, d_obs, w_o_l, N_l)
        n_sweep_plain.append(mse_plain_acc / n_trials)
        n_sweep_init.append(mse_init_acc / n_trials)
        n_sweep_learned.append(mse_learned_acc / n_trials)
        n_sweep_oracle.append(mse_oracle_acc / n_trials)
        n_sweep_huber.append(mse_huber_acc / n_trials)
        n_sweep_hampel.append(mse_hampel_acc / n_trials)
        n_sweep_streaming_plain.append(mse_str_plain_acc / n_trials)
        n_sweep_streaming_init.append(mse_str_init_acc / n_trials)
        n_sweep_streaming_learned.append(mse_str_learned_acc / n_trials)
        print(f"  N={N_l:4d}: plain={n_sweep_plain[-1]:.4e}  "
              f"init={n_sweep_init[-1]:.4e}  "
              f"learned={n_sweep_learned[-1]:.4e}  "
              f"oracle={n_sweep_oracle[-1]:.4e}  "
              f"huber={n_sweep_huber[-1]:.4e}  "
              f"hampel={n_sweep_hampel[-1]:.4e}  "
              f"str_plain={n_sweep_streaming_plain[-1]:.4e}  "
              f"str_init={n_sweep_streaming_init[-1]:.4e}  "
              f"str_learned={n_sweep_streaming_learned[-1]:.4e}", flush=True)

    # ---- Out-of-sample sweeps (Phase 0 oracle-oos-headroom) ----
    # For every (K, N) train block, generate a fresh test block with the
    # SAME plant w_o but a large seed offset; fit each batch contender on
    # the train block and score the fitted w on the test block.  This
    # measures generalization (does the learned/fixed/oracle curve fit on
    # data it never saw), which the in-sample metric cannot see.
    print(f"\n=== Block robust: out-of-sample sweeps "
          f"(test-seed offset {_OOS_SEED_OFFSET}) ===", flush=True)

    def oos_contenders(contenders, seed_tr, N_l, K_l):
        """Fit all batch contenders on a train block, score each on a
        fresh test block; return dict of OOS MSEs."""
        w_o_l = make_block_fixed(seed_tr)
        X_tr, d_tr, _, burst_tr = make_block(
            w_o_l, N_l, args.sigma, mode='iid',
            noise=args.noise, p_burst=args.p_burst,
            kappa=args.kappa, seed=seed_tr,
            dtype=dtype, device=device, return_noise=True)
        X_te, d_te = make_block(
            w_o_l, N_l, args.sigma, mode='iid',
            noise=args.noise, p_burst=args.p_burst,
            kappa=args.kappa, seed=seed_tr + _OOS_SEED_OFFSET,
            dtype=dtype, device=device)
        # The oracle is fit on the same train block, so its burst mask
        # comes from the train block's noise (no double generation).
        oracle_w = oracle_weighter(burst_tr, dtype=dtype, device=device)
        out = {}
        for name, wgt in contenders.items():
            out[name] = oos_block_mse(wgt, X_tr, d_tr, X_te, w_o_l, K_l,
                                      args.block_delta)
        out['oracle'] = oos_block_mse(oracle_w, X_tr, d_tr, X_te, w_o_l, K_l,
                                      args.block_delta)
        return out

    # Fit at the train-block seed used by the in-sample sweep, so the
    # OOS score shares the exact same training data.
    oos_k = {k: [] for k in ('plain', 'learned', 'init', 'oracle', 'huber', 'hampel')}
    for K_l in K_values:
        acc = {k: 0.0 for k in oos_k}
        for trial in range(n_trials):
            res = oos_contenders(
                {'plain': const_w, 'learned': weighter, 'init': init_w,
                 'huber': huber_w, 'hampel': hampel_w},
                seed_tr=seed + trial * 1000, N_l=args.block_N, K_l=K_l)
            for k in acc:
                acc[k] += res[k]
        for k in oos_k:
            oos_k[k].append(acc[k] / n_trials)
        print(f"  OOS K={K_l:2d}: " + "  ".join(
            f"{k}={oos_k[k][-1]:.4e}" for k in oos_k), flush=True)

    oos_n = {k: [] for k in ('plain', 'learned', 'init', 'oracle', 'huber', 'hampel')}
    for N_l in N_values:
        acc = {k: 0.0 for k in oos_n}
        for trial in range(n_trials):
            res = oos_contenders(
                {'plain': const_w, 'learned': weighter, 'init': init_w,
                 'huber': huber_w, 'hampel': hampel_w},
                seed_tr=seed + trial * 1000 + N_l, N_l=N_l,
                K_l=args.block_K)
            for k in acc:
                acc[k] += res[k]
        for k in oos_n:
            oos_n[k].append(acc[k] / n_trials)
        print(f"  OOS N={N_l:4d}: " + "  ".join(
            f"{k}={oos_n[k][-1]:.4e}" for k in oos_n), flush=True)

    # ---- Oracle + OOS headroom verdict ----
    # In-sample k_sweep at K=args.block_K: how much of the plain->oracle
    # improvement does the learned weighter capture?
    try:
        eval_idx = K_values.index(args.block_K)
    except ValueError:
        eval_idx = len(K_values) - 1
    plain_k = k_sweep_plain[eval_idx]
    learned_k = k_sweep_learned[eval_idx]
    oracle_k = k_sweep_oracle[eval_idx]
    gap_total = plain_k - oracle_k
    gap_learned = plain_k - learned_k
    headroom_gap = learned_k - oracle_k
    headroom_ratio = headroom_gap / max(gap_total, 1e-30)
    # Threshold at 10%: learned within 10% of the plain->oracle span is
    # effectively at the ceiling; otherwise headroom remains.
    headroom_verdict = 'CEILING' if headroom_ratio <= 0.10 else 'HEADROOM'
    oos_fixed_ratio = oos_k['learned'][eval_idx] / max(oos_k['init'][eval_idx], 1e-30)
    learned_oracle_ratio = learned_k / max(oracle_k, 1e-30)
    print(f"\n=== Oracle + OOS headroom verdict (in-sample k_sweep @ "
          f"K={args.block_K}) ===", flush=True)
    print(f"  plain={plain_k:.4e}  learned={learned_k:.4e}  "
          f"oracle={oracle_k:.4e}", flush=True)
    print(f"  learned/oracle ratio={learned_oracle_ratio:.4f} "
          f"(~1.0 => learned ~ oracle => CEILING)", flush=True)
    print(f"  OOS learned-vs-fixed ratio={oos_fixed_ratio:.4f} "
          f"(<1.0 => learned generalizes better than fixed curve)", flush=True)
    print(f"  headroom_ratio={headroom_ratio:.4f} "
          f"verdict={headroom_verdict}", flush=True)

    plot_block_mse_vs_N(block_dir, N_values,
                         n_sweep_plain, n_sweep_init, n_sweep_learned,
                         n_sweep_streaming_plain, n_sweep_streaming_init,
                         n_sweep_streaming_learned,
                         mse_oracle=n_sweep_oracle,
                         mse_huber=n_sweep_huber, mse_hampel=n_sweep_hampel,
                         oos={k: oos_n[k] for k in
                              ('plain', 'init', 'learned', 'oracle', 'huber', 'hampel')})

    # ---- Settle iter histogram ----
    if k_sweep_iters:
        plot_block_settle_iter_histogram(block_dir, k_sweep_iters,
                                          args.block_K, args.block_N)

    # ---- Phase 1.5 gate 5: phantom-vs-exact bias measurement ----
    print(f"\n=== Phantom-vs-exact implicit gradient bias ===", flush=True)
    bias_results = _measure_phantom_vs_exact_bias(
        weighter=weighter, d=args.d, N=args.block_N, K=args.block_K,
        delta=args.block_delta, sigma=args.sigma, p_burst=args.p_burst,
        kappa=args.kappa, seed=seed, device=device, dtype=dtype)
    print(f"  rel_bias={bias_results['rel_bias']:.4e}  "
          f"phantom_grad={bias_results['phantom_grad']:.4e}  "
          f"exact_grad={bias_results['exact_grad']:.4e}",
          flush=True)

    # ---- Best fixed-curve depth (init weighter) vs learned at the same depth ----
    # The k_sweep evaluates all three block contenders (plain / init / learned)
    # on the same K grid and same data per K; the init column's argmin is the
    # best classical fixed-curve depth. Comparing learned at that same K
    # isolates "better curve" from "more depth".
    best_init_idx = min(range(len(K_values)), key=lambda i: k_sweep_init[i])
    best_init_K = K_values[best_init_idx]
    best_init_mse = k_sweep_init[best_init_idx]
    learned_at_best_init_K_mse = k_sweep_learned[best_init_idx]
    print(f"\n=== Fixed-curve depth profile (k_sweep @ N={args.block_N}) ===",
          flush=True)
    print(f"  best fixed-curve (init) K={best_init_K}: "
          f"init={best_init_mse:.4e}  "
          f"learned@same K={learned_at_best_init_K_mse:.4e}", flush=True)

    # ---- Save metrics ----
    metrics = {
        'd': args.d,
        'N': args.block_N,
        'K': args.block_K,
        'delta': args.block_delta,
        'noise': args.noise,
        'p_burst': args.p_burst,
        'kappa': args.kappa,
        'sigma': args.sigma,
        'n_trials': n_trials,
        'device': str(device),
        'training': {
            'epochs': args.block_epochs,
            'N': args.block_N,
            'K': args.block_K,
            'lr': args.block_lr,
        },
        'weighter': {
            'c': float(weighter.c.item()),
            'alpha': float(weighter.alpha.item()),
        },
        'k_sweep': {
            'K_values': K_values,
            'plain': k_sweep_plain,
            'learned': k_sweep_learned,
            'init': k_sweep_init,
            # Phase 0: oracle ceiling column (true burst mask).
            'oracle': k_sweep_oracle,
            # Classical robust baselines (MAD-adaptive, per-block scale).
            'huber': k_sweep_huber,
            'hampel': k_sweep_hampel,
        },
        'n_sweep': {
            'N_values': N_values,
            'plain': n_sweep_plain,
            'init': n_sweep_init,
            'learned': n_sweep_learned,
            'oracle': n_sweep_oracle,
            'huber': n_sweep_huber,
            'hampel': n_sweep_hampel,
            'streaming_plain': n_sweep_streaming_plain,
            'streaming_init': n_sweep_streaming_init,
            'streaming_learned': n_sweep_streaming_learned,
            # Back-compat alias: pre-fix the learned-streaming curve was
            # written under the key 'streaming'.  Keep it so downstream
            # readers of the old key continue to work.
            'streaming': n_sweep_streaming_learned,
        },
        'oos_k_sweep': {
            'note': ('out-of-sample k_sweep (Phase 0): fit each batch '
                     'contender on the train block (same seeds as k_sweep), '
                     'score the fitted w on a fresh test block with the '
                     'same plant; OOS MSE = ||X_test w - X_test w_o||^2 / N.'),
            'K_values': K_values,
            'test_seed_offset': _OOS_SEED_OFFSET,
            'plain': oos_k['plain'],
            'learned': oos_k['learned'],
            'init': oos_k['init'],
            'oracle': oos_k['oracle'],
            'huber': oos_k['huber'],
            'hampel': oos_k['hampel'],
        },
        'oos_n_sweep': {
            'note': ('out-of-sample n_sweep (Phase 0): same protocol as '
                     'oos_k_sweep at K=args.block_K.'),
            'N_values': N_values,
            'test_seed_offset': _OOS_SEED_OFFSET,
            'plain': oos_n['plain'],
            'learned': oos_n['learned'],
            'init': oos_n['init'],
            'oracle': oos_n['oracle'],
            'huber': oos_n['huber'],
            'hampel': oos_n['hampel'],
        },
        'verdict': {
            'note': ('Phase 0 headroom verdict on the in-sample k_sweep at '
                     'K=args.block_K.  headroom_ratio = '
                     '(learned - oracle) / (plain - oracle): ~0 means the '
                     'learned weighter captures essentially all the '
                     'plain->oracle improvement (CEILING); ~1 means most of '
                     'the gap remains (HEADROOM).  CEILING/HEADROOM split '
                     'at headroom_ratio <= 0.10.'),
            'at_K': args.block_K,
            'plain_mse': float(plain_k),
            'learned_mse': float(learned_k),
            'oracle_mse': float(oracle_k),
            'learned_oracle_ratio': float(learned_oracle_ratio),
            'oos_learned_vs_init_ratio': float(oos_fixed_ratio),
            'headroom_ratio': float(headroom_ratio),
            'headroom_verdict': headroom_verdict,
        },
        'fixed_curve_vs_learned': {
            'note': ('best fixed-curve depth (min over init column) vs '
                     'learned block at the same depth; both evaluated '
                     'on the k_sweep (N=args.block_N).'),
            'best_init_K': int(best_init_K),
            'best_init_mse': float(best_init_mse),
            'learned_at_best_init_K_mse': float(learned_at_best_init_K_mse),
        },
        'phantom_vs_exact': bias_results,
    }
    with open(os.path.join(block_dir, 'block_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  [block] Wrote metrics to {block_dir}/block_metrics.json")
    print(f"  [block] trained weighter: c={metrics['weighter']['c']:.4f}, "
          f"alpha={metrics['weighter']['alpha']:.4f}")
    print(f"  Figures in {block_dir}/")


def _measure_phantom_vs_exact_bias(weighter, d, N, K, delta, sigma, p_burst,
                                   kappa, seed, device, dtype):
    """Measure the phantom-gradient bias at the *trained* weighter's
    operating point (Phase 1.5 prerequisite #3).

    Note: since the 2026-08-13 fix, training defaults to
    ``backward_mode='exact'`` (the true implicit gradient) — the phantom
    gradient is no longer the one used for learning.  This measurement
    is still reported for diagnostic purposes: it quantifies the
    approximation error of the phantom (Geng et al. 2021) VJP at the
    trained weighter's operating point, decoupled from the training
    signal.  A measurement, not a pass/fail bound.

    ``weighter`` is the fully-trained weighter; its params are copied into
    two fresh instances by ``utils.learned_robust.measure_phantom_vs_exact_bias``
    so the only difference between the phantom and exact runs is the
    backward mode.
    """
    from utils.learned_robust import (
        block_robust_rls, LearnedRobustWeighter, make_block,
        measure_phantom_vs_exact_bias,
    )

    torch.manual_seed(seed)
    w_o = torch.randn(d, device=device, dtype=dtype)
    w_o = w_o / w_o.norm()
    X, d_obs = make_block(w_o, N, sigma, mode='iid',
                          noise='impulsive', p_burst=p_burst,
                          kappa=kappa, seed=seed,
                          dtype=dtype, device=device)

    res = measure_phantom_vs_exact_bias(
        X, d_obs, w_o, weighter.to(dtype), delta=delta, K=K,
        max_iter=100, tol=1e-8)
    res.update({
        'd': d,
        'N': N,
        'K': K,
        'delta': delta,
        'sigma': sigma,
        'p_burst': p_burst,
        'kappa': kappa,
        'precision': 'float64 (eval dtype)',
        'operating_point': 'trained weighter',
    })
    return res


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)

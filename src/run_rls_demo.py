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

Metrics:
  - Ensemble-mean ||w_t - w_o||^2 vs t (log scale), all 4 contenders on one plot
  - Discrepancy: ||w_fabric - w_digital_RLS|| / ||w_digital_RLS|| vs t (should be 1e-4..1e-6)
  - Settling budget: iterations per sample for LinearSolveLayer
  - Tracking: time-varying w_o with process noise, steady-state misadjustment vs lambda

Also produces the Ljung overlay: fabric-LMS (ODE dw/dt = mu*e*x) and digital
LMS at decreasing step sizes mu*dt converge onto the continuous trajectory.

Results are saved to ./results/rls_demo/.
"""
import os
import sys
import argparse
import json
import time
import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from utils.circuit_block import LinearSolveLayer, EquilibriumSolve


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
        self.R = (R0.to(self.device) if R0 is not None else torch.eye(d, device=self.device))
        self.w = (torch.zeros(d, device=self.device) if w_init is None
                  else w_init.clone().to(self.device))
        self.p = torch.zeros(d, device=self.device)
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

def make_stream(w_o, T, sigma, mode='iid', seed=0, dtype=torch.float64, device='cpu'):
    """Generate (x_t, d_t) stream of length T.

    mode='iid':   x_t ~ N(0, I)
    mode='ar1':   x_t = rho * x_{t-1} + sqrt(1-rho^2) * eps_t, eps_t ~ N(0, I)
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
    nu = sigma * torch.randn(T, generator=g, dtype=dtype, device=device)
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
                 mode='iid', seed_base=0, dtype=torch.float64, device='cpu',
                 name='', log_every=1, quiet=False):
    """`contender_factory` is a zero-arg callable returning a fresh contender
    instance.  Returns (W_mean[T,d], W_runs[n_trials, T, d], per-trial extras list).

    `name` and `log_every` control progress logging: prints every `log_every`
    trials with the elapsed wall time.  Set `quiet=True` to suppress all logging
    (used when called from inside another logged loop)."""
    device = torch.device(device) if not isinstance(device, torch.device) else device
    d = w_o.shape[0]
    W_runs = torch.zeros(n_trials, T, d, dtype=dtype, device=device)
    extras_all = []
    if not quiet:
        print(f"  [{name}] {n_trials} trials, T={T}, mode={mode}", flush=True)
    loop_start = time.time()
    for trial in range(n_trials):
        x, d_obs = make_stream(w_o, T, sigma, mode=mode,
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
    print(f"True plant ||w_o|| = {w_o.norm().item():.4f}, d = {args.d}")

    if args.batch_only:
        run_batch_experiment(args, out_dir, w_o, device, dtype, args.seed)
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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)

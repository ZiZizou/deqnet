"""Phase 1 + Phase 1.5 - Learned Robust IR-RLS / Block Robust IRLS.

Implements the learned robust influence function and its RLS variants:

  * ``LearnedRobustWeighter``  - per-sample influence v(e) in (0, 1].
    Generalized-Cauchy family with parameters ensuring v(0)=1, v->0
    for |e|>>c, and v->v_max (flat) when alpha->0 (the Gaussian-control
    signature per plan decision 1).

  * ``FabricRobustRLS``        - per-sample weighted RLS with the
    LeastSquares fabric solve.  Carries the autograd graph through R
    and p when ``training=True`` so the weighter receives gradients,
    otherwise detaches v_t to keep the frozen-weighter Monte-Carlo
    graph-free (plan decision 5).

  * ``DigitalRobustRLS``       - O(d^2) recursion with the corrected
    v_t weighting (plan finding A, decision 3).  Byte-identical to
    ``DigitalRLS`` at v_t=1; v_t->0 freezes w and divides P by lam.

Phase 1.5 additions (block-parallel single-settle reconcile):

  * ``make_block``              - block generator (T, d) regressor and
    (T,) observations; reuses ``_make_noise`` for the noise model.
  * ``block_robust_rls``        - block IRLS with K+1 settles (one
    plain batch LS + K outer iterations).  Weights are block-parallel
    v = weighter(d - X w).  Optional ``settle_log`` collects the n_iter
    of every settle for the histogram metric.
  * ``digital_block_robust_rls`` - the digital twin using
    ``torch.linalg.solve``; reduces to plain batch LS at v=1.
  * ``measure_phantom_vs_exact_bias`` - measures the phantom-gradient
    bias at a given (typically trained) weighter operating point
    (Phase 1.5 prerequisite #3; measurement, not pass/fail).

Design decisions referenced by line numbers in the plan
``LEARNED_RLS_ISTA_PLAN.md``: 1 (parameterization), 3 (recursion),
4 (truncated BPTT detaches R, p, AND w), 5 (training flag, v_t.detach),
11 (no_grad on _chebyshev_beta).

KIMI audit corrections applied:
  #1  raw_c init = -2.25 (c ~ 0.107, ~10 sigma at the demo defaults
      sigma=0.01, kappa=20 -> burst magnitude ~ 0.2).  The plan
      default raw_c=2.0 would put the knee at |e|>>2, leaving the
      weighter blind to bursts at init and forcing Adam to traverse
      ~20x in raw_c on a 2% signal.
  #2  Gate 3 tolerance relaxed to 1e-5 (feedback-amplification floor
      at lambda=0.99 is ~1e-6 with default tol).
  #3  Implicit-vs-unrolled gradient gate runs in float64 with tol 1e-10.
  #4  Gaussian-control gate tests Var_e[v(e)] over a fixed e-grid.
  #5  Standalone v_t (0, 1] bounds test over a wide e-grid including
      |e| -> 1e3 (overload robustness).
  #6  Warm-starting the linear solve via init=w.detach() cuts
      Anderson iters substantially (the fabric's w_t moves slowly
      between samples; see LinearSolveLayer.forward init= kwarg).
"""
import math
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.circuit_block import LinearSolveLayer, EquilibriumSolve


# ----------------------------------------------------------------------------
# Robust parent-class lookup: handle both ``import run_rls_demo`` and the
# ``python run_rls_demo.py`` ``__main__`` case (plan decision 9).
# ----------------------------------------------------------------------------


def _lookup_attr(attr_name):
    """Return attr_name from ``run_rls_demo`` (imported) or ``__main__``
    (running as a script).  Falls back to ``import run_rls_demo`` last
    so tests that import ``run_rls_demo`` first are unaffected.
    """
    for mod_name in ('run_rls_demo', '__main__'):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, attr_name):
            return getattr(mod, attr_name)
    # Fallback: import as a module.  This is the path the test suite
    # takes; tests do ``sys.path.insert(0, src/)`` then ``import
    # run_rls_demo``.
    import importlib
    mod = importlib.import_module('run_rls_demo')
    return getattr(mod, attr_name)


# Late-bound base classes.  Evaluated at *class definition* time (i.e.
# at first import of this module), so the import context must be set up
# before importing ``learned_robust``.  The test file imports
# ``run_rls_demo`` first; ``run_rls_demo.py`` only imports
# ``learned_robust`` from inside ``train_robust_weighter`` and
# ``run_robust_experiment`` (plan decision 9).
FabricRLS = _lookup_attr('FabricRLS')
DigitalRLS = _lookup_attr('DigitalRLS')


class LearnedRobustWeighter(nn.Module):
    """Per-sample robust influence v(e) in [0, v_max].

    Parameterization (Phase B / plan decision B1, ``log_exp_v1``):

        c     = exp(log_c)                              # > 0 strictly
        alpha = exp(log_alpha)                          # > 0 strictly
        v(e)  = v_max * exp(-alpha * log1p((e / c)^2))

    Identity: v(0) = v_max.  Descends toward 0 for |e| >> c.
    Gaussian-control: alpha -> 0 collapses to v ~ v_max constant.

    ``log_c`` and ``log_alpha`` are the trainable coordinates so a fixed
    optimizer step changes c and alpha multiplicatively — appropriate
    for quantities spanning orders of magnitude (alpha may traverse
    0.13 to >= 1, potentially to > 20).  log1p-square is monotone and
    non-negative; the entire expression is monotonic in both |e| and
    alpha, so the local derivative scale at init matches the legacy
    softplus parameterization to within a small constant.

    Finite-precision positivity (plan B2): in float32,
    ``alpha * log1p((e/c)^2)`` may underflow to ``-inf`` and ``exp`` to
    zero; we clamp to ``[0, v_max]``.  The numerical statement is

        0 <= v(e) <= v_max  in finite precision;
        delta > 0  preserves R = X^T diag(v) X + delta I  SPD.

    The ``v_max`` cap is a fixed buffer in (0, 1] (not a learnable
    parameter) for backward compatibility with tests and scripts
    that read ``weighter.v_max`` (e.g.
    ``test_weighter_init_unit_check``).  Setting ``v_max`` away from 1
    is allowed but only a constant multiplicative rescale of v(e).

    Backward compatibility (plan B3):
      * ``raw_c`` / ``raw_alpha`` keyword args are accepted as deprecated
        legacy softplus coordinates and converted to ``log_c`` /
        ``log_alpha`` (a one-shot ``UserWarning`` is emitted).  New code
        should use ``c_init`` and ``alpha_init`` (the natural log-space
        coordinates).
      * ``load_state_dict`` migrates legacy checkpoints
        (``raw_c`` / ``raw_alpha``) to the new coordinates with a
        ``UserWarning`` and saves new checkpoints alongside the
        parameterization tag ``"log_exp_v1"``.
    """

    def __init__(self, c_init=0.10, alpha_init=0.13, raw_c=None,
                 raw_alpha=None, v_max=1.0):
        super().__init__()
        if raw_c is not None or raw_alpha is not None:
            warnings.warn(
                "LearnedRobustWeighter: legacy raw_c/raw_alpha kwargs are "
                "deprecated; pass c_init/alpha_init (log-space) instead. "
                "Converting legacy softplus coords to log_c/log_alpha.",
                UserWarning,
            )
            if raw_c is not None:
                c_legacy = F.softplus(torch.tensor(float(raw_c))) + 1e-3
                c_init = c_legacy.item()
            if raw_alpha is not None:
                a_legacy = F.softplus(torch.tensor(float(raw_alpha)))
                alpha_init = a_legacy.item()
        # Learned parameters in log-space (positive coordinates after
        # exponentiation).  Init at float64 so a later ``.to(dtype)``
        # preserves full precision (float32 init would round the value
        # before any cast); default float32 training still works via
        # ``.to(torch.float32)``.
        self.log_c = nn.Parameter(torch.tensor(math.log(max(float(c_init), 1e-30)),
                                               dtype=torch.float64))
        self.log_alpha = nn.Parameter(torch.tensor(math.log(max(float(alpha_init), 1e-30)),
                                                   dtype=torch.float64))
        # Fixed buffer (decision 1: v_max is fixed, not learned).
        self.register_buffer('v_max', torch.tensor(float(v_max)))

    @property
    def c(self):
        return self.log_c.exp()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def forward(self, e):
        """Compute v(e) = v_max * exp(-alpha * log1p((e/c)^2)).

        Parameters
        ----------
        e : scalar or (...,) tensor
            Per-sample prior error e_prior = d_t - w^T x_t.

        Returns
        -------
        v_t : shape broadcastable to ``e``
            Per-sample influence in ``[0, v_max]`` (in finite precision).
            Mathematically ``(0, v_max]``; underflows at saturation.
        """
        c = self.c
        alpha = self.alpha
        log_term = torch.log1p((e / c).square())
        v = self.v_max * torch.exp(-alpha * log_term)
        # Numerical floor / ceiling: in float32, exp can underflow to
        # exactly 0 for sufficiently large alpha*log_term; that is fine
        # because delta*I keeps the Gram matrix strictly positive definite.
        # Clip the rare fp-rounding over-vmax defensively.
        return v.clamp(min=0.0, max=self.v_max)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Migrate legacy softplus checkpoints on load (plan B3).

        Legacy state-dicts contain ``raw_c`` and ``raw_alpha``; the new
        parameterization uses ``log_c`` and ``log_alpha``.  If either
        legacy key is present we convert to log-space and emit a
        visible ``UserWarning``.  ``v_max`` is kept as a buffer (the new
        module also has a ``v_max`` buffer).
        """
        sd = state_dict
        legacy_keys = ('raw_c', 'raw_alpha')
        if any(k in sd for k in legacy_keys):
            warnings.warn(
                "Migrated legacy softplus weighter checkpoint "
                "(raw_c/raw_alpha -> log_c/log_alpha)",
                UserWarning,
            )
            sd = dict(sd)
            if 'raw_c' in sd:
                raw_c_t = sd.pop('raw_c')
                if isinstance(raw_c_t, torch.Tensor):
                    c_old = F.softplus(raw_c_t.to(torch.float64)) + 1e-3
                else:
                    c_old = F.softplus(torch.tensor(float(raw_c_t))) + 1e-3
                sd['log_c'] = torch.log(c_old.clamp_min(1e-30))
            if 'raw_alpha' in sd:
                raw_a_t = sd.pop('raw_alpha')
                if isinstance(raw_a_t, torch.Tensor):
                    a_old = F.softplus(raw_a_t.to(torch.float64))
                else:
                    a_old = F.softplus(torch.tensor(float(raw_a_t)))
                sd['log_alpha'] = torch.log(a_old.clamp_min(1e-30))
        return super().load_state_dict(sd, strict=strict, assign=assign)

    # ------------------------------------------------------------------
    # Metadata helpers (plan B3).
    # ------------------------------------------------------------------

    @torch.no_grad()
    def metadata(self, *, parameterization='log_exp_v1'):
        return {
            'weighter_parameterization': parameterization,
            'c': float(self.c.item()),
            'alpha': float(self.alpha.item()),
            'v_max': float(self.v_max.item()),
        }


# ----------------------------------------------------------------------------
# Constant weighter (used for byte-exact identity checks: gates 2a, 3).
# ----------------------------------------------------------------------------


def constant_weighter(value=1.0, dtype=None):
    """Stand-in weighter that returns a constant ``value`` (no params).

    Used for the byte-exact identity checks (gate 2a, gate 3): a
    DigitalRobustRLS instantiated with a constant weighter should match
    the plain DigitalRLS / FabricRLS trajectory to the bit.

    Parameters
    ----------
    value : float
        The constant influence value (typically 1.0).
    dtype : torch.dtype, optional
        Buffer dtype.  Defaults to the current default dtype so the
        forward path preserves whatever dtype the test stream uses
        (float32 for training, float64 for parity gates).
    """

    class _Const(nn.Module):
        def __init__(self, value, dtype):
            super().__init__()
            self.register_buffer('v_const', torch.tensor(float(value), dtype=dtype))

        def forward(self, e):
            # ``expand_as`` preserves the source dtype; casting to the
            # input dtype avoids silent promotion when the test stream
            # is float64 but the buffer is float32.
            return self.v_const.to(e.dtype).expand_as(e).clone()

    return _Const(value, dtype if dtype is not None else torch.get_default_dtype())


class FixedCauchyWeighter(nn.Module):
    """Stateless generalized-Cauchy weighter with FIXED (c, alpha) buffers.

        v(e) = v_max / (1 + (e/c)^2)^alpha

    The same family as ``LearnedRobustWeighter`` but with frozen,
    directly-specified curve parameters (no softplus indirection).  Two
    purposes:

      * the honest ablation contender ("trained vs its own init"): the
        plan's ``init_w`` is a ``LearnedRobustWeighter`` at its *training*
        init, which is equivalent but obfuscated.  A ``FixedCauchyWeighter``
        with the init's (c, alpha) makes the baseline explicit.
      * the (c, alpha) grid sweep: evaluating ``block_robust_rls`` over a
        dense grid of fixed curves without inverting softplus.
    """

    def __init__(self, c=0.107, alpha=0.127, v_max=1.0, dtype=None):
        super().__init__()
        dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.register_buffer('c', torch.tensor(float(c), dtype=dtype))
        self.register_buffer('alpha', torch.tensor(float(alpha), dtype=dtype))
        self.register_buffer('v_max', torch.tensor(float(v_max), dtype=dtype))

    def forward(self, e):
        c = self.c.to(device=e.device, dtype=e.dtype)
        alpha = self.alpha.to(device=e.device, dtype=e.dtype)
        v_max = self.v_max.to(device=e.device, dtype=e.dtype)
        ratio = e / c
        denom = (1.0 + ratio * ratio).clamp_min(1e-30)
        v = v_max / denom.pow(alpha)
        return v.clamp(min=1e-30, max=v_max)


class MADRobustWeighter(nn.Module):
    """Classical robust (Huber / Hampel) weighter with MAD scale normalization.

    The classical state-of-the-art fixed method: it adapts the influence
    function's scale to the data *per block*, which a static learned curve
    cannot do.  ``forward(e)`` recomputes, on every call,

        sigma_hat = 1.4826 * median(|e|)

    and returns the IRLS weight for the chosen robust loss:

      mode='huber':   v = 1            if |u| <= a
                      v = a / |u|      if |u| >  a        (u = e / sigma_hat)

      mode='hampel':  the 3-segment redescending weight with (a, b, r),
                      b ~ 3a, r ~ 8a by default:
                          v = 1                 |u| <= a
                          v = a / |u|           a < |u| <= b
                          v = a/b * (r-|u|)/(r-b)   b < |u| <= r
                          v = 0                 |u| > r

    Because ``block_robust_rls`` calls ``weighter(e)`` fresh every outer
    IRLS iteration, the scale is re-derived from the *current* residual
    block each call -- the per-block adaptivity property.  No parameters
    and no cross-call state, so the weight vector for a block depends only
    on that block's residuals.

    The weights are clamped to (0, 1] so ``R = X^T diag(v) X + delta*I``
    stays SPD (same invariant as the learned weighter).
    """

    def __init__(self, mode='huber', a=1.345, b=None, r=None, dtype=None):
        super().__init__()
        if mode not in ('huber', 'hampel'):
            raise ValueError(f"unknown MAD mode: {mode!r}")
        self.mode = mode
        # Hampel segment defaults (classical choices, b ~ 3a, r ~ 6-10a).
        if b is None:
            b = 3.0 * a
        if r is None:
            r = 8.0 * a
        dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.register_buffer('a', torch.tensor(float(a), dtype=dtype))
        self.register_buffer('b', torch.tensor(float(b), dtype=dtype))
        self.register_buffer('r', torch.tensor(float(r), dtype=dtype))

    def forward(self, e):
        a = self.a.to(device=e.device, dtype=e.dtype)
        b = self.b.to(device=e.device, dtype=e.dtype)
        r = self.r.to(device=e.device, dtype=e.dtype)
        # MAD-based scale, re-derived per call (no caching).
        sigma_hat = 1.4826 * e.abs().median()
        sigma_hat = sigma_hat.clamp_min(1e-12)
        u = e / sigma_hat
        au = u.abs()
        v = torch.ones_like(u)
        if self.mode == 'huber':
            v = torch.where(au > a, a / au.clamp_min(1e-30), v)
        else:  # hampel
            v = torch.where(au > a, a / au.clamp_min(1e-30), v)
            ramp = (a / b) * ((r - au) / (r - b)).clamp_min(0.0)
            v = torch.where(au > b, ramp, v)
            v = torch.where(au > r, torch.zeros_like(v), v)
        return v.clamp(min=1e-30, max=1.0)


# ----------------------------------------------------------------------------
# Fabric Robust RLS
# ----------------------------------------------------------------------------


class FabricRobustRLS(FabricRLS):
    """Per-sample weighted fabric RLS with a learned influence function.

    Per-step:
        e_prior = d_t - w^T x_t          # uses current w (with graph if training)
        v_t     = weighter(e_prior)      # (graph) if training else detached
        R_{t+1} = lambda * R_t + v_t * outer(x_t, x_t)
        p_{t+1} = lambda * p_t + v_t * d_t * x_t
        w_{t+1} = LinearSolveLayer(p_{t+1}, R_{t+1}, init=w.detach())  # warm-start

    The training flag controls whether v_t is detached inside ``step``
    (plan decision 5).  When training=True, v_t and the resulting R, p
    carry the autograd graph through the weighter, so
    ``.backward()`` at the loss gives non-zero grads on
    ``raw_c`` and ``raw_alpha`` (the mechanism exercised in
    Phase 0's ``test_weighter_grad_flow_simulation``).  When
    training=False, v_t is detached so the frozen-weighter
    Monte-Carlo accumulates a T-step autograd graph through R/p.

    The chebyshev step size is computed under ``torch.no_grad()`` --
    eigvalsh through the graph is pure waste (decision 11).
    """

    def __init__(self, d, weighter, lam=0.99, R0=None, w_init=None,
                 max_iter=100, tol=1e-8, beta='chebyshev', device='cpu',
                 training=False, log_every=0, log_name='',
                 backward_mode=None):
        # backward_mode:
        #   None (default) -> 'phantom' when training else 'exact'
        #   'exact' / 'phantom' / 'auto' (auto picks phantom iff T>backward_threshold)
        # The training mode is heavy because the chain through R/p grows
        # with the truncation window.  Phantom backward does one VJP per
        # implicit gradient step (instead of CG on J^T y = grad_out), so
        # the cost per step is O(chain_depth) rather than O(chain_depth *
        # cg_iter).  For the gate-4 implicit-vs-unrolled gradient test
        # (T=8, exact required) the test passes a `Backward_mode='exact'`
        # override; for the long-horizon training loop the default
        # `phantom` is the right choice.
        if backward_mode is None:
            backward_mode = 'phantom' if training else 'exact'
        super().__init__(d=d, lam=lam, R0=R0, w_init=w_init,
                         max_iter=max_iter, tol=tol, beta=beta,
                         device=device, log_every=log_every,
                         log_name=log_name)
        self.weighter = weighter
        self.training_mode = bool(training)
        self._backward_mode = backward_mode

    def step(self, x_t, d_t):
        """One weighted fabric RLS step.

        Parameters
        ----------
        x_t : (d,) tensor
            Input regressor vector.
        d_t : scalar tensor
            Observation d_t = w_o^T x_t + nu_t.

        Returns
        -------
        w : (d,) tensor
            New equalizer taps ``w_{t+1} = R_{t+1}^{-1} p_{t+1}``.
        """
        # Predict prior error using the current w.  This naturally
        # inherits the autograd graph from w (training) -- threading
        # through the weighter.
        e_prior = d_t - self.w @ x_t
        v_t = self.weighter(e_prior)
        if not self.training_mode:
            v_t = v_t.detach()

        # Weighted RLS recurrences.  Both R and p grow the graph through
        # v_t when training; the toolchain in EquilibriumSolve handles
        # the shared-graph re-traversal via retain_graph=True.
        self.R = self.lam * self.R + v_t * torch.outer(x_t, x_t)
        self.p = self.lam * self.p + v_t * d_t * x_t

        # Chebyshev step under no_grad: R carries a graph during training,
        # but eigvalsh through the graph is wasteful (decision 11).
        if self.beta_mode == 'chebyshev':
            with torch.no_grad():
                beta = self._chebyshev_beta(self.R)
        else:
            beta = float(self.beta_mode)

        # Solve via the linear equilibrium, warm-started at the previous
        # w.detach() (KIMI correction #6).  Gradient-safe: init carries
        # no graph, so the implicit backward is unaffected.
        # During training we use the cheap 'phantom' implicit gradient
        # (Geng et al. 2021): one VJP step instead of the full CG on
        # J^T y = grad_out.  This is biased but bounded; it lets us
        # chain T=128+ training steps without the exact-mode backward
        # becoming exponential in T.  During eval (training=False) the
        # graph is detached anyway, so backward_mode is moot.
        bm = self._backward_mode
        layer = LinearSolveLayer(max_iter=self.max_iter, tol=self.tol, beta=beta,
                                 backward_mode=bm)
        w_star = layer(self.p.unsqueeze(0), self.R, init=self.w.detach().unsqueeze(0)).squeeze(0)

        # Track settle iterations for the histogram metric.
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


# ----------------------------------------------------------------------------
# Digital Robust RLS
# ----------------------------------------------------------------------------


class DigitalRobustRLS(DigitalRLS):
    """O(d^2) digital RLS with per-sample influence weighting.

    Corrected recursion per plan finding A and decision 3:

        Px      = P @ x_t
        k       = Px / (lam / v_t + x_t^T Px)         # == v_t Px / (lam + v_t x^T Px)
        e       = d_t - w^T x_t
        w       = w + k * e                            # no v_t factor
        P       = (P - outer(k, Px)) / lam             # no v_t factor

    Limits:
        v_t = 1  -> byte-identical to ``DigitalRLS``.
        v_t -> 0 -> k = 0, w frozen, P -> P / lam (sample ignored,
        matching the fabric forgetting).

    The training flag is applied to v_t detach (decision 5).  The
    Woodbury update above is purely numerical; there is no graph to
    grow through ``P``.
    """

    def __init__(self, d, weighter, lam=0.99, delta=1.0, w_init=None,
                 device='cpu', training=False):
        super().__init__(d=d, lam=lam, delta=delta, w_init=w_init,
                         device=device)
        self.weighter = weighter
        self.training_mode = bool(training)

    def step(self, x_t, d_t):
        e_prior = d_t - self.w @ x_t
        v_t = self.weighter(e_prior)
        if not self.training_mode:
            v_t = v_t.detach()

        Px = self.P @ x_t
        # Corrected denom: lam / v_t + x_t^T Px; v_t -> 0 freezes k.
        denom = self.lam / v_t + x_t @ Px
        k = Px / denom
        e = d_t - self.w @ x_t
        self.w = self.w + k * e
        self.P = (self.P - torch.outer(k, Px)) / self.lam
        return self.w.clone()


# ----------------------------------------------------------------------------
# Phase 1.5 - Block Robust IRLS (block-parallel reconcile)
# ----------------------------------------------------------------------------


def make_block(w_o, T, sigma, *, mode='iid', noise='gaussian', p_burst=0.02,
               kappa=20.0, seed=0, dtype=torch.float64, device='cpu',
               return_noise=False):
    """Generate (X, d_obs) block of shape (T, d) and (T,).

    Mirrors ``run_rls_demo.make_stream`` but without the streaming
    accumulation; uses ``_make_noise`` for the noise model so the
    contaminated-Gaussian / impulsive mode is reusable in the block
    setting.  The block regressor is the row-stacked collection of T
    i.i.d. N(0, I) samples (or AR(1) for the ``mode='ar1'`` case) --
    a (T, d) observation matrix.

    Parameters
    ----------
    w_o : (d,) tensor
        True plant (unit-norm by convention).
    T : int
        Number of samples in the block.
    sigma : float
        Nominal Gaussian noise scale.
    return_noise : bool, optional
        If True, also return the true additive noise ``nu`` and the true
        burst mask ``burst`` (Phase 0 oracle-oos-headroom).  In
        simulation these are available exactly (``nu = d_obs - X @ w_o``
        and the generator's Bernoulli burst gate), so the oracle weighter
        needs no generator-state replay.  Default False keeps the return
        type backward-compatible.

    Returns
    -------
    X : (T, d) tensor
    d_obs : (T,) tensor, ``X @ w_o + nu``
    nu : (T,) tensor, optional (if ``return_noise=True``)
        True additive noise.
    burst : (T,) tensor, optional (if ``return_noise=True``)
        True burst indicator (1 = burst, 0 = clean; all-zero for a clean
        block).
    """
    # Lazy import to avoid the circular dependency (plan decision 9).
    _make_noise = _lookup_attr('_make_noise')
    device = torch.device(device) if not isinstance(device, torch.device) else device
    g = torch.Generator(device=device).manual_seed(seed)
    d = w_o.shape[0]
    X = torch.zeros(T, d, dtype=dtype, device=device)
    if mode == 'iid':
        X = torch.randn(T, d, generator=g, dtype=dtype, device=device)
    elif mode == 'ar1':
        rho = 0.9
        eps = torch.randn(T, d, generator=g, dtype=dtype, device=device)
        X[0] = eps[0]
        for t in range(1, T):
            X[t] = rho * X[t - 1] + (1.0 - rho * rho) ** 0.5 * eps[t]
    else:
        raise ValueError(f"unknown mode: {mode!r}")
    nu, burst = _make_noise(T, sigma, mode=noise, p_burst=p_burst, kappa=kappa,
                            generator=g, dtype=dtype, device=device,
                            return_burst=True)
    d_obs = X @ w_o + nu
    if return_noise:
        return X, d_obs, nu, burst
    return X, d_obs


def oracle_weighter(burst, *, floor=1e-3, dtype=None, device='cpu'):
    """Oracle influence weighter: v=1 on clean rows, v~0 on burst rows.

    Phase 0 oracle ceiling (plan ``oracle-oos-headroom``).  The oracle
    knows the TRUE burst mask (in simulation, the noise generator's
    Bernoulli gate, returned by ``make_block(..., return_noise=True)``)
    and down-weights exactly those rows to ``floor``.  It is an
    idealized upper bound on any learned/fixed curve: perfect
    contamination knowledge, no reliance on residual magnitudes.

    ``floor`` is kept strictly positive (default 1e-3) so that
    ``R = X^T diag(v) X + delta*I`` stays SPD (the same invariant the
    ``LearnedRobustWeighter`` guarantees via v_t > 0).  As ``floor -> 0``
    the oracle approaches the exact reweighted-LS "discard the bursts"
    solution.

    On a clean block (``burst`` all-zero, e.g. ``noise='gaussian'``) the
    mask is empty and the oracle reduces to plain batch LS -- the
    ``test_oracle_bound`` gate checks exactly this reduction.

    Parameters
    ----------
    burst : (T,) tensor
        True burst indicator (1 = burst, 0 = clean).
    floor : float, optional
        Weight applied to burst rows (default 1e-3).
    dtype : torch.dtype, optional
        Buffer dtype (defaults to the current default dtype).
    device : str or torch.device, optional
        Buffer device (defaults to 'cpu').

    Returns
    -------
    nn.Module
        Callable ``(e) -> v`` returning per-sample weights in
        ``[floor, 1]`` broadcastable to the residual shape (T,).
    """
    dtype = dtype if dtype is not None else torch.get_default_dtype()
    device = torch.device(device) if not isinstance(device, torch.device) else device

    class _Oracle(nn.Module):
        def __init__(self, burst, floor, dtype, device):
            super().__init__()
            self.register_buffer('mask', burst.to(device).bool())
            self.register_buffer('floor', torch.tensor(float(floor), dtype=dtype))

        def forward(self, e):
            # v = 1 everywhere, then floor exactly the burst rows.
            v = torch.ones_like(e)
            v = torch.where(self.mask.to(e.device), self.floor.to(e.dtype), v)
            return v

    return _Oracle(burst, floor, dtype, device)


def block_robust_rls(X, d, weighter, delta, K, *, settle=None, w_init=None,
                     max_iter=100, tol=1e-8, beta=1.0,
                     backward_tol=None, backward_max_iter=None,
                     backward_mode='exact', settle_log=None,
                     settle_info_log=None,
                     backend='dense', solver='deq'):
    """Block-parallel robust IRLS (the Phase 1.5 construction).

    Per plan "The construction": the sequential dependency is across
    K outer iterations, not T samples.  Weights are block-parallel:

        R0 = X^T X + delta * I
        w  = settle(R0, X^T d)                              # settle 1: plain batch LS
        for k in range(K):
            e  = d - X @ w                                  # (T,) vectorized
            v  = weighter(e)                                # (T,) block-parallel
            R  = X^T (v[:, None] * X) + delta * I
            p  = X^T (v * d)
            w  = settle(R, p, v0=w.detach())                # warm-start
        return w                                            # K+1 settles, independent of T

    Weighter graphs are carried through (no detach): backprop through
    the K outer iterations is the training signal.  The settle uses
    ``LinearSolveLayer`` with ``backward_mode='phantom'`` by default
    (cheap VJP per impl step; the implicit gradient is biased but
    bounded, and the bias is measured in the Phase 1.5 gate 5).

    Two backends are supported (selected by ``backend``):

    * ``'dense'`` (default, backward-compatible): forms ``R`` and ``p``
      every outer iteration and solves ``R w = p`` via the supplied
      ``settle`` (typically ``LinearSolveLayer``).  This is the legacy
      path from Phase 1.5.  In this backend, ``solver='direct'``
      replaces the equilibrium settle with ``torch.linalg.solve`` —
      the decisive training diagnostic for whether the failure mode is
      solver accuracy (Phase C5).  ``solver='direct'`` is fully
      differentiable (``linalg.solve`` is differentiable for
      non-singular R).

    * ``'circuit_stamp'``: assembles the KCL residual
      ``f(w) = X^T [v * (y - X w)] - delta w`` from
      ``TransformerConductanceBank`` + ``LeakageToGround`` and solves
      ``f(w) = 0`` via ``WeightedGramCircuitSolve``.  ``R`` is never
      formed in the forward path; it is implicit in the
      ``EquilibriumSolve`` Jacobian used during the backward pass.
      ``conductance`` (= v) is the explicit ``u`` argument to
      ``EquilibriumSolve`` so the weighter receives gradients.

    For the circuit backend the wrapper requires ``backward_mode='exact'``
    on ``WeightedGramCircuitSolve`` for training -- the cheap
    ``phantom`` mode is catastrophically biased at production scale
    (rel_bias ~ 1e29 at the 2026-08-17 operating point).  For the
    ``dense`` backend the same caution applies.

    Parameters
    ----------
    X : (T, d) tensor
        Block regressor.
    d : (T,) tensor
        Observations ``X @ w_o + nu``.
    weighter : callable
        Maps (T,) residuals to (T,) influence weights v(e) in [0, v_max].
    delta : float
        Identity regularization strength for ``R = X^T diag(v) X + delta * I``.
    K : int
        Number of outer IRLS iterations.
    settle : callable, optional
        ``(p, R, init=None) -> w`` solver.  Defaults to
        ``LinearSolveLayer(...)`` with the specified max_iter/tol/beta.
        Ignored when ``backend='circuit_stamp'`` (the solver is then
        built into ``WeightedGramCircuitSolve``).  Ignored when
        ``solver='direct'`` (the closed-form solve replaces it).
    w_init : (d,) tensor, optional
        Initial guess for the first settle.  Defaults to ``torch.zeros(d)``.
    solver : {'deq', 'direct'}
        Inner solve mode (only when ``backend='dense'``).  ``'deq'``
        uses the supplied ``settle`` (default ``LinearSolveLayer``);
        ``'direct'`` uses ``torch.linalg.solve`` (the Phase C5
        diagnostic for isolating DEQ-specific gradient errors).
        settle_log : list, optional
        If given, appends the settle-iteration count (``n_iter`` from
        ``EquilibriumSolve.last_info``) after each of the K+1 settles so
        the caller can histogram the full settle cost.  For
        ``solver='direct'``, ``-1`` is appended at each settle (the
        closed-form solve has no iteration count).  Best-effort: a
        custom ``settle`` that does not update ``EquilibriumSolve.last_info``
        simply contributes nothing.
        backend : {'dense', 'circuit_stamp'}
        Selects the residual-assembly and inner-solver path.  Default
        ``'dense'`` preserves the prior behavior exactly.

    Returns
    -------
    w : (d,) tensor
        Final settled tap voltages after K+1 outer iterations.
    """
    if backend not in ('dense', 'circuit_stamp'):
        raise ValueError(
            f"backend must be 'dense' or 'circuit_stamp', got {backend!r}")
    if solver not in ('deq', 'direct'):
        raise ValueError(
            f"solver must be 'deq' or 'direct', got {solver!r}")
    if backend == 'circuit_stamp' and solver != 'deq':
        raise ValueError(
            "backend='circuit_stamp' only supports solver='deq'")
    d_dim = X.shape[1]
    device = X.device
    dtype = X.dtype
    def _record_settle():
        if settle_info_log is not None:
            info = EquilibriumSolve.last_info
            settle_info_log.append(dict(info) if info is not None else {})
    if backend == 'circuit_stamp':
        if backward_mode != 'exact':
            raise ValueError(
                "backend='circuit_stamp' requires backward_mode='exact'; "
                "phantom backward is not valid for the weighted-Gram circuit path")
        # Lazy import: circuit_stamp itself lazy-imports EquilibriumSolve.
        from utils.circuit_stamp import WeightedGramCircuitSolve
        solver_cfg = {
            'method': 'anderson',
            'max_iter': max_iter,
            'tol': tol,
            'beta': beta,
            'backward_mode': backward_mode,
            'backward_tol': tol,
            'backward_max_iter': max_iter,
        }
        # ``auto_beta=True`` makes the wrapper compute the chebyshev step
        # from R's spectrum each IRLS round (consistent with the dense
        # backend's behavior in ``_chebyshev_beta``).  The forward
        # residual itself still does not form R; only the solver's step
        # size does (under no_grad).
        circuit = WeightedGramCircuitSolve(delta=delta, solver_cfg=solver_cfg,
                                           auto_beta=True)

        if w_init is None:
            w_init = torch.zeros(d_dim, dtype=dtype, device=device)

        # settle 1: plain batch LS using ones for v.
        v0 = torch.ones(d.shape[0], dtype=dtype, device=device)
        w = circuit(X=X, y=d, conductance=v0, w0=w_init)
        if settle_log is not None and EquilibriumSolve.last_info is not None:
            settle_log.append(EquilibriumSolve.last_info.get('n_iter', -1))
        _record_settle()

        for _ in range(K):
            e = d - X @ w
            v = weighter(e)  # graph carried through; do not detach.
            w = circuit(X=X, y=d, conductance=v,
                        w0=w.detach())
            if settle_log is not None and EquilibriumSolve.last_info is not None:
                settle_log.append(EquilibriumSolve.last_info.get('n_iter', -1))
            _record_settle()
        return w

    # backend == 'dense'
    if solver == 'direct':
        # Closed-form decide: compute R, p each outer iteration and solve
        # exactly via torch.linalg.solve (differentiable, no iteration
        # budget).  The Phase C5 diagnostic: if this direct path reaches
        # the grid valley while the DEQ path does not, the failure mode
        # is solver accuracy, not training geometry.
        if w_init is None:
            w_init = torch.zeros(d_dim, dtype=dtype, device=device)
        eye = delta * torch.eye(d_dim, dtype=dtype, device=device)

        R0 = X.t() @ X + eye
        p0 = X.t() @ d
        w = torch.linalg.solve(R0, p0)
        if settle_log is not None:
            settle_log.append(-1)  # closed-form: no iter count

        for _ in range(K):
            e = d - X @ w
            v = weighter(e)
            R = X.t() @ (v.unsqueeze(-1) * X) + eye
            p = X.t() @ (v * d)
            # Warm-start via w.detach() is irrelevant for direct solve
            # (no fixed-point iteration) but kept consistent.
            w = torch.linalg.solve(R, p)
            if settle_log is not None:
                settle_log.append(-1)
            # Direct solves have no EquilibriumSolve info to record.
        return w

    # solver == 'deq' (default dense path, behavior preserved exactly)
    if settle is None:
        # Default: chebyshev-style beta computed per settle from the
        # spectral radius of R, so the iteration converges for any
        # (positive-definite) R.  The fixed-beta=1.0 default inherited
        # from LinearSolveLayer is unsafe for large R (N >> d).
        def _chebyshev_beta(R):
            with torch.no_grad():
                eigs = torch.linalg.eigvalsh(R)
                lam_min = eigs[0].item()
                lam_max = eigs[-1].item()
            if lam_max <= 0:
                return 1.0
            return float(2.0 / (lam_min + lam_max))
        settle_fn = lambda p_, R_, init=None: LinearSolveLayer(
            max_iter=max_iter, tol=tol, beta=_chebyshev_beta(R_),
            backward_mode=backward_mode,
            backward_tol=backward_tol,
            backward_max_iter=backward_max_iter,
        )(p_, R_, init=init)
    else:
        settle_fn = settle
    if w_init is None:
        w_init = torch.zeros(d_dim, dtype=dtype, device=device)
    eye = delta * torch.eye(d_dim, dtype=dtype, device=device)

    # settle 1: plain batch LS (unweighted).
    R0 = X.t() @ X + eye
    p0 = X.t() @ d
    w = settle_fn(p0.unsqueeze(0), R0, init=w_init.unsqueeze(0)).squeeze(0)
    if settle_log is not None and EquilibriumSolve.last_info is not None:
        settle_log.append(EquilibriumSolve.last_info.get('n_iter', -1))
    _record_settle()

    for _ in range(K):
        e = d - X @ w
        v = weighter(e)
        # R = X^T diag(v) X + delta * I -- block-parallel accumulate.
        R = X.t() @ (v.unsqueeze(-1) * X) + eye
        p = X.t() @ (v * d)
        w = settle_fn(p.unsqueeze(0), R, init=w.detach().unsqueeze(0)).squeeze(0)
        if settle_log is not None and EquilibriumSolve.last_info is not None:
            settle_log.append(EquilibriumSolve.last_info.get('n_iter', -1))
        _record_settle()
    return w


def digital_block_robust_rls(X, d, weighter, delta, K, *, w_init=None):
    """Block-parallel robust IRLS in digital form (no autograd graph).

    Same construction as ``block_robust_rls`` but uses
    ``torch.linalg.solve`` for the inner solve.  Useful as the
    byte-comparable digital twin: in the ``v(e) == 1`` limit this
    reduces to plain batch LS.
    """
    d_dim = X.shape[1]
    device = X.device
    dtype = X.dtype
    if w_init is None:
        w_init = torch.zeros(d_dim, dtype=dtype, device=device)
    eye = delta * torch.eye(d_dim, dtype=dtype, device=device)

    R0 = X.t() @ X + eye
    p0 = X.t() @ d
    w = torch.linalg.solve(R0, p0)

    for _ in range(K):
        e = d - X @ w
        v = weighter(e)
        R = X.t() @ (v.unsqueeze(-1) * X) + eye
        p = X.t() @ (v * d)
        w = torch.linalg.solve(R, p)
    return w


def _block_settle_iter_count(settle):
    """Return the n_iter from the most recent ``settle`` call (``EquilibriumSolve.last_info``).

    Block IRLS invokes the settle ``K+1`` times; the caller can sum
    these to populate the settle-iter histogram metric.
    """
    info = EquilibriumSolve.last_info
    if info is None:
        return -1
    return info.get('n_iter', -1)


def measure_phantom_vs_exact_bias(X, d_obs, w_o, weighter, delta, K, *,
                                  max_iter=100, tol=1e-8):
    """Measure the phantom-gradient bias at the given weighter's operating point.

    Training defaults to ``backward_mode='exact'`` (CG adjoint, validated
    to ~1e-10 and re-verified at production scale by the FD gradcheck) and
    the chained-K phantom VJP is catastrophically wrong at production
    scale (rel_bias ~1e29 at the 2026-08-17 operating point).  This
    measurement exists to keep that phantom-vs-exact gap quantified;
    ``weighter`` is typically the fully-trained weighter: its params are
    copied into two fresh instances so the only difference between the
    phantom and exact runs is ``backward_mode``.

    The loss is the block-training loss ``||X w^K - X w_o||^2`` (the
    noiseless supervision signal of Phase 1.5 decision 1).

    This is a MEASUREMENT, not a pass/fail bound (phantom is biased by
    construction; report the bias, don't tune it away).

    Returns dict(rel_bias, phantom_grad, exact_grad) with
    ``rel_bias = |g_phantom - g_exact| / |g_exact|``.
    """
    def _clone(w):
        w2 = LearnedRobustWeighter(c_init=0.10, alpha_init=0.13)
        device = next(w.parameters()).device
        dtype = next(w.parameters()).dtype
        w2 = w2.to(device).to(dtype)
        with torch.no_grad():
            w2.log_c.copy_(w.log_c.to(device=device, dtype=dtype))
            w2.log_alpha.copy_(w.log_alpha.to(device=device, dtype=dtype))
        return w2

    w_phantom = _clone(weighter)
    w_K_p = block_robust_rls(X, d_obs, w_phantom, delta=delta, K=K,
                             max_iter=max_iter, tol=tol,
                             backward_mode='phantom')
    loss_p = (X @ w_K_p - X @ w_o).pow(2).sum()
    loss_p.backward()
    g_phantom = w_phantom.log_c.grad.item()

    w_exact = _clone(weighter)
    w_K_e = block_robust_rls(X, d_obs, w_exact, delta=delta, K=K,
                             max_iter=max_iter, tol=tol,
                             backward_mode='exact')
    loss_e = (X @ w_K_e - X @ w_o).pow(2).sum()
    loss_e.backward()
    g_exact = w_exact.log_c.grad.item()

    rel_bias = abs(g_phantom - g_exact) / max(abs(g_exact), 1e-12)
    return {
        'rel_bias': float(rel_bias),
        'phantom_grad': float(g_phantom),
        'exact_grad': float(g_exact),
    }

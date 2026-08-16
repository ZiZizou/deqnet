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
import sys

import torch
import torch.nn as nn

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
    """Per-sample robust influence v(e) in (0, 1].

    Parameterization (plan decision 1, KIMI correction #1):

        c     = softplus(raw_c) + 1e-3          # > 1e-3 strictly
        alpha = softplus(raw_alpha)            # > 0 strictly
        v(e)  = v_max / (1 + (e/c)^2)^alpha

    Identity: v(0) = v_max = 1.  Descends toward 0 for |e| >> c.
    Gaussian-control: alpha -> 0 collapses to v ~ v_max constant.

    ``raw_c`` is initialized to -2.25 (c ~ 0.107) so the knee sits
    between nominal errors (sigma ~ 0.01) and the impulsive burst
    magnitude (kappa * sigma ~ 0.2).  ``raw_alpha`` is initialized to
    -2.0 (alpha ~ 0.127) to match the plan's "alpha small, v_max near 1
    near e ~ 0" init.

    The ``v_max`` cap is a fixed buffer in (0, 1] (not a learnable
    parameter) so v_t > 0 strictly and R = lambda R + v_t xx^T stays
    SPD with probability 1.
    """

    def __init__(self, raw_c=-2.25, raw_alpha=-2.0, v_max=1.0):
        super().__init__()
        # Learned parameters (decision 1: raw_c, raw_alpha are learned).
        self.raw_c = nn.Parameter(torch.tensor(float(raw_c)))
        self.raw_alpha = nn.Parameter(torch.tensor(float(raw_alpha)))
        # Fixed buffer (decision 1: v_max is fixed, not learned).
        self.register_buffer('v_max', torch.tensor(float(v_max)))

    @property
    def c(self):
        # softplus(raw_c) + 1e-3 keeps c > 1e-3 strictly.
        # softplus is monotone, so backprop is smooth and unconstrained.
        return torch.nn.functional.softplus(self.raw_c) + 1e-3

    @property
    def alpha(self):
        return torch.nn.functional.softplus(self.raw_alpha)

    def forward(self, e):
        """Compute v(e) = v_max / (1 + (e/c)^2)^alpha.

        Parameters
        ----------
        e : scalar or (...,) tensor
            Per-sample prior error e_prior = d_t - w^T x_t.

        Returns
        -------
        v_t : shape broadcastable to ``e``
            Per-sample influence in (0, v_max] (strictly positive).
        """
        c = self.c
        alpha = self.alpha
        ratio = e / c
        denom = (1.0 + ratio * ratio).clamp_min(1e-30)
        v = self.v_max / denom.pow(alpha)
        # Numerical floor / ceiling: v_t must be in (0, v_max].
        # softplus + clamping guarantees analytically, but rounding
        # during the eigvalsh-adjacent matmuls can occasionally push
        # the value slightly above v_max; clip defensively.
        return v.clamp(min=1e-30, max=self.v_max)


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
                     backward_mode='phantom', settle_log=None):
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

    Parameters
    ----------
    X : (T, d) tensor
        Block regressor.
    d : (T,) tensor
        Observations ``X @ w_o + nu``.
    weighter : callable
        Maps (T,) residuals to (T,) influence weights v(e) in (0, 1].
    delta : float
        Identity regularization strength for ``R = X^T diag(v) X + delta * I``.
    K : int
        Number of outer IRLS iterations.
    settle : callable, optional
        ``(p, R, init=None) -> w`` solver.  Defaults to
        ``LinearSolveLayer(...)`` with the specified max_iter/tol/beta.
    w_init : (d,) tensor, optional
        Initial guess for the first settle.  Defaults to ``torch.zeros(d)``.
    settle_log : list, optional
        If given, appends the settle-iteration count (``n_iter`` from
        ``EquilibriumSolve.last_info``) after each of the K+1 settles so
        the caller can histogram the full settle cost.  Best-effort: a
        custom ``settle`` that does not update ``EquilibriumSolve.last_info``
        simply contributes nothing.
    """
    d_dim = X.shape[1]
    device = X.device
    dtype = X.dtype
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

    for _ in range(K):
        e = d - X @ w
        v = weighter(e)
        # R = X^T diag(v) X + delta * I -- block-parallel accumulate.
        R = X.t() @ (v.unsqueeze(-1) * X) + eye
        p = X.t() @ (v * d)
        w = settle_fn(p.unsqueeze(0), R, init=w.detach().unsqueeze(0)).squeeze(0)
        if settle_log is not None and EquilibriumSolve.last_info is not None:
            settle_log.append(EquilibriumSolve.last_info.get('n_iter', -1))
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

    Training uses ``backward_mode='phantom'`` (one cheap VJP per implicit
    step, Geng et al. 2021) but every existing gate validates only the
    exact (CG) adjoint (~1e-10).  The gradient actually used for learning
    is biased by construction; this measures that bias on the *trained*
    configuration.  ``weighter`` is typically the fully-trained weighter:
    its params are copied into two fresh instances so the only difference
    between the phantom and exact runs is ``backward_mode``.

    The loss is the block-training loss ``||X w^K - X w_o||^2`` (the
    noiseless supervision signal of Phase 1.5 decision 1).

    This is a MEASUREMENT, not a pass/fail bound (phantom is biased by
    construction; report the bias, don't tune it away).

    Returns dict(rel_bias, phantom_grad, exact_grad) with
    ``rel_bias = |g_phantom - g_exact| / |g_exact|``.
    """
    def _clone(w):
        w2 = LearnedRobustWeighter(raw_c=0.0, raw_alpha=0.0)
        w2 = w2.to(w.raw_c.dtype).to(w.raw_c.device)
        with torch.no_grad():
            w2.raw_c.copy_(w.raw_c)
            w2.raw_alpha.copy_(w.raw_alpha)
        return w2

    w_phantom = _clone(weighter)
    w_K_p = block_robust_rls(X, d_obs, w_phantom, delta=delta, K=K,
                             max_iter=max_iter, tol=tol,
                             backward_mode='phantom')
    loss_p = (X @ w_K_p - X @ w_o).pow(2).sum()
    loss_p.backward()
    g_phantom = w_phantom.raw_c.grad.item()

    w_exact = _clone(weighter)
    w_K_e = block_robust_rls(X, d_obs, w_exact, delta=delta, K=K,
                             max_iter=max_iter, tol=tol,
                             backward_mode='exact')
    loss_e = (X @ w_K_e - X @ w_o).pow(2).sum()
    loss_e.backward()
    g_exact = w_exact.raw_c.grad.item()

    rel_bias = abs(g_phantom - g_exact) / max(abs(g_exact), 1e-12)
    return {
        'rel_bias': float(rel_bias),
        'phantom_grad': float(g_phantom),
        'exact_grad': float(g_exact),
    }

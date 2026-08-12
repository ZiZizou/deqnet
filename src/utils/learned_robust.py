"""Phase 1 - Learned Robust IR-RLS.

Implements the learned robust influence function and its two RLS variants:

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

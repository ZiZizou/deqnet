"""Element-level circuit-stamp realization of the weighted Gram-matrix IRLS residual.

This module implements the mapping documented in
``docs/weighted_gram_circuit.md``: replace the abstract dense residual

    f(w) = p - R w,    R = X^T diag(v) X + delta I,    p = X^T (v * y)

with an algebraically identical but element-level KCL assembly

    f(w) = X^T [ v * (y - X w) ] - delta w,

distributed across three physical primitives:

* ``TransformerConductanceBank``  -- ideal signed multiport transformer per
  sample row, followed by a positive per-sample conductance v_t.
* ``LeakageToGround``            -- scalar shunt conductance delta from
  each tap node to ground.
* ``WeightedGramCircuitLayer``   -- assembles the KCL residual from the
  bank and the leakage.

The forward path does **not** form ``R`` or ``p``.  It only computes
branch voltage drops, branch currents, the transformer-transpose KCL
injection, and the per-tap shunt leakage.  These three pieces give the
same answer as ``p - R w`` to within float-rounding (Gate 1).

The DEQ wrapper ``WeightedGramCircuitSolve`` passes ``v`` (the
per-sample conductance) as the explicit ``u`` argument to
``EquilibriumSolve.apply``.  This is required so ``EquilibriumSolve.backward``
returns a ``v.grad`` that propagates through the learned weighter back to
``raw_c`` and ``raw_alpha``.  Closure-captured ``v`` would not receive a
gradient through the standard Function return signature.

Implementation notes
====================

* ``X`` and ``y`` are treated as data (closure-captured constants) for
  this milestone.  If the team later requires differentiation through
  them, ``EquilibriumSolve`` must be extended to accept multiple
  explicit tensor inputs and return gradients for all of them -- do
  not silently fall back to ``.grad`` side effects on non-leaf
  closures.

* The shared ``linear_conductance_current`` helper is the I-V law
  ``current = gain * voltage_drop`` -- the same shape as the existing
  ``Conductance`` device uses.  Both can call it for stylistic
  symmetry; we do not refactor ``Conductance`` here.

* ``TransformerConductanceBank.forward`` accepts unbatched tensors
  ``X: (T, d)``, ``y: (T,)``, ``conductance: (T,)``, ``w: (d,)``.  Batched
  shapes ``X: (B, T, d)`` etc. are not implemented in the first
  milestone; if a batched use case arrives, extend ``forward`` with
  explicit ``einsum`` calls and Gate 9-shape-validation tests.
"""
from __future__ import annotations

from typing import Callable, Dict, Any, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Shared I-V law helper
# ---------------------------------------------------------------------------


def linear_conductance_current(drop: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
    """Linear (ohmic) I-V law shared by the existing ``Conductance`` device
    and the new ``TransformerConductanceBank`` branch primitive.

    ``current = gain * voltage_drop`` elementwise; both arguments must be
    broadcastable.  Used here only by the transformer-bank branch current;
    the existing ``Conductance`` device (which is a two-terminal edge)
    inlines the same expression but does not call this helper -- keeping
    it as a separate, optional convenience for future refactors.
    """
    return gain * drop


# ---------------------------------------------------------------------------
# TransformerConductanceBank: ideal signed transformer + positive conductance
# ---------------------------------------------------------------------------


class TransformerConductanceBank(nn.Module):
    """Ideal signed transformer bank followed by one positive conductance per row.

    For each sample ``t``:

        q_t = x_t^T w - y_t                  (branch voltage drop)
        i_t = v_t * q_t                      (branch current, ohmic)
        j   = -X^T i                         (current injected into taps)

    The transformer is the transpose of the per-tap coupling that
    appears in the regression: the forward path sees the row ``x_t``
    mixed onto the taps (a voltage-mode transformer reading the tap
    voltages), the inverse path couples the branch current back into
    the taps.  Both directions use the same ``X`` coefficients.

    Requirements (asserted at forward time):

    * ``conductance.shape == y.shape``  -- one conductance per branch.
    * ``X.shape == (len(y), len(w))``   -- one signed coupling per
      (tap, branch) pair.
    * ``conductance >= 0``               -- rejected in debug mode
      (negative conductance breaks the SPD contract).

    The forward is autograd-safe; do not wrap in ``torch.no_grad()``.
    """

    def __init__(self, validate_nonneg: bool = False):
        super().__init__()
        # Default: no runtime validation (production fast path).  Tests
        # set validate_nonneg=True to assert the "positive conductance"
        # contract (Gate 9).
        self.validate_nonneg = bool(validate_nonneg)

    def branch_drop(self, w: torch.Tensor, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """``q[t] = sum_i X[t, i] * w[i] - y[t]``."""
        return X @ w - y

    def branch_current(self, branch_drop: torch.Tensor, conductance: torch.Tensor) -> torch.Tensor:
        """``i[t] = v[t] * q[t]`` via the shared ohmic helper."""
        return linear_conductance_current(branch_drop, conductance)

    def kcl_injection(self, branch_current: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """``j[i] = -sum_t X[t, i] * i[t]`` (transformer-transpose KCL)."""
        return -(X.t() @ branch_current)

    def forward(
        self,
        w: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
        conductance: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the KCL injection ``-X^T (v * (X w - y))``.

        Parameters
        ----------
        w : (d,) tensor
            Tap-node voltages.
        X : (T, d) tensor
            Per-sample signed transformer coupling.
        y : (T,) tensor
            Per-sample branch reference / source voltage.
        conductance : (T,) tensor
            Per-sample positive branch conductance.
        """
        if w.dim() != 1:
            raise ValueError(f"w must be 1-D (d,), got shape {tuple(w.shape)}")
        if X.dim() != 2:
            raise ValueError(f"X must be 2-D (T, d), got shape {tuple(X.shape)}")
        if y.dim() != 1:
            raise ValueError(f"y must be 1-D (T,), got shape {tuple(y.shape)}")
        if conductance.dim() != 1:
            raise ValueError(f"conductance must be 1-D (T,), got shape {tuple(conductance.shape)}")
        T, d = X.shape
        if w.shape[0] != d:
            raise ValueError(f"w has dim {w.shape[0]} but X has {d} columns")
        if y.shape[0] != T:
            raise ValueError(f"y has length {y.shape[0]} but X has {T} rows")
        if conductance.shape[0] != T:
            raise ValueError(f"conductance has length {conductance.shape[0]} but X has {T} rows")
        # Reject non-finite values at the element boundary so NaN/Inf cannot
        # turn into a misleading fixed-point convergence failure downstream.
        for name, value in (("w", w), ("X", X), ("y", y),
                            ("conductance", conductance)):
            if not torch.isfinite(value).all().item():
                raise ValueError(f"{name} must contain only finite values")
        if self.validate_nonneg and (conductance < 0).any().item():
            raise ValueError("TransformerConductanceBank: negative conductance rejected")

        q = self.branch_drop(w, X, y)
        i = self.branch_current(q, conductance)
        return self.kcl_injection(i, X)


# ---------------------------------------------------------------------------
# LeakageToGround: scalar shunt conductance from each tap node to ground
# ---------------------------------------------------------------------------


class LeakageToGround(nn.Module):
    """Scalar shunt leakage from each tap node to ground.

    Holds ``delta`` as a Python float (registered as a buffer only when
    needed by serialization; for the first milestone it is a fixed
    scalar that does not depend on the call).  Per-element tap leakage
    ``delta_diag`` is intentionally not supported in this milestone --
    it is a hardware-fidelity concern and would change the sign of
    ``f`` per row of the data matrix.
    """

    def __init__(self, delta: float):
        super().__init__()
        if float(delta) <= 0.0:
            raise ValueError(f"delta must be > 0, got {delta!r}")
        self.delta = float(delta)

    def current(self, w: torch.Tensor) -> torch.Tensor:
        """Current leaving each tap node to ground: ``delta * w``."""
        return self.delta * w

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return self.current(w)


# ---------------------------------------------------------------------------
# WeightedGramCircuitLayer: assembles the KCL residual
# ---------------------------------------------------------------------------


class WeightedGramCircuitLayer(nn.Module):
    """Circuit-stamped realization of weighted ridge least squares.

    .. math::

        f(w) = X^T [ v * (y - X w) ] - delta * w

    The forward path computes the KCL injection from
    ``TransformerConductanceBank`` and subtracts the per-tap shunt
    leakage from ``LeakageToGround``.  ``R = X^T diag(v) X + delta I``
    is **never** formed in ``forward``.
    """

    def __init__(self, delta: float, validate_nonneg: bool = False):
        super().__init__()
        self.delta = float(delta)
        if self.delta <= 0.0:
            raise ValueError(f"delta must be > 0, got {self.delta!r}")
        self.bank = TransformerConductanceBank(validate_nonneg=validate_nonneg)
        self.leakage = LeakageToGround(self.delta)

    def rhs(
        self,
        w: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
        conductance: torch.Tensor,
    ) -> torch.Tensor:
        """KCL residual ``f(w) = X^T [ v * (y - X w) ] - delta w``.

        Note the sign of the transformer-bank output: the bank returns
        ``j = -X^T (v * (X w - y)) = X^T [ v * (y - X w) ]`` -- this
        is exactly the first term of ``f``.  Subtracting the leakage
        completes the residual.
        """
        branch_injection = self.bank(w, X, y, conductance)
        leakage_current = self.leakage.current(w)
        return branch_injection - leakage_current

    def forward(
        self,
        w: torch.Tensor,
        X: torch.Tensor,
        y: torch.Tensor,
        conductance: torch.Tensor,
    ) -> torch.Tensor:
        return self.rhs(w, X, y, conductance)


# ---------------------------------------------------------------------------
# Dense-equivalence helpers (for tests and certificates, NOT for forward)
# ---------------------------------------------------------------------------


def dense_residual(w: torch.Tensor, X: torch.Tensor, y: torch.Tensor,
                   conductance: torch.Tensor, delta: float) -> torch.Tensor:
    """Reference dense residual ``p - R w`` (FOR TESTS / DIAGNOSTICS ONLY).

    The dense path is mathematically equivalent to the element-level stamp
    but it materializes ``R = X^T diag(v) X + delta I`` and ``p = X^T (v *
    y)``.  It must never be used inside the production forward path --
    it defeats the architectural motivation of the circuit interpretation.
    Used by Gate 1 and Gate 6 to assert that the element-stamped residual
    matches the dense one to numerical tolerance.
    """
    d = X.shape[-1]
    eye = delta * torch.eye(d, dtype=X.dtype, device=X.device)
    R = X.t() @ (conductance.unsqueeze(-1) * X) + eye
    p = X.t() @ (conductance * y)
    return p - R @ w


def weighted_gram_certificate(X: torch.Tensor, conductance: torch.Tensor,
                              delta: float) -> Dict[str, Any]:
    """Contraction certificate for the weighted Gram solve.

    Reports the spectrum of the stiffness matrix

        M = X^T diag(v) X + delta I,

    which is SPD whenever ``conductance >= 0`` and ``delta > 0``.  The
    eigenvalues are computed exactly via ``torch.linalg.eigvalsh`` --
    only feasible for small d (typical for this project, d = 8..32).
    For large systems, switch to a matrix-free estimate.

    The guaranteed lower bound on ``lambda_min(M)`` is ``delta``
    (regardless of ``X`` and ``v``), which is what the DEQ solver's
    contraction guarantee relies on.
    """
    d = X.shape[-1]
    eye = delta * torch.eye(d, dtype=X.dtype, device=X.device)
    R = X.t() @ (conductance.unsqueeze(-1) * X) + eye
    eigvals = torch.linalg.eigvalsh(R)
    lam_min = float(eigvals.min().item())
    lam_max = float(eigvals.max().item())
    return {
        "lambda_min_M": lam_min,
        "lambda_max_M": lam_max,
        "condition_number": float(lam_max / max(lam_min, 1e-30)),
        "guaranteed_lower_bound": float(delta),
    }


# ---------------------------------------------------------------------------
# WeightedGramCircuitSolve: DEQ wrapper that exposes ``conductance`` explicitly
# ---------------------------------------------------------------------------


class WeightedGramCircuitSolve(nn.Module):
    """Implicit solve of the circuit-stamped weighted Gram residual.

    ``conductance`` (``v``) is the explicit ``u`` argument to
    ``EquilibriumSolve.apply``, not a closure-captured tensor.  This
    guarantees that ``EquilibriumSolve.backward`` returns a
    ``conductance.grad`` that propagates through the outer autograd
    graph (the weighter) to the learnable parameters.

    ``X`` and ``y`` remain closure-captured because they are data, not
    parameters.  If a future use case requires gradients through them,
    extend ``EquilibriumSolve`` rather than relying on closure-side
    ``.grad`` effects.

    Step-size policy
    ----------------

    The legacy fixed-point map ``g(w) = w + beta * f(w)`` for the circuit
    residual ``f(w) = p - R w`` converges iff
    ``beta < 2 / lambda_max(R)`` where
    ``R = X^T diag(v) X + delta I``.  For typical
    ``T ~ d ~ 8..32``, ``lambda_max(R)`` can be ~10..100, so a default
    ``beta = 1.0`` is unstable and the solver diverges.  We follow the
    legacy dense backend's policy: compute ``beta`` from the spectrum of
    ``R`` under ``torch.no_grad()`` (``torch.linalg.eigvalsh`` of ``R``).
    This is a one-time ``O(d^3)`` cost that does NOT enter the residual
    assembly -- the forward path still avoids forming ``R``.

    Set ``auto_beta=False`` (or pass ``solver_cfg`` with an explicit
    ``beta``) to override this and use a fixed step.  This is the
    ``-no-chebyshev`` style used by some debug tests.

    For very large systems where ``eigvalsh`` is too expensive, a
    conservative bound

        lambda_max(R) <= delta + sum_t v_t ||x_t||^2

    yields the step

        beta = 2 / (delta + delta + sum_t v_t ||x_t||^2).

    This option is exposed via ``conservative_beta=True``.
    """

    def __init__(self, delta: float, solver_cfg: Optional[Dict[str, Any]] = None,
                 validate_nonneg: bool = False, auto_beta: bool = True,
                 conservative_beta: bool = False):
        super().__init__()
        self.layer = WeightedGramCircuitLayer(delta, validate_nonneg=validate_nonneg)
        self.auto_beta = bool(auto_beta)
        self.conservative_beta = bool(conservative_beta)
        # Default to a strict solver config (small max_iter, very small tol)
        # so the float64 parity gates converge to the direct-solve answer.
        # ``beta`` is computed at forward time (see ``auto_beta``).
        self.solver_cfg = solver_cfg if solver_cfg is not None else {
            "method": "anderson",
            "max_iter": 100,
            "tol": 1e-12,
            "beta": 1.0,
            "backward_mode": "exact",
            "backward_tol": 1e-12,
            "backward_max_iter": 100,
        }

    def _chebyshev_beta_from_R(self, X: torch.Tensor, y: torch.Tensor,
                               conductance: torch.Tensor) -> float:
        """Compute ``2 / (lambda_min + lambda_max)`` of ``R`` under no_grad.

        Builds ``R = X^T diag(v) X + delta I`` for spectrum extraction
        only.  The residual ``forward`` path does not form ``R``.
        """
        d = X.shape[-1]
        eye = self.layer.delta * torch.eye(d, dtype=X.dtype, device=X.device)
        with torch.no_grad():
            R = X.t() @ (conductance.unsqueeze(-1) * X) + eye
            eigs = torch.linalg.eigvalsh(R)
            lam_min = eigs[0].item()
            lam_max = eigs[-1].item()
        if lam_max <= 0:
            return 1.0
        return float(2.0 / (lam_min + lam_max))

    def _conservative_beta(self, X: torch.Tensor, conductance: torch.Tensor) -> float:
        """Conservative upper-bound beta (no eigvalsh).

        Uses ``lambda_max(R) <= delta + sum_t v_t ||x_t||^2``, so
        ``beta = 2 / (delta + (delta + sum_t v_t ||x_t||^2))``.
        Suitable for large d where eigvalsh is too expensive.
        """
        delta = self.layer.delta
        with torch.no_grad():
            row_norms_sq = (X * X).sum(dim=-1)
            ub = delta + (conductance * row_norms_sq).sum().item()
        return float(2.0 / (delta + max(ub, 1e-30)))

    def forward(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        conductance: torch.Tensor,
        w0: torch.Tensor,
    ) -> torch.Tensor:
        """Solve the circuit-stamped residual to its fixed point.

        Returns ``w*`` with the same shape as ``w0``.
        """
        layer = self.layer
        cfg = dict(self.solver_cfg)  # shallow copy, we may override beta
        if self.auto_beta:
            if self.conservative_beta:
                cfg["beta"] = self._conservative_beta(X, conductance)
            else:
                cfg["beta"] = self._chebyshev_beta_from_R(X, y, conductance)

        def rhs_fn(w: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
            return layer.rhs(w, X=X, y=y, conductance=g)

        # EquilibriumSolve is imported lazily here to avoid a hard
        # dependency cycle at module-import time (circuit_block.py
        # itself imports this module via the dense-path wrapper).
        from utils.circuit_block import EquilibriumSolve
        return EquilibriumSolve.apply(rhs_fn, w0, conductance, cfg)
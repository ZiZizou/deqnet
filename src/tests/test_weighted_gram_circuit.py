"""Test gates for the circuit-stamped weighted-Gram IRLS residual.

These gates verify the element-level circuit interpretation described in
``docs/weighted_gram_circuit.md`` and the implementation in
``src/utils/circuit_stamp.py``.  All exact-algebra and gradient tests run
in float64 on CPU (Gate 9 keeps the production float32 path intact by
construction: the validation only fires when ``validate_nonneg=True``).

Gate list (mirrors §7 of the design note):
    1. element RHS matches dense RHS in float64
    2. energy-gradient identity f = -grad E
    3. Jacobian symmetry and (N)SD
    4. solve parity with direct solve and LinearSolveLayer
    5. v = 1 circuit parity with plain ridge LS
    6. multi-round block IRLS parity (K in {0, 1, 4, 8})
    7. learned-weighter gradient parity (raw_c, raw_alpha)
    8. finite-difference gradient check through circuit path
    9. invalid input safety checks

Run all gates:
    cd ~/Documents/deqnet/src && python tests/test_weighted_gram_circuit.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch

from utils.circuit_block import LinearSolveLayer
from utils.circuit_stamp import (
    TransformerConductanceBank,
    LeakageToGround,
    WeightedGramCircuitLayer,
    WeightedGramCircuitSolve,
    weighted_gram_certificate,
    linear_conductance_current,
    dense_residual,
)
from utils.learned_robust import (
    block_robust_rls,
    LearnedRobustWeighter,
    constant_weighter,
)


# Force float64 for the exact-algebra gates. Tests are designed to run in
# float64; production paths stay float32 (the element-level operations
# are dtype-agnostic).
torch.set_default_dtype(torch.float64)


# ----------------------------------------------------------------------------
# Common helpers
# ----------------------------------------------------------------------------


def _seeded_inputs(T: int, d: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(T, d, generator=g)
    # Signed X is the hard case (transformer must handle both polarities).
    w = torch.randn(d, generator=g)
    y = torch.randn(T, generator=g)
    conductance = torch.rand(T, generator=g) + 0.05   # strictly > 0
    return X, w, y, conductance


def _signed_inputs(T: int, d: int, seed: int = 0):
    """Fully signed X (uniform on [-1, 1]) to exercise the transformer polarity."""
    g = torch.Generator().manual_seed(seed)
    X = (torch.rand(T, d, generator=g) * 2.0 - 1.0)
    w = torch.randn(d, generator=g)
    y = torch.randn(T, generator=g)
    conductance = torch.rand(T, generator=g) + 0.05
    return X, w, y, conductance


# ----------------------------------------------------------------------------
# Gate 1: element RHS matches dense RHS
# ----------------------------------------------------------------------------


def test_gate1_element_matches_dense():
    """Element-stamped RHS == dense RHS in float64.

    Uses signed X so the transformer polarity (the hard case) is exercised.
    """
    X, w, y, conductance = _signed_inputs(T=32, d=8, seed=0)
    delta = 0.01
    layer = WeightedGramCircuitLayer(delta=delta)

    rhs_circuit = layer.rhs(w, X, y, conductance)
    rhs_dense = dense_residual(w, X, y, conductance, delta)

    err = (rhs_circuit - rhs_dense).abs().max().item()
    print(f"  Gate 1: max abs diff = {err:.3e}")
    assert err < 1e-10, f"element vs dense mismatch: {err}"
    print("  PASS")
    return err


# ----------------------------------------------------------------------------
# Gate 2: energy-gradient identity
# ----------------------------------------------------------------------------


def test_gate2_energy_gradient_identity():
    """f(w) = -grad E(w) where E = 1/2 sum v*(Xw-y)^2 + delta/2 ||w||^2.

    This is the strongest local circuit-stamp correctness test: any
    sign or shape error in the bank/leakage assembly is caught here.
    """
    X, w0, y, conductance = _signed_inputs(T=24, d=6, seed=1)
    delta = 0.01
    layer = WeightedGramCircuitLayer(delta=delta)

    w = w0.detach().clone().requires_grad_(True)
    energy = 0.5 * (conductance * (X @ w - y).square()).sum()
    energy = energy + 0.5 * delta * w.square().sum()

    grad_energy = torch.autograd.grad(energy, w)[0]
    rhs = layer.rhs(w.detach(), X, y, conductance)

    err = (rhs + grad_energy).abs().max().item()
    print(f"  Gate 2: |f + grad E| = {err:.3e}")
    assert err < 1e-10, f"energy-gradient identity violated: {err}"
    print("  PASS")
    return err


# ----------------------------------------------------------------------------
# Gate 3: Jacobian symmetry and (N)SD
# ----------------------------------------------------------------------------


def test_gate3_jacobian_symmetric_nsd():
    """J_f = -(X^T diag(v) X + delta I) is symmetric and negative-definite."""
    X, w, y, conductance = _signed_inputs(T=16, d=6, seed=2)
    delta = 0.01
    layer = WeightedGramCircuitLayer(delta=delta)

    J = torch.autograd.functional.jacobian(
        lambda ww: layer.rhs(ww, X, y, conductance), w,
    )

    # Symmetry
    sym_err = (J - J.t()).abs().max().item()
    print(f"  Gate 3 symmetry: |J - J^T| = {sym_err:.3e}")
    assert sym_err < 1e-10, f"Jacobian not symmetric: {sym_err}"

    # Negative-definite: -J is SPD, lambda_min > 0, lambda_min >= delta.
    eigvals = torch.linalg.eigvalsh(-J)
    lam_min = eigvals.min().item()
    print(f"  Gate 3 -J eigvals: min={eigvals[0].item():.4e}"
          f" max={eigvals[-1].item():.4e} delta={delta:.4e}")
    assert lam_min > 0, f"-J not positive-definite: min={lam_min}"
    # Tolerate a small amount of float-rounding below delta.
    assert lam_min >= delta - 1e-10, (
        f"lambda_min(-J) = {lam_min} < delta = {delta} (numerical bound violated)")
    print("  PASS")


# ----------------------------------------------------------------------------
# Gate 4: solve parity with direct solve and LinearSolveLayer
# ----------------------------------------------------------------------------


def test_gate4_solve_parity():
    """Circuit solve agrees with torch.linalg.solve and LinearSolveLayer.

    Tolerance: relative error < 1e-8 (not byte-identical; different
    operation order, Anderson history, etc.).
    """
    X, _, y, conductance = _signed_inputs(T=32, d=8, seed=3)
    delta = 0.01

    d = X.shape[-1]
    eye = delta * torch.eye(d, dtype=X.dtype)
    R = X.t() @ (conductance.unsqueeze(-1) * X) + eye
    p = X.t() @ (conductance * y)
    w_ref = torch.linalg.solve(R, p)

    # LinearSolveLayer (dense backend, fixed-beta for safety on small system).
    layer_dense = LinearSolveLayer(max_iter=200, tol=1e-12, beta=0.5,
                                   backward_mode='exact')
    w_dense = layer_dense(p.unsqueeze(0), R, init=torch.zeros(1, d)).squeeze(0)

    # Circuit-stamped solve via WeightedGramCircuitSolve.
    solver_cfg = {
        'method': 'anderson', 'max_iter': 500, 'tol': 1e-12,
        'beta': 0.5, 'm': 5,
        'backward_mode': 'exact', 'backward_tol': 1e-12, 'backward_max_iter': 200,
    }
    solver = WeightedGramCircuitSolve(delta=delta, solver_cfg=solver_cfg)
    w_circuit = solver(X=X, y=y, conductance=conductance,
                       w0=torch.zeros(d))

    rel_dense = (w_dense - w_ref).norm().item() / w_ref.norm().item()
    rel_circuit = (w_circuit - w_ref).norm().item() / w_ref.norm().item()
    print(f"  Gate 4: rel(LinearSolveLayer - direct) = {rel_dense:.3e}"
          f"  rel(circuit - direct) = {rel_circuit:.3e}")
    assert rel_dense < 1e-8, f"LinearSolveLayer parity: {rel_dense}"
    assert rel_circuit < 1e-8, f"circuit parity: {rel_circuit}"
    print("  PASS")


# ----------------------------------------------------------------------------
# Gate 5: v = 1 circuit parity with plain ridge LS
# ----------------------------------------------------------------------------


def test_gate5_v_equals_one_parity():
    """With v = ones, the circuit solve matches plain ridge batch LS:
    w = (X^T X + delta I)^{-1} X^T y.
    """
    X, _, y, _ = _signed_inputs(T=64, d=8, seed=4)
    delta = 0.01

    d = X.shape[-1]
    eye = delta * torch.eye(d, dtype=X.dtype)
    R = X.t() @ X + eye
    p = X.t() @ y
    w_ref = torch.linalg.solve(R, p)

    v = torch.ones_like(y)
    solver_cfg = {
        'method': 'anderson', 'max_iter': 500, 'tol': 1e-12,
        'beta': 0.5, 'm': 5,
        'backward_mode': 'exact', 'backward_tol': 1e-12, 'backward_max_iter': 200,
    }
    solver = WeightedGramCircuitSolve(delta=delta, solver_cfg=solver_cfg)
    w_circuit = solver(X=X, y=y, conductance=v, w0=torch.zeros(d))

    rel = (w_circuit - w_ref).norm().item() / w_ref.norm().item()
    print(f"  Gate 5: rel(circuit - plain-ridge) = {rel:.3e}")
    assert rel < 1e-8, f"v=1 circuit parity: {rel}"
    print("  PASS")


# ----------------------------------------------------------------------------
# Gate 6: multi-round block IRLS parity
# ----------------------------------------------------------------------------


def test_gate6_multi_round_irls_parity():
    """Both backends produce the same w and same MSE for K in {0, 1, 4, 8}.

    Uses a FixedCauchyWeighter (deterministic, no graph surprises) and the
    same (X, y, init) for both backends.  Strict float64 tolerance.
    """
    from utils.learned_robust import FixedCauchyWeighter

    T, d = 48, 8
    g = torch.Generator().manual_seed(5)
    X = torch.randn(T, d, generator=g)
    w_o = torch.randn(d, generator=g)
    w_o = w_o / w_o.norm()
    nu = torch.randn(T, generator=g) * 0.01
    y = X @ w_o + nu
    delta = 0.01
    wgt = FixedCauchyWeighter(c=0.107, alpha=0.127)

    for K in (0, 1, 4, 8):
        w_d = block_robust_rls(X, y, wgt, delta=delta, K=K,
                               max_iter=200, tol=1e-12,
                               backward_mode='exact',
                               backend='dense')
        w_cs = block_robust_rls(X, y, wgt, delta=delta, K=K,
                                max_iter=200, tol=1e-12,
                                backward_mode='exact',
                                backend='circuit_stamp')

        rel_w = (w_d - w_cs).norm().item() / max(w_d.norm().item(), 1e-12)
        mse_d = (X @ w_d - y).pow(2).mean().item()
        mse_cs = (X @ w_cs - y).pow(2).mean().item()
        rel_mse = abs(mse_d - mse_cs) / max(abs(mse_d), 1e-30)
        print(f"  Gate 6 K={K}: rel(w)={rel_w:.3e}  rel(mse)={rel_mse:.3e}")
        assert rel_w < 1e-6, f"K={K} rel(w)={rel_w}"
        assert rel_mse < 1e-9, f"K={K} rel(mse)={rel_mse}"
    print("  PASS")


# ----------------------------------------------------------------------------
# Gate 7: learned-weighter gradient parity
# ----------------------------------------------------------------------------


def test_gate7_learned_weighter_grad_parity():
    """Gradients w.r.t. log_c and log_alpha match between dense and circuit backends.

    A scalar loss is used: ``||X w_hat - X w_true||^2``.  Both gradients
    must be finite, nonzero, and agree to ``1e-5`` relative.
    """
    T, d = 32, 6
    g = torch.Generator().manual_seed(7)
    X = torch.randn(T, d, generator=g)
    w_o = torch.randn(d, generator=g)
    w_o = w_o / w_o.norm()
    nu = torch.randn(T, generator=g) * 0.05
    y = X @ w_o + nu
    delta = 0.01
    K = 3

    def _run(backend: str):
        # c_init/alpha_init match the legacy raw_c=-2.25 / raw_alpha=-2.0
        # operating point (softplus(-2.25)+1e-3 ~ 0.1012, softplus(-2) ~ 0.1269).
        weighter = LearnedRobustWeighter(c_init=0.10119, alpha_init=0.12693)
        w_K = block_robust_rls(X, y, weighter, delta=delta, K=K,
                               max_iter=200, tol=1e-12,
                               backward_mode='exact',
                               backend=backend)
        loss = (X @ w_K - X @ w_o).pow(2).mean()
        loss.backward()
        return {
            'log_c_grad': weighter.log_c.grad.item(),
            'log_alpha_grad': weighter.log_alpha.grad.item(),
            'loss': loss.item(),
        }

    res_d = _run('dense')
    res_cs = _run('circuit_stamp')

    for name in ('log_c_grad', 'log_alpha_grad'):
        g_d = res_d[name]
        g_cs = res_cs[name]
        assert math.isfinite(g_d) and math.isfinite(g_cs), f"non-finite {name}"
        assert abs(g_d) > 0 and abs(g_cs) > 0, f"zero {name}"
        rel = abs(g_d - g_cs) / max(abs(g_d), 1e-12)
        print(f"  Gate 7 {name}: dense={g_d:.4e}  circuit={g_cs:.4e}  rel={rel:.3e}")
        assert rel < 1e-5, f"{name} parity: {rel}"
    print(f"  Gate 7 loss: dense={res_d['loss']:.4e}  circuit={res_cs['loss']:.4e}")
    print("  PASS")


# ----------------------------------------------------------------------------
# Gate 8: finite-difference gradient check through circuit path
# ----------------------------------------------------------------------------


def test_gate8_finite_difference_gradcheck():
    """Exact implicit grad on log_c matches central FD through the full circuit pipeline.

    Small problem (d=4, T=16, K=3, float64, strict solver) so FD noise is small.
    """
    T, d = 16, 4
    g = torch.Generator().manual_seed(8)
    X = torch.randn(T, d, generator=g)
    w_o = torch.randn(d, generator=g)
    w_o = w_o / w_o.norm()
    nu = torch.randn(T, generator=g) * 0.05
    y = X @ w_o + nu
    delta = 0.01
    K = 2

    def _loss_for(log_c_val: float) -> float:
        # c = exp(log_c); the legacy raw_c=-2.25 corresponds to
        # c ~ softplus(-2.25)+1e-3 ~ 0.1012, so log_c ~ -2.2906.
        weighter = LearnedRobustWeighter(c_init=math.exp(log_c_val),
                                         alpha_init=0.12693)
        w_K = block_robust_rls(X, y, weighter, delta=delta, K=K,
                               max_iter=300, tol=1e-13,
                               backward_mode='exact',
                               backend='circuit_stamp')
        return float((X @ w_K - X @ w_o).pow(2).mean().item())

    # Central finite difference over log_c (single-parameter perturbation).
    h = 1e-3
    center = math.log(0.1012)
    L_plus = _loss_for(center + h)
    L_minus = _loss_for(center - h)
    fd_grad = (L_plus - L_minus) / (2 * h)

    # Implicit-grad path on the same perturbed weighter.
    weighter = LearnedRobustWeighter(c_init=math.exp(center), alpha_init=0.12693)
    w_K = block_robust_rls(X, y, weighter, delta=delta, K=K,
                           max_iter=300, tol=1e-13,
                           backward_mode='exact',
                           backend='circuit_stamp')
    loss = (X @ w_K - X @ w_o).pow(2).mean()
    loss.backward()
    impl_grad = weighter.log_c.grad.item()

    rel = abs(impl_grad - fd_grad) / max(abs(fd_grad), 1e-12)
    print(f"  Gate 8: implicit grad on log_c = {impl_grad:.4e}"
          f"  FD grad = {fd_grad:.4e}  rel = {rel:.3e}")
    assert math.isfinite(impl_grad), "implicit grad not finite"
    assert abs(impl_grad) > 0, "implicit grad is zero"
    # FD is itself noisy at 1e-3; allow generous tolerance (~1e-2 relative).
    assert rel < 1e-2, f"FD gradcheck: {rel}"
    print("  PASS")


# ----------------------------------------------------------------------------
# Gate 9: invalid input safety checks
# ----------------------------------------------------------------------------


def test_gate9_invalid_input_checks():
    """Reject negative conductance, non-positive delta, bad shapes, NaN/Inf."""
    layer = WeightedGramCircuitLayer(delta=0.01, validate_nonneg=True)

    # Bad delta at construction.
    raised = False
    try:
        WeightedGramCircuitLayer(delta=0.0)
    except ValueError:
        raised = True
    assert raised, "delta=0 must raise"
    print("  Gate 9: delta <= 0 rejected")

    # Negative conductance rejected.
    X, w, y, _ = _signed_inputs(T=8, d=4, seed=9)
    bad_v = -torch.ones(8) * 0.1
    raised = False
    try:
        layer.rhs(w, X, y, bad_v)
    except ValueError:
        raised = True
    assert raised, "negative conductance must raise when validate_nonneg=True"
    print("  Gate 9: negative conductance rejected")

    # Incompatible shapes.
    X_bad = torch.randn(8, 4)            # T=8 OK
    w_bad = torch.randn(5)                # d mismatch
    raised = False
    try:
        layer.rhs(w_bad, X_bad, y[:8], torch.rand(8))
    except ValueError:
        raised = True
    assert raised, "shape mismatch (w vs X) must raise"
    print("  Gate 9: shape mismatch (w) rejected")

    y_bad = torch.randn(7)
    raised = False
    try:
        layer.rhs(w[:4], X_bad, y_bad, torch.rand(7))
    except ValueError:
        raised = True
    assert raised, "shape mismatch (y vs X) must raise"
    print("  Gate 9: shape mismatch (y) rejected")

    # Non-finite X / y / v are rejected at the element boundary.
    bad_cases = []
    X_nan = torch.randn(8, 4)
    X_nan[0, 0] = float("nan")
    bad_cases.append(("X", torch.zeros(4), X_nan, torch.zeros(8), torch.ones(8)))
    y_nan = torch.zeros(8)
    y_nan[0] = float("inf")
    bad_cases.append(("y", torch.zeros(4), torch.randn(8, 4), y_nan, torch.ones(8)))
    v_nan = torch.ones(8)
    v_nan[0] = float("nan")
    bad_cases.append(("conductance", torch.zeros(4), torch.randn(8, 4),
                      torch.zeros(8), v_nan))
    for label, ww, XX, yy, vv in bad_cases:
        raised = False
        try:
            layer.rhs(ww, XX, yy, vv)
        except ValueError:
            raised = True
        assert raised, f"non-finite {label} must raise"
    print("  Gate 9: non-finite inputs rejected")
    
    # Without validate_nonneg, the bank silently allows negative v
    # (this is the production fast path used by the trained weighter's
    # strict-positive invariant).
    layer_loose = WeightedGramCircuitLayer(delta=0.01, validate_nonneg=False)
    try:
        layer_loose.rhs(torch.zeros(4), torch.randn(8, 4), torch.zeros(8),
                        -torch.ones(8) * 0.1)
    except Exception as e:
        raise AssertionError(
            f"validate_nonneg=False must allow negative v, got {e!r}")
    print("  Gate 9: validate_nonneg=False allows negative v (production path)")

    print("  PASS")


# ----------------------------------------------------------------------------
# Bonus: weighted_gram_certificate
# ----------------------------------------------------------------------------


def test_weighted_gram_certificate():
    """Certificate matches the eigvalsh of R and the lambda_min >= delta bound."""
    X, _, _, conductance = _signed_inputs(T=32, d=6, seed=10)
    delta = 0.01

    cert = weighted_gram_certificate(X, conductance, delta)

    d = X.shape[-1]
    eye = delta * torch.eye(d, dtype=X.dtype)
    R = X.t() @ (conductance.unsqueeze(-1) * X) + eye
    eigvals = torch.linalg.eigvalsh(R)

    print(f"  Certificate: {cert}")
    print(f"  Eigvalsh:    min={eigvals[0].item():.4e} max={eigvals[-1].item():.4e}")
    assert abs(cert["lambda_min_M"] - eigvals[0].item()) < 1e-10
    assert abs(cert["lambda_max_M"] - eigvals[-1].item()) < 1e-10
    assert cert["guaranteed_lower_bound"] == delta
    assert cert["lambda_min_M"] >= delta - 1e-10, (
        f"lambda_min_M = {cert['lambda_min_M']} < delta = {delta}")
    print("  PASS")


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------


def run_all():
    print("#" * 60)
    print("# Circuit-Stamp Test Gates (float64 CPU)")
    print("#" * 60)
    failed = []
    for name, fn in [
        ("Gate 1 element == dense", test_gate1_element_matches_dense),
        ("Gate 2 energy-gradient", test_gate2_energy_gradient_identity),
        ("Gate 3 Jacobian sym/NSD", test_gate3_jacobian_symmetric_nsd),
        ("Gate 4 solve parity", test_gate4_solve_parity),
        ("Gate 5 v=1 parity", test_gate5_v_equals_one_parity),
        ("Gate 6 multi-round IRLS", test_gate6_multi_round_irls_parity),
        ("Gate 7 weighter grad", test_gate7_learned_weighter_grad_parity),
        ("Gate 8 FD gradcheck", test_gate8_finite_difference_gradcheck),
        ("Gate 9 invalid input", test_gate9_invalid_input_checks),
        ("weighted_gram_certificate", test_weighted_gram_certificate),
    ]:
        print(f"\n[{name}]")
        try:
            fn()
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed.append(name)
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            failed.append(name)
    if failed:
        print(f"\n!!! {len(failed)} gates failed:")
        for n in failed:
            print(f"  - {n}")
        sys.exit(1)
    print("\nALL CIRCUIT-STAMP GATES PASSED.")


if __name__ == '__main__':
    run_all()
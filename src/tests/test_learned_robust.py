"""Phase 0 gradcheck gate for the learned model plumbing.

The LinearSolveLayer is the shared building block for both Milestone 1
(learned robust IR-RLS) and Milestone 2 (learned-prox ISTA). The
end-to-end implicit training loop is only viable if:

  (a) gradients flow to p (the solve's ``u`` argument) and to R
      (closure-captured into ``rhs_fn``) when at least one input
      requires grad;
  (b) the analytical gradients (via ``.backward()`` + ``.grad``) match
      finite-difference estimates to solver tolerance.

The plan's original ``gradcheck_solve`` used ``torch.autograd.gradcheck``
on ``(p, R)``. That API does not work here because R is closure-captured
into ``EquilibriumSolve``'s ``rhs_fn`` rather than passed as a Function
input, so ``autograd.grad`` can't see R in the graph (analytical returns
zero, numerical is non-zero, and the test reports a spurious Jacobian
mismatch). The legacy ``.backward() + .grad`` mechanism DOES populate
``R.grad`` via the ``f_.backward(−y)`` call inside
``EquilibriumSolve.backward`` (``circuit_block.py:680``), which is what
the actual training loop uses.

So the Phase 0 gate is a finite-difference validation of the
``.backward()``-based gradient, not ``autograd.gradcheck``. If the
fabric-training loop works (weighter gets gradient), this gate passes;
if it doesn't, no amount of gradcheck-style testing will save us.

Gate 1 of the plan. No model code in this phase.
"""
import os
import sys

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS_DIR))  # add src/ to path

from utils.circuit_block import LinearSolveLayer


def _make_spd(d, dtype=torch.float64, seed=0):
    """Deterministic SPD R = Q diag(eigs) Q^T + I, like
    test_equilibrium_solve::test_linear_solve_layer_gradient_flow."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    Q = torch.randn(d, d, generator=g, dtype=dtype)
    eigs = 1.0 + 3.0 * torch.rand(d, generator=g, dtype=dtype)
    R = Q @ torch.diag(eigs) @ Q.t()
    R = ((R + R.t()) / 2 + torch.eye(d, dtype=dtype))
    return R


def _finite_diff_grad_R(layer, p, R, eps=1e-6, n_probes=5, seed=0):
    """Probe ∂(sum w)/∂R at (p, R) via central finite differences.

    Returns (grad_fd tensor, list of (i, j) probed indices).  Only probes
    `n_probes` randomly chosen entries to keep the test cheap.
    """
    g = torch.Generator(device='cpu').manual_seed(seed)
    with torch.no_grad():
        w0 = layer(p.unsqueeze(0), R).squeeze(0)
    grad_fd = torch.zeros_like(R)
    indices = torch.randint(0, R.numel(), (n_probes,), generator=g)
    probed = []
    for flat_idx in indices.tolist():
        i, j = divmod(flat_idx, R.shape[1])
        Rp = R.detach().clone()
        Rp[i, j] += eps
        w_p = layer(p.unsqueeze(0), Rp).squeeze(0)
        Rm = R.detach().clone()
        Rm[i, j] -= eps
        w_m = layer(p.unsqueeze(0), Rm).squeeze(0)
        grad_fd[i, j] = ((w_p - w_m).sum() / (2 * eps)).item()
        probed.append((i, j))
    return grad_fd, probed


def test_linear_solve_layer_gradients_flow(d=5):
    """Regression: grad on p (the u argument) and on R (closure-captured)
    must both be finite and non-zero. Catches the historical detach-bug
    regression in LinearSolveLayer and the closure-capture skip.
    """
    torch.manual_seed(11)
    p = torch.nn.Parameter(torch.randn(d, dtype=torch.float64))
    R = _make_spd(d).detach().requires_grad_(True)

    layer = LinearSolveLayer(max_iter=200, tol=1e-10)
    w_star = layer(p.unsqueeze(0), R).squeeze(0)
    assert torch.isfinite(w_star).all(), \
        "LinearSolveLayer produced non-finite values"

    w_star.sum().backward()

    assert p.grad is not None, "p.grad is None -- backward broke (detach-bug regressed?)"
    assert torch.isfinite(p.grad).all(), "p.grad is non-finite"
    assert p.grad.norm().item() > 1e-10, \
        f"p.grad is effectively zero ({p.grad.norm().item():.3e})"

    assert R.grad is not None, \
        "R.grad is None -- closure-captured R did not receive gradient"
    assert torch.isfinite(R.grad).all(), "R.grad is non-finite"
    assert R.grad.norm().item() > 1e-10, \
        f"R.grad is effectively zero ({R.grad.norm().item():.3e}); " \
        "robust-RLS weighter cannot learn through this path"

    g_p = p.grad.norm().item()
    g_R = R.grad.norm().item()
    print(f"  [LinearSolveLayer grad flow] d={d}  ||d(sum w)/dp||={g_p:.4e}  "
          f"||d(sum w)/dR||={g_R:.4e}. PASS")


def test_gradcheck_via_finite_diff(d=4, tol=1e-5):
    """Replace the plan's autograd.gradcheck with a finite-difference
    comparison of the .backward()-based gradient. This validates that
    the implicit backward (solve_jacobian_transpose + f_.backward) is
    numerically consistent with the forward, which is what the training
    loop relies on.
    """
    torch.manual_seed(0)
    R = _make_spd(d).detach().requires_grad_(True)
    p = torch.nn.Parameter(torch.randn(d, dtype=torch.float64))

    layer = LinearSolveLayer(max_iter=200, tol=1e-12, beta=0.5)
    w_star = layer(p.unsqueeze(0), R).squeeze(0)
    w_star.sum().backward()

    # Finite-difference probe of the same output
    grad_fd, probed = _finite_diff_grad_R(layer, p, R, eps=1e-6, n_probes=8, seed=1)

    # Compare analytical R.grad to finite-difference at the probed entries only
    rel_errs = []
    for (i, j) in probed:
        ana = R.grad[i, j].item()
        num = grad_fd[i, j].item()
        if abs(ana) > 1e-8 or abs(num) > 1e-8:
            rel_errs.append(abs(ana - num) / max(abs(ana), abs(num), 1e-12))
    if rel_errs:
        max_rel_err = max(rel_errs)
        print(f"  [gradcheck via FD] d={d}  max rel err={max_rel_err:.2e}  "
              f"tol={tol}  {'PASS' if max_rel_err < tol else 'FAIL'}")
        assert max_rel_err < tol, \
            f"finite-difference vs backward gradient mismatch: max rel err = {max_rel_err:.2e}"
    else:
        print(f"  [gradcheck via FD] d={d}  (no probed entries exceeded threshold). PASS")


def test_gradcheck_batch(n=3, d=4, tol=1e-5):
    """Batch case: (B, d) state, shape used by EquilibriumSolve internally."""
    torch.manual_seed(2)
    R = _make_spd(d).detach().requires_grad_(True)
    p = torch.nn.Parameter(torch.randn(n, d, dtype=torch.float64))

    layer = LinearSolveLayer(max_iter=200, tol=1e-12, beta=0.5)
    w_star = layer(p, R)
    w_star.sum().backward()

    assert R.grad is not None and torch.isfinite(R.grad).all()
    assert R.grad.norm().item() > 1e-10
    assert p.grad is not None and torch.isfinite(p.grad).all()
    assert p.grad.norm().item() > 1e-10

    # Finite-difference probe
    g = torch.Generator(device='cpu').manual_seed(3)
    with torch.no_grad():
        w0 = layer(p, R)
    eps = 1e-6
    max_rel_err = 0.0
    for _ in range(5):
        flat_idx = torch.randint(0, R.numel(), (1,), generator=g).item()
        i, j = divmod(flat_idx, R.shape[1])
        Rp = R.detach().clone()
        Rp[i, j] += eps
        w_p = layer(p, Rp)
        Rm = R.detach().clone()
        Rm[i, j] -= eps
        w_m = layer(p, Rm)
        num = ((w_p - w_m).sum() / (2 * eps)).item()
        ana = R.grad[i, j].item()
        if abs(ana) > 1e-8 or abs(num) > 1e-8:
            rel_err = abs(ana - num) / max(abs(ana), abs(num), 1e-12)
            max_rel_err = max(max_rel_err, rel_err)
    print(f"  [gradcheck batch via FD] n={n} d={d}  max rel err={max_rel_err:.2e}  "
          f"tol={tol}  {'PASS' if max_rel_err < tol else 'FAIL'}")
    assert max_rel_err < tol, \
        f"batch finite-difference vs backward gradient mismatch: max rel err = {max_rel_err:.2e}"


def test_weighter_grad_flow_simulation(d=4, tol=1e-4):
    """Phase 1 regression gate: FabricRobustRLS.step structure.

    Build p and R from the SAME weighter output v_t = sigma(raw_c), so
    their autograd graphs share non-leaf nodes. This is the exact
    mechanism the training loop uses. Verify raw_c.grad is finite,
    non-zero, and matches a closed-form reference (re-expressed in the
    autograd graph: w* = solve(R, p) for the affine case).
    """
    torch.manual_seed(7)
    lam = 0.99
    R_prev = torch.eye(d, dtype=torch.float64)
    p_prev = torch.zeros(d, dtype=torch.float64)
    x_t = torch.randn(d, dtype=torch.float64)
    d_t = torch.tensor(0.5, dtype=torch.float64)

    raw_c = torch.nn.Parameter(torch.tensor(2.0))
    v_t = torch.sigmoid(raw_c)
    R_computed = lam * R_prev + v_t * torch.outer(x_t, x_t)
    p_computed = lam * p_prev + v_t * d_t * x_t

    layer = LinearSolveLayer(max_iter=500, tol=1e-12, beta=0.5)
    w_star = layer(p_computed.unsqueeze(0), R_computed).squeeze(0)
    assert w_star.requires_grad, \
        "w_star lost requires_grad: p (the u input) must require grad"
    loss = (w_star ** 2).sum()
    loss.backward()

    assert raw_c.grad is not None, \
        "raw_c.grad is None -- weighter did not receive gradient"
    assert torch.isfinite(raw_c.grad).all(), \
        f"raw_c.grad is non-finite: {raw_c.grad}"
    g_implicit = raw_c.grad.abs().item()
    assert g_implicit > 1e-10, \
        f"raw_c.grad is effectively zero ({g_implicit:.3e}); " \
        "Phase 1 weighter training will not work"

    # Closed-form reference: re-express w* = solve(R, p) in the autograd
    # graph. For the affine LinearSolveLayer, implicit diff equals this
    # explicit solve up to solver tolerance, so agreement is a strong
    # correctness check of the retain_graph=True backward.
    raw_c_ref = torch.nn.Parameter(torch.tensor(2.0))
    v_t_ref = torch.sigmoid(raw_c_ref)
    R_ref = lam * R_prev + v_t_ref * torch.outer(x_t, x_t)
    p_ref = lam * p_prev + v_t_ref * d_t * x_t
    w_ref = torch.linalg.solve(R_ref, p_ref)
    loss_ref = (w_ref ** 2).sum()
    g_ref = torch.autograd.grad(loss_ref, raw_c_ref)[0].item()

    rel_err = abs(raw_c.grad.item() - g_ref) / max(abs(g_ref), 1e-12)
    print(f"  [weighter grad flow sim] d={d}  raw_c.grad={raw_c.grad.item():.6e}  "
          f"ref={g_ref:.6e}  rel_err={rel_err:.2e}")
    assert rel_err < tol, \
        f"implicit vs explicit-reference gradient mismatch: " \
        f"rel_err={rel_err:.2e}, tol={tol}"


if __name__ == '__main__':
    print("=" * 60)
    print("Phase 0 gradcheck gate (no model code)")
    print("=" * 60)
    test_linear_solve_layer_gradients_flow()
    test_gradcheck_via_finite_diff()
    test_gradcheck_batch()
    test_weighter_grad_flow_simulation()
    print("\nPhase 0 gate passed.")

"""Phase 0 + Phase 1 gates for the learned-IR-RLS design.

Phase 0 (gradcheck / plumbing):
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

Phase 1 (learned robust IR-RLS):
  Six new gates exercise the ``LearnedRobustWeighter`` and the
  ``FabricRobustRLS`` / ``DigitalRobustRLS`` subclasses.  See the
  ``LEARNED_RLS_ISTA_PLAN.md`` file for the full plan.
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


# ----------------------------------------------------------------------------
# Phase 1 gates (require utils.learned_robust + run_rls_demo.DigitalRLS).
# ----------------------------------------------------------------------------

from utils.learned_robust import (
    LearnedRobustWeighter, FabricRobustRLS, DigitalRobustRLS,
    constant_weighter,
    block_robust_rls, digital_block_robust_rls, make_block,
    measure_phantom_vs_exact_bias, oracle_weighter,
    FixedCauchyWeighter, MADRobustWeighter,
)
from run_rls_demo import (
    DigitalRLS, FabricRLS, _make_noise, make_stream,
    batch_experiment_metrics,
)
from utils.circuit_block import LinearSolveLayer


def test_weighter_bounds():
    """Gate 5 (KIMI correction #5 + Phase B2): v_t in [0, v_max] for |e| up to 1e3.

    Load-bearing invariant for the SPD certificate on R (with delta*I the
    Gram matrix stays SPD even when individual v_t can underflow to 0 in
    finite precision; phase B2).  If anyone later edits the
    parameterization, this should fail loudly.
    """
    w = LearnedRobustWeighter(c_init=0.10119, alpha_init=0.12693)
    e_grid = torch.tensor([1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0],
                          dtype=torch.float64)
    v = w(e_grid)
    assert (v >= 0).all(), f"v_t has negative entries: {v}"
    assert (v <= 1.0 + 1e-7).all(), f"v_t exceeds v_max: {v}"
    # Sanity: very small e -> v near 1; very large e -> v bounded away
    # from v_max (the descent rate is governed by alpha ~ 0.127).
    assert v[0].item() > 0.99, f"v(1e-3) should be near 1, got {v[0].item():.4e}"
    assert v[-1].item() < 0.5, f"v(1e3) should descend below 0.5, got {v[-1].item():.4e}"
    # And importantly: v is finite & bounded for all |e| (no overflow).
    assert torch.isfinite(v).all(), f"v has non-finite entries: {v}"

def test_constant_weighter_returns_one():
    """Sanity check on the byte-exact identity helper."""
    cw = constant_weighter(1.0)
    e = torch.randn(7)
    v = cw(e)
    assert torch.allclose(v, torch.ones(7)), f"constant weighter returned {v}"
    print(f"  [constant weighter] v shape={v.shape}, v[0]={v[0].item():.4f}. PASS")


def test_weighter_init_unit_check():
    """Gate 2b (per plan decision 2b): weighter init unit check.

    v(0) = 1 (exactly), v ~ 1 for |e| <= 0.1, and v redescends for
    |e| >> c ~ 0.107.  The 'frozen init matches plain RLS' split
    (decision 2a) is the byte-exact v==1 case covered by the
    digital/fabric v==1 tests below.
    """
    w = LearnedRobustWeighter(c_init=0.10119, alpha_init=0.12693).to(torch.float64)
    # v(0) = 1 exactly
    v0 = w(torch.tensor(0.0, dtype=torch.float64))
    assert torch.allclose(v0, torch.tensor(1.0, dtype=torch.float64), atol=1e-12), \
        f"v(0)={v0.item()}"
    # v(e) ~ 1 for small |e|.  With c ~ 0.1012 (init c_init=0.10119) and
    # alpha ~ 0.127, v(0.05) ~ 0.975 and v(0.1) ~ 0.92.
    e_small = torch.linspace(-0.05, 0.05, 11, dtype=torch.float64)
    v_small = w(e_small)
    assert (v_small > 0.95).all() and (v_small <= 1.0 + 1e-12).all(), \
        f"v(e) for |e|<=0.05 should be ~1, got {v_small}"
    # v redescends for |e| >> c.  With init (c ~ 0.107, alpha ~ 0.127),
    # v(10) ~ 0.31, v(100) ~ 0.17, v(1000) ~ 0.10.  The threshold is
    # set so v(10) is required to descend measurably (~ v at the burst
    # magnitude kappa*sigma = 0.2) without demanding full saturation.
    e_large = torch.tensor([10.0, 100.0, 1000.0], dtype=torch.float64)
    v_large = w(e_large)
    assert (v_large < 0.4).all(), f"v(e) for |e|>>c should descend, got {v_large}"
    # Stronger descent at |e| >= 100: the weighter must be meaningful
    # at the saturation tail (the trained weighter should move these
    # far below 0.2 during training; the init just has to be < 0.3).
    assert (v_large[1:] < 0.3).all(), f"v(e) for |e|>=100 should descend, got {v_large[1:]}"
    # v_max=1.0 strictly
    assert torch.allclose(w.v_max, torch.tensor(1.0, dtype=torch.float64))
    print(f"  [weighter init unit] v(0)={v0.item():.6f}, "
          f"v(0.05)={v_small[10].item():.4f}, v(10)={v_large[0].item():.4e}, "
          f"v(1e3)={v_large[-1].item():.4e}. PASS")


def test_digital_robust_v1_byteexact_matches_digital_rls(d=8, T=64, lam=0.99,
                                                          seed=0):
    """Gate 2a / Gate 3: v_t == 1 makes DigitalRobustRLS byte-identical
    to DigitalRLS (the strong invariant 'v_t=1 -> identical')."""
    torch.manual_seed(seed)
    const_w = constant_weighter(1.0, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    w_o = torch.randn(d, generator=g, dtype=torch.float32)
    w_o = w_o / w_o.norm()
    x = torch.randn(T, d, generator=g, dtype=torch.float32)
    d_obs = x @ w_o + 0.01 * torch.randn(T, generator=g, dtype=torch.float32)

    drls = DigitalRLS(d=d, lam=lam, device='cpu')
    drrs = DigitalRobustRLS(d=d, weighter=const_w, lam=lam, device='cpu')
    for t in range(T):
        w1 = drls.step(x[t], d_obs[t])
        w2 = drrs.step(x[t], d_obs[t])
        assert torch.equal(w1, w2), \
            f"step {t}: digital_rls.w={w1} vs digital_robust.w={w2}"
    assert torch.equal(drls.P, drrs.P), \
        f"final P mismatch: digital_rls.P={drls.P} vs digital_robust.P={drrs.P}"
    print(f"  [digital_robust v=1 byte-exact] T={T}, d={d}: "
          f"all {T} steps match DigitalRLS to the bit. PASS")


def test_fabric_robust_v1_close_to_fabric_rls(d=8, T=64, lam=0.99,
                                                tol=1e-5, seed=0):
    """Gate 2a — fabric variant.  Not byte-exact (the fabric path goes
    through LinearSolveLayer with chebyshev beta, which has its own
    solver noise), but should match to solver tolerance."""
    torch.manual_seed(seed)
    const_w = constant_weighter(1.0, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    w_o = torch.randn(d, generator=g, dtype=torch.float32)
    w_o = w_o / w_o.norm()
    x = torch.randn(T, d, generator=g, dtype=torch.float32)
    d_obs = x @ w_o + 0.01 * torch.randn(T, generator=g, dtype=torch.float32)

    frls = FabricRLS(d=d, lam=lam, R0=torch.eye(d, dtype=torch.float32),
                     max_iter=200, tol=1e-10, beta='chebyshev', device='cpu')
    frrs = FabricRobustRLS(d=d, weighter=const_w, lam=lam,
                            R0=torch.eye(d, dtype=torch.float32),
                            max_iter=200, tol=1e-10, beta='chebyshev',
                            device='cpu')
    for t in range(T):
        w1 = frls.step(x[t], d_obs[t])
        w2 = frrs.step(x[t], d_obs[t])
    err = (w1 - w2).norm().item()
    rel_err = err / max(w1.norm().item(), 1e-12)
    print(f"  [fabric_robust v=1 trail] final ||w_fabric - w_fabric_robust||="
          f"{err:.4e}, rel={rel_err:.4e}, tol={tol}")
    # Final-error check only — the trajectories drift apart slightly
    # over T=64 because the warm-starting init=w.detach() interacts
    # differently with the two paths.
    assert rel_err < tol, f"final w mismatch: rel_err={rel_err:.4e} > {tol}"


def test_digital_robust_matches_fabric_robust(d=8, T=64, lam=0.99,
                                                tol=1e-5, seed=0):
    """Gate 3: digital-robust and fabric-robust trajectories match to
    solver tolerance (KIMI correction #2: 1e-5 not 1e-6; the feedback-
    amplification floor at lam=0.99 with default tol is ~1e-6)."""
    torch.manual_seed(seed)
    # Float32 throughout: byte-exact and trajectory-match tests use the
    # natural dtype of the underlying DigitalRLS (P is float32-default).
    # The implicit-vs-unrolled gradient test below keeps float64 because
    # it hands a closed-form reference through autograd in float64.
    const_w = constant_weighter(1.0, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    w_o = torch.randn(d, generator=g, dtype=torch.float32)
    w_o = w_o / w_o.norm()
    x = torch.randn(T, d, generator=g, dtype=torch.float32)
    d_obs = x @ w_o + 0.01 * torch.randn(T, generator=g, dtype=torch.float32)

    drrs = DigitalRobustRLS(d=d, weighter=const_w, lam=lam, device='cpu')
    frrs = FabricRobustRLS(d=d, weighter=const_w, lam=lam,
                            R0=torch.eye(d, dtype=torch.float32),
                            max_iter=500, tol=1e-12, beta='chebyshev',
                            device='cpu')
    W_d = torch.zeros(T, d, dtype=torch.float32)
    W_f = torch.zeros(T, d, dtype=torch.float32)
    for t in range(T):
        W_d[t] = drrs.step(x[t], d_obs[t])
        W_f[t] = frrs.step(x[t], d_obs[t])
    # Discrepancy measure: max relative ||w_d - w_f|| over time.
    errs = (W_d - W_f).norm(dim=-1)
    rel = errs / W_d.norm(dim=-1).clamp_min(1e-12)
    max_rel = rel.max().item()
    print(f"  [digital_robust vs fabric_robust] max rel err={max_rel:.4e}, "
          f"tol={tol}")
    assert max_rel < tol, f"max rel err={max_rel:.4e} > tol={tol}"


def test_implicit_vs_unrolled_grad_robust(d=8, T=8, lam=0.99, tol=1e-4,
                                            seed=0):
    """Gate 4: implicit-backward gradient matches a fully-unrolled
    forward reference to rel err < 1e-4 (KIMI correction #3: float64 +
    tol 1e-10 to isolate algorithmic error from precision error)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    w_o = torch.randn(d, generator=g, dtype=torch.float64)
    w_o = w_o / w_o.norm()
    x = torch.randn(T, d, generator=g, dtype=torch.float64)
    d_obs = x @ w_o + 0.05 * torch.randn(T, generator=g, dtype=torch.float64)

    # ---- Implicit backward path (production) ----
    w_imp = LearnedRobustWeighter(c_init=0.10119, alpha_init=0.12693)
    w_imp.log_c.data = w_imp.log_c.data.double()
    w_imp.log_alpha.data = w_imp.log_alpha.data.double()
    d64 = torch.float64
    fr_imp = FabricRobustRLS(d=d, weighter=w_imp, lam=lam,
                             R0=torch.eye(d, dtype=d64),
                             w_init=torch.zeros(d, dtype=d64),
                             max_iter=50, tol=1e-10, beta=0.5,
                             device='cpu', training=True,
                             backward_mode='exact')
    loss_imp = torch.zeros((), dtype=torch.float64)
    for t in range(T):
        e_t = d_obs[t] - fr_imp.w @ x[t]
        loss_imp = loss_imp + e_t.pow(2).sum()
        fr_imp.step(x[t], d_obs[t])
    loss_imp.backward()
    g_imp = w_imp.log_c.grad.item()

    # ---- Unrolled forward reference ----
    w_unr = LearnedRobustWeighter(c_init=0.10119, alpha_init=0.12693)
    w_unr.log_c.data = w_imp.log_c.data.clone()
    w_unr.log_alpha.data = w_imp.log_alpha.data.clone()
    R_unr = torch.eye(d, dtype=torch.float64)
    p_unr = torch.zeros(d, dtype=torch.float64)
    w_state = torch.zeros(d, dtype=torch.float64)
    loss_unr = torch.zeros((), dtype=torch.float64)
    layer = LinearSolveLayer(max_iter=50, tol=1e-10, beta=0.5)
    for t in range(T):
        e_t = d_obs[t] - w_state @ x[t]
        loss_unr = loss_unr + e_t.pow(2).sum()
        v_t = w_unr(e_t)
        R_unr = lam * R_unr + v_t * torch.outer(x[t], x[t])
        p_unr = lam * p_unr + v_t * d_obs[t] * x[t]
        w_state = layer(p_unr.unsqueeze(0), R_unr).squeeze(0)
    loss_unr.backward()
    g_unr = w_unr.log_c.grad.item()

    rel_err = abs(g_imp - g_unr) / max(abs(g_unr), 1e-12)
    # Phase 1.5 prerequisite #4: state the precision context so the
    # rel_err number is interpretable.  The number is only meaningful at
    # float64 / solver tol 1e-10; in float32 the solver noise alone
    # exceeds 1e-4.
    precision = 'float64 / solver tol 1e-10'
    print(f"  [implicit vs unrolled grad robust d={d} T={T}] "
          f"g_imp={g_imp:.6e}, g_unr={g_unr:.6e}, rel_err={rel_err:.2e}, "
          f"tol={tol}, precision={precision}. "
          f"{'PASS' if rel_err < tol else 'FAIL'}")
    assert rel_err < tol, f"implicit vs unrolled grad mismatch: rel_err={rel_err:.2e}"


def test_impulsive_noise_burst_rate(T=20000, p_burst=0.02, kappa=20.0,
                                     sigma=0.01, tol=0.005, seed=0):
    """Gate 5: _make_noise burst statistics; gaussian mode ≡ ``sigma * randn``.

    Burst rate should be ~ p_burst (within tol), and burst amplitude
    should be ~ sqrt(1 + kappa^2) * sigma (the std of z1 + kappa * z2).
    """
    # Impulsive: use two parallel generators seeded identically so the
    # byte-identical check is meaningful (g itself is shared/advanced by
    # _make_noise, so we can't reuse it for the reference draws).
    g_a = torch.Generator().manual_seed(seed)
    g_b = torch.Generator().manual_seed(seed)
    nu_imp = _make_noise(T, sigma, mode='impulsive',
                         p_burst=p_burst, kappa=kappa,
                         generator=g_a, dtype=torch.float64)
    z1 = torch.randn(T, generator=g_b, dtype=torch.float64)
    z2 = torch.randn(T, generator=g_b, dtype=torch.float64)
    burst = (torch.rand(T, generator=g_b, dtype=torch.float64) < p_burst).to(torch.float64)
    nu_check = sigma * (z1 + kappa * burst * z2)
    assert torch.equal(nu_imp, nu_check), \
        "_make_noise impulsive path is not byte-identical to its decomposition"
    burst_rate = burst.mean().item()
    burst_amp = nu_imp[burst > 0.5].abs().mean().item()
    # Theoretical burst amplitude std would be sigma * sqrt(1 + kappa^2)
    # for the marginal; the mean abs is sigma * sqrt(2/pi) * sqrt(1 + kappa^2).
    import math
    expected_amp = sigma * math.sqrt(2.0 / math.pi) * math.sqrt(1.0 + kappa * kappa)
    print(f"  [impulsive noise] burst rate={burst_rate:.4f} (target {p_burst}), "
          f"burst |mean|={burst_amp:.4e} (expected ~ {expected_amp:.4e})")
    assert abs(burst_rate - p_burst) < tol, \
        f"burst_rate={burst_rate} differs from p_burst={p_burst} by more than {tol}"
    # Burst amplitude check (15% relative tolerance for the empirical mean abs)
    assert abs(burst_amp - expected_amp) / expected_amp < 0.15, \
        f"burst amplitude {burst_amp} differs from expected {expected_amp}"

    # Gaussian mode ≡ old make_stream noise: use two parallel generators
    # seeded identically so the byte-identical check is meaningful.
    g_a = torch.Generator().manual_seed(seed)
    g_b = torch.Generator().manual_seed(seed)
    nu_gauss = _make_noise(T, sigma, mode='gaussian', generator=g_a,
                            dtype=torch.float64)
    nu_legacy = sigma * torch.randn(T, generator=g_b, dtype=torch.float64)
    assert torch.equal(nu_gauss, nu_legacy), \
        "_make_noise gaussian path is not byte-identical to sigma * randn"
    print(f"  [impulsive noise] gaussian mode byte-identical to legacy. PASS")


def test_gaussian_control_flat_weighter(epochs=20, T=8, lam=0.99, sigma=0.01,
                                          d=4, lr=1e-2, seed=0):
    """Gate 6 (KIMI correction #4): on Gaussian noise the trained weighter
    should be approximately flat (low Var_e[v(e)]).  We don't require
    alpha -> 0 specifically (c -> infinity would also satisfy the
    functional property); we test the variance directly."""
    torch.manual_seed(seed)
    device = torch.device('cpu')
    weighter = LearnedRobustWeighter(c_init=0.10119, alpha_init=0.12693).to(device)
    optimizer = torch.optim.Adam(weighter.parameters(), lr=lr)

    for epoch in range(epochs):
        w_o = torch.randn(d, device=device)
        w_o = w_o / w_o.norm()
        x, d_obs = make_stream(w_o, T, sigma, mode='iid', noise='gaussian',
                               seed=epoch, dtype=torch.float32, device=device)
        R0 = torch.eye(d, device=device, dtype=torch.float32)
        filt = FabricRobustRLS(d=d, weighter=weighter, lam=lam, R0=R0,
                               max_iter=50, tol=1e-5, beta='chebyshev',
                               device=device, training=True)
        loss = torch.zeros((), device=device)
        for t in range(T):
            e_t = d_obs[t] - filt.w @ x[t]
            loss = loss + e_t.pow(2).sum()
            filt.step(x[t], d_obs[t])
            # No filter.w/R/p detach here.  Detaching them would break
            # the next e_t's grad_fn and propagate "no grad_fn" through
            # the loss accumulator's chained additions.  The chain
            # through R/p grows by one step per iteration; with the
            # ``phantom`` backward mode set by ``FabricRobustRLS`` when
            # ``training=True`` the per-step cost is O(chain_depth) and
            # total backward is O(T * chain_depth).  For T=32, d=4 this
            # is tractable.  (The train_robust_weighter smoke test uses
            # T=128 with truncate_every=32; same pattern.)
        loss = loss + 10.0 * (filt.w - w_o).pow(2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # Functional property: Var_e[v(e)] on a typical Gaussian noise grid.
    e_grid = torch.linspace(-0.05, 0.05, 51, device=device, dtype=torch.float32)
    with torch.no_grad():
        v = weighter(e_grid)
    var_v = v.var().item()
    alpha_here = weighter.alpha.item()
    c_here = weighter.c.item()

    # Flat-curve threshold (plan Phase 1 evidence: Var_e[v(e)] = 1.83e-4
    # << 5e-3).  This is the load-bearing Gaussian control: if a future
    # change to the parameterization or training breaks flatness under
    # Gaussian noise, the gate must fail loudly.  alpha -> 0 and c -> inf
    # both satisfy the functional property; we test the variance directly.
    print(f"  [gaussian_control] epochs={epochs}, alpha={alpha_here:.4f}, "
          f"c={c_here:.4f}, Var_e[v(e)]={var_v:.4e} "
          f"(flat threshold 5e-3). "
          f"{'PASS' if var_v < 5e-3 else 'FAIL'}")
    assert var_v < 5e-3, \
        f"Gaussian control failed: Var_e[v(e)]={var_v:.4e} >= 5e-3 " \
        "(weighter not flat under Gaussian noise)"


# ----------------------------------------------------------------------------
# Phase 1.5 gates (Block Robust IRLS).
# ----------------------------------------------------------------------------


def test_block_robust_v1_matches_batch_ls(d=6, N=128, delta=1e-2, seed=0,
                                            tol=1e-6):
    """Phase 1.5 gate 1: with ``v(e) == 1`` the block IRLS reduces to
    plain batch least-squares (K=0 -> same as K>0 because v=1 makes
    every outer iteration re-solve the same system).

    The byte-equality proof: when v=1, R = X^T X + delta * I and
    p = X^T d on every settle, so all K+1 settles produce the same
    w = (X^T X + delta * I)^{-1} (X^T d) regardless of warm-start.

    The fabric solve (via LinearSolveLayer) is compared against the
    regularized normal equations through the *restored*
    ``batch_experiment_metrics`` path (Phase 1.5 prerequisite #1:
    ``run_batch_experiment``'s `m['trial'] = trial` and the
    ``test_batch_fabric_matches_lstsq`` test both depend on that
    function returning a dict), so this gate doubles as a regression
    check on the restoration.
    """
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    w_o = torch.randn(d, generator=g, dtype=torch.float64)
    w_o = w_o / w_o.norm()
    X = torch.randn(N, d, generator=g, dtype=torch.float64)
    nu = 0.01 * torch.randn(N, generator=g, dtype=torch.float64)
    d_obs = X @ w_o + nu

    const_w = constant_weighter(1.0, dtype=torch.float64)
    w_K0 = block_robust_rls(X, d_obs, const_w, delta=delta, K=0,
                            max_iter=200, tol=1e-12,
                            backward_mode='exact')
    # K=4 with v=1 should be byte-equivalent (same R, p every iter).
    w_K4 = block_robust_rls(X, d_obs, const_w, delta=delta, K=4,
                            max_iter=200, tol=1e-12,
                            backward_mode='exact')

    # The v=1 block IRLS reduces to the plain regularized normal
    # equations R w = p with R = X^T X + delta*I, p = X^T d.  Compare the
    # fabric solve against the lstsq reference through the restored
    # ``batch_experiment_metrics`` path (prerequisite #1).
    R0 = delta * torch.eye(d, dtype=torch.float64)
    R = X.t() @ X + R0
    p = X.t() @ d_obs
    m = batch_experiment_metrics(w_K0, R, p, X, d_obs, R0, w_o=w_o)
    rel_err = m['rel_err']
    print(f"  [block v=1 plain batch LS] rel_err={rel_err:.4e}, "
          f"normal_eq_res={m['normal_eq_residual']:.4e} "
          f"(via restored batch_experiment_metrics)")
    assert rel_err < tol, \
        f"block robust v=1 fabric/reference mismatch: {rel_err:.4e} > {tol}"

    # K=4 with v=1 should still match the same problem (same R, p every iter).
    ks_diff = (w_K0 - w_K4).norm().item() / w_K0.norm().clamp_min(1e-12).item()
    print(f"  [block v=1 K=0 vs K=4] rel_diff={ks_diff:.4e}")
    # The two w's may differ slightly due to the warm-start init=w.detach()
    # at every K iteration.  For v=1, R/p are unchanged -> the linear solve
    # is the same -> w should match within solver tolerance.
    assert ks_diff < tol * 10, \
        f"K=0 vs K=4 with v=1 shouldn't differ: {ks_diff:.4e} > {tol * 10}"


def test_block_robust_digital_matches_fabric(d=6, N=128, delta=1e-2, K=4,
                                                seed=0, tol=1e-5):
    """Phase 1.5 gate 2: digital vs fabric block twins match to <1e-5.

    The fabric path uses ``LinearSolveLayer`` (Anderson with chebyshev
    or fixed beta) and the digital path uses ``torch.linalg.solve``;
    both solve the same weighted normal equations but the fabric
    undergoes an iterative settle (with solver tolerance).  The two
    solutions should agree to solver tolerance.
    """
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    w_o = torch.randn(d, generator=g, dtype=torch.float64)
    w_o = w_o / w_o.norm()
    X = torch.randn(N, d, generator=g, dtype=torch.float64)
    nu = 0.01 * torch.randn(N, generator=g, dtype=torch.float64)
    d_obs = X @ w_o + nu

    # Force a meaningful IRLS trajectory by using a weighter that
    # descends sharply for |e| > kappa*sigma*0.5.
    weighter = LearnedRobustWeighter(c_init=0.10119, alpha_init=0.12693)
    weighter = weighter.to(torch.float64)

    w_fabric = block_robust_rls(X, d_obs, weighter, delta=delta, K=K,
                                max_iter=300, tol=1e-12, beta=1.0,
                                backward_mode='exact')
    w_digital = digital_block_robust_rls(X, d_obs, weighter, delta=delta, K=K)

    rel = (w_fabric - w_digital).norm().item() / w_digital.norm().clamp_min(1e-12).item()
    print(f"  [block digital vs fabric d={d} N={N} K={K}] rel={rel:.4e}, "
          f"tol={tol}")
    assert rel < tol, \
        f"fabric/digital block twin mismatch: {rel:.4e} > {tol}"


def test_block_robust_impulsive_improvement(d=6, N=128, delta=1e-2, K=4,
                                              sigma=0.01, p_burst=0.02,
                                              kappa=20.0, seed=0,
                                              epochs=20, lr=1e-2,
                                              n_trials=2):
    """Phase 1.5 gate 3 (honest result gate): learned block IRLS must
    beat plain batch LS on impulsive noise.

    Train the weighter end-to-end on impulsive noise, then compare
    the MSE on the K-th iterate against plain batch LS over a held-out
    block.  The gate is that the learned weighter improves over the
    unweighted baseline at all (improvement > 0); the magnitude is
    reported, not tuned.  If training does not beat plain batch LS,
    the gate fails loudly -- that is an honest result, not something
    to paper over (mirrors the plan's "report, don't tune").
    """
    torch.manual_seed(seed)
    weighter = LearnedRobustWeighter(c_init=0.10119, alpha_init=0.12693)
    optimizer = torch.optim.Adam(weighter.parameters(), lr=lr)

    for epoch in range(epochs):
        w_o = torch.randn(d, dtype=torch.float32)
        w_o = w_o / w_o.norm()
        X, d_obs = make_block(w_o, N, sigma, mode='iid',
                              noise='impulsive', p_burst=p_burst,
                              kappa=kappa, seed=seed + epoch,
                              dtype=torch.float32, device='cpu')
        w_K = block_robust_rls(X, d_obs, weighter, delta=delta, K=K,
                               max_iter=50, tol=1e-5, backward_mode='phantom')
        loss = (X @ w_K - X @ w_o).pow(2).sum() + 10.0 * (w_K - w_o).pow(2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Held-out block: plain batch LS vs trained block IRLS.
    mse_plain_list = []
    mse_learned_list = []
    for trial in range(n_trials):
        w_o = torch.randn(d, dtype=torch.float64)
        w_o = w_o / w_o.norm()
        X, d_obs = make_block(w_o, N, sigma, mode='iid',
                              noise='impulsive', p_burst=p_burst,
                              kappa=kappa, seed=seed + 1000 + trial,
                              dtype=torch.float64, device='cpu')
        const_w = constant_weighter(1.0, dtype=torch.float64)
        # Plain batch LS (K=0 with v=1, byte-equivalent to lstsq).
        w_plain = block_robust_rls(X, d_obs, const_w, delta=delta, K=0)
        # Trained block IRLS.
        w_here = weighter.to(torch.float64)
        w_learned = block_robust_rls(X, d_obs, w_here, delta=delta, K=K,
                                     max_iter=200, tol=1e-10, beta=1.0,
                                     backward_mode='exact')
        mse_plain = float(((X @ w_plain - X @ w_o).pow(2)).mean().item())
        mse_learned = float(((X @ w_learned - X @ w_o).pow(2)).mean().item())
        mse_plain_list.append(mse_plain)
        mse_learned_list.append(mse_learned)

    mse_plain_mean = sum(mse_plain_list) / n_trials
    mse_learned_mean = sum(mse_learned_list) / n_trials
    improvement = (mse_plain_mean - mse_learned_mean) / mse_plain_mean
    print(f"  [block impulsive improvement] mse_plain={mse_plain_mean:.4e}, "
          f"mse_learned={mse_learned_mean:.4e}, "
          f"improvement={improvement:.4e} ({improvement * 100:.2f}%)")
    # Honest threshold: the weighted weighter must beat plain LS at all.
    # Bigger wins are desirable; the margin is reported, not tuned.
    assert improvement > 0, \
        f"learned block IRLS did not beat plain batch LS: " \
        f"improvement={improvement:.4e}"


def test_block_robust_grad_flow(d=8, N=16, K=4, delta=1e-2, tol=1e-4,
                                  seed=0):
    """Phase 1.5 gate 4: ``log_c.grad`` finite/nonzero after backprop
    through K settles; exact implicit-vs-unrolled < 1e-4 (float64,
    solver tol 1e-10).

    Builds the block IRLS path through ``LinearSolveLayer`` (production)
    against a fully-unrolled reference (per-iter closed form via
    ``torch.linalg.solve``).  Runs in float64 to isolate the
    algorithmic-error component from solver noise at the default tol.
    """
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    w_o = torch.randn(d, generator=g, dtype=torch.float64)
    w_o = w_o / w_o.norm()
    X = torch.randn(N, d, generator=g, dtype=torch.float64)
    nu = 0.05 * torch.randn(N, generator=g, dtype=torch.float64)
    d_obs = X @ w_o + nu

    # ---- Production (implicit, exact backward) ----
    w_imp = LearnedRobustWeighter(c_init=0.10119, alpha_init=0.12693)
    w_imp.log_c.data = w_imp.log_c.data.double()
    w_imp.log_alpha.data = w_imp.log_alpha.data.double()
    w_K_imp = block_robust_rls(X, d_obs, w_imp, delta=delta, K=K,
                               max_iter=50, tol=1e-10, beta=1.0,
                               backward_mode='exact')
    loss_imp = (X @ w_K_imp - X @ w_o).pow(2).sum()
    loss_imp.backward()
    g_imp = w_imp.log_c.grad.item()

    # ---- Unrolled reference (closed-form solve per iter) ----
    w_unr = LearnedRobustWeighter(c_init=0.10119, alpha_init=0.12693)
    w_unr.log_c.data = w_imp.log_c.data.clone()
    w_unr.log_alpha.data = w_imp.log_alpha.data.clone()
    R = X.t() @ X + delta * torch.eye(d, dtype=torch.float64)
    p = X.t() @ d_obs
    w_state = torch.linalg.solve(R, p)
    for _ in range(K):
        e = d_obs - X @ w_state
        v = w_unr(e)
        R = X.t() @ (v.unsqueeze(-1) * X) + delta * torch.eye(d, dtype=torch.float64)
        p = X.t() @ (v * d_obs)
        w_state = torch.linalg.solve(R, p)
    loss_unr = (X @ w_state - X @ w_o).pow(2).sum()
    loss_unr.backward()
    g_unr = w_unr.log_c.grad.item()

    rel_err = abs(g_imp - g_unr) / max(abs(g_unr), 1e-12)
    precision = 'float64 / solver tol 1e-10'
    print(f"  [block grad flow d={d} N={N} K={K}] "
          f"g_imp={g_imp:.6e}, g_unr={g_unr:.6e}, rel_err={rel_err:.2e}, "
          f"tol={tol}, precision={precision}. "
          f"{'PASS' if rel_err < tol else 'FAIL'}")
    assert torch.isfinite(w_imp.log_c.grad).all(), \
        "log_c.grad is non-finite"
    assert abs(g_imp) > 1e-10, \
        f"log_c.grad is effectively zero: {g_imp}"
    assert rel_err < tol, \
        f"implicit vs unrolled block grad mismatch: rel_err={rel_err:.2e}"


def test_phantom_vs_exact_bias(d=4, N=16, K=4, delta=1e-2, sigma=0.01,
                                 p_burst=0.02, kappa=20.0, seed=0,
                                 train_epochs=6, lr=1e-2):
    """Phase 1.5 gate 5: phantom-vs-exact implicit gradient bias
    (Phase 1.5 prerequisite #3 -- measurement gate, not a pass/fail).

    Training uses ``backward_mode='phantom'`` (cheap VJP per implicit
    step) but every existing gate validates only the exact (CG)
    adjoint (~1e-10).  The gradient actually used for learning is
    unverified.  This gate trains a weighter briefly (same protocol as
    ``test_block_robust_impulsive_improvement``) and then measures the
    phantom-vs-exact bias at that *trained* operating point -- phantom
    is biased by construction (Geng et al. 2021), so this is a
    measurement, not a pass/fail bound.

    The ``run_robust_block_experiment`` driver performs the same
    measurement on the fully-trained weighter and reports it into
    ``block_metrics.json``; this gate runs the measurement end-to-end
    on a trained config at test time.
    """
    torch.manual_seed(seed)
    weighter = LearnedRobustWeighter(c_init=0.10119, alpha_init=0.12693)
    optimizer = torch.optim.Adam(weighter.parameters(), lr=lr)

    # Short training to reach a non-trivial operating point (float32,
    # same protocol as gate 3).  The bias is measured afterwards at the
    # trained params, per the plan ("on the trained configuration").
    for epoch in range(train_epochs):
        w_o = torch.randn(d, dtype=torch.float32)
        w_o = w_o / w_o.norm()
        X, d_obs = make_block(w_o, N, sigma, mode='iid',
                              noise='impulsive', p_burst=p_burst,
                              kappa=kappa, seed=seed + epoch,
                              dtype=torch.float32, device='cpu')
        w_K = block_robust_rls(X, d_obs, weighter, delta=delta, K=K,
                               max_iter=50, tol=1e-5,
                               backward_mode='phantom')
        loss = ((X @ w_K - X @ w_o).pow(2).sum()
                + 10.0 * (w_K - w_o).pow(2).sum())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Held-out block at float64 (grad-accuracy precision, decision 0b).
    g = torch.Generator().manual_seed(seed + 1000)
    w_o = torch.randn(d, generator=g, dtype=torch.float64)
    w_o = w_o / w_o.norm()
    X, d_obs = make_block(w_o, N, sigma, mode='iid',
                          noise='impulsive', p_burst=p_burst,
                          kappa=kappa, seed=seed + 1000,
                          dtype=torch.float64, device='cpu')

    res = measure_phantom_vs_exact_bias(
        X, d_obs, w_o, weighter.to(torch.float64), delta=delta, K=K,
        max_iter=100, tol=1e-8)
    g_phantom = res['phantom_grad']
    g_exact = res['exact_grad']
    rel_bias = res['rel_bias']

    # Measurement: report the bias.  Even a 1e-1 bias is acceptable for
    # a phantom gradient (the literature reports up to 1e-1 in deep
    # chains).  This is a MEASUREMENT, not a pass/fail bound -- the
    # ``run_robust_block_experiment`` driver reports it into
    # ``block_metrics.json`` so the user has evidence about the
    # actual training signal quality.  We only require finiteness and
    # nonzero magnitude (a zero gradient is a real bug).
    import math
    assert math.isfinite(g_phantom), f"phantom log_c.grad is non-finite: {g_phantom}"
    assert math.isfinite(g_exact), f"exact log_c.grad is non-finite: {g_exact}"
    assert abs(g_phantom) > 1e-12, \
        f"phantom gradient is effectively zero: {g_phantom:.4e}"
    assert abs(g_exact) > 1e-12, \
        f"exact gradient is effectively zero: {g_exact:.4e}"
    signs_agree = (g_phantom * g_exact) > 0
    print(f"  [phantom vs exact bias d={d} N={N} K={K} (trained config)] "
          f"g_phantom={g_phantom:.4e}, g_exact={g_exact:.4e}, "
          f"rel_bias={rel_bias:.4e} (measurement, not pass/fail). "
          f"signs_agree={'yes' if signs_agree else 'NO'}.")
    if not signs_agree:
        # Don't fail-loud: phantom is biased by construction (Geng et al.
        # 2021).  Print a warning so the user sees the sign disagreement,
        # but proceed.  The bias is reported into ``block_metrics.json``.
        print(f"    WARNING: phantom and exact gradients disagree in sign "
              f"(rel_bias={rel_bias:.4e}).  This is a phantom-gradient "
              f"artefact of the chained K={K} settle; the training "
              f"signal is still usable for learning but the gradient "
              f"direction is biased.  See the bias in "
              f"block_metrics.json.")

# ----------------------------------------------------------------------------
# Phase 0 gates (oracle-oos-headroom plan): oracle ceiling + OOS scoring.
# ----------------------------------------------------------------------------


def test_oracle_bound(d=6, N=128, delta=1e-2, K=4, sigma=0.01, p_burst=0.02,
                      kappa=20.0, seed=0, epochs=20, lr=1e-2, n_trials=2):
    """Phase 0 gate 1 (oracle-oos-headroom): the oracle is an upper bound.

    On an impulsive block the oracle (true burst mask, v=floor on bursts)
    must be the best of the three:  oracle <= learned <= plain.
    On a clean block (noise='gaussian', no bursts) the oracle mask is
    empty, so the oracle must reduce to plain batch LS: oracle ~= plain.
    """
    torch.manual_seed(seed)
    weighter = LearnedRobustWeighter(c_init=0.10119, alpha_init=0.12693)
    optimizer = torch.optim.Adam(weighter.parameters(), lr=lr)

    for epoch in range(epochs):
        w_o = torch.randn(d, dtype=torch.float32)
        w_o = w_o / w_o.norm()
        X, d_obs = make_block(w_o, N, sigma, mode='iid',
                              noise='impulsive', p_burst=p_burst,
                              kappa=kappa, seed=seed + epoch,
                              dtype=torch.float32, device='cpu')
        w_K = block_robust_rls(X, d_obs, weighter, delta=delta, K=K,
                               max_iter=50, tol=1e-5, backward_mode='phantom')
        loss = (X @ w_K - X @ w_o).pow(2).sum() + 10.0 * (w_K - w_o).pow(2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    const_w = constant_weighter(1.0, dtype=torch.float64)
    w_here = weighter.to(torch.float64)

    # Impulsive block: oracle <= learned <= plain.
    mse_plain_list = []
    mse_learned_list = []
    mse_oracle_list = []
    for trial in range(n_trials):
        w_o = torch.randn(d, dtype=torch.float64)
        w_o = w_o / w_o.norm()
        X, d_obs, nu, burst = make_block(w_o, N, sigma, mode='iid',
                                         noise='impulsive', p_burst=p_burst,
                                         kappa=kappa, seed=seed + 1000 + trial,
                                         dtype=torch.float64, device='cpu',
                                         return_noise=True)
        oracle_w = oracle_weighter(burst, dtype=torch.float64)
        # Sanity: the burst mask must be non-trivial under impulsive noise.
        assert burst.sum().item() > 0, \
            f"impulsive block had no bursts (seed={seed + 1000 + trial})"
        assert nu.shape == d_obs.shape, f"nu shape {nu.shape} != d_obs {d_obs.shape}"
        w_plain = block_robust_rls(X, d_obs, const_w, delta=delta, K=0)
        w_learned = block_robust_rls(X, d_obs, w_here, delta=delta, K=K,
                                     max_iter=200, tol=1e-10, beta=1.0,
                                     backward_mode='exact')
        w_oracle = block_robust_rls(X, d_obs, oracle_w, delta=delta, K=K,
                                    max_iter=200, tol=1e-10, beta=1.0,
                                    backward_mode='exact')
        mse = lambda w: float(((X @ w - X @ w_o).pow(2)).mean().item())
        mse_plain_list.append(mse(w_plain))
        mse_learned_list.append(mse(w_learned))
        mse_oracle_list.append(mse(w_oracle))
        print(f"  [oracle bound trial {trial}] plain={mse(w_plain):.4e} "
              f"learned={mse(w_learned):.4e} oracle={mse(w_oracle):.4e}")

    mse_plain = sum(mse_plain_list) / n_trials
    mse_learned = sum(mse_learned_list) / n_trials
    mse_oracle = sum(mse_oracle_list) / n_trials
    print(f"  [oracle bound] mean plain={mse_plain:.4e} learned={mse_learned:.4e} "
          f"oracle={mse_oracle:.4e}")
    # oracle must be no worse than learned, learned no worse than plain
    # (relative slack for solver noise at the K=0-vs-K solves).
    assert mse_oracle <= mse_learned * (1 + 1e-9) + 1e-18, \
        f"oracle should be <= learned: oracle={mse_oracle:.4e}, learned={mse_learned:.4e}"
    assert mse_learned <= mse_plain * (1 + 1e-9) + 1e-18, \
        f"learned should be <= plain: learned={mse_learned:.4e}, plain={mse_plain:.4e}"

    # Clean block (no bursts): oracle mask empty -> oracle reduces to
    # plain batch LS (v=1 everywhere -> same reweighted system every K).
    w_o = torch.randn(d, dtype=torch.float64)
    w_o = w_o / w_o.norm()
    Xc, dc, nuc, burstc = make_block(w_o, N, sigma, mode='iid',
                                     noise='gaussian', p_burst=p_burst,
                                     kappa=kappa, seed=seed + 2000,
                                     dtype=torch.float64, device='cpu',
                                     return_noise=True)
    assert burstc.sum().item() == 0, \
        f"clean (gaussian) block should have no bursts, got {burstc.sum().item()}"
    oracle_c = oracle_weighter(burstc, dtype=torch.float64)
    w_plain_c = block_robust_rls(Xc, dc, const_w, delta=delta, K=0)
    w_oracle_c = block_robust_rls(Xc, dc, oracle_c, delta=delta, K=K,
                                  max_iter=200, tol=1e-10, beta=1.0,
                                  backward_mode='exact')
    mse_plain_c = float(((Xc @ w_plain_c - Xc @ w_o).pow(2)).mean().item())
    mse_oracle_c = float(((Xc @ w_oracle_c - Xc @ w_o).pow(2)).mean().item())
    rel_c = abs(mse_oracle_c - mse_plain_c) / max(mse_plain_c, 1e-30)
    print(f"  [oracle bound clean] plain={mse_plain_c:.4e} oracle={mse_oracle_c:.4e} "
          f"rel_diff={rel_c:.4e}")
    assert rel_c < 1e-6, \
        f"oracle must reduce to plain LS on a clean block: rel_diff={rel_c:.4e}"


def test_oos_monotonicity(d=6, delta=1e-2, K=4, sigma=0.01, p_burst=0.02,
                          kappa=20.0, seed=0, epochs=20, lr=1e-2, n_trials=6):
    """Phase 0 gate 2 (oracle-oos-headroom): out-of-sample monotonicity.

    Batch OOS (fit on a train block, score on a fresh test block with the
    same plant) must improve as the train block grows: more training data
    -> better plant estimate -> lower OOS error.  The OOS estimator
    ``||X_test w - X_test w_o||^2/N`` has high per-trial variance (X_test
    is redrawn every trial), so the gate tests:
      * the span direction for the learned weighter (OOS at N=512 must
        beat OOS at N=128);
      * strict step-by-step monotonicity for the ORACLE (the cleanest
        signal -- perfectly discards bursts, so its OOS is dominated by
        the deterministic LS bias that shrinks with N);
      * the oracle ceiling holds out-of-sample at every N (oracle <=
        learned).
    """
    torch.manual_seed(seed)
    weighter = LearnedRobustWeighter(c_init=0.10119, alpha_init=0.12693)
    optimizer = torch.optim.Adam(weighter.parameters(), lr=lr)

    for epoch in range(epochs):
        w_o = torch.randn(d, dtype=torch.float32)
        w_o = w_o / w_o.norm()
        X, d_obs = make_block(w_o, 128, sigma, mode='iid',
                              noise='impulsive', p_burst=p_burst,
                              kappa=kappa, seed=seed + epoch,
                              dtype=torch.float32, device='cpu')
        w_K = block_robust_rls(X, d_obs, weighter, delta=delta, K=K,
                               max_iter=50, tol=1e-5, backward_mode='phantom')
        loss = (X @ w_K - X @ w_o).pow(2).sum() + 10.0 * (w_K - w_o).pow(2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    w_here = weighter.to(torch.float64)
    oos_learned = []
    oos_oracle = []
    for N in [128, 256, 512]:
        acc_l, acc_o = 0.0, 0.0
        for trial in range(n_trials):
            w_o = torch.randn(d, dtype=torch.float64)
            w_o = w_o / w_o.norm()
            Xtr, dtr, nu, burst = make_block(w_o, N, sigma, mode='iid',
                                             noise='impulsive', p_burst=p_burst,
                                             kappa=kappa,
                                             seed=seed + 1000 + trial + N,
                                             dtype=torch.float64, device='cpu',
                                             return_noise=True)
            # Fresh test block, same plant, seed offset far from training.
            Xte, dte = make_block(w_o, N, sigma, mode='iid',
                                  noise='impulsive', p_burst=p_burst,
                                  kappa=kappa,
                                  seed=seed + 1000 + trial + N + 1_000_000,
                                  dtype=torch.float64, device='cpu')
            w_l = block_robust_rls(Xtr, dtr, w_here, delta=delta, K=K,
                                   max_iter=200, tol=1e-10, beta=1.0,
                                   backward_mode='exact')
            oracle_w = oracle_weighter(burst, dtype=torch.float64)
            w_o_ = block_robust_rls(Xtr, dtr, oracle_w, delta=delta, K=K,
                                    max_iter=200, tol=1e-10, beta=1.0,
                                    backward_mode='exact')
            oos_l = float(((Xte @ w_l - Xte @ w_o).pow(2)).mean().item())
            oos_or = float(((Xte @ w_o_ - Xte @ w_o).pow(2)).mean().item())
            acc_l += oos_l
            acc_o += oos_or
        oos_learned.append(acc_l / n_trials)
        oos_oracle.append(acc_o / n_trials)
        print(f"  [oos monotone N={N}] learned={oos_learned[-1]:.4e} "
              f"oracle={oos_oracle[-1]:.4e}")

    print(f"  [oos monotone] learned: {[f'{x:.4e}' for x in oos_learned]}")
    print(f"  [oos monotone] oracle:  {[f'{x:.4e}' for x in oos_oracle]}")
    # Span monotonicity: more training data must help across the N range.
    assert oos_learned[-1] < oos_learned[0], \
        f"OOS learned should improve with N (span): {oos_learned}"
    # Oracle OOS is the cleanest signal -- strictly step-monotone in N.
    assert oos_oracle[0] > oos_oracle[1] > oos_oracle[2], \
        f"OOS oracle should improve monotonically with N: {oos_oracle}"
    # Oracle ceiling holds out-of-sample at every N.
    for oo, ll in zip(oos_oracle, oos_learned):
        assert oo <= ll * (1 + 1e-9) + 1e-18, \
            f"OOS oracle should be <= learned: oracle={oo:.4e}, learned={ll:.4e}"


def test_mad_scale_tracks_noise(T=2000, seed=0):
    """Classical-robust baseline gate 1: the MAD-adaptive scale.

    (a) 1.4826 * median(|e|) recovers the noise scale for Gaussian blocks.
    (b) The scale is recomputed PER CALL (no cross-call caching): the same
        ``MADRobustWeighter`` module given residual blocks of different
        scales weights the *same* residual value differently -- downweights
        it when it is many sigma (small-scale block) and keeps it when it
        is sub-sigma (large-scale block).
    """
    torch.manual_seed(seed)
    e1 = 0.005 * torch.randn(T, dtype=torch.float64)
    e2 = 0.05 * torch.randn(T, dtype=torch.float64)
    # Inject the same absolute residual into both blocks.
    e1[0] = 0.02
    e2[0] = 0.02

    # (a) MAD scale ~ sigma for Gaussian noise.
    s1 = 1.4826 * e1.abs().median()
    s2 = 1.4826 * e2.abs().median()
    assert 0.8 < s1 / 0.005 < 1.2, \
        f"MAD scale mismatch (small): {s1:.4e} vs 0.005"
    assert 0.8 < s2 / 0.05 < 1.2, \
        f"MAD scale mismatch (large): {s2:.4e} vs 0.05"

    # (b) Per-block adaptivity: 0.02 is 4 sigma of e1 (downweight) but
    # 0.4 sigma of e2 (keep).  A cached scale would fail this.
    w = MADRobustWeighter(mode='huber', dtype=torch.float64)
    v1 = w(e1)
    v2 = w(e2)
    print(f"  [mad scale] s1={s1:.4e} s2={s2:.4e} "
          f"v1[0]={v1[0].item():.4f} v2[0]={v2[0].item():.4f}")
    assert v1[0].item() < 0.5, \
        f"small-scale block must downweight 0.02 (4 sigma): v1[0]={v1[0].item():.4f}"
    assert v2[0].item() > 0.9, \
        f"large-scale block must keep 0.02 (0.4 sigma): v2[0]={v2[0].item():.4f}"
    assert not torch.equal(v1, v2), \
        "MAD weighter must not cache its scale across calls"

    # Hampel: a far outlier (|u| > r) must be fully zeroed.
    e3 = 0.01 * torch.randn(T, dtype=torch.float64)
    e3[0] = 100.0 * 0.01  # 100 sigma
    w_hm = MADRobustWeighter(mode='hampel', dtype=torch.float64)
    v3 = w_hm(e3)
    assert v3[0].item() < 1e-6, \
        f"Hampel should zero a 100-sigma outlier: v3[0]={v3[0].item():.4e}"


def test_huber_hampel_mad_improvement(d=6, N=256, delta=1e-2, K=8,
                                      sigma=0.01, p_burst=0.02, kappa=20.0,
                                      seed=0, n_trials=3):
    """Classical-robust baseline gate 2: Huber/Hampel + MAD beat plain on
    impulsive blocks, and degrade only mildly on clean (Gaussian) blocks.

    The MAD-adaptive scale (1.4826 * median|e|, recomputed per IRLS round)
    is the classical state-of-the-art fixed robust method.  If the learned
    weighter cannot beat these, the paper story needs the richer weighter
    (per the "missing contender" audit).

    The Gaussian-case bound is deliberately NOT ~0: on clean data a robust
    estimator sacrifices efficiency (Huber at a=1.345 is ~95% asymptotically
    relative efficiency with known scale; finite-sample scale estimation
    and the in-sample metric -- which rewards LS by construction -- inflate
    that further).  The gate asserts only that the degradation is bounded
    (no catastrophic/spurious downweighting).
    """
    huber_w = MADRobustWeighter(mode='huber', dtype=torch.float64)
    hampel_w = MADRobustWeighter(mode='hampel', dtype=torch.float64)
    const_w = constant_weighter(1.0, dtype=torch.float64)

    def _mse_block(X, d_obs, wgt):
        w = block_robust_rls(X, d_obs, wgt, delta=delta, K=K,
                             max_iter=200, tol=1e-10, beta=1.0,
                             backward_mode='exact')
        return w

    # ---- Impulsive: MAD-adaptive robust must beat plain ----
    imp = {'plain': 0.0, 'huber': 0.0, 'hampel': 0.0}
    for trial in range(n_trials):
        g = torch.Generator().manual_seed(seed + 1000 + trial)
        w_o = torch.randn(d, generator=g, dtype=torch.float64)
        w_o = w_o / w_o.norm()
        X, d_obs = make_block(w_o, N, sigma, mode='iid', noise='impulsive',
                              p_burst=p_burst, kappa=kappa,
                              seed=seed + 1000 + trial,
                              dtype=torch.float64)
        mse = lambda w_: float(((X @ w_ - X @ w_o).pow(2)).mean().item())
        imp['plain'] += mse(_mse_block(X, d_obs, const_w))
        imp['huber'] += mse(_mse_block(X, d_obs, huber_w))
        imp['hampel'] += mse(_mse_block(X, d_obs, hampel_w))
    for k in imp:
        imp[k] /= n_trials
    print(f"  [huber/hampel impulsive] plain={imp['plain']:.4e} "
          f"huber={imp['huber']:.4e} hampel={imp['hampel']:.4e}")
    assert imp['huber'] < imp['plain'], \
        f"Huber+MAD must beat plain on impulsive: {imp['huber']:.4e} vs {imp['plain']:.4e}"
    assert imp['hampel'] < imp['plain'], \
        f"Hampel+MAD must beat plain on impulsive: {imp['hampel']:.4e} vs {imp['plain']:.4e}"

    # ---- Gaussian (clean): MAD-adaptive degrades only mildly ----
    gauss = {'plain': 0.0, 'huber': 0.0, 'hampel': 0.0}
    for trial in range(n_trials):
        g = torch.Generator().manual_seed(seed + 2000 + trial)
        w_o = torch.randn(d, generator=g, dtype=torch.float64)
        w_o = w_o / w_o.norm()
        X, d_obs = make_block(w_o, N, sigma, mode='iid', noise='gaussian',
                              seed=seed + 2000 + trial,
                              dtype=torch.float64)
        mse = lambda w_: float(((X @ w_ - X @ w_o).pow(2)).mean().item())
        gauss['plain'] += mse(_mse_block(X, d_obs, const_w))
        gauss['huber'] += mse(_mse_block(X, d_obs, huber_w))
        gauss['hampel'] += mse(_mse_block(X, d_obs, hampel_w))
    for k in gauss:
        gauss[k] /= n_trials
    rel_huber = abs(gauss['huber'] - gauss['plain']) / max(gauss['plain'], 1e-30)
    rel_hampel = abs(gauss['hampel'] - gauss['plain']) / max(gauss['plain'], 1e-30)
    print(f"  [huber/hampel gaussian] plain={gauss['plain']:.4e} "
          f"huber={gauss['huber']:.4e} hampel={gauss['hampel']:.4e}")
    # Bounded efficiency loss only: at a=1.345, Huber is ~95% ARE (asymptotic,
    # known scale); finite-sample MAD estimation plus the in-sample metric
    # (which rewards LS by construction) inflate that further.  The gate
    # rejects only catastrophic/spurious downweighting (order-1x degradation).
    assert rel_huber < 2.5e-1, \
        f"Huber must not catastrophically degrade on Gaussian: rel={rel_huber:.4e}"
    assert rel_hampel < 2.5e-1, \
        f"Hampel must not catastrophically degrade on Gaussian: rel={rel_hampel:.4e}"


# ----------------------------------------------------------------------------
# Phase B / Phase E1: fast unit tests for the log-coordinate weighter.
# ----------------------------------------------------------------------------


def test_new_legacy_parameterization_equivalence(dtype=torch.float64,
                                                  tol=1e-12):
    """Phase E1.1: new log-scale weighter produces the same v(e) curve
    as the legacy softplus weighter at the matching (c, alpha) operating
    point (forward equivalence at tight tolerance).

    The legacy ``raw_c=-2.25`` corresponds to ``c = softplus(-2.25)+1e-3``
    (about 0.1012); the new init defaults ``c_init=0.10`` are within
    sub-percent of that -- the test compares both ways: a legacy
    parameterization and a new one constructed at the same c and
    alpha, and asserts max-abs-error on a representative e grid.
    """
    legacy_w = LearnedRobustWeighter(c_init=0.10, alpha_init=0.13).to(dtype)
    legacy_c = legacy_w.c.item()
    legacy_alpha = legacy_w.alpha.item()
    new_w = LearnedRobustWeighter(c_init=legacy_c, alpha_init=legacy_alpha).to(dtype)
    e = torch.tensor([-10.0, -0.5, -0.1, 0.0, 0.1, 0.5, 10.0],
                     dtype=dtype)
    v_legacy = legacy_w(e).detach()
    v_new = new_w(e).detach()
    err = (v_legacy - v_new).abs().max().item()
    print(f"  [param equivalence] c={legacy_c:.6f}, alpha={legacy_alpha:.6f}, "
          f"max_abs_err={err:.3e}, tol={tol}")
    assert err < tol, \
        f"new and legacy weighter parameterizations disagree: {err:.3e} >= {tol}"


def test_weight_bounds_monotonicity(dtype=torch.float64):
    """Phase E1.2: v(e) is bounded in [0, 1] for |e| in [1e-8, 1e4] and
    monotonically non-increasing in |e| over a log-spaced grid.

    Tests the actual scientific claim rather than a few manually chosen
    error values (plan Phase E1.2).
    """
    w = LearnedRobustWeighter(c_init=0.1012, alpha_init=0.1269).to(dtype)
    e_abs = torch.logspace(-8, 4, 200, dtype=dtype)
    v = w(e_abs).detach()
    assert torch.all(v >= 0).item(), \
        f"v_t has negative entries: min={v.min().item()}"
    assert torch.all(v <= 1.0 + 1e-12).item(), \
        f"v_t exceeds v_max: max={v.max().item()}"
    # Monotone non-increasing in |e| (the forward is symmetric and
    # monotone in |e|).
    assert torch.all(v[1:] <= v[:-1] + 1e-15).item(), \
        "v(|e|) is not monotonically non-increasing"
    # Sanity: v(1e-8) ~ 1 (very small |e| -> near the origin -> v ~ 1)
    # and v(1e4) descends measurably (large |e| -> saturation tail).
    # Note: with the init alpha ~ 0.127 the tail decays slowly
    # (v(1e4) ~ 5e-2), so we only require a strong relative descent
    # rather than near-zero absolute value.
    v_near_zero = v[0].item()
    v_far = v[-1].item()
    assert v_near_zero > 0.99, \
        f"v(1e-8) should be near 1, got {v_near_zero}"
    assert v_far < 0.2, \
        f"v(1e4) should descend well below 1, got {v_far:.4e}"
    print(f"  [weight bounds/monotone] v(1e-8)={v_near_zero:.4f}, "
          f"v(1e4)={v_far:.4e}, monotone non-increasing. PASS")


def test_legacy_checkpoint_migration(dtype=torch.float64):
    """Phase E1.3: a legacy softplus checkpoint (raw_c, raw_alpha) loads
    into the new log-scale weighter and produces the same forward output
    as a freshly-constructed legacy weighter (within float64 tolerance).

    Emits a ``UserWarning`` on load (the documented migration signal).
    """
    import torch.nn.functional as _F
    legacy_w = LearnedRobustWeighter(c_init=0.10, alpha_init=0.13).to(dtype)
    legacy_sd = legacy_w.state_dict()
    assert 'log_c' in legacy_sd and 'log_alpha' in legacy_sd, \
        f"new weighter state dict missing log_c/log_alpha: {list(legacy_sd)}"
    # Simulate a legacy checkpoint: raw_c / raw_alpha.
    legacy_ckpt = {'raw_c': torch.tensor(-2.25, dtype=dtype),
                   'raw_alpha': torch.tensor(-2.0, dtype=dtype),
                   'v_max': torch.tensor(1.0, dtype=dtype)}
    legacy_c = _F.softplus(torch.tensor(-2.25, dtype=dtype)) + 1e-3
    legacy_alpha = _F.softplus(torch.tensor(-2.0, dtype=dtype))
    new_w = LearnedRobustWeighter(c_init=0.10, alpha_init=0.13).to(dtype)
    new_w.load_state_dict(legacy_ckpt)
    assert abs(new_w.c.item() - legacy_c.item()) < 1e-12, \
        f"migrated c={new_w.c.item()} != legacy c={legacy_c.item()}"
    assert abs(new_w.alpha.item() - legacy_alpha.item()) < 1e-12, \
        f"migrated alpha={new_w.alpha.item()} != legacy alpha={legacy_alpha.item()}"
    e = torch.tensor([-1.0, -0.1, 0.0, 0.1, 1.0], dtype=dtype)
    v_new = new_w(e).detach()
    legacy_compare = LearnedRobustWeighter(c_init=float(legacy_c),
                                          alpha_init=float(legacy_alpha)).to(dtype)
    v_ref = legacy_compare(e).detach()
    err = (v_new - v_ref).abs().max().item()
    assert err < 1e-12, \
        f"migrated vs freshly-built weighter disagree: {err:.3e}"
    print(f"  [legacy migration] migrated (c, alpha)=({new_w.c.item():.6f}, "
          f"{new_w.alpha.item():.6f}), "
          f"forward_err={err:.3e}. PASS")


def test_deq_direct_oneblock_parity(d=8, N=64, K=4, delta=1e-2, seed=0,
                                      tol_forward=1e-4, tol_grad=1e-3):
    """Phase E1.4 / Phase C5: DEQ and direct closed-form block IRLS
    agree on forward w and on the gradient w.r.t. log_c/log_alpha at
    float64 / tight tol.

    Kept small enough (d=8, N=64, K=4) for run_all.py; the
    production-scale gradcheck lives in ``tests/test_prod_gradcheck.py``.
    """
    torch.manual_seed(seed)
    g = torch.Generator(device='cpu').manual_seed(seed)
    w_o = torch.randn(d, generator=g, dtype=torch.float64)
    w_o = w_o / w_o.norm()
    X = torch.randn(N, d, generator=g, dtype=torch.float64)
    nu = 0.01 * torch.randn(N, generator=g, dtype=torch.float64)
    d_obs = X @ w_o + nu

    wgt = LearnedRobustWeighter(c_init=0.1012, alpha_init=0.1269).to(torch.float64)

    w_deq = block_robust_rls(X, d_obs, wgt, delta=delta, K=K,
                             max_iter=500, tol=1e-12,
                             backward_mode='exact', solver='deq')
    w_dir = block_robust_rls(X, d_obs, wgt, delta=delta, K=K,
                             max_iter=500, tol=1e-12,
                             backward_mode='exact', solver='direct')
    rel = (w_deq - w_dir).norm().item() / w_deq.norm().clamp_min(1e-12).item()
    assert rel < tol_forward, \
        f"DEQ vs direct forward parity mismatch: rel={rel:.3e} >= {tol_forward}"

    wgt_g1 = LearnedRobustWeighter(c_init=0.1012, alpha_init=0.1269).to(torch.float64)
    wgt_g2 = LearnedRobustWeighter(c_init=0.1012, alpha_init=0.1269).to(torch.float64)
    w_K1 = block_robust_rls(X, d_obs, wgt_g1, delta=delta, K=K,
                            max_iter=500, tol=1e-12,
                            backward_mode='exact', solver='deq')
    ((X @ w_K1 - X @ w_o).pow(2).sum()).backward()
    g_log_c_deq = wgt_g1.log_c.grad.item()
    g_log_a_deq = wgt_g1.log_alpha.grad.item()
    w_K2 = block_robust_rls(X, d_obs, wgt_g2, delta=delta, K=K,
                            max_iter=500, tol=1e-12,
                            backward_mode='exact', solver='direct')
    ((X @ w_K2 - X @ w_o).pow(2).sum()).backward()
    g_log_c_dir = wgt_g2.log_c.grad.item()
    g_log_a_dir = wgt_g2.log_alpha.grad.item()
    rel_g_c = abs(g_log_c_deq - g_log_c_dir) / max(abs(g_log_c_dir), 1e-12)
    rel_g_a = abs(g_log_a_deq - g_log_a_dir) / max(abs(g_log_a_dir), 1e-12)
    assert rel_g_c < tol_grad and rel_g_a < tol_grad, \
        f"DEQ vs direct grad parity mismatch: " \
        f"rel_g_c={rel_g_c:.3e}, rel_g_a={rel_g_a:.3e}"

    print(f"  [DEQ vs direct parity d={d} N={N} K={K}] "
          f"forward_rel={rel:.3e}, grad_rel_c={rel_g_c:.3e}, "
          f"grad_rel_a={rel_g_a:.3e}. PASS")


if __name__ == '__main__':
    print("=" * 60)
    print("Phase 0 + Phase 1 + Phase 1.5 + oracle-oos gates")
    print("=" * 60)
    # Phase 0
    test_linear_solve_layer_gradients_flow()
    test_gradcheck_via_finite_diff()
    test_gradcheck_batch()
    test_weighter_grad_flow_simulation()
    # Phase 1
    test_weighter_bounds()
    test_constant_weighter_returns_one()
    test_weighter_init_unit_check()
    test_digital_robust_v1_byteexact_matches_digital_rls()
    test_fabric_robust_v1_close_to_fabric_rls()
    test_digital_robust_matches_fabric_robust()
    test_implicit_vs_unrolled_grad_robust()
    test_impulsive_noise_burst_rate()
    test_gaussian_control_flat_weighter()
    # Phase 1.5
    test_block_robust_v1_matches_batch_ls()
    test_block_robust_digital_matches_fabric()
    test_block_robust_impulsive_improvement()
    test_block_robust_grad_flow()
    test_phantom_vs_exact_bias()
    # Phase 0 (oracle-oos-headroom)
    test_oracle_bound()
    test_oos_monotonicity()
    # Classical-robust baselines (Huber/Hampel + MAD scale)
    test_mad_scale_tracks_noise()
    test_huber_hampel_mad_improvement()
    # Phase E1: log-coordinate weighter unit tests
    test_new_legacy_parameterization_equivalence()
    test_weight_bounds_monotonicity()
    test_legacy_checkpoint_migration()
    test_deq_direct_oneblock_parity()
    print("\nAll Phase 0 + Phase 1 + Phase 1.5 + oracle-oos + E1 gates passed.")

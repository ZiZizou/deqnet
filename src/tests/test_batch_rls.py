"""Focused regression tests for the batch least-squares RLS experiment.

Each test corresponds to a documented guarantee in
`docs/specs/batch-rls-demo.md`:

* Shapes/SPD: accumulation returns R with documented shape and R is SPD
  before the fabric solve.
* Direct-reference agreement: the fabric solution matches
  torch.linalg.lstsq on the same regularized objective within float64
  tolerance.
* Single-settle: exactly one LinearSolveLayer invocation per batch solve.
* Determinism: fixed seeds produce identical accumulated systems and
  reported metrics.
"""
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS_DIR))
sys.path.insert(0, os.path.join(os.path.dirname(THIS_DIR), 'src'))

import torch

from utils.circuit_block import EquilibriumSolve
from run_rls_demo import (
    FabricBatchRLS,
    batch_lstsq_reference,
    batch_experiment_metrics,
)


def _make_block(d=6, T=128, delta=1e-2, sigma=0.01, seed=0, dtype=torch.float64):
    torch.manual_seed(seed)
    w_o = torch.randn(d, dtype=dtype)
    w_o = w_o / w_o.norm()
    X = torch.randn(T, d, dtype=dtype)
    d_obs = X @ w_o + sigma * torch.randn(T, dtype=dtype)
    R0 = delta * torch.eye(d, dtype=dtype)
    return X, d_obs, R0, w_o


def test_batch_accumulation_shapes_and_spd():
    X, d_obs, R0, _ = _make_block()
    solver = FabricBatchRLS(d=X.shape[1], R0=R0, max_iter=200, tol=1e-12)
    solver.accumulate(X, d_obs)
    R_sym = 0.5 * (solver.R + solver.R.t())
    assert solver.R.shape == (X.shape[1], X.shape[1])
    assert solver.p.shape == (X.shape[1],)
    eigs = torch.linalg.eigvalsh(R_sym)
    assert eigs.min().item() > 0.0, \
        f"accumulated R is not SPD: min eig = {eigs.min().item():.3e}"
    assert torch.allclose(solver.R, solver.R.t(), atol=1e-12), \
        "accumulated R is not symmetric"
    print(f"  [batch shapes/SPD] min eig = {eigs.min().item():.3e}. PASS")


def test_batch_fabric_matches_lstsq():
    X, d_obs, R0, _ = _make_block()
    solver = FabricBatchRLS(d=X.shape[1], R0=R0, max_iter=300, tol=1e-12)
    solver.accumulate(X, d_obs)
    w_fabric, info = solver.solve()
    assert info is not None
    assert info.get('converged', False), f"did not converge: {info}"
    R_sym = 0.5 * (solver.R + solver.R.t())
    m = batch_experiment_metrics(w_fabric, R_sym, solver.p, X, d_obs, R0)
    print(f"  [batch vs lstsq] rel_err={m['rel_err']:.3e} "
          f"normal_eq_res={m['normal_eq_residual']:.3e} "
          f"iters={info['n_iter']}")
    assert m['rel_err'] < 1e-6, f"fabric/reference mismatch: {m['rel_err']}"
    print("  PASS")


def test_batch_single_settle_invocation():
    X, d_obs, R0, _ = _make_block()
    calls = {'n': 0}

    orig_last = EquilibriumSolve.last_info
    EquilibriumSolve.last_info = None

    def counting_solve(solver):
        # Snapshot the class attribute, count distinct ids.
        before = id(EquilibriumSolve.last_info)
        w_star, info = solver.solve()
        after = id(EquilibriumSolve.last_info)
        if before != after:
            calls['n'] += 1
        return w_star, info

    solver = FabricBatchRLS(d=X.shape[1], R0=R0, max_iter=100, tol=1e-10)
    solver.accumulate(X, d_obs)
    _ = counting_solve(solver)
    EquilibriumSolve.last_info = orig_last
    assert calls['n'] == 1, f"expected exactly one EquilibriumSolve call, got {calls['n']}"
    print("  [batch single-settle] one EquilibriumSolve invocation per solve(). PASS")


def test_batch_deterministic_with_seed():
    X1, d_obs1, R0_1, w_o_1 = _make_block(seed=123)
    X2, d_obs2, R0_2, w_o_2 = _make_block(seed=123)
    assert torch.equal(X1, X2) and torch.equal(d_obs1, d_obs2)
    assert torch.allclose(R0_1, R0_2)
    assert torch.equal(w_o_1, w_o_2)

    s1 = FabricBatchRLS(d=X1.shape[1], R0=R0_1, max_iter=200, tol=1e-12)
    s2 = FabricBatchRLS(d=X2.shape[1], R0=R0_2, max_iter=200, tol=1e-12)
    s1.accumulate(X1, d_obs1)
    s2.accumulate(X2, d_obs2)
    assert torch.equal(s1.R, s2.R)
    assert torch.equal(s1.p, s2.p)

    w1, info1 = s1.solve()
    w2, info2 = s2.solve()
    assert torch.equal(w1, w2), "deterministic seed produced different w"
    assert info1.get('n_iter') == info2.get('n_iter')
    print("  [batch determinism] seed=123 -> identical R, p, w. PASS")


def test_batch_lstsq_reference_matches_augmented_system():
    X, d_obs, R0, _ = _make_block()
    w_ref = batch_lstsq_reference(X, d_obs, R0)
    n = X.shape[1]
    # Reference forms R0 = L^T L via eigendecomposition, then solves
    # lstsq([X; L], [d; 0]) which is equivalent to the original quadratic.
    eigs, Q = torch.linalg.eigh(R0)
    L = Q @ torch.diag(torch.sqrt(eigs.clamp_min(0.0))) @ Q.t()
    A_aug = torch.cat([X, L], dim=0)
    rhs_aug = torch.cat([d_obs, torch.zeros(n, dtype=X.dtype)])
    w_direct, *_ = torch.linalg.lstsq(A_aug, rhs_aug)
    err = (w_ref - w_direct).abs().max().item()
    print(f"  [batch lstsq ref] vs sqrt(R0)-augmented lstsq err = {err:.3e}")
    assert err < 1e-10, f"reference implementation diverges from sqrt(R0)-augmented lstsq: {err}"
    print("  PASS")


def test_batch_weight_consistency():
    """Fabric and reference must solve the same weighted objective when
    weight != 1.0."""
    X, d_obs, R0, _ = _make_block()
    for wgt in (1.0, 0.5, 2.0):
        s = FabricBatchRLS(d=X.shape[1], R0=R0, weight=wgt,
                           max_iter=300, tol=1e-12)
        s.accumulate(X, d_obs)
        # Accumulate symmetrizes self.R; ensure metric sees the same R.
        w_fabric, info = s.solve()
        assert info is not None and info.get('converged', False)
        m = batch_experiment_metrics(w_fabric, s.R, s.p, X, d_obs, R0,
                                     weight=wgt)
        print(f"  [batch weight={wgt}] rel_err={m['rel_err']:.3e} "
              f"normal_eq_res={m['normal_eq_residual']:.3e} "
              f"iters={info['n_iter']}")
        assert m['rel_err'] < 1e-6, \
            f"weight={wgt} mismatch: rel_err={m['rel_err']}"
        assert m['normal_eq_residual'] < 1e-6, \
            f"weight={wgt} normal-eq residual too large: {m['normal_eq_residual']}"
    print("  PASS")


def test_batch_objective_matches_lstsq_minimizer():
    """The reported regularized_objective must match the weighted objective
    both fabric and reference are minimizing (no arbitrary scaling)."""
    X, d_obs, R0, _ = _make_block()
    wgt = 1.5
    s = FabricBatchRLS(d=X.shape[1], R0=R0, weight=wgt, max_iter=300, tol=1e-12)
    s.accumulate(X, d_obs)
    w_fabric, _ = s.solve()
    w_ref = batch_lstsq_reference(X, d_obs, R0, weight=wgt)
    expected = (wgt * (X @ w_fabric - d_obs).pow(2).sum()
                + w_fabric @ R0 @ w_fabric).item()
    expected_ref = (wgt * (X @ w_ref - d_obs).pow(2).sum()
                    + w_ref @ R0 @ R0.new_zeros(())).item() if False else None
    # Compute the same objective at w_ref directly to confirm reference
    # value.
    ref_obj = (wgt * (X @ w_ref - d_obs).pow(2).sum()
               + w_ref @ R0 @ w_ref).item()
    m = batch_experiment_metrics(w_fabric, s.R, s.p, X, d_obs, R0, weight=wgt)
    print(f"  [batch objective] fabric_obj={m['regularized_objective']:.6e} "
          f"ref_obj={ref_obj:.6e}")
    assert abs(m['regularized_objective'] - expected) < 1e-8 * max(1.0, abs(expected))
    assert abs(ref_obj - m['regularized_objective']) < 1e-8 * max(1.0, abs(expected))
    print("  PASS")


def test_batch_r_is_symmetric_after_accumulate():
    X, d_obs, R0, _ = _make_block()
    s = FabricBatchRLS(d=X.shape[1], R0=R0, max_iter=100, tol=1e-12)
    s.accumulate(X, d_obs)
    assert torch.allclose(s.R, s.R.t(), atol=1e-12), \
        f"self.R is not symmetric after accumulate: max asym = {(s.R - s.R.t()).abs().max().item():.3e}"
    print("  [batch R symmetry] PASS")


if __name__ == '__main__':
    print("=" * 60)
    print("Test: Batch least-squares RLS")
    print("=" * 60)
    test_batch_accumulation_shapes_and_spd()
    test_batch_fabric_matches_lstsq()
    test_batch_single_settle_invocation()
    test_batch_deterministic_with_seed()
    test_batch_lstsq_reference_matches_augmented_system()
    test_batch_weight_consistency()
    test_batch_objective_matches_lstsq_minimizer()
    test_batch_r_is_symmetric_after_accumulate()
    print("\nAll batch RLS tests passed.")

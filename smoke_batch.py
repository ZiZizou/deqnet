"""Smoke test for the batch least-squares experiment."""
import os
import sys
import tempfile
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(THIS_DIR, 'src'))

from run_rls_demo import (
    FabricBatchRLS, batch_lstsq_reference, batch_experiment_metrics,
)


def main():
    torch.manual_seed(0)
    dtype = torch.float64
    d = 8
    T = 512
    delta = 1e-2
    sigma = 0.01
    w_o = torch.randn(d, dtype=dtype)
    w_o = w_o / w_o.norm()
    X = torch.randn(T, d, dtype=dtype)
    d_obs = X @ w_o + sigma * torch.randn(T, dtype=dtype)

    R0 = delta * torch.eye(d, dtype=dtype)
    solver = FabricBatchRLS(d=d, R0=R0, max_iter=300, tol=1e-12, beta='chebyshev')
    solver.accumulate(X, d_obs)
    R_sym = 0.5 * (solver.R + solver.R.t())
    eigs = torch.linalg.eigvalsh(R_sym)
    assert eigs.min().item() > 0.0, "R is not SPD"

    w_fabric, info = solver.solve()
    assert info is not None and info.get('n_iter', -1) > 0, \
        f"info missing or zero iters: {info}"
    assert info.get('converged', False), f"did not converge: {info}"

    m = batch_experiment_metrics(w_fabric, R_sym, solver.p, X, d_obs, R0, w_o=w_o)
    print(f"abs_err={m['abs_err']:.3e} rel_err={m['rel_err']:.3e}"
          f" normal_eq_res={m['normal_eq_residual']:.3e}"
          f" objective={m['regularized_objective']:.6e}"
          f" plant_err={m['plant_error']:.3e}"
          f" iters={info['n_iter']} converged={info['converged']}")
    assert m['rel_err'] < 1e-5, f"fabric/reference mismatch: {m['rel_err']}"
    assert m['normal_eq_residual'] < 1e-6, \
        f"normal-equation residual too large: {m['normal_eq_residual']}"
    print("PASS")


if __name__ == '__main__':
    main()

"""Tests for the DEQ implementation.

Run with:
    cd src && python -m pytest tests/ -v
or:
    cd src && python tests/test_deq_solver.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

from utils.deq_solver import (
    fixed_point, anderson, solve_jacobian_transpose,
    check_contraction, estimate_lipschitz,
)


def linear_residual_factory(A, b):
    """Build f(v) = b - A v so that J = -A is negative definite.

    Matches the circuit residual form f(v) = -B^T g - Gamma v + Su, whose
    Jacobian is J = -M = -(B^T D B + Gamma).  Under this sign convention,
    the fixed-point iteration g(v) = v + beta * f(v) has Jacobian
    I - beta*A with spectral radius < 1 for beta in (0, 2 / lambda_max(A)).
    """
    def f(v):
        return b - A @ v
    return f


def test_linear_anderson_matches_solve():
    """Anderson on a random SPD system must match torch.linalg.solve.

    Build a controlled SPD with eigvals in [1, 5]; choose beta at the
    Chebyshev-optimum for the fixed-point map.
    """
    torch.manual_seed(0)
    n = 8
    Q = torch.randn(n, n)
    D = 1.0 + 4.0 * torch.rand(n)
    A = Q @ torch.diag(D) @ Q.t()
    A = (A + A.t()) / 2 + torch.eye(n)
    eigs = torch.linalg.eigvalsh(A)
    lam_min, lam_max = eigs.min().item(), eigs.max().item()
    b = torch.randn(n)
    f = linear_residual_factory(A, b)
    v0 = torch.zeros(n)
    # Chebyshev/optimal fixed-point beta: 2 / (lam_min + lam_max).
    beta_opt = 2.0 / (lam_min + lam_max)
    v_star, info = anderson(f, v0, m=5, beta=beta_opt, tol=1e-6, max_iter=5000)
    v_ref = torch.linalg.solve(A, b)
    err = (v_star - v_ref).abs().max().item()
    print(f"[linear_anderson] err={err:.3e} iter={info['n_iter']} conv={info['converged']} beta={beta_opt:.3f}")
    assert err < 1e-4, f"Anderson mismatch: err={err:.3e}"
    print("  PASS")


def test_linear_fixed_point_matches_solve():
    """Fixed-point on the same system must also match."""
    torch.manual_seed(0)
    n = 8
    Q = torch.randn(n, n)
    D = 1.0 + 4.0 * torch.rand(n)
    A = Q @ (torch.diag(D) @ Q.t()) + torch.eye(n) * 0.0
    A = (A + A.t()) / 2 + torch.eye(n)
    lam_max = torch.linalg.eigvalsh(A).max().item()
    b = torch.randn(n)
    f = linear_residual_factory(A, b)
    v0 = torch.zeros(n)
    beta = 0.5 / lam_max
    v_star, info = fixed_point(f, v0, beta=beta, tol=1e-9, max_iter=5000)
    v_ref = torch.linalg.solve(A, b)
    err = (v_star - v_ref).abs().max().item()
    print(f"[linear_fixed_point] err={err:.3e} iter={info['n_iter']} conv={info['converged']} beta={beta:.3f}")
    assert err < 1e-4, f"fixed_point mismatch: err={err:.3e}"
    print("  PASS")


def test_anderson_unique_attractor():
    """Strongly monotone f(v) = -Av + b - u with fixed u must have unique fixed point."""
    torch.manual_seed(1)
    n = 6
    A = torch.eye(n) + 0.5 * torch.randn(n, n).abs()
    A = (A + A.t()) / 2 + 2.0 * torch.eye(n)
    lam_max = torch.linalg.eigvalsh(A).max().item()
    b = torch.randn(n)
    u = torch.zeros(n)
    f = lambda v: -A @ v + b + u
    targets = []
    for seed in range(5):
        torch.manual_seed(seed + 100)
        v0 = torch.randn(n)
        beta = 0.5 / lam_max
        v_star, info = anderson(f, v0, m=5, beta=beta, tol=1e-8, max_iter=500)
        targets.append(v_star)
    targets = torch.stack(targets)
    spread = targets.std(dim=0).max().item()
    print(f"[unique_attractor] spread={spread:.3e}")
    assert spread < 1e-4, f"Unique attractor failed: spread={spread:.3e}"
    print("  PASS")


def test_solve_jacobian_transpose_linear():
    """On a linear SPD function f(v) = A v - b with v_star=0, J=A.

    solve_jacobian_transpose must converge to A^{-1} rhs.
    """
    torch.manual_seed(2)
    n = 4
    Q = torch.randn(n, n)
    D = 1.0 + torch.rand(n)
    A = Q @ torch.diag(D) @ Q.t()  # SPD
    b = torch.randn(n)

    def f(v):
        return A @ v - b

    lam_max = torch.linalg.eigvalsh(A).max().item()
    beta = 0.5 / lam_max
    rhs = torch.randn(n)
    y = solve_jacobian_transpose(f, torch.zeros(n), rhs, tol=1e-7, max_iter=500, beta=beta)
    y_ref = torch.linalg.solve(A, rhs)
    err = (y - y_ref).abs().max().item()
    print(f"[solve_jt_linear] err={err:.3e}")
    assert err < 1e-3, f"solve_jt mismatch: {err}"
    print("  PASS")


def test_check_contraction_positive_margin():
    """A passive SPD stiffness matrix should yield positive margin."""
    n = 5
    src = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    des = torch.tensor([2, 3, 4, 5], dtype=torch.long)
    D = torch.tensor([2.0, 2.0, 2.0, 2.0])
    gamma = torch.full((n,), 1.0)
    result = check_contraction(src, des, n, D, gamma)
    print(f"[check_contraction] lambda_max={result['lambda_max_J']:.4f} "
          f"margin={result['contraction_margin']:.4f} passive={result['passive']}")
    assert result['passive'], "J should be negative definite for passive D"
    print("  PASS")


def test_check_contraction_fails_when_gamma_zero():
    """A chain topology has a Laplacian with at least one zero eigenvalue.

    With gamma=0 the J = -L_g is negative semi-definite, so margin = 0
    and we should NOT certify strict passivity.
    """
    n = 5
    src = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    des = torch.tensor([2, 3, 4, 5], dtype=torch.long)
    D = torch.tensor([2.0, 2.0, 2.0, 2.0])
    gamma = torch.zeros(n)
    result = check_contraction(src, des, n, D, gamma)
    print(f"[chain_with_zero_gamma] lambda_max={result['lambda_max_J']:.4e}")
    assert not result['passive'], "J should not be strictly negative definite when gamma=0 on a chain graph"
    print("  PASS")


def test_estimate_lipschitz():
    """Sanity check on L estimation."""
    A = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    f = lambda v: A @ v
    L = estimate_lipschitz(f, torch.zeros(2), n_steps=10, eps=1e-3)
    print(f"[estimate_lipschitz] L~={L:.3f} (should be ~3)")
    assert 1.5 < L < 5.0
    print("  PASS")


if __name__ == '__main__':
    print("=" * 60)
    print("Test: deq_solver")
    print("=" * 60)
    test_linear_anderson_matches_solve()
    test_linear_fixed_point_matches_solve()
    test_anderson_unique_attractor()
    test_solve_jacobian_transpose_linear()
    test_check_contraction_positive_margin()
    test_check_contraction_fails_when_gamma_zero()
    test_estimate_lipschitz()
    print("\nAll deq_solver tests passed.")

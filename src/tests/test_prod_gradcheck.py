"""Production-scale finite-difference gradient check for the exact
implicit backward of the block-robust IRLS weighter.

This is the correctness gate everything downstream (oracle, OOS, channel
retraining, grid sweep) currently *infers* rather than directly verifies:
the gradient of the block-training loss w.r.t. the weighter's raw
parameters at PRODUCTION scale (d=16, N=512, K=8), compared against
central finite differences.

The Phase 0 / Phase 1.5 gates validate the same plumbing at small scale
(``test_gradcheck_via_finite_diff`` d=4 probing R entries;
``test_block_robust_grad_flow`` d=8/N=16/K=4 implicit-vs-unrolled).  This
script is deliberately separate from the lightweight suite (``run_all.py``
/ ``test_learned_robust.py``): a single production-scale run costs roughly
an hour on CPU, so it is invoked explicitly, e.g.

    python tests/test_prod_gradcheck.py                     # full scale, CPU
    python tests/test_prod_gradcheck.py --d 8 --N 128 --K 4 # fast smoke
    python tests/test_prod_gradcheck.py --gpu               # GPU (Kaggle)

Method: one fixed block (float64).  The analytic gradient uses the exact
(CG) implicit backward through ``block_robust_rls``; the numerical
gradient perturbs ``raw_c`` / ``raw_alpha`` by +-eps and re-runs the
FORWARD block IRLS only (no autograd).  Central difference; several eps
are tried and the best agreement reported (the truncation-vs-roundoff
crossover).
"""
import os
import sys
import argparse

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS_DIR))  # add src/ to path

from utils.learned_robust import (
    LearnedRobustWeighter, block_robust_rls, make_block,
)


def _forward_loss(weighter, X, d_obs, w_o, delta, K, max_iter, tol):
    """Forward-only block-IRLS loss at the given weighter (no backward)."""
    w = block_robust_rls(X, d_obs, weighter, delta=delta, K=K,
                         max_iter=max_iter, tol=tol,
                         backward_mode='exact')
    return (X @ w - X @ w_o).pow(2).sum()


def _make_weighter(raw_c, raw_alpha, dtype=torch.float64):
    w = LearnedRobustWeighter(raw_c=raw_c, raw_alpha=raw_alpha)
    return w.to(dtype)


def fd_gradcheck_prod(d=16, N=512, K=8, delta=1e-2, sigma=0.01,
                      p_burst=0.02, kappa=20.0, seed=0, tol=1e-4,
                      max_iter=50, solve_tol=1e-10, device='cpu'):
    """FD-vs-exact gradient check at production scale (float64).

    Returns a dict with the exact/FD gradients and best relative errors for
    ``raw_c`` and ``raw_alpha``.  Raises AssertionError if either best
    relative error exceeds ``tol``.
    """
    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)
    w_o = torch.randn(d, generator=g, dtype=torch.float64, device=device)
    w_o = w_o / w_o.norm()
    X, d_obs = make_block(w_o, N, sigma, mode='iid',
                          noise='impulsive', p_burst=p_burst, kappa=kappa,
                          seed=seed, dtype=torch.float64, device=device)

    base_c, base_a = -2.25, -2.0  # the training init operating point

    # ---- Analytic: exact implicit backward ----
    wgt = _make_weighter(base_c, base_a).to(device)
    loss = _forward_loss(wgt, X, d_obs, w_o, delta, K, max_iter, solve_tol)
    loss.backward()
    g_c_exact = wgt.raw_c.grad.item()
    g_a_exact = wgt.raw_alpha.grad.item()

    # ---- Numerical: central finite differences (forward-only) ----
    def _loss_at(raw_c, raw_alpha):
        wgt_ = _make_weighter(raw_c, raw_alpha).to(device)
        return _forward_loss(wgt_, X, d_obs, w_o, delta, K,
                             max_iter, solve_tol).item()

    best_c, best_a = float('inf'), float('inf')
    best_eps_c = best_eps_a = None
    for eps in (1e-3, 1e-4, 1e-5):
        g_c_fd = (_loss_at(base_c + eps, base_a)
                  - _loss_at(base_c - eps, base_a)) / (2 * eps)
        g_a_fd = (_loss_at(base_c, base_a + eps)
                  - _loss_at(base_c, base_a - eps)) / (2 * eps)
        rel_c = abs(g_c_fd - g_c_exact) / max(abs(g_c_exact), 1e-12)
        rel_a = abs(g_a_fd - g_a_exact) / max(abs(g_a_exact), 1e-12)
        if rel_c < best_c:
            best_c, best_eps_c = rel_c, eps
        if rel_a < best_a:
            best_a, best_eps_a = rel_a, eps

    res = {
        'd': d, 'N': N, 'K': K, 'delta': delta, 'sigma': sigma,
        'p_burst': p_burst, 'kappa': kappa, 'seed': seed,
        'precision': 'float64',
        'solver': {'max_iter': max_iter, 'tol': solve_tol},
        'loss': float(loss.item()),
        'raw_c': {'exact_grad': g_c_exact, 'best_fd_grad': None,
                  'best_eps': best_eps_c, 'best_rel_err': best_c},
        'raw_alpha': {'exact_grad': g_a_exact, 'best_fd_grad': None,
                      'best_eps': best_eps_a, 'best_rel_err': best_a},
    }

    print(f"=== Production-scale FD gradcheck (d={d}, N={N}, K={K}, "
          f"float64, {device}) ===")
    print(f"  loss={res['loss']:.6e}")
    print(f"  raw_c:     exact={g_c_exact:.6e}  best FD rel err="
          f"{best_c:.3e} @ eps={best_eps_c}")
    print(f"  raw_alpha: exact={g_a_exact:.6e}  best FD rel err="
          f"{best_a:.3e} @ eps={best_eps_a}")
    ok_c = best_c < tol
    ok_a = best_a < tol
    print(f"  raw_c     {'PASS' if ok_c else 'FAIL'} (tol={tol})")
    print(f"  raw_alpha {'PASS' if ok_a else 'FAIL'} (tol={tol})")
    assert ok_c, f"raw_c FD-vs-exact rel err {best_c:.3e} >= {tol}"
    assert ok_a, f"raw_alpha FD-vs-exact rel err {best_a:.3e} >= {tol}"
    return res


def _resolve_device(use_gpu, gpu_id):
    if use_gpu and torch.cuda.is_available():
        return f'cuda:{gpu_id}'
    return 'cpu'


def main():
    p = argparse.ArgumentParser(
        description='Production-scale FD gradcheck for block robust IRLS.')
    p.add_argument('--d', type=int, default=16)
    p.add_argument('--N', type=int, default=512)
    p.add_argument('--K', type=int, default=8)
    p.add_argument('--delta', type=float, default=1e-2)
    p.add_argument('--sigma', type=float, default=0.01)
    p.add_argument('--p_burst', type=float, default=0.02)
    p.add_argument('--kappa', type=float, default=20.0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--tol', type=float, default=1e-4)
    p.add_argument('--solver_max_iter', type=int, default=50)
    p.add_argument('--solver_tol', type=float, default=1e-10)
    p.add_argument('--gpu', action='store_true',
                   help='Run on CUDA (default CPU).')
    p.add_argument('--gpu_id', type=int, default=0)
    args = p.parse_args()

    device = _resolve_device(args.gpu, args.gpu_id)
    print(f"Device: {device}")
    fd_gradcheck_prod(d=args.d, N=args.N, K=args.K, delta=args.delta,
                      sigma=args.sigma, p_burst=args.p_burst,
                      kappa=args.kappa, seed=args.seed, tol=args.tol,
                      max_iter=args.solver_max_iter,
                      solve_tol=args.solver_tol, device=device)
    print("\nProduction-scale FD gradcheck PASSED.")


if __name__ == '__main__':
    main()
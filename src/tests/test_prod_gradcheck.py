"""Production-scale finite-difference gradient check for the exact
implicit backward of the block-robust IRLS weighter (Phase E2).

This is the correctness gate everything downstream (oracle, OOS, channel
retraining, grid sweep) currently *infers* rather than directly verifies:
the gradient of the block-training loss w.r.t. the weighter's log-space
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
gradient perturbs ``log_c`` / ``log_alpha`` by +-eps and re-runs the
FORWARD block IRLS only (no autograd).  Central difference; an eps
sweep is tried and the best agreement reported (the
truncation-vs-roundoff crossover).

Phase B (``log_exp_v1``): the trainable coordinates are now
``log_c`` and ``log_alpha`` (log-scale reparameterization).  The
finite-difference step must be tuned in the new coordinates: eps in
``{3e-2, 1e-2, 3e-3, 1e-3, 3e-4}``.  Output includes the parameterization
tag ``log_exp_v1`` for traceability (plan B3).
"""
import argparse
import json
import os
import sys
import time

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS_DIR))  # add src/ to path

from utils.learned_robust import (
    LearnedRobustWeighter, block_robust_rls, make_block,
)


# Phase B init: the legacy softplus init operating point,
# expressed in log-space (matches the existing run_rls_demo constants).
import math as _math
import torch.nn.functional as _F
_BASE_LOG_C = float(_math.log(
    _F.softplus(torch.tensor(-2.25)).item() + 1e-3))
_BASE_LOG_ALPHA = float(_math.log(
    _F.softplus(torch.tensor(-2.0)).item()))


def _forward_loss(weighter, X, d_obs, w_o, delta, K, max_iter, tol):
    """Forward-only block-IRLS loss at the given weighter (no backward)."""
    w = block_robust_rls(X, d_obs, weighter, delta=delta, K=K,
                         max_iter=max_iter, tol=tol,
                         backward_mode='exact')
    return (X @ w - X @ w_o).pow(2).sum()


def _make_weighter(log_c, log_alpha, dtype=torch.float64):
    """New-parameterization factory: c = exp(log_c), alpha = exp(log_alpha)."""
    c_init = float(_math.exp(log_c))
    alpha_init = float(_math.exp(log_alpha))
    w = LearnedRobustWeighter(c_init=c_init, alpha_init=alpha_init)
    return w.to(dtype)


def fd_gradcheck_prod(d=16, N=512, K=8, delta=1e-2, sigma=0.01,
                      p_burst=0.02, kappa=20.0, seed=0, tol=1e-4,
                      max_iter=50, solve_tol=1e-10, device='cpu'):
    """FD-vs-exact gradient check at production scale (float64).

    Returns a dict with the exact/FD gradients and best relative errors for
    ``log_c`` and ``log_alpha``.  Raises AssertionError if either best
    relative error exceeds ``tol``.  The eps sweep is
    ``{3e-2, 1e-2, 3e-3, 1e-3, 3e-4}`` (Phase B4: tuned in the new
    log-space coordinates).
    """
    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)
    w_o = torch.randn(d, generator=g, dtype=torch.float64, device=device)
    w_o = w_o / w_o.norm()
    X, d_obs = make_block(w_o, N, sigma, mode='iid',
                          noise='impulsive', p_burst=p_burst, kappa=kappa,
                          seed=seed, dtype=torch.float64, device=device)

    base_log_c = _BASE_LOG_C
    base_log_alpha = _BASE_LOG_ALPHA

    # ---- Analytic: exact implicit backward (log-space coords) ----
    wgt = _make_weighter(base_log_c, base_log_alpha).to(device)
    loss = _forward_loss(wgt, X, d_obs, w_o, delta, K, max_iter, solve_tol)
    loss.backward()
    g_c_exact = wgt.log_c.grad.item()
    g_a_exact = wgt.log_alpha.grad.item()

    # ---- Numerical: central finite differences (forward-only, log-space) ----
    def _loss_at(log_c, log_alpha):
        wgt_ = _make_weighter(log_c, log_alpha).to(device)
        return _forward_loss(wgt_, X, d_obs, w_o, delta, K,
                             max_iter, solve_tol).item()

    eps_grid = (3e-2, 1e-2, 3e-3, 1e-3, 3e-4)
    best_c, best_a = float('inf'), float('inf')
    best_eps_c = best_eps_a = None
    best_fd_c = best_fd_a = None
    for eps in eps_grid:
        g_c_fd = (_loss_at(base_log_c + eps, base_log_alpha)
                  - _loss_at(base_log_c - eps, base_log_alpha)) / (2 * eps)
        g_a_fd = (_loss_at(base_log_c, base_log_alpha + eps)
                  - _loss_at(base_log_c, base_log_alpha - eps)) / (2 * eps)
        rel_c = abs(g_c_fd - g_c_exact) / max(abs(g_c_exact), 1e-12)
        rel_a = abs(g_a_fd - g_a_exact) / max(abs(g_a_exact), 1e-12)
        if rel_c < best_c:
            best_c, best_eps_c, best_fd_c = rel_c, eps, g_c_fd
        if rel_a < best_a:
            best_a, best_eps_a, best_fd_a = rel_a, eps, g_a_fd

    res = {
        'parameterization': 'log_exp_v1',
        'd': d, 'N': N, 'K': K, 'delta': delta, 'sigma': sigma,
        'p_burst': p_burst, 'kappa': kappa, 'seed': seed,
        'precision': 'float64',
        'solver': {'max_iter': max_iter, 'tol': solve_tol},
        'loss': float(loss.item()),
        'weighter': {'log_c': base_log_c, 'log_alpha': base_log_alpha,
                     'c': float(_math.exp(base_log_c)),
                     'alpha': float(_math.exp(base_log_alpha))},
        'log_c': {'exact_grad': g_c_exact, 'best_fd_grad': best_fd_c,
                  'best_eps': best_eps_c, 'best_rel_err': best_c},
        'log_alpha': {'exact_grad': g_a_exact, 'best_fd_grad': best_fd_a,
                      'best_eps': best_eps_a, 'best_rel_err': best_a},
    }

    print(f"=== Production-scale FD gradcheck (d={d}, N={N}, K={K}, "
          f"float64, {device}, parameterization=log_exp_v1) ===")
    print(f"  loss={res['loss']:.6e}")
    print(f"  weighter at op point: c={res['weighter']['c']:.6f}, "
          f"alpha={res['weighter']['alpha']:.6f}")
    print(f"  log_c:     exact={g_c_exact:.6e}  best FD rel err="
          f"{best_c:.3e} @ eps={best_eps_c}")
    print(f"  log_alpha: exact={g_a_exact:.6e}  best FD rel err="
          f"{best_a:.3e} @ eps={best_eps_a}")
    ok_c = best_c < tol
    ok_a = best_a < tol
    print(f"  log_c     {'PASS' if ok_c else 'FAIL'} (tol={tol})")
    print(f"  log_alpha {'PASS' if ok_a else 'FAIL'} (tol={tol})")
    assert ok_c, f"log_c FD-vs-exact rel err {best_c:.3e} >= {tol}"
    assert ok_a, f"log_alpha FD-vs-exact rel err {best_a:.3e} >= {tol}"
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
    p.add_argument('--out', type=str, default=None,
                   help='Optional JSON file to dump the result dict.')
    args = p.parse_args()

    device = _resolve_device(args.gpu, args.gpu_id)
    print(f"Device: {device}")
    t0 = time.time()
    res = fd_gradcheck_prod(d=args.d, N=args.N, K=args.K, delta=args.delta,
                            sigma=args.sigma, p_burst=args.p_burst,
                            kappa=args.kappa, seed=args.seed, tol=args.tol,
                            max_iter=args.solver_max_iter,
                            solve_tol=args.solver_tol, device=device)
    elapsed = time.time() - t0
    res['elapsed_s'] = elapsed
    print(f"\nProduction-scale FD gradcheck PASSED ({elapsed:.1f}s).")

    if args.out is not None:
        # Strip inf/nan-prone fields and serialize.
        with open(args.out, 'w') as f:
            json.dump(res, f, indent=2)
        print(f"Wrote {args.out}")


if __name__ == '__main__':
    main()
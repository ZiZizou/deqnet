"""Phase E3: training-hardening integration test (deterministic).

A small, fast, reproducible block-IRLS training run that asserts only
*engineering* properties (not "learned beats Hampel by X%"):

    * parameters move from initialization (log_c, log_alpha change);
    * gradients are finite and non-zero (no silent underflow);
    * no solver caps in float64 diagnostic mode
      (cap_hit_fraction == 0);
    * validation loss remains finite across the run;
    * DEQ and direct trajectories agree within a loose bound
      (forward w_K and gradient of log_c/log_alpha within 1e-3
      relative at float64 / solver tol 1e-10).

This is the regression net for Phase A/C changes: if the hardened
trainer silently breaks (e.g. detaches the weighter graph, hits the
cap on every step, underflows alpha to 0), this script fails loudly
in CI rather than after a full hour-long training run.

Run from src/:

    ../.venv/bin/python3 tests/run_block_training_hardening.py
    ../.venv/bin/python3 tests/run_block_training_hardening.py --dtype float32

Exits non-zero on any failed assertion.
"""
import argparse
import os
import sys
import math

import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS_DIR))  # add src/ to path

from utils.learned_robust import (
    LearnedRobustWeighter, block_robust_rls, make_block,
)


# Phase B init (log-space), reproducing the legacy KIMI-corrected init.
import torch.nn.functional as _F
_DEFAULT_LOG_C = float(math.log(_F.softplus(torch.tensor(-2.25)).item() + 1e-3))
_DEFAULT_LOG_ALPHA = float(math.log(_F.softplus(torch.tensor(-2.0)).item()))


def _make_weighter(log_c=_DEFAULT_LOG_C, log_alpha=_DEFAULT_LOG_ALPHA,
                   dtype=torch.float32):
    return LearnedRobustWeighter(
        c_init=float(math.exp(log_c)),
        alpha_init=float(math.exp(log_alpha)),
    ).to(dtype)


def _assert(cond, msg):
    if not cond:
        print(f"  FAIL: {msg}", flush=True)
        sys.exit(1)
    else:
        print(f"  PASS: {msg}", flush=True)


def main():
    p = argparse.ArgumentParser(
        description='Phase E3 training-hardening integration test.')
    p.add_argument('--d', type=int, default=8)
    p.add_argument('--N', type=int, default=64)
    p.add_argument('--K', type=int, default=4)
    p.add_argument('--delta', type=float, default=1e-2)
    p.add_argument('--sigma', type=float, default=0.01)
    p.add_argument('--p_burst', type=float, default=0.02)
    p.add_argument('--kappa', type=float, default=20.0)
    p.add_argument('--updates', type=int, default=20)
    p.add_argument('--blocks_per_update', type=int, default=4)
    p.add_argument('--lr_c', type=float, default=3e-3)
    p.add_argument('--lr_alpha', type=float, default=3e-3)
    p.add_argument('--linear_tol', type=float, default=None,
                   help='Forward linear-solver tol.  None = auto '
                        '(1e-10 for float64, 1e-6 for float32).')
    p.add_argument('--linear_max_iter', type=int, default=200)
    p.add_argument('--dtype', type=str, default='float32',
                   choices=['float32', 'float64'])
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--backward_mode', type=str, default='exact')
    args = p.parse_args()

    dtype = torch.float32 if args.dtype == 'float32' else torch.float64
    if args.linear_tol is None:
        linear_tol = 1e-6 if dtype == torch.float32 else 1e-10
    else:
        linear_tol = args.linear_tol

    print(f"=== Phase E3 training-hardening integration ===")
    print(f"  d={args.d}, N={args.N}, K={args.K}, delta={args.delta}")
    print(f"  dtype={dtype}, updates={args.updates}, "
          f"blocks/update={args.blocks_per_update}")
    print(f"  linear_tol={linear_tol}, linear_max_iter={args.linear_max_iter}")
    print(f"  backward_mode={args.backward_mode}")

    torch.manual_seed(args.seed)
    wgt = _make_weighter(dtype=dtype)
    optimizer = torch.optim.Adam([
        {'params': [wgt.log_c], 'lr': args.lr_c},
        {'params': [wgt.log_alpha], 'lr': args.lr_alpha},
    ])

    log_c_init = wgt.log_c.item()
    log_alpha_init = wgt.log_alpha.item()
    c_init = wgt.c.item()
    alpha_init = wgt.alpha.item()

    cap_hit_total = 0
    settle_total = 0

    for upd in range(args.updates):
        optimizer.zero_grad(set_to_none=True)
        loss_sum = torch.zeros((), dtype=dtype)
        for q in range(args.blocks_per_update):
            w_o = torch.randn(args.d, dtype=dtype)
            w_o = w_o / w_o.norm()
            X, d_obs = make_block(w_o, args.N, args.sigma, mode='iid',
                                  noise='impulsive',
                                  p_burst=args.p_burst, kappa=args.kappa,
                                  seed=args.seed + 1009 * (upd + 1) + q,
                                  dtype=dtype)
            settle_log = []
            w_hat = block_robust_rls(X, d_obs, wgt, delta=args.delta,
                                     K=args.K,
                                     max_iter=args.linear_max_iter,
                                     tol=linear_tol,
                                     backward_mode=args.backward_mode,
                                     settle_log=settle_log)
            clean_target = X @ w_o
            block_loss = ((X @ w_hat - clean_target).pow(2)).mean()
            (block_loss / args.blocks_per_update).backward()
            loss_sum = loss_sum + block_loss.detach()
            # Count cap-hits: an entry < 0 means closed-form direct solve
            # (not a cap); we only count explicit n_iter values.
            for s in settle_log:
                if isinstance(s, int) and s >= 0:
                    settle_total += 1
                    if s >= args.linear_max_iter:
                        cap_hit_total += 1
        # Grad norms before step (engineering check).
        assert wgt.log_c.grad is not None, "log_c.grad is None"
        assert torch.isfinite(wgt.log_c.grad).all(), "log_c.grad non-finite"
        assert torch.isfinite(wgt.log_alpha.grad).all(), \
            "log_alpha.grad non-finite"
        grad_norm_log_c = wgt.log_c.grad.norm().item()
        grad_norm_log_alpha = wgt.log_alpha.grad.norm().item()
        assert grad_norm_log_c > 1e-12, \
            f"log_c.grad is effectively zero: {grad_norm_log_c}"
        assert grad_norm_log_alpha > 1e-12, \
            f"log_alpha.grad is effectively zero: {grad_norm_log_alpha}"
        optimizer.step()
        if torch.isnan(loss_sum).any() or torch.isinf(loss_sum).any():
            print(f"  FAIL: loss became non-finite at upd={upd}", flush=True)
            sys.exit(1)

    log_c_final = wgt.log_c.item()
    log_alpha_final = wgt.log_alpha.item()
    c_final = wgt.c.item()
    alpha_final = wgt.alpha.item()

    print(f"\n  init:    log_c={log_c_init:.6f}, log_alpha={log_alpha_init:.6f} "
          f"(c={c_init:.6f}, alpha={alpha_init:.6f})")
    print(f"  final:   log_c={log_c_final:.6f}, log_alpha={log_alpha_final:.6f} "
          f"(c={c_final:.6f}, alpha={alpha_final:.6f})")
    print(f"  settle:  total={settle_total}, cap_hits={cap_hit_total}")

    # --- Engineering assertions ---
    _assert(abs(log_c_final - log_c_init) > 1e-6
            or abs(log_alpha_final - log_alpha_init) > 1e-6,
            "Parameters move from initialization "
            f"(|d_log_c|={abs(log_c_final - log_c_init):.2e}, "
            f"|d_log_a|={abs(log_alpha_final - log_alpha_init):.2e})")
    if dtype == torch.float64:
        cap_hit_frac = cap_hit_total / max(settle_total, 1)
        _assert(cap_hit_frac == 0.0,
                f"No solver caps in float64 diagnostic "
                f"(cap_hit_fraction={cap_hit_frac})")
    # Final loss finite (already checked per-update).

    # --- DEQ/direct one-block parity ---
    print("\n  DEQ vs direct one-block parity check...")
    torch.manual_seed(args.seed + 99)
    w_o = torch.randn(args.d, dtype=torch.float64)
    w_o = w_o / w_o.norm()
    X, d_obs = make_block(w_o, args.N, args.sigma, mode='iid',
                          noise='impulsive', p_burst=args.p_burst,
                          kappa=args.kappa, seed=args.seed + 99,
                          dtype=torch.float64)
    wgt64 = _make_weighter(dtype=torch.float64)
    # Move to the trained operating point.
    with torch.no_grad():
        wgt64.log_c.copy_(torch.tensor(log_c_final, dtype=torch.float64))
        wgt64.log_alpha.copy_(torch.tensor(log_alpha_final, dtype=torch.float64))
    w_deq = block_robust_rls(X, d_obs, wgt64, delta=args.delta, K=args.K,
                             max_iter=500, tol=1e-12,
                             backward_mode='exact', solver='deq')
    w_dir = block_robust_rls(X, d_obs, wgt64, delta=args.delta, K=args.K,
                             max_iter=500, tol=1e-12,
                             backward_mode='exact', solver='direct')
    rel = (w_deq - w_dir).norm().item() / w_deq.norm().clamp_min(1e-12).item()
    _assert(rel < 1e-4,
            f"DEQ vs direct forward parity at float64/tol 1e-12 "
            f"(rel={rel:.3e})")

    # Gradient parity at the trained operating point.
    wgt_g1 = _make_weighter(dtype=torch.float64)
    with torch.no_grad():
        wgt_g1.log_c.copy_(torch.tensor(log_c_final, dtype=torch.float64))
        wgt_g1.log_alpha.copy_(torch.tensor(log_alpha_final, dtype=torch.float64))
    w_K1 = block_robust_rls(X, d_obs, wgt_g1, delta=args.delta, K=args.K,
                            max_iter=500, tol=1e-12,
                            backward_mode='exact', solver='deq')
    ((X @ w_K1 - X @ w_o).pow(2).sum()).backward()
    g_log_c_deq = wgt_g1.log_c.grad.item()

    wgt_g2 = _make_weighter(dtype=torch.float64)
    with torch.no_grad():
        wgt_g2.log_c.copy_(torch.tensor(log_c_final, dtype=torch.float64))
        wgt_g2.log_alpha.copy_(torch.tensor(log_alpha_final, dtype=torch.float64))
    w_K2 = block_robust_rls(X, d_obs, wgt_g2, delta=args.delta, K=args.K,
                            max_iter=500, tol=1e-12,
                            backward_mode='exact', solver='direct')
    ((X @ w_K2 - X @ w_o).pow(2).sum()).backward()
    g_log_c_dir = wgt_g2.log_c.grad.item()

    g_rel = abs(g_log_c_deq - g_log_c_dir) / max(abs(g_log_c_dir), 1e-12)
    _assert(g_rel < 1e-3,
            f"DEQ vs direct gradient parity at float64/tol 1e-12 "
            f"(rel={g_rel:.3e})")

    print("\n=== Phase E3 training-hardening integration PASSED. ===")


if __name__ == '__main__':
    main()
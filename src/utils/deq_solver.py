import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Tuple, Dict, Any
import warnings


class ConvergenceWarning(UserWarning):
    pass


warnings.simplefilter('always', ConvergenceWarning)


def fixed_point(
    f: Callable[[torch.Tensor], torch.Tensor],
    v0: torch.Tensor,
    beta: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 200,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Plain fixed-point iteration v <- v + beta * f(v) on a strongly monotone f.

    Solves f(v) = 0.  v0 and the returned v have the same shape.  f must be
    a callable mapping a tensor of that shape to the same shape.

    Convergence requires beta in (0, 2 / lambda_max(J)) where J = D f.  For
    f(v) = -M v + b with M symmetric positive definite, this corresponds to
    a forward-Euler discretization of dv/dt = f(v) with step size beta.
    """
    v = v0.detach().clone()
    info = {'n_iter': 0, 'final_residual': float('inf'), 'converged': False}
    last_good = v.detach().clone()
    for k in range(max_iter):
        with torch.no_grad():
            fv = f(v)
            res = fv.abs().max().item()
        if not torch.isfinite(fv).all():
            info['n_iter'] = k
            info['final_residual'] = res
            warnings.warn(f"fixed_point: NaN/Inf at iter {k}; returning best iterate", ConvergenceWarning)
            return last_good, info
        g = v + beta * fv
        if torch.isfinite(g).all():
            last_good = g.detach().clone()
        info['n_iter'] = k + 1
        info['final_residual'] = res
        if res < tol:
            info['converged'] = True
            return g, info
        v = g
    warnings.warn(f"fixed_point: did not converge in {max_iter} iters, final residual = {res:.3e}", ConvergenceWarning)
    return last_good, info


def anderson(
    f: Callable[[torch.Tensor], torch.Tensor],
    v0: torch.Tensor,
    m: int = 5,
    beta: float = 1.0,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Anderson acceleration (type-I) on the fixed-point map g(v) = v + beta * f(v).

    Solves f(v) = 0.  At each step we have m past iterates V_0..V_{m-1}
    (oldest..newest) and residuals F_0..F_{m-1}.  Find alpha_0..alpha_{m-1}
    with sum=1 minimizing  ||sum alpha_i F_i||.  Standard reformulation:

        alpha_0 = 1 - sum_{i>=1} alpha_i
        D_F[:, i] = F_{i+1} - F_0,    D_V[:, i] = V_{i+1} - V_0
        solve D_F @ x = -F_0   (least squares)
        v_new = V_0 + D_V @ x

    Then drop oldest entry and repeat.  Returns (v_star, info_dict).

    Convergence requires beta in (0, 2 / lambda_max(J)) where J = D f.  For
    f(v) = -M v + b with M SPD, this matches a forward-Euler discretization
    of dv/dt = f(v) with step size beta.  The fixed-point map g(v) = (I -
    beta M) v + beta b has eigenvalues 1 - beta * lambda_i(M), all in (-1, 1)
    for beta < 2 / lambda_max(M).
    """
    v = v0.detach().clone()
    shape = v.shape
    flat_dim = v.numel()

    V_buf = torch.zeros(m, flat_dim, dtype=v.dtype, device=v.device)
    F_buf = torch.zeros(m, flat_dim, dtype=v.dtype, device=v.device)
    head = 0
    nstored = 0
    last_good = v.detach().clone().reshape(-1)
    info = {'n_iter': 0, 'final_residual': float('inf'), 'converged': False}

    for k in range(max_iter):
        with torch.no_grad():
            fv = f(v)
            fv_flat = fv.detach().reshape(-1)
            if not torch.isfinite(fv_flat).all():
                warnings.warn(
                    f"anderson: NaN/Inf at iter {k}; returning best iterate",
                    ConvergenceWarning,
                )
                info['n_iter'] = k
                info['final_residual'] = float('nan')
                return last_good.view(shape), info
            res = fv_flat.abs().max().item()

        g = v + beta * fv
        g_flat = g.detach().reshape(-1)
        if torch.isfinite(g_flat).all():
            last_good = g_flat.clone()

        info['n_iter'] = k + 1
        info['final_residual'] = res
        if res < tol:
            info['converged'] = True
            return g, info

        V_buf[head] = g_flat
        F_buf[head] = fv_flat
        head = (head + 1) % m
        nstored = min(nstored + 1, m)

        if nstored < 2:
            v = g
            continue

        idxs = [(head - nstored + i) % m for i in range(nstored)]
        Vs = V_buf[idxs]
        Fs = F_buf[idxs]
        F0 = Fs[0]
        V0 = Vs[0]
        D_F = (Fs[1:] - F0).t()
        D_V = (Vs[1:] - V0).t()

        try:
            x, *_ = torch.linalg.lstsq(D_F, -F0)
            x = x.flatten()
            if x.numel() < nstored - 1:
                pad = torch.zeros(nstored - 1 - x.numel(),
                                  dtype=x.dtype, device=x.device)
                x = torch.cat([x, pad], dim=0)
            elif x.numel() > nstored - 1:
                x = x[:nstored - 1]
            v_new_flat = V0 + D_V @ x
        except Exception:
            v_new_flat = g_flat

        if torch.isfinite(v_new_flat).all():
            v = v_new_flat.view(shape)
        else:
            v = g.detach().clone()

    warnings.warn(
        f"anderson: did not converge in {max_iter} iters, final residual = {res:.3e}",
        ConvergenceWarning,
    )
    return last_good.view(shape), info


def solve_jacobian_transpose(
    func: Callable[[torch.Tensor], torch.Tensor],
    v_star: torch.Tensor,
    rhs: torch.Tensor,
    tol: float = 1e-6,
    max_iter: int = 50,
    beta: float = 1.0,
) -> torch.Tensor:
    """Solve J^T y = rhs where J = D_func(v_star), matrix-free.

    Strategy:
      * Small problems (n <= 64): construct J^T explicitly via forward-mode
        finite differences (one column per unit vector in R^n), then solve
        via torch.linalg.lstsq.  Robust and fast for n in the dozens.
      * Larger problems: conjugate gradient on the normal equation with
        matrix-free matvec.  Bails to a fixed-point iteration if CG
        detects non-positive-definiteness.

    beta is the fixed-point step size used in the fallback iteration
    y <- y + beta * (J^T y - rhs).  For J = -M with M SPD (the circuit case),
    contractive if beta in (0, 2 / lambda_max(M)).
    """
    n_flat = rhs.numel()
    z0 = v_star.detach()
    n_v = z0.numel()
    device = z0.device
    dtype = z0.dtype

    if n_flat <= 64:
        # Compute J explicitly via torch.autograd.functional.jacobian.
        z = z0.detach().clone().requires_grad_(True)
        J = torch.autograd.functional.jacobian(func, z, vectorize=False)
        J_flat = J.reshape(n_flat, n_flat)
        rhs_flat = rhs.detach().reshape(-1)
        try:
            sol = torch.linalg.solve(J_flat.t(), rhs_flat)
        except RuntimeError:
            sol = torch.linalg.lstsq(J_flat.t(), rhs_flat).solution
        y_flat = torch.atleast_1d(sol).reshape(-1)
        if y_flat.numel() < n_flat:
            pad = torch.zeros(n_flat - y_flat.numel(), device=device, dtype=dtype)
            y_flat = torch.cat([y_flat, pad], dim=0)
        if y_flat.numel() > n_flat:
            y_flat = y_flat[:n_flat]
        return y_flat.view_as(rhs)

    # General case: matrix-free CG.
    rhs_flat = rhs.detach().reshape(-1)
    target_norm = rhs_flat.abs().max().item()

    def At(y_flat):
        y_view = y_flat.view_as(v_star)
        with torch.enable_grad():
            z = v_star.detach().clone().requires_grad_(True)
            fv = func(z)
        g = torch.autograd.grad(
            outputs=fv, inputs=z,
            grad_outputs=y_view, retain_graph=False, allow_unused=True,
        )[0]
        if g is None:
            return torch.zeros_like(y_flat)
        return g.reshape(-1).detach()

    # Circuit Jacobian J = df/dv = -(B^T D B + Gamma) is symmetric NSD.
    # CG requires the system matrix to be SPD, so we solve M y = -rhs
    # with M = -J = B^T D B + Gamma, which is SPD.  The resulting y
    # equals J^{-1} rhs (same answer, convergent solver).
    M_times = lambda y_flat: -At(y_flat)

    y = torch.zeros_like(rhs_flat)
    r = -rhs_flat - M_times(y)
    p = r.clone()
    rs_old = (r * r).sum().item()
    initial_resid = (r * r).sum().item() ** 0.5
    cg_failed = False

    for k in range(max_iter):
        Mp = M_times(p)
        pMp = (p * Mp).sum().item()
        if abs(pMp) <= 1e-14:
            return y.view_as(rhs)  # CG numerically done
        if pMp < 0:
            cg_failed = True
            break
        alpha = rs_old / pMp
        y = y + alpha * p
        r = r - alpha * Mp
        rs_new = (r * r).sum().item()
        if rs_new ** 0.5 < tol * max(1.0, target_norm):
            return y.view_as(rhs)
        if rs_new > initial_resid * 1e6:
            cg_failed = True
            break
        p = r + (rs_new / (rs_old + 1e-30)) * p
        rs_old = rs_new

    if not cg_failed:
        warnings.warn(
            f"solve_jacobian_transpose: CG hit max_iter, residual sqrt = {rs_new ** 0.5:.3e}",
            ConvergenceWarning,
        )
        return y.view_as(rhs)

    # Fallback: fixed-point iteration
    last_good = y.clone()
    res = float('inf')
    for k in range(max_iter):
        with torch.no_grad():
            Jty = At(y)
            r = Jty - rhs_flat
            res = r.abs().max().item()
        if not torch.isfinite(r).all():
            return last_good.view_as(rhs)
        y = y + beta * r
        if torch.isfinite(y).all():
            last_good = y.clone()
        if res < tol * max(1.0, target_norm):
            return y.view_as(rhs)
    warnings.warn(
        f"solve_jacobian_transpose: did not converge in {max_iter} iters, final res = {res:.3e}",
        ConvergenceWarning,
    )
    return last_good.view_as(rhs)


def estimate_lipschitz(
    f: Callable[[torch.Tensor], torch.Tensor],
    v0: torch.Tensor,
    n_steps: int = 20,
    eps: float = 1e-3,
) -> float:
    """Rough estimate of operator Lipschitz constant by probing with random directions."""
    v = v0.detach().clone()
    with torch.no_grad():
        fv = f(v)
    estimate = 0.0
    rng = torch.Generator(device=v.device)
    rng.manual_seed(0)
    for _ in range(n_steps):
        d = torch.randn(v.shape, generator=rng, device=v.device, dtype=v.dtype)
        d = d / (d.norm() + 1e-12)
        with torch.no_grad():
            fv_d = f(v + eps * d)
            diff = (fv_d - fv).reshape(-1)
            d_flat = d.reshape(-1)
            num = (diff * d_flat).sum().abs()
            denom = (d_flat * d_flat).sum()
            if denom.item() > 0:
                ratio = (num / (denom.sqrt() + 1e-12)).item()
                estimate = max(estimate, ratio / eps)
    return max(estimate, 1e-6)


def check_contraction(
    src_indices: torch.Tensor,
    des_indices: torch.Tensor,
    num_nodes: int,
    worst_case_D: torch.Tensor,
    gamma_diag: torch.Tensor,
    n_power_iter: int = 30,
    eps: float = 1e-6,
) -> Dict[str, float]:
    """Analytic certificate on J = -M where M = B^T D B + Gamma is SPD.

    Reports the spectrum of the worst-case J (over input-independent slope
    bounds).  The useful quantity is the contraction rate
    ``lambda_min_M`` (= min eigenvalue of M = -lambda_max_J): it predicts
    solver iteration count and the backward CG condition number.  When
    devices saturate into dead zones, ``lambda_min_M`` -> ``gamma_floor``
    and iteration counts blow up — that is the early-warning signal for
    the late-training collapse described in §"known failure modes".

    Note on the legacy ``passive`` flag: with ``gamma = softplus(raw) +
    gamma_floor`` and ``gamma_floor > 0``, ``J`` is NSD by construction,
    so ``passive=True`` unconditionally — it certifies nothing the
    parameterization didn't already guarantee.  The flag is retained for
    backward compatibility but ``lambda_min_M`` is the meaningful number.

    For small n (<= 128), the spectrum is computed exactly via
    ``torch.linalg.eigvalsh``.  For larger n, power iteration on -J
    approximates ``lambda_max_J`` (closest-to-zero eigenvalue).
    """
    n = num_nodes
    device = src_indices.device
    E = src_indices.shape[0]
    B = torch.zeros(E, n + 1, dtype=worst_case_D.dtype, device=device)
    arange_e = torch.arange(E, device=device)
    B[arange_e, src_indices] = -1.0
    B[arange_e, des_indices] = 1.0
    Bn = B[:, 1:]

    def matvec_M(v):
        """v -> M v = B^T D B v + Gamma v  (SPD)."""
        return Bn.t() @ (worst_case_D * (Bn @ v)) + gamma_diag * v

    def matvec_J(v):
        """v -> J v = -M v."""
        return -matvec_M(v)

    if n <= 128:
        # Build M explicitly by applying matvec_M to each basis vector.
        M_mat = torch.zeros((n, n), dtype=worst_case_D.dtype, device=device)
        for i in range(n):
            ei = torch.zeros(n, dtype=worst_case_D.dtype, device=device)
            ei[i] = 1.0
            M_mat[:, i] = matvec_M(ei)
        eigs_M = torch.linalg.eigvalsh(M_mat)
        lam_min_M = eigs_M.min().item()
        lam_max_M = eigs_M.max().item()
        lam_max_J = (-lam_min_M)
    else:
        # Power iteration on M (SPD) to get lam_max_M, and on -J=M to
        # get lam_min_M via inverse power iteration.  For n>128 this is
        # approximate; users needing exact values should switch to a
        # smaller topology or use an explicit eigensolver.
        v = torch.randn(n, dtype=worst_case_D.dtype, device=device)
        v = v / (v.norm() + 1e-12)
        prev_lam = 0.0
        for _ in range(n_power_iter):
            Mv = matvec_M(v)
            lam = (v * Mv).sum().item()
            if abs(lam - prev_lam) < eps * max(1.0, abs(prev_lam)):
                break
            prev_lam = lam
            v = Mv / (Mv.norm() + 1e-12)
        lam_max_M = (v * matvec_M(v)).sum().item()
        # Approximate lam_min_M via inverse power iteration on M (CG
        # on M @ x = v each step).
        lam_min_M = float('nan')
        v = torch.randn(n, dtype=worst_case_D.dtype, device=device)
        v = v / (v.norm() + 1e-12)
        try:
            y = v.clone()
            r = matvec_M(y) - v
            p = r.clone()
            rs_old = (r * r).sum().item()
            for _ in range(n_power_iter):
                Mp = matvec_M(p)
                pMp = (p * Mp).sum().item()
                if pMp <= 0:
                    break
                alpha = rs_old / pMp
                y = y + alpha * p
                r = r - alpha * Mp
                rs_new = (r * r).sum().item()
                if rs_new ** 0.5 < eps * max(1.0, v.norm().item()):
                    break
                p = r + (rs_new / (rs_old + 1e-30)) * p
                rs_old = rs_new
            # Rayleigh quotient on M at y approximates 1 / lam_min_M.
            ray = (y * matvec_M(y)).sum().item() / max((y * y).sum().item(), 1e-30)
            lam_min_M = 1.0 / max(ray, 1e-30)
        except Exception:
            pass
        lam_max_J = -lam_min_M

    return {
        'lambda_max_J': lam_max_J,
        'lambda_min_M': lam_min_M,
        'lambda_max_M': lam_max_M,
        'contraction_margin': -lam_max_J,
        'passive': lam_max_J < 0.0,
    }

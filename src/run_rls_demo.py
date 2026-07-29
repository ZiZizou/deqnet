"""Streaming RLS adaptive-filtering demo.

System identification: recover an unknown plant w_o in R^d from a stream of
noisy linear observations  d_t = w_o^T x_t + nu_t, with four contenders:

  1. digital_lms   : w <- w + mu * e * x                       (1st-order, digital)
  2. digital_rls   : standard RLS recursion with forgetting lambda
  3. fabric_lms    : continuous-time gradient flow dw/dt = mu * e * x,
                     integrated by forward Euler at fine dt          (1st-order, fabric)
  4. fabric_rls    : per sample:
                       R <- lambda * R + outer(x, x)              # write phase (digital, v1)
                       p <- lambda * p + d_t * x                  # observation, NOT error
                       w = LinearSolveLayer(p, R)                 # settle phase (the fabric)

RLS recursion note: the exponentially-weighted least-squares solution is
  w_t = R_t^{-1} p_t
where  R_t = sum_{k<=t} lambda^{t-k} x_k x_k^T  and  p_t = sum_{k<=t} lambda^{t-k} d_k x_k.
Note the *observation* d_k enters p, not the error e_k.  The README's earlier
p <- decay*p + e*x sketch was wrong; this is the correct form.

Metrics:
  - Ensemble-mean ||w_t - w_o||^2 vs t (log scale), all 4 contenders on one plot
  - Discrepancy: ||w_fabric - w_digital_RLS|| / ||w_digital_RLS|| vs t (should be 1e-4..1e-6)
  - Settling budget: iterations per sample for LinearSolveLayer
  - Tracking: time-varying w_o with process noise, steady-state misadjustment vs lambda

Also produces the Ljung overlay: fabric-LMS (ODE dw/dt = mu*e*x) and digital
LMS at decreasing step sizes mu*dt converge onto the continuous trajectory.

Results are saved to ./results/rls_demo/.
"""
import os
import sys
import argparse
import json
import numpy as np
import torch

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from utils.circuit_block import LinearSolveLayer


# ----------------------------------------------------------------------------
# 1. Contenders
# ----------------------------------------------------------------------------

class DigitalLMS:
    """w <- w + mu * e * x, scalar mu."""
    def __init__(self, d, mu=0.05, w_init=None):
        self.d = d
        self.mu = mu
        self.w = torch.zeros(d) if w_init is None else w_init.clone()

    def step(self, x_t, d_t):
        e = d_t - self.w @ x_t
        self.w = self.w + self.mu * e * x_t
        return self.w.clone()


class DigitalRLS:
    """Standard RLS with forgetting factor lambda.  Maintains P = R^{-1}
    for O(d^2) updates instead of O(d^3) explicit solve per step."""
    def __init__(self, d, lam=0.99, delta=1.0, w_init=None):
        self.d = d
        self.lam = lam
        self.w = torch.zeros(d) if w_init is None else w_init.clone()
        self.P = (1.0 / delta) * torch.eye(d)

    def step(self, x_t, d_t):
        # Standard RLS: k = P x / (lam + x^T P x); w += k * e; P = (P - k x^T P) / lam
        Px = self.P @ x_t
        denom = self.lam + x_t @ Px
        k = Px / denom
        e = d_t - self.w @ x_t
        self.w = self.w + k * e
        self.P = (self.P - torch.outer(k, Px)) / self.lam
        return self.w.clone()


class FabricLMS:
    """Continuous-time LMS: dw/dt = mu * e(t) * x(t).

    Integrated by forward Euler at fine dt.  When mu*dt -> 0, this converges
    onto the same trajectory as DigitalLMS with the same mu*dt — the Ljung
    experiment demonstrates this directly."""
    def __init__(self, d, mu=0.05, dt=0.1, w_init=None):
        self.d = d
        self.mu = mu
        self.dt = dt
        self.w = torch.zeros(d) if w_init is None else w_init.clone()

    def step(self, x_t, d_t):
        # Hold x and d constant over the inner Euler substep.
        e = d_t - self.w @ x_t
        self.w = self.w + self.dt * self.mu * e * x_t
        return self.w.clone()


class FabricRLS:
    """Per-sample:
         R <- lam * R + outer(x, x)
         p <- lam * p + d_t * x
         w = LinearSolveLayer(p, R)            # analog settle

    beta for the fixed-point map is chosen adaptively as
    2 / (lambda_min(R) + lambda_max(R)) (Chebyshev step) so that the
    solver converges in O(1) Anderson iterations regardless of how large
    R has grown through accumulation.  R is symmetric PSD so we use the
    eigvalsh branch.
    """
    def __init__(self, d, lam=0.99, R0=None, w_init=None,
                 max_iter=100, tol=1e-8, beta='chebyshev'):
        self.d = d
        self.lam = lam
        self.R = R0 if R0 is not None else torch.eye(d)
        self.w = torch.zeros(d) if w_init is None else w_init.clone()
        self.p = torch.zeros(d)
        self.max_iter = max_iter
        self.tol = tol
        self.beta_mode = beta
        self.last_iters = []

    def _chebyshev_beta(self, R):
        eigs = torch.linalg.eigvalsh(R)
        lam_min = eigs[0].item()
        lam_max = eigs[-1].item()
        if lam_max <= 0:
            return 1.0
        return float(2.0 / (lam_min + lam_max))

    def step(self, x_t, d_t):
        self.R = self.lam * self.R + torch.outer(x_t, x_t)
        self.p = self.lam * self.p + d_t * x_t
        if self.beta_mode == 'chebyshev':
            beta = self._chebyshev_beta(self.R)
        else:
            beta = float(self.beta_mode)
        layer = LinearSolveLayer(max_iter=self.max_iter, tol=self.tol, beta=beta)
        # Solve w* = R^{-1} p via analog settle.
        w_star = layer(self.p.unsqueeze(0), self.R).squeeze(0)
        # Track iterations from the last EquilibriumSolve.
        from utils.circuit_block import EquilibriumSolve
        if EquilibriumSolve.last_info is not None:
            self.last_iters.append(EquilibriumSolve.last_info.get('n_iter', -1))
        self.w = w_star
        return self.w.clone()


# ----------------------------------------------------------------------------
# 2. Stream generators
# ----------------------------------------------------------------------------

def make_stream(w_o, T, sigma, mode='iid', seed=0, dtype=torch.float64):
    """Generate (x_t, d_t) stream of length T.

    mode='iid':   x_t ~ N(0, I)
    mode='ar1':   x_t = rho * x_{t-1} + sqrt(1-rho^2) * eps_t, eps_t ~ N(0, I)
    """
    g = torch.Generator().manual_seed(seed)
    d = w_o.shape[0]
    x = torch.zeros(T, d, dtype=dtype)
    if mode == 'iid':
        x = torch.randn(T, d, generator=g, dtype=dtype)
    elif mode == 'ar1':
        rho = 0.9
        eps = torch.randn(T, d, generator=g, dtype=dtype)
        x[0] = eps[0]
        for t in range(1, T):
            x[t] = rho * x[t - 1] + np.sqrt(1 - rho ** 2) * eps[t]
    else:
        raise ValueError(f"unknown mode: {mode}")
    nu = sigma * torch.randn(T, generator=g, dtype=dtype)
    d_obs = x @ w_o + nu
    return x, d_obs


def make_tracking_stream(w_o0, T, sigma, process_std, seed=0, dtype=torch.float64):
    """Stream with time-varying plant: w_o(t+1) = w_o(t) + q_t, q_t ~ N(0, process_std^2 * I)."""
    g = torch.Generator().manual_seed(seed)
    d = w_o0.shape[0]
    x = torch.randn(T, d, generator=g, dtype=dtype)
    w_o = torch.zeros(T, d, dtype=dtype)
    w_o[0] = w_o0
    for t in range(1, T):
        w_o[t] = w_o[t - 1] + process_std * torch.randn(d, generator=g, dtype=dtype)
    nu = sigma * torch.randn(T, generator=g, dtype=dtype)
    d_obs = (x * w_o).sum(dim=-1) + nu
    return x, d_obs, w_o


# ----------------------------------------------------------------------------
# 3. Single-trial runner
# ----------------------------------------------------------------------------

def run_trial(contender, x, d_obs, w_o):
    """Returns (W[T,d]) and (extra_info, dict).  `contender` is an
    already-constructed contender instance.  `w_o` may be shape (d,) for
    static plant or (T, d) for tracking."""
    T = x.shape[0]
    d = x.shape[1]
    W = torch.zeros(T, d, dtype=x.dtype)
    for t in range(T):
        W[t] = contender.step(x[t], d_obs[t])
    extras = {}
    if hasattr(contender, 'last_iters'):
        extras['last_iters'] = list(contender.last_iters)
    return W, extras


# ----------------------------------------------------------------------------
# 4. Monte Carlo
# ----------------------------------------------------------------------------

def monte_carlo(contender_factory, w_o, T, sigma, n_trials,
                 mode='iid', seed_base=0, dtype=torch.float64):
    """`contender_factory` is a zero-arg callable returning a fresh contender
    instance.  Returns (W_mean[T,d], W_runs[n_trials, T, d], per-trial extras list)."""
    d = w_o.shape[0]
    W_runs = torch.zeros(n_trials, T, d, dtype=dtype)
    extras_all = []
    for trial in range(n_trials):
        x, d_obs = make_stream(w_o, T, sigma, mode=mode,
                               seed=seed_base + trial, dtype=dtype)
        contender = contender_factory()
        W, extras = run_trial(contender, x, d_obs, w_o)
        W_runs[trial] = W
        extras_all.append(extras)
    W_mean = W_runs.mean(dim=0)
    return W_mean, W_runs, extras_all


# ----------------------------------------------------------------------------
# 5. Metrics
# ----------------------------------------------------------------------------

def w_err_sq(W, w_o):
    """Squared error ||w_t - w_o||^2 over time, shape (T,)."""
    diff = W - w_o.unsqueeze(0)
    return (diff * diff).sum(dim=-1)


def relative_discrepancy(W_a, W_b):
    """||w_a - w_b|| / ||w_b|| over time, shape (T,).  Uses norm of b
    as denominator so it's meaningful even when w_b -> 0."""
    denom = W_b.norm(dim=-1).clamp_min(1e-12)
    return (W_a - W_b).norm(dim=-1) / denom


def steady_state_misadjustment(W, w_o_track, tail_frac=0.5):
    """Mean ||w_t - w_o(t)||^2 over the last `tail_frac` of the stream.
    Used for the tracking experiment."""
    T = W.shape[0]
    tail = int(T * tail_frac)
    err = w_err_sq(W[tail:], w_o_track[tail:])
    return err.mean().item()


# ----------------------------------------------------------------------------
# 6. Plotting
# ----------------------------------------------------------------------------

def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def plot_learning_curves(out_dir, results, w_o, suptitle):
    """results: dict[name -> (W_mean, W_runs)].  Plots ensemble-mean
    ||w_t - w_o||^2 vs t on log scale."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    T = next(iter(results.values()))[0].shape[0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = {'digital_lms': 'C0', 'digital_rls': 'C1',
              'fabric_lms': 'C2', 'fabric_rls': 'C3'}
    styles = {'digital_lms': '-', 'digital_rls': '-',
              'fabric_lms': '--', 'fabric_rls': '--'}
    for name, (W_mean, _) in results.items():
        err = w_err_sq(W_mean, w_o).cpu().numpy()
        ax.semilogy(np.arange(T), err,
                    label=name, color=colors.get(name, 'k'),
                    linestyle=styles.get(name, '-'), linewidth=1.5)
    ax.set_xlabel('sample t')
    ax.set_ylabel(r'$\|w_t - w_o\|^2$')
    ax.set_title(suptitle)
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'learning_curves.png'), dpi=140)
    plt.close(fig)


def plot_discrepancy(out_dir, W_fabric_rls_mean, W_digital_rls_mean):
    """Discrepancy ||w_fabric - w_digital|| / ||w_digital|| vs t."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    T = W_fabric_rls_mean.shape[0]
    disc = relative_discrepancy(W_fabric_rls_mean, W_digital_rls_mean).cpu().numpy()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(np.arange(T), disc, color='C3', linewidth=1.5)
    ax.set_xlabel('sample t')
    ax.set_ylabel(r'$\|w_\mathrm{fabric} - w_\mathrm{digital\,RLS}\| / \|w_\mathrm{digital\,RLS}\|$')
    ax.set_title('Fabric-RLS vs Digital-RLS discrepancy')
    ax.grid(True, which='both', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'discrepancy.png'), dpi=140)
    plt.close(fig)


def plot_settling_budget(out_dir, iters_list, label):
    """Histogram of iterations per sample for LinearSolveLayer."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    iters = np.array(iters_list)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(iters, bins=np.arange(iters.min(), iters.max() + 2) - 0.5,
            color='C3', edgecolor='black', alpha=0.7)
    ax.set_xlabel('iterations per sample')
    ax.set_ylabel('count')
    ax.set_title(f'{label}  (mean={iters.mean():.2f}, max={iters.max()}, median={np.median(iters):.0f})')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'settling_budget.png'), dpi=140)
    plt.close(fig)


def plot_misadjust_vs_lambda(out_dir, lambdas, misadj_rls, misadj_fabric):
    """Steady-state misadjustment vs lambda for digital RLS and fabric RLS."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(lambdas, misadj_rls, 'o-', color='C1', label='digital RLS')
    ax.plot(lambdas, misadj_fabric, 's--', color='C3', label='fabric RLS')
    ax.set_xlabel(r'forgetting factor $\lambda$')
    ax.set_ylabel('steady-state misadjustment')
    ax.set_title('Tracking experiment: misadjustment vs $\\lambda$')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'misadjust_vs_lambda.png'), dpi=140)
    plt.close(fig)


def plot_ljung_overlay(out_dir, w_o, x, d_obs, mu_values, dt_values):
    """Three panels, one per (mu*dt) value: fabric-LMS (continuous flow at
    fine dt) vs digital-LMS (discrete at coarse dt) trajectories.

    For each panel we also overlay a "continuous reference" path: digital
    LMS at very small step size (proxy for the ODE flow)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = len(mu_values)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
    T = x.shape[0]
    # Continuous reference: digital LMS at very small dt (proxy for ODE flow).
    mu_ref = 0.05
    dt_ref = 0.001
    n_substeps = max(1, int(round(1.0 / dt_ref)))  # 1000 substeps per "sample"
    w_ref = torch.zeros_like(w_o)
    err_ref = np.zeros(T)
    for t in range(T):
        for _ in range(n_substeps):
            e = d_obs[t] - w_ref @ x[t]
            w_ref = w_ref + dt_ref * mu_ref * e * x[t]
        err_ref[t] = ((w_ref - w_o) ** 2).sum().item()
    for i, (mu, dt) in enumerate(zip(mu_values, dt_values)):
        ax = axes[i]
        # Digital LMS with step mu*dt at the sample rate.
        w_d = torch.zeros_like(w_o)
        err_d = np.zeros(T)
        for t in range(T):
            e = d_obs[t] - w_d @ x[t]
            w_d = w_d + mu * dt * e * x[t]
            err_d[t] = ((w_d - w_o) ** 2).sum().item()
        # Fabric LMS (continuous-time ODE): integrate by forward Euler at fine sub-dt.
        n_sub = max(1, int(round(dt / 0.001)))
        dt_inner = dt / n_sub
        w_f = torch.zeros_like(w_o)
        err_f = np.zeros(T)
        for t in range(T):
            for _ in range(n_sub):
                e = d_obs[t] - w_f @ x[t]
                w_f = w_f + dt_inner * mu * e * x[t]
            err_f[t] = ((w_f - w_o) ** 2).sum().item()
        ax.semilogy(np.arange(T), err_d, '-', color='C0', label=f'digital LMS ($\\mu\\Delta t$={mu*dt:.3f})')
        ax.semilogy(np.arange(T), err_f, '--', color='C2', label=f'fabric LMS (ODE, dt={dt})')
        ax.semilogy(np.arange(T), err_ref, ':', color='gray', label='reference ($\\Delta t$=0.001)')
        ax.set_xlabel('sample t')
        if i == 0:
            ax.set_ylabel(r'$\|w_t - w_o\|^2$')
        ax.set_title(f'$\\mu$={mu}, $\\Delta t$={dt}')
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=0.3)
    fig.suptitle('Ljung: digital LMS -> continuous LMS as $\\mu\\Delta t \\to 0$')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'ljung_overlay.png'), dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 7. Main experiment flow
# ----------------------------------------------------------------------------

def run_experiment(args):
    out_dir = _ensure_dir(args.out_dir)
    torch.set_default_dtype(torch.float64)
    dtype = torch.float64

    # ---- Common plant ----
    g = torch.Generator().manual_seed(args.seed)
    w_o = torch.randn(args.d, generator=g, dtype=dtype)
    w_o = w_o / w_o.norm()  # unit-norm plant
    print(f"True plant ||w_o|| = {w_o.norm().item():.4f}, d = {args.d}")

    # ---- 4 contenders ----
    def digital_lms_factory():
        return DigitalLMS(d=args.d, mu=args.mu_lms)
    def digital_rls_factory():
        return DigitalRLS(d=args.d, lam=args.lam_rls)
    def fabric_lms_factory():
        return FabricLMS(d=args.d, mu=args.mu_lms, dt=args.dt_fabric_lms)
    def fabric_rls_factory():
        return FabricRLS(d=args.d, lam=args.lam_rls, R0=torch.eye(args.d),
                         max_iter=args.linear_max_iter, tol=args.linear_tol,
                         beta=args.linear_beta)

    contenders = {
        'digital_lms': digital_lms_factory,
        'digital_rls': digital_rls_factory,
        'fabric_lms': fabric_lms_factory,
        'fabric_rls': fabric_rls_factory,
    }

    # ---- Part 1: iid ----
    print(f"\n=== iid input, T={args.T}, sigma={args.sigma}, "
          f"n_trials={args.n_trials} ===")
    results_iid = {}
    extras_iid = {}
    for name, factory in contenders.items():
        print(f"  running {name}...")
        W_mean, _, extras = monte_carlo(factory, w_o, args.T, args.sigma,
                                       args.n_trials, mode='iid',
                                       seed_base=args.seed, dtype=dtype)
        results_iid[name] = (W_mean, None)
        extras_iid[name] = extras
    plot_learning_curves(out_dir, results_iid, w_o,
                          f'iid: ensemble-mean $\\|w-w_o\\|^2$ ({args.n_trials} trials)')

    # Discrepancy plot
    W_fabric_rls_mean = results_iid['fabric_rls'][0]
    W_digital_rls_mean = results_iid['digital_rls'][0]
    plot_discrepancy(out_dir, W_fabric_rls_mean, W_digital_rls_mean)

    # Settling budget: collect iterations from fabric_rls
    iters = []
    for extras in extras_iid['fabric_rls']:
        iters.extend(extras.get('last_iters', []))
    plot_settling_budget(out_dir, iters, label='LinearSolveLayer iterations per sample')
    if iters:
        print(f"  settling budget: mean={np.mean(iters):.2f}, max={max(iters)}, "
              f"median={np.median(iters):.0f}, n={len(iters)}")

    # ---- Part 2: AR(1) correlated ----
    print(f"\n=== AR(1) input (rho=0.9), T={args.T} ===")
    results_ar1 = {}
    for name, factory in contenders.items():
        print(f"  running {name}...")
        W_mean, _, _ = monte_carlo(factory, w_o, args.T, args.sigma,
                                  args.n_trials, mode='ar1',
                                  seed_base=args.seed, dtype=dtype)
        results_ar1[name] = (W_mean, None)
    plot_learning_curves(out_dir, results_ar1, w_o,
                          f'AR(1) $\\rho$=0.9: ensemble-mean $\\|w-w_o\\|^2$')

    # ---- Part 3: Tracking experiment ----
    print(f"\n=== Tracking experiment: time-varying w_o, sweep lambda ===")
    lambdas = args.tracking_lambdas
    misadj_rls, misadj_fabric = [], []
    for lam in lambdas:
        def rls_factory():
            return DigitalRLS(d=args.d, lam=lam)
        def frls_factory():
            return FabricRLS(d=args.d, lam=lam, R0=torch.eye(args.d),
                             max_iter=args.linear_max_iter, tol=args.linear_tol,
                             beta=args.linear_beta)
        # Single trial per lambda (averaging in tracking would average over
        # the time-varying trajectory, which we want).
        x, d_obs, w_o_track = make_tracking_stream(
            w_o, args.T, args.sigma, args.process_std,
            seed=args.seed, dtype=dtype)
        W_rls, _ = run_trial(rls_factory(), x, d_obs, w_o_track)
        W_frls, _ = run_trial(frls_factory(), x, d_obs, w_o_track)
        misadj_rls.append(steady_state_misadjustment(W_rls, w_o_track))
        misadj_fabric.append(steady_state_misadjustment(W_frls, w_o_track))
        print(f"  lambda={lam}: digital RLS misadj={misadj_rls[-1]:.4e}, "
              f"fabric RLS misadj={misadj_fabric[-1]:.4e}")
    plot_misadjust_vs_lambda(out_dir, lambdas, misadj_rls, misadj_fabric)

    # ---- Part 4: Ljung overlay ----
    print(f"\n=== Ljung overlay: digital LMS -> continuous LMS as mu*dt -> 0 ===")
    x_lj, d_lj = make_stream(w_o, args.T, args.sigma, mode='iid',
                              seed=args.seed, dtype=dtype)
    mu_values = [0.05, 0.05, 0.05]
    dt_values = [1.0, 0.5, 0.1]  # decreasing step sizes
    plot_ljung_overlay(out_dir, w_o, x_lj, d_lj, mu_values, dt_values)

    # ---- Save metrics ----
    metrics = {
        'd': args.d,
        'T': args.T,
        'sigma': args.sigma,
        'n_trials': args.n_trials,
        'settling': {'mean': float(np.mean(iters)) if iters else None,
                     'max': int(max(iters)) if iters else None,
                     'median': float(np.median(iters)) if iters else None,
                     'n': len(iters)},
        'tracking': {'lambdas': lambdas,
                     'digital_rls_misadj': [float(x) for x in misadj_rls],
                     'fabric_rls_misadj': [float(x) for x in misadj_fabric]},
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nWrote metrics to {out_dir}/metrics.json")
    print(f"Figures in {out_dir}/")


def parse_args():
    parser = argparse.ArgumentParser(description='Streaming RLS adaptive-filtering demo')
    parser.add_argument('--out_dir', type=str, default='./results/rls_demo')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--d', type=int, default=8)
    parser.add_argument('--T', type=int, default=500)
    parser.add_argument('--sigma', type=float, default=0.01)
    parser.add_argument('--n_trials', type=int, default=50)
    parser.add_argument('--mu_lms', type=float, default=0.05)
    parser.add_argument('--lam_rls', type=float, default=0.99)
    parser.add_argument('--dt_fabric_lms', type=float, default=0.1)
    parser.add_argument('--linear_max_iter', type=int, default=100)
    parser.add_argument('--linear_tol', type=float, default=1e-8)
    parser.add_argument('--linear_beta', type=str, default='chebyshev',
                        help="Beta for LinearSolveLayer: 'chebyshev' for auto "
                             "Chebyshev step, or a float for fixed step.")
    parser.add_argument('--process_std', type=float, default=1e-3)
    parser.add_argument('--tracking_lambdas', type=float, nargs='+',
                        default=[0.90, 0.95, 0.98, 0.99, 0.995, 0.999])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_experiment(args)

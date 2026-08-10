# Implementation Plan: Learned IR-RLS + Learned-Prox ISTA

Status: agreed plan (2026-08-07). Storage location decided: split into `utils/` + `tests/` with
training/eval wiring in `run_rls_demo.py`. Framing and wireline generator resolve to the
`kirchhoffnet_deq_kimi_convo_summary.md` context.

Progress: **Phase 0 gate PASSED** (2026-08-07). `src/tests/test_learned_robust.py` lives, all 4 tests
green on `ssr0`, including the weighter-grad simulation that reproduces the Phase 1 loop structure. It
exposed and fixed one real bug in `EquilibriumSolve.backward` (finding E) and one divergence from the
plan's `gradcheck_solve` spec (see 0a). Next: Phase 1.

Two critiques before the plan, because they change the design:

1. **`LearnedProxDevice` is not a prox and breaks its own contraction claim.** `gain ∈ (0, max_gain=10)`
   makes the map Lipschitz-10, so "bounded derivative guarantees contraction" is false. The correct
   construction is the **resolvent form**: parameterize a monotone penalty gradient `ψ_θ` and define the
   fixed point as `w = z − η·ψ_θ(w)`, which *is* `prox_{ηg}(z)` with `g' = ψ`. Contraction then reduces
   to strong monotonicity of `η(HᵀH + diag(ψ'(w)))`, the non-vacuous margin.
2. **Robust weighting is vacuous under Gaussian noise.** With the current `make_stream`, the Bayes-optimal
   weight is `v ≡ 1` and training converges to "do nothing". Impulsive/burst noise mode is required or
   the robustness experiment has no signal.

## Findings that diverge from the plan (from the codebase audit)

### A. `DigitalRobustRLS` recursion is off by one power of `v_t`
The fabric accumulates `R ← λR + v_t·xxᵀ`, `p ← λp + v_t·dx`. The plan sketch
`w += v_t·k·e` and `P = (P − v_t·outer(k,Px))/λ` is wrong (extra `v_t` / `v_t²`). Matrix-inversion-lemma
derivation with `A = λR_{t-1}`, identity `(A+vxxᵀ)⁻¹vx = A⁻¹x·v/(1+v·xᵀA⁻¹x)` gives the correct recursion:

```python
Px = self.P @ x_t
k = Px / (self.lam / v_t + x_t @ Px)      # == v_t·Px/(λ + v_t·xᵀPx)  — plan's k is correct
e = d_t - self.w @ x_t
self.w = self.w + k * e                    # no v_t factor
self.P = (self.P - torch.outer(k, Px)) / self.lam   # no v_t factor
```

Identical to `DigitalRLS` except `λ → λ/v_t` in the gain denominator. `v_t=1` ⇒ byte-identical to
`DigitalRLS`; `v_t→0` ⇒ no update (sample ignored, `P→P/λ`), matching the fabric exactly.

### B. `FabricLearnedISTA.rhs` computes the fixed-point map, not the residual
`EquilibriumSolve` iterates `v ← v + β·f(v)` and stops when `‖f(v)‖ < tol`; `rhs_fn` must be the
**residual** (zero at equilibrium, symmetric NSD Jacobian) — that's what `LinearSolveLayer.rhs(w,p,R)=p−Rw`
does. The plan's `rhs(v,u) = u + v − η(HᵀH v + ψ(v))` is the map `G(v)`; feeding it to `EquilibriumSolve`
solves `G(v)=0`, not `G(v*)=v*`. The layer must pass the residual:

```python
def residual(self, v, u):   # f(v) = G(v) - v = 0  at equilibrium
    eta = self.eta
    return u - eta * (v @ self.HtH) - eta * self.penalty.psi(v)
```

`J_f = −η(HᵀH + diag(ψ′))` is symmetric NSD — matches the existing CG machinery and the plan's own margin
formula (§2.3). The map form is only used for init `v0 = u` and the analytical-ISTA reference.

### C. Repo assumptions that don't hold
- `generate_wireline_block`/"benchmark doc" does not exist in the repo, but the recipe exists in
  `kirchhoffnet_deq_kimi_convo_summary.md` (PAM-4, `conv1d` ISI, SNR-scaled noise). Implement per that.
- Hardware is CPU-only (`ssr0` has no CUDA). Precision policy: train float32 CPU, parity/gradcheck
  float64 CPU.

### D. Honest-claim adjustments
- `MonotonePenalty` is smooth shrinkage (quadratic tail for `|v| ≫ thr`), not hard soft-threshold. Gate 3
  must compare against **hard soft-threshold ISTA with λ tuned** (small grid); any init deviation is
  reported, not masked.
- PAM-4 symbols are dense, so the L1 prior is a mismatch training is *supposed* to correct. Frame the
  trained device as a **learned monotone equalizer**, not a sparse decoder.
- Truncated BPTT detaches `R, p` but not `w`; `e_prior` next window threads the old graph through `w`.
  Detach `w` too at the truncation point.

### E. `EquilibriumSolve.backward` must `retain_graph=True` when u and the closure share graph nodes
Phase 0's weighter simulation exposed a latent crash. `EquilibriumSolve.backward` ran
`f_.backward(gradient=y, retain_graph=False)` then returned `u_grad` for the outer engine to traverse
`u`'s graph. When `p` and `R` are BOTH built from the same weighter output `v_t = σ(raw_c)` (the real
`FabricRobustRLS.step`), the inner backward frees the shared node `v_t`'s saved tensors and the outer
traversal of `p`'s graph re-enters it → `RuntimeError: Trying to backward through the graph a second
time`. Phase 0's original independent-`p` test missed it; the real loop structure crashes. Fix
(`circuit_block.py:680`): `retain_graph=True` — shared nodes survive until the outer backward finishes,
both paths accumulate correctly (verified rel err 3e-6 vs a closed-form reference). Also removed the
dead `v_star.detach().requires_grad_(True)` trick in `forward`: the engine strips the output when no
explicit Function input (v0/u) requires grad, so the line was a silent no-op.

## ISTA framing (decided, per summary doc)
`w` = equalizer taps (d = 16), `X` = sliding-window Toeplitz regressor built from the received signal,
`s` = known PAM-4 symbols. Equilibrium solves `min_w ‖Xw − s‖² + Φ(w)`. Readout `ŝ = X·w*`. The channel
enters only through generating `rx`. Soft-threshold L1 is the epoch-0 ablation baseline only.

---

## Phase 0 — Prerequisite gates (no model code) — **DONE**

**0a. Gradient-plumbing gate. — DONE, with divergence from `gradcheck_solve`.**

The plan's `gradcheck_solve` used `torch.autograd.gradcheck` on `(p, R)`. That API does not work here:
`R` is closure-captured into `EquilibriumSolve`'s `rhs_fn`, not a Function input, so `autograd.grad`
sees no `R` in the graph (analytical 0, numerical nonzero → spurious Jacobian mismatch). The
`.backward() + .grad` mechanism DOES populate `R.grad` via `f_.backward(−y)` and is what the training
loop actually uses, so the gate is a finite-difference comparison of that gradient.

`src/tests/test_learned_robust.py` (4 tests, all green on `ssr0`):

1. `test_linear_solve_layer_gradients_flow` — `p.grad` and closure-captured `R.grad` both finite/nonzero.
2. `test_gradcheck_via_finite_diff` — `.backward()` gradient vs central finite differences, max rel err ~1e-7.
3. `test_gradcheck_batch` — same for batched `(B, d)` state.
4. `test_weighter_grad_flow_simulation` — the real `FabricRobustRLS.step` graph structure (`p` and `R`
   both from `v_t = σ(raw_c)`); asserts `raw_c.grad` finite/nonzero and matches a closed-form reference
   (`w = solve(R, p)` re-expressed in the autograd graph) to rel err 0.00e+00. Required the
   `retain_graph=True` fix (finding E).

Run: `ssh ssr0 "python /home/annaik/Documents/deqnet/src/tests/test_learned_robust.py"`. Existing 40
tests in `tests/run_all.py` still pass unchanged.

**0b. Precision policy.** Document in `run_rls_demo.py` header + training entrypoints: train float32 CPU,
parity/gradcheck float64 CPU.

## Phase 1 — Learned Robust IR-RLS

New **`src/utils/learned_robust.py`**:
1. `LearnedRobustWeighter(nn.Module)` — scalar `raw_c=2.0`, `raw_alpha=-2.0`, `v_max=1.0` fixed (not
   learned). Init ⇒ `v≈1` near `e≈0`, redescends for `|e| ≫ c`. Generalized Cauchy influence.
2. `FabricRobustRLS(FabricRLS)` — per-sample `e_prior = d_t − wᵀx_t`, `v_t = weighter(e_prior)`,
   `R = λR + v_t·outer(x_t,x_t)`, `p = λp + v_t·d_t·x_t`, solve via `LinearSolveLayer`.
   `_chebyshev_beta` wrapped in `torch.no_grad()` (R carries a graph during training).
3. `DigitalRobustRLS(DigitalRLS)` — corrected recursion per finding A.

Extend **`src/run_rls_demo.py`**:
- `make_stream(..., noise='gaussian', p_burst=0.02, kappa=20.0)` + `_make_noise` (contaminated
  Gaussian: rare large bursts). Backward-compatible defaults.
- `monte_carlo(..., noise=..., p_burst=..., kappa=...)` passthrough.
- `train_robust_weighter(...)`: plant-per-epoch randomization, T=128, prequential MSE +
  `10·‖w−w_o‖²` terminal term, truncation every 32 **detaching R, p, and w** (finding D).
- `run_robust_experiment(...)`: contenders `{digital_rls, fabric_rls, digital_robust_rls,
  fabric_robust_rls}` under `noise='impulsive'`, frozen trained weighter; plots: robust learning curves,
  fabric↔digital-robust discrepancy, settle-iter histogram, learned `v(e)` influence curve.
- New argparse: `--noise --p_burst --kappa --train_robust --robust_epochs --robust_T --truncate
  --robust_lr`.

Tests (gates):
1. `test_gradcheck_solve` — gradcheck on `LinearSolveLayer(p, R)`.
2. `test_frozen_weighter_matches_plain_rls` — frozen-init `FabricRobustRLS` ≡ `FabricRLS` on identical
   seeds to solver tolerance.
3. `test_digital_robust_matches_fabric_robust` — discrepancy `<1e-6`; `v≡1` ⇒ byte-identical to
   `DigitalRLS`.
4. `test_implicit_vs_unrolled_grad_robust` — relative grad error `<1e-4` on d=8, ~20 Anderson iters.
5. `test_impulsive_noise_burst_rate` — `_make_noise` statistics.
6. `test_gaussian_control_flat_weighter` — Gaussian-noise control ⇒ `α→0`, `v≈const`, no improvement.

## Phase 1 — TODO & design decisions (locked 2026-08-07)

Build order: noise plumbing → `learned_robust.py` → training/eval wiring → gates 2–6 → verify.
**Do not touch `run_all.py` registration (Phase 3).**

### Design decisions (resolved with user; these override ambiguous plan text)

1. **Weighter parameterization.** `LearnedRobustWeighter`:
   `raw_c=2.0`, `raw_alpha=-2.0` are **learned** `nn.Parameter`s (the plan phrase "fixed (not
   learned)" applies only to `v_max=1.0`, a fixed buffer clipped to (0,1]). Forward:
   `v(e) = v_max / (1 + (e/c)^2)^alpha`, `c = softplus(raw_c) + 1e-3`, `alpha = softplus(raw_alpha)`.
   Init ⇒ `v(0)=1`, `v≈1` near `e≈0`, redescends for `|e|≫c`; `alpha→0` ⇒ `v→v_max` flat
   (this is the Gaussian-control signal for gate 6). `v_t∈(0,1]` preserves SPD of R by
   construction. This is the generalized-Cauchy family with the exponent chosen so the
   identity limit is `alpha→0` (NOT the canonical `(1+(e/c)^2)^{p/2-1}` form, whose `p→0`
   limit redescends and would break gate 6).
2. **Gate 2 resolution.** "frozen-init ≡ plain RLS to solver tolerance" is unachievable as
   written: init `alpha=0.127` ⇒ `v(e)≈1` only for `e≈0`; early-stream errors are `O(1)` so
   `v` dips to ~0.93–0.97 and trajectories genuinely diverge a few percent. Split it:
   (a) `v≡1` `FabricRobustRLS` matches `FabricRLS` **byte-exact** on identical seeds
   (the strong invariant "v_t=1 ⇒ identical", encodes finding A);
   (b) weighter-init unit check: `v(0)=1`, `v≈1` for `|e|≤0.1`, redescension toward 0.
3. **`DigitalRobustRLS` recursion (finding A, verified vs Woodbury).** The user's inline
   snippet (`w += v_t*k*e`; `P = (P − v_t*outer(k,Px))/λ` with `k = Px/(λ/v_t + xᵀPx)`)
   double-counts `v_t`: `k` already contains one factor (`1/(λ/v_t+·) = v_t/(λ+v_t·)`).
   Correct form (byte-identical to `DigitalRLS` at `v_t=1`, `v_t→0` ⇒ sample ignored):

   ```python
   Px = self.P @ x_t
   k  = Px / (self.lam / v_t + x_t @ Px)      # == v_t·Px/(λ + v_t·xᵀPx)
   e  = d_t - self.w @ x_t
   self.w = self.w + k * e                    # no v_t factor
   self.P = (self.P - torch.outer(k, Px)) / self.lam   # no v_t factor
   ```

4. **Truncated BPTT detaches R, p, AND w** (finding D). The user's pseudocode detached only
   `R,p`; the next window's `e_prior = d − wᵀx` would thread the old graph through `w`.
5. **Eval detaches `v_t`.** `FabricRobustRLS`/`DigitalRobustRLS` get a `training` flag
   (default False): keep the graph when training, `v_t.detach()` otherwise — otherwise the
   frozen-weighter Monte-Carlo accumulates a T-step autograd graph through R/p.
6. **Contaminated-Gaussian noise (user-confirmed form).** `_make_noise`:
   `gaussian` ⇒ `σ·randn` byte-identical to current `make_stream`;
   `impulsive` ⇒ `σ·(randn + κ·burst·randn)`, `burst = (rand < p_burst)`, `p_burst=0.02`,
   `κ=20.0`. Backward-compatible defaults everywhere.
7. **Training protocol (user-confirmed).** Per-epoch randomized plant `w_o` (unit norm),
   `T=128`, loss = prequential `Σ_t (d_{t+1} − w_tᵀx_{t+1})²` + `10·‖w_T − w_o‖²`, Adam.
   Truncation every 32 (decision 4). Each `FabricRobustRLS.step` is one `LinearSolveLayer`
   implicit solve; p (the `u` input) requires grad via `v_t`, so `w_star` carries the graph
   and the `retain_graph=True` fix (finding E) handles the p/R shared-`v_t` node.
8. **Precision policy (0b).** Train float32 CPU (solver tol `1e-5`, `max_iter≈50`);
   parity/gradcheck gates float64 CPU (solver tol `1e-8`). Document in `run_rls_demo.py`
   header + training entrypoint.
9. **Circular-import handling.** `learned_robust.py` imports `FabricRLS`/`DigitalRLS` from
   `run_rls_demo` at module top; `run_rls_demo` imports `learned_robust` *inside*
   `train_robust_weighter`/`run_robust_experiment`.
10. **Eval dtype.** Trained weighter is float32; `run_robust_experiment` runs float64
    (demo convention) and casts the frozen weighter with `.double()`.
11. **`_chebyshev_beta` under `torch.no_grad()`** (R carries a graph during training).

### TODO

- [ ] `run_rls_demo.py`: `_make_noise` + `noise/p_burst/kappa` args on `make_stream`,
      `monte_carlo` passthrough (decision 6).
- [ ] `src/utils/learned_robust.py`: `LearnedRobustWeighter` (decision 1).
- [ ] `src/utils/learned_robust.py`: `FabricRobustRLS(FabricRLS)` — `training` flag,
      `no_grad` chebyshev beta, graph-carrying solve (decisions 5, 11).
- [ ] `src/utils/learned_robust.py`: `DigitalRobustRLS(DigitalRLS)` (decision 3).
- [ ] `run_rls_demo.py`: `train_robust_weighter` — float32, plant-per-epoch, T=128,
      prequential + terminal loss, truncate-32 detaching R,p,w, Adam (decisions 4, 7, 8).
- [ ] `run_rls_demo.py`: `run_robust_experiment` — frozen weighter, 4 contenders under
      `noise='impulsive'`; plots: learning curves, fabric↔digital-robust discrepancy,
      settle-iter histogram, learned `v(e)`/`ψ(e)=v(e)·e` curve (init vs trained overlay);
      results → `results/rls_demo/robust/`.
- [ ] `run_rls_demo.py`: argparse `--noise --p_burst --kappa --train_robust
      --robust_epochs --robust_T --truncate --robust_lr`; header precision-policy note.
- [ ] `tests/test_learned_robust.py`: gate 2 (identity byte-exact + init unit check).
- [ ] `tests/test_learned_robust.py`: gate 3 (digital-robust vs fabric-robust `<1e-6`;
      `v≡1` byte-identical to `DigitalRLS`).
- [ ] `tests/test_learned_robust.py`: gate 4 (implicit vs unrolled grad, d=8, ~20–30
      Anderson iters, rel err `<1e-4`).
- [ ] `tests/test_learned_robust.py`: gate 5 (`_make_noise` statistics; gaussian ≡ old).
- [ ] `tests/test_learned_robust.py`: gate 6 (Gaussian control ⇒ `alpha` decreases,
      `v≈const`, no improvement).
- [ ] Verify: `python tests/test_learned_robust.py`;
      `python run_rls_demo.py --noise impulsive --train_robust`;
      `python tests/run_all.py` (existing 40 unchanged).

### Risks (report honestly, do not tune away)

- float32 implicit backward precision — mitigated by training solver tol `1e-5`, grad
  finiteness checked.
- Weighter may not beat plain RLS under impulsive noise — that is a reported result, not a
  bug to silence (gate 6 is the control that keeps this honest).

## Phase 2 — Learned-Prox ISTA (equalizer framing)

New **`src/utils/learned_ista.py`**:
1. `MonotonePenalty(nn.Module)` — per-coordinate `raw_gain`, `threshold`;
   `psi = g·(softplus(v−thr) − softplus(−v−thr))`, `slope = g·(sigmoid(v−thr)+sigmoid(−v−thr)) ≥ 0`.
2. `FabricLearnedISTA(nn.Module)` — solves for equalizer taps:
   - `u = η·(Xᵀ s)`, `eta = 2·sigmoid(raw_c)/λ_max(XᵀX)` (c ∈ (0,2)).
   - passes `lambda w: u − η·(w @ XᵀX) − η·penalty.psi(w)` to `EquilibriumSolve` (**residual
     convention**, finding B). `X` per-call (data-dependent, no fixed buffer).
   - `contraction_margin = eigvalsh(η(XᵀX + diag(slope(w*))))[0]`, logged per settle, asserted
     `> δ_min = 1e-4`; optional disclosed ridge if margin collapses. `M = N−d+1 ≥ d` ⇒ rank-d `XᵀX`,
     margin primarily from the regressor.
3. `generate_wireline_block(batch, block_len=512, snr_db, channel_taps)` + `build_toeplitz(rx, d)` —
   per summary doc recipe: PAM-4 `∈{±1,±3}`, `conv1d` ISI, noise scaled to `signal_pwr/10^(snr/10)`.
   Default channel: 16-tap, 25 dB-loss response.

Extend **`src/run_rls_demo.py`**:
- `train_learned_ista(...)`: fixed 16-tap channel, SNR curriculum uniform(10–25) dB, loss = MSE +
  PAM-4 cross-entropy on `ŝ = X·w*`, margin + `n_iter` logged per settle.
- `run_ista_experiment(...)`: BER-vs-SNR (analytical hard soft-threshold ISTA with λ-tuned grid,
  learned-at-init, trained), margin-vs-epoch, settle-iters-vs-tol.
- New argparse: `--train_ista --ista_epochs --snr_lo --snr_hi --wireline_N --wireline_taps --ista_lr
  --ridge`.

Tests (gates):
1. `test_penalty_monotone` — `slope ≥ 0`, bounded.
2. `test_ista_residual_matches_direct` — equilibrium ≈ direct regularized solve at init.
3. `test_ista_init_matches_analytical_ista` — BER at init ≈ λ-tuned hard soft-threshold ISTA
   (statistical; deviations reported, not masked).
4. `test_implicit_vs_unrolled_grad_ista` — relative grad error `<1e-4` on d=8, ~20 Anderson iters.
5. `test_margin_positive` — margin `> 0` for well-posed `X`; margin-collapse case reported as a result.
6. `test_wireline_generator_shapes` — `generate_wireline_block` / `build_toeplitz` shapes + BER sanity.

## Phase 3 — Consolidation

- Register both new test modules in `src/tests/run_all.py`.
- Full acceptance on `ssr0`:
  - `ssh ssr0 "cd ~/Documents/deqnet/src && python tests/run_all.py"` (existing 40 + new).
  - `ssh ssr0 "cd ~/Documents/deqnet/src && python run_rls_demo.py --noise impulsive --train_robust"`
  - `ssh ssr0 "cd ~/Documents/deqnet/src && python run_rls_demo.py --train_ista"`
- Save figures + JSON to `results/rls_demo/robust/` and `results/rls_demo/ista/`.
- Update `.opencode_memory.md` with new modules and the corrected recursions.

**Order:** Phase 0 → 1 → 2 → 3. Phase 0 is done; the predicted gradcheck risk materialized as an API
incompatibility (not a bug) plus one real bug in `EquilibriumSolve.backward` (finding E), both resolved.
Phase 1 is next.

## Test checklist (gates, not suggestions)

1. Gradient-plumbing gate passes (Phase 0a; finite-difference form of `gradcheck_solve`, see 0a).
2. Weighter frozen at init (`v≈1`): `FabricRobustRLS` matches `FabricRLS` trajectory to solver tolerance
   on identical seeds.
3. Penalty at init: learned-ISTA BER statistically indistinguishable from λ-tuned analytical
   soft-threshold ISTA.
4. Implicit vs. unrolled backward: relative gradient error `<1e-4` on d=8, ~20 Anderson iterations,
   for both models.
5. Margin assertion `λ_min > 0` never fires during training; if it does, that's a result, not a bug to
   silence.
6. Gaussian-noise control run: trained weighter learns `v ≈ const` (flat curve).

## File map

```
src/
├── utils/
│   ├── learned_robust.py       # LearnedRobustWeighter, FabricRobustRLS, DigitalRobustRLS (new)
│   └── learned_ista.py         # MonotonePenalty, FabricLearnedISTA,
│                               #   generate_wireline_block, build_toeplitz (new)
├── tests/
│   ├── test_learned_robust.py  # Phase 0 gate: grad flow + FD gradcheck + weighter sim (DONE; gates 1,2,4,6
│   │                           #   + impulsive-noise stats come later in Phase 1)
│   ├── test_learned_ista.py    # gates 3,4,5 + generator shapes (new)
│   └── run_all.py              # register the two new modules
└── run_rls_demo.py             # _make_noise/make_stream noise arg, train_robust_weighter,
                                #   run_robust_experiment, train_learned_ista, run_ista_experiment,
                                #   argparse flags, plotting, header precision policy
```

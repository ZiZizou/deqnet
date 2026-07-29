## KirchhoffNet: A Scalable Ultra Fast Analog Neural Network

### Introduction

This repo contains the code for our paper titled: [KirchhoffNet: A Scalable Ultra Fast Analog Neural Network](https://arxiv.org/pdf/2310.15872), accepted by the top Electronic Design Automation (EDA) conference, ICCAD'24. In one sentence, we argue an interesting fact that an analog integrated circuit mathematically corresponds to a [Neural ODE](https://arxiv.org/pdf/1806.07366). Thus, if we fabricate such an analog circuit as a specific hardware accelerator for Neural ODE, it can run extremely fast (compared to running Neural ODE on GPUs). This is the first work in this area.

### DEQ Mode (Deep Equilibrium Networks)

This repo also contains a **Deep Equilibrium (DEQ)** extension: instead of integrating the circuit ODE forward in time, we solve for the **steady-state equilibrium** directly via Anderson-accelerated fixed-point iteration. This replaces sequential time integration with an implicit solve.

**Why DEQ?** The circuit's right-hand side `f(v)` naturally satisfies a strong monotonicity property: its Jacobian `J = -M` is symmetric negative definite, where `M = BᵀDB + Γ` is assembled from edge conductances `D` and nodal leakage `Γ`. This guarantees:
- A unique, globally attracting equilibrium for any input.
- Convergence of forward-Euler fixed-point `v ← v + β·f(v)` for `β < 2/λ_max(M)`.
- Well-conditioned backward pass via the implicit function theorem: `Jᵀ y = g` is solved matrix-free with conjugate gradients.

**Contraction rate certificate.** Because the slopes of monotone activation functions are bounded, the worst-case spectrum of `M = BᵀDB + Γ` can be computed analytically without data — `eigvalsh` (or power iteration) gives `lambda_min_M` and `lambda_max_M`.  The contraction rate `lambda_min_M` predicts solver iteration count and the backward CG condition number; when devices saturate into dead zones (`D ≈ 0`), `lambda_min_M → gamma_floor` and iteration counts blow up — the early-warning signal for the late-training collapse.  The legacy `passive: True` flag is vacuous as constructed (always True when `gamma_floor > 0`) and is retained for backward compatibility; `lambda_min_M` is the meaningful number.

### DEQ Implementation Status

Audited against [`ORIGINAL_PLAN.md`](./ORIGINAL_PLAN.md) — **32/32 tests pass**.

| Section | Status | Notes |
|---|---|---|
| §1. `deq_solver.py` — Anderson, fixed-point, CG backward, contraction check | ✅ Complete | Sign convention intentionally diverged from plan (§1 said `v-β·f(v)` but §2 defines `f(v) = -Bᵀg - Γv + Su` with `J=-M` NSD; `v+β·f(v)` is the correct contractive map — plan was internally inconsistent) |
| §2. Device passivity — positivity reparameterization, `max_slope()`, monotone whitelist | ✅ Near-complete | `Device.negation=False` branch retained under `reparam=False` + `DeprecationWarning` for ODE backwards compatibility (plan said "delete it") |
| §3. Circuit RHS — `gamma=softplus(raw)+1e-2`, `input_map`, `rhs(v,u)` | ✅ Complete | Original `forward(t,x)` preserved for ODE mode |
| §4. `EquilibriumSolve` — implicit backward via `f_.backward(-y)`, phantom gradients | ✅ Complete | Both `exact` and `phantom` modes implemented |
| §5. `EquilibriumBlock` — composed multi-layer DEQ | ✅ Partial | Composed mode works; fused mode (`merge_topologies`) is a `NotImplementedError` stub — acknowledged as `TODO(v2)`. Inter-layer gradient flow (differentiable `v_{k-1}^*` → `u_k`) also deferred to v2. |
| §6. `LinearSolveLayer` — second-order analog Newton/RLS | ◐ Component done, missing demo | `LinearSolveLayer` class implemented and tested. Full streaming RLS script (`run_rls_demo.py`) with baselines (digital RLS, LMS, analog LMS) not yet written — this is the plan's central novelty claim. |
| §7. `model.py` — `CircuitNet` ODE/DEQ branching, `solver_stats()` | ✅ Complete | `torchdiffeq` lazy-imported only inside ODE branch |
| §8. Config & training scripts — YAML sections, `contract_check_every`, certificate logging | ✅ Complete | Three example configs: twomoon, mnist, toyregress |

### DEQ Training Results

Smoke training on **toyregress** (5-dimensional regression, 5 input nodes, 1 output node, 50 samples, 5 epochs).

| Config | Params | Train loss | Test loss | Solver | Certificate |
|--------|--------|-----------|----------|--------|-------------|
| 1-layer `[4,4]` | 0.96 KiB | 0.124 → **0.021** | **0.019** | Layer 0 converges in 25-60 iter (residual ~1e-5) ✓ | passive (λ_max=-40.9) |
| 2-layer `[4,4],[4,4]` | 0.96 KiB | 0.186 → **0.030** | **0.028** | Layer 0 converges ~23-49 iter ✓; Layer 1 hits 60-iter limit (residual ~6e-4, above tol=1e-5) ✗ | passive (λ_max=-40.9) |

No NaN/Inf, no CG failures, no ill-conditioned warnings in either run.

**Convergence status summary:**
- **1-layer toyregress:** fully verified — forward solver reaches tol within budget on every batch.
- **2-layer toyregress:** Layer 1 consistently hits the max-iter ceiling with residual ~6e-4 (60× above tol).  Training still improves loss because the residual is small enough to drive meaningful gradients, but formal convergence is not reached.  Likely causes: composed multi-layer dynamics shift the effective spectrum; need more iterations, smaller `β`, or fused-mode settling.
- **twomoon** and **MNIST:** not yet run — convergence behaviour on these configs is unverified.

**Key fixes that resolved prior divergence:**

1. **Device-param init for `reparam=True`** (`circuit_block.py`). `ShiftRelu1` (and all `reparam=True` devices) were never initialized — `CircuitLayer.__init__` only called `_init_model_param` on `self.model.param`, which only exists when `reparam=False`. Result: `raw_gain=0` everywhere → `gain=max_gain/2` constant, `bias=0` (no dead zone) → purely-linear dynamics → circuit diverged numerically. Fixed by adding initialization for `raw_gain`, `raw_gain_src`, `raw_gain_des`, `bias` attributes when `model.reparam=True`.

2. **Backward CG on SPD `M` instead of NSD `J`** (`deq_solver.py`). `solve_jacobian_transpose` was solving `J y = rhs` with CG, but `J = ∂f/∂v = -(BᵀDB + Γ)` is symmetric negative definite — CG only converges on SPD systems. Fixed by solving `M y = -rhs` where `M = -J` is SPD, via `M_times = lambda y: -At(y)`, with residual/step updates adjusted for the sign flip.

**Remaining gaps:**

1. **Fused-mode netlist** [`topology.py:merge_topologies`, §5 of the plan] — the plan's biggest unimplemented feature. For now only composed (per-layer) DEQ works via `merge: composed`. `merge_topologies` offsets node indices of each layer by the cumulative node count of previous layers, concatenates edge lists, and appends inter-layer edges — producing one `CircuitLayer` that settles as a single physical circuit. Requires:
   - Implementing the offset + concat logic.
   - Building merged `Gamma`, `S`, `input_map` for the combined netlist.
   - Adding a `mode: fused` branch to `EquilibriumBlock.__init__` and `forward`.
   - A `NotImplementedError` stub currently blocks this.

2. **Inter-layer gradient flow in composed mode** [§5, implied] — Currently each `EquilibriumSolve` is independent; only the last layer's parameters receive gradients. The plan's v2 design feeds `v_{k-1}^*` as the `u` argument of layer `k` (not just warm-start init), making the autograd chain differentiable across layers. This is `TODO(v2)` alongside fused mode in `EquilibriumBlock`.

3. **Streaming RLS adaptive-filtering demo** [`LinearSolveLayer`, §6] — `LinearSolveLayer` is implemented and tested in isolation, but the plan's full streaming demo script does not yet exist:
   ```
   for each sample (x_t, d_t):
       R ← decay·R + outer(x_t, x_t)      # conductance update (write phase)
       p ← decay·p + e_t·x_t              # current-source update
       w = LinearSolveLayer(p, R)          # settle phase (the fabric)
       e_{t+1} = d_{t+1} - w @ x_{t+1}
   ```
   This container demo is the plan's entire novelty argument (§6, bullet 4 of the validation sequence). Requires a new script (e.g. `run_rls_demo.py`) and baselines (digital RLS, LMS, analog-LMS first-order fabric).

4. **Jacobian regularizer for late-training collapse** [Known failure modes, §"What to watch for"] — When ReLU biases push devices into the dead zone over most of the input range, `D ≈ 0` and only `Γ` contracts, slowing the solver. The plan proposes a loss term `λ·relu(μ(J) + c)` to keep the effective contraction rate bounded away from zero. Not yet implemented.

5. **`Device.negation=False` branch cleanup** [§2] — ~15 lines of dead code guarded by `reparam=False`, kept only for ODE backward compatibility and guarded by a `DeprecationWarning`. Does not affect DEQ mode. Could be deleted in a cleanup pass.

6. **Fused vs. composed ablation** [Validation step 5] — Compare accuracy and iteration count between fused and composed DEQ at matched parameter count. If fused ≠ composed, that gap is a finding about inter-layer settling dynamics. This is an experiment (not code) but the infrastructure to run it requires items 1–2 above.

7. **CNF / log-density via DEQ** [Plan §5, point (a)] — Currently density tasks use the ODE path (FFJORD). The plan notes that the equilibrium map is invertible with `log|det(dv*/du)|` computable via stochastic trace estimation on `J⁻¹`, which would allow a pure-DEQ flow model. This is a research contribution, not a coding TODO, but worth flagging as a direction.

### Known Issues (from July 2026 audit)

1. **`EquilibriumBlock.forward` has no direct end-to-end test.** `test_multi_layer_grad_flows_to_earlier_layers` manually chains `EquilibriumSolve.apply` calls — it tests the mechanism, not the class. A regression in `prepare`/`forward`/input_map wiring would pass silently. Should add `test_equilibrium_block_2layer` that builds an `EquilibriumBlock`, calls `prepare`+`forward`, and checks non-zero grad on layer 0 params.

2. **`check_contraction` can silently return NaN for large n (>128).** The inverse-power-iteration fallthrough initializes `lam_min_M = float('nan')` and the CG loop can break before computing the Rayleigh quotient. The caller logs `lambda_min(M)=nan` with no distinct warning. Should detect the NaN and report `gamma_floor` as lower bound.

3. **`contract_check` only audits layer 0.** `main_deq.py` hardcodes `layer_list[0]`. For multi-layer configs, layers ≥ 1 have their own spectrum but are never checked. Should iterate all layers.

4. **Test/doc counts** were stale (27→32 tests). Now updated.

5. **`v0` parameter in `EquilibriumBlock.forward` is accepted but ignored.** `init` is always `torch.zeros(batch, n)`. Harmless today (init is a speed knob per §7.5), but if warm-start across training steps is desired, `v0` should be wired through.

6. **`test_linear_solve_layer_matches_direct_solve`** has a precision-losing `.double()` then `.float()` cast at `test_equilibrium_solve.py:267-268`. Stays in float32, test passes at `err < 1e-5`.

7. **`test_implicit_grad_matches_unrolled`** has a dead `solver_cfg` assignment at line 63 (immediately overwritten at lines 66-67).

### Prerequisite

You need to have [`pytorch`](https://pytorch.org/get-started/locally/) and [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq) installed; please use the embedded links for instructions on installations. Next, clone the code to your local machine. `torchdiffeq` is only needed for ODE mode; the DEQ path uses it only as an optional lazy import.

### DEQ Architecture

```
src/
├── utils/
│   ├── deq_solver.py          # Solver machinery
│   │   ├── fixed_point         # Plain fixed-point iteration
│   │   ├── anderson            # Anderson acceleration (type-I, ring-buffer)
│   │   ├── solve_jacobian_transpose  # Matrix-free Jᵀ y = g (CG + fallback)
│   │   ├── check_contraction   # Analytic passivity certificate
│   │   └── estimate_lipschitz  # Rough Lipschitz estimate
│   ├── circuit_block.py        # Circuit model & DEQ extensions
│   │   ├── Device / ShiftRelu1 / etc.  # Monotone device models (positivity-reparameterized)
│   │   ├── CircuitLayer        # ODE RHS + DEQ residual f(v, u)
│   │   ├── EquilibriumSolve    # torch.autograd.Function with implicit backward
│   │   ├── EquilibriumBlock    # Composed multi-layer DEQ block
│   │   └── LinearSolveLayer    # Second-order analog Newton/RLS layer
│   ├── model.py                # CircuitNet: ODE/DEQ wiring
│   └── topology.py             # Graph generators + merge_topologies (stub)
├── configs/
│   ├── config_deq_twomoon.yaml
│   ├── config_deq_mnist.yaml
│   └── config_deq_toyregress.yaml
├── tests/
│   ├── run_all.py              # Test runner (all 32 tests)
│   ├── test_deq_solver.py      # Forward/backward solver correctness (8 tests)
│   ├── test_devices.py         # Device passivity & gain ranges (8 tests)
│   ├── test_rhs.py             # Circuit residual & gamma (6 tests)
│   ├── test_equilibrium_solve.py # Implicit diff & gradcheck (7 tests)
│   └── test_deq_end_to_end.py  # Full pipeline train step (3 tests)
├── main_deq.py                 # DEQ training entry point
├── main_image_classify.py      # ODE image classification
├── main_gen.py                 # ODE generative modeling
└── main_den.py                 # ODE density matching
```

**Device passivity.** Every edge device uses `gain = max_gain · sigmoid(raw_gain)`, keeping the slope in `(0, max_gain)` and ensuring monotonicity. The leakage parameter uses `γ = softplus(raw_γ) + 1e-2` for a hard positivity floor.

**Equilibrium residual.** The circuit RHS is:

```
f(v, u) = -Bᵀ g(B·v̂ ; θ) - Γ·v + S·u
```

where `B` is the incidence matrix, `g` are the edge conductances, `Γ` is nodal leakage, and `S` selects input injection nodes.

**Implicit backward.** `EquilibriumSolve.forward` calls `anderson(f, v0)` under `torch.no_grad()`. `backward` solves `Jᵀ y = grad_out` via a matrix-free CG (or explicit solve for n ≤ 64), then calls `f_.backward(-y)` to populate parameter gradients via autograd. No autograd graph through the forward solver is built — memory is constant in solver iterations.

**Phantom gradients.** Set `backward_mode: phantom` in the config for a cheap biased gradient (`y ← grad_out - damp · (Jᵀ y_prev - grad_out)`) that is robust when `J` is ill-conditioned early in training.

**Second-order layer.** `LinearSolveLayer` solves `R·w = p` (SPD `R`) at equilibrium, providing a clean building block for adaptive filtering (RLS-style) demos.

### DEQ Usage

Train a DEQ two-moon classifier:
```shell
cd src
python main_deq.py --config_path ./configs/config_deq_twomoon.yaml --gpu -1
```

DEQ configs extend the ODE config with a `solver` section and a `fabric` section:

```yaml
mode: deq
solver:
  method: anderson          # anderson | fixedpoint
  tol: 1.0e-5
  max_iter: 80
  anderson_m: 5
  beta: 1.0
  backward_mode: exact      # exact | phantom
  phantom_damp: 0.5
  phantom_steps: 1
fabric:
  gamma_floor: 0.05
  max_gain: 5.0
  input_nodes: [1, 2]
  merge: composed           # composed | fused (v2)
contract_check_every: 5
```

### Testing

Run all 32 DEQ tests:
```shell
cd src
python tests/run_all.py
```

Or run individual test modules:
```shell
cd src
python tests/test_deq_solver.py
python tests/test_equilibrium_solve.py
```

### Reproduce ODE Results

To reproduce the results reported in the ICCAD paper, please refer to the `src` folder. To validate our results, we have stored the trained models, allowing users to directly run the inference procedure. To do this, change the directory to the `src` folder and use the provided trained models (under `src/results/xxx`) to reproduce the results for generation and density matching in our paper by running:

```shell
python inference_gen_den.py --exp_name xxx --gpu 0
```

Here `xxx` can be `genmnist`, `2spirals`, `8guassians`, `pinwheel`, `swissroll`, `twomoon`, `potential1`, `potential2`. Additionally, training such a model is also straightforward using the `main_gen.py` for generation, and `main_den.py` for density matching. Take twomoon generation as an example:

```shell
python main_gen.py --config_path ./configs/config_twomoon.yaml --gpu 0
```

Inside `src/results/twomoon`, we provide the `config.yaml` file, which was used to train the model stored there. Users can also use the `config.yaml` under other `src/results/xxx` as the command-line argument for the script `main_gen.py` and `main_den.py` to re-run the training procedure.

Similarly, we also provide trained model checkpoints for the image classification task. But first please revise the data path in lines 22-28 in `src/utils/testbench.py` for the program to successfully locate the data. Afterwards:

```shell
python inference_image_classify.py --exp_name xxx --gpu 0
```

Here `xxx` can be `mnist`, `svhn`, and `cifar10`. The `config.yaml` file under `src/results/xxx` can be used along with `main_image_classify.py` to do training.

### Remarks

We have been asked a few questions frequently, please refer to this [Q&A](https://zhengqigao.github.io/articles/what_is_kirchhoffnet.pdf) for some common questions and our answers. Also, here is a one-page summary of [our work](https://zhengqigao.github.io/articles/kirchhoffnet.pdf). There are many future works we want to explore, such as power and area of such an analog integrated circuit. Also, we want to redo this work based on commercial simulators such as Hspice and Spectre. We are also considering fabricating a real hardware. We are always looking for collaborators on this topic.

### Citation

At this moment, the ICCAD proceeding is not publicly available. Please cite our Arxiv paper if you use it in your research:

```bibtex
@misc{gao2024kirchhoffnet,
      title={KirchhoffNet: A Scalable Ultra Fast Analog Neural Network}, 
      author={Zhengqi Gao and Fan-Keng Sun and Ron Rohrer and Duane S. Boning},
      year={2024},
      eprint={2310.15872},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2310.15872}, 
}
```

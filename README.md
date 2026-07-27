## KirchhoffNet: A Scalable Ultra Fast Analog Neural Network

### Introduction

This repo contains the code for our paper titled: [KirchhoffNet: A Scalable Ultra Fast Analog Neural Network](https://arxiv.org/pdf/2310.15872), accepted by the top Electronic Design Automation (EDA) conference, ICCAD'24. In one sentence, we argue an interesting fact that an analog integrated circuit mathematically corresponds to a [Neural ODE](https://arxiv.org/pdf/1806.07366). Thus, if we fabricate such an analog circuit as a specific hardware accelerator for Neural ODE, it can run extremely fast (compared to running Neural ODE on GPUs). This is the first work in this area.

### DEQ Mode (Deep Equilibrium Networks)

This repo also contains a **Deep Equilibrium (DEQ)** extension: instead of integrating the circuit ODE forward in time, we solve for the **steady-state equilibrium** directly via Anderson-accelerated fixed-point iteration. This replaces sequential time integration with an implicit solve.

**Why DEQ?** The circuit's right-hand side `f(v)` naturally satisfies a strong monotonicity property: its Jacobian `J = -M` is symmetric negative definite, where `M = BᵀDB + Γ` is assembled from edge conductances `D` and nodal leakage `Γ`. This guarantees:
- A unique, globally attracting equilibrium for any input.
- Convergence of forward-Euler fixed-point `v ← v + β·f(v)` for `β < 2/λ_max(M)`.
- Well-conditioned backward pass via the implicit function theorem: `Jᵀ y = g` is solved matrix-free with conjugate gradients.

**Contraction certificate.** Because the slopes of monotone activation functions are bounded, the worst-case `λ_max(J)` can be computed analytically without data — a single power iteration per parameter update certifies passivity.

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
│   ├── run_all.py              # Test runner (all 27 tests)
│   ├── test_deq_solver.py      # Forward/backward solver correctness (7 tests)
│   ├── test_devices.py         # Device passivity & gain ranges (8 tests)
│   ├── test_rhs.py             # Circuit residual & gamma (6 tests)
│   ├── test_equilibrium_solve.py # Implicit diff & gradcheck (6 tests)
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

Run all 27 DEQ tests:
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

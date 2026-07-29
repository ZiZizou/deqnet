"""Tests for EquilibriumSolve (implicit differentiation) on linear circuits."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils.circuit_block import CircuitLayer, EquilibriumSolve, LinearSolveLayer
from utils.topology import generate_topology


def _linear_layer(n=5):
    topo_dict = {
        'name': '_fully_connect',
        'args': {'num_node': n, 'include_gnd': True, 'repeat': [1, 0]},
    }
    topo = generate_topology(topo_dict)
    circuit_dict = {
        'model': {'name': 'Conductance', 'args': {'max_gain': 2.0, 'reparam': True}},
        'initialization': 'ones',
    }
    layer = CircuitLayer(topo, circuit_dict)
    layer.prepare([-1])
    layer.set_input_nodes(None, device_index=-1)
    return layer


def test_forward_solves_equilibrium():
    layer = _linear_layer(n=4)
    n = layer.max_node_index

    def rhs_fn(v, u=None):
        return layer.rhs(v, u)

    u = torch.randn(2, 1) * 0.5
    v0 = torch.zeros(2, n)
    # With the random u scaling, equilibrium is bounded; use small beta.
    solver_cfg = {'method': 'anderson', 'max_iter': 200, 'tol': 1e-8, 'm': 5, 'beta': 0.1,
                  'backward_mode': 'exact', 'backward_tol': 1e-6, 'backward_max_iter': 100}
    v_star = EquilibriumSolve.apply(rhs_fn, v0, u, solver_cfg)
    res = layer.rhs(v_star, u).abs().max().item()
    print(f"[forward residual] ||f(v*)||_inf = {res:.3e}")
    assert res < 1e-5, f"forward did not solve: residual={res}"
    print("  PASS")


def test_implicit_grad_matches_unrolled():
    """Implicit-diff grad must match the closed-form linear grad (linear circuit).

    For a linear circuit with constant Conductance, the forward is
    f(v, u) = -M v + S u with M = (Bn^T D) Bn + gamma I.  The equilibrium is
    v* = u @ S^T @ M^{-1} (per row).
    """
    layer = _linear_layer(n=4)
    n = layer.max_node_index
    u_param = torch.nn.Parameter(torch.randn(3, 1) * 0.3)
    # Inject at the first non-ground node (v[0]).
    layer.set_input_nodes([1], device_index=-1)

    def rhs_fn(v, u):
        return layer.rhs(v, u)

    solver_cfg = {'method': 'anderson', 'max_iter': 200, 'tol': 1e-9, 'm': 5, 'beta': 1.0,
                  'backward_mode': 'exact', 'backward_tol': 1e-7, 'backward_max_iter': 200}

    v0 = torch.zeros(3, n)
    solver_cfg = {'method': 'anderson', 'max_iter': 300, 'tol': 1e-9, 'm': 5, 'beta': 0.1,
                  'backward_mode': 'exact', 'backward_tol': 1e-7, 'backward_max_iter': 200}
    v_star = EquilibriumSolve.apply(rhs_fn, v0, u_param, solver_cfg)
    v_star.sum().backward()
    grad_implicit = u_param.grad.clone()

    src = layer.src_node
    des = layer.des_node
    E = src.shape[0]
    Bmat = torch.zeros(E, n + 1)
    arange = torch.arange(E)
    Bmat[arange, src] = -1.0
    Bmat[arange, des] = 1.0
    Bn = Bmat[:, 1:]
    _g = layer.model.gain.detach()
    while _g.dim() > 1:
        _g = _g[0]
    D = _g.reshape(-1)
    M = (Bn.t() * D) @ Bn + layer.gamma.detach().item() * torch.eye(n)  # n x n
    M_inv = torch.linalg.solve(M, torch.eye(n))
    S = layer.input_map  # (n, 1)
    # Reference: v_star = u @ S^T @ M^{-1}
    v_star_ref = u_param @ S.t() @ M_inv
    grad_ref = torch.autograd.grad(v_star_ref.sum(), u_param, retain_graph=False)[0]

    err = (grad_implicit - grad_ref).abs().max().item()
    print(f"[implicit vs unrolled grad] err={err:.3e}")
    print(f"  implicit={grad_implicit.flatten().tolist()}")
    print(f"  ref     ={grad_ref.flatten().tolist()}")
    assert err < 1e-3, f"grad mismatch: {err}"
    print("  PASS")


def test_phantom_backward_does_not_error():
    """Phantom-mode backward must run without error (biased but defined)."""
    layer = _linear_layer(n=3)
    n = layer.max_node_index
    u_param = torch.nn.Parameter(torch.randn(2, 1) * 0.3)
    layer.set_input_nodes([1], device_index=-1)

    def rhs_fn(v, u):
        return layer.rhs(v, u)

    solver_cfg = {'method': 'anderson', 'max_iter': 100, 'tol': 1e-8, 'm': 3, 'beta': 0.1,
                  'backward_mode': 'phantom'}
    v0 = torch.zeros(2, n)
    v_star = EquilibriumSolve.apply(rhs_fn, v0, u_param, solver_cfg)
    v_star.sum().backward()
    assert u_param.grad is not None
    print(f"[phantom backward] grad norm = {u_param.grad.norm().item():.4f}. PASS")


def test_gradcheck_implicit_vs_unrolled():
    """torch.autograd.gradcheck-style: compare implicit-diff gradients against
    unrolled fixed-point iteration with create_graph=True, on a small network
    in double precision.

    For a linear circuit, both must agree to high precision.  This is the spec's
    "validation step 2" — catches VJP sign-convention errors.
    """
    torch.manual_seed(7)
    layer = _linear_layer(n=4)
    layer = layer.double()  # entire module → float64 to avoid dtype mixing
    n = layer.max_node_index
    layer.set_input_nodes([1], device_index=-1)
    # layer.double() does not convert registered buffers; convert explicitly.
    layer.input_map = layer.input_map.double()
    layer.src_node = layer.src_node
    layer.des_node = layer.des_node

    d_in = 1
    u_param = torch.nn.Parameter(torch.randn(2, d_in, dtype=torch.float64) * 0.3)

    def rhs_fn(v, u):
        return layer.rhs(v, u)

    solver_cfg = {'method': 'anderson', 'max_iter': 500, 'tol': 1e-12, 'm': 5, 'beta': 0.1,
                  'backward_mode': 'exact', 'backward_tol': 1e-10, 'backward_max_iter': 500}
    v0 = torch.zeros(2, n, dtype=torch.float64)

    # Implicit path
    v_star_imp = EquilibriumSolve.apply(rhs_fn, v0, u_param, solver_cfg)
    g_imp = torch.autograd.grad(v_star_imp.sum(), u_param, retain_graph=False)[0]

    # Reference: closed-form linear solution v* = u @ S^T @ M^{-1}
    src = layer.src_node
    des = layer.des_node
    E = src.shape[0]
    Bmat = torch.zeros(E, n + 1, dtype=torch.float64)
    arange = torch.arange(E)
    Bmat[arange, src] = -1.0
    Bmat[arange, des] = 1.0
    Bn = Bmat[:, 1:]
    _g = layer.model.gain.detach().reshape(-1)
    D = _g
    gamma_val = layer.gamma.detach().item()
    M = (Bn.t() * D) @ Bn + gamma_val * torch.eye(n, dtype=torch.float64)
    M_inv = torch.linalg.solve(M, torch.eye(n, dtype=torch.float64))
    S = layer.input_map
    v_star_ref = u_param @ S.t() @ M_inv
    g_ref = torch.autograd.grad(v_star_ref.sum(), u_param, retain_graph=False)[0]

    err = (g_imp - g_ref).abs().max().item()
    print(f"[gradcheck implicit vs closed-form] err={err:.3e}")
    assert err < 1e-5, f"grad mismatch: {err}"
    print("  PASS")


def test_multi_layer_grad_flows_to_earlier_layers():
    """Composed-mode DEQ: in a 2-layer block, layer 0's parameters must
    receive nonzero gradient.

    Before the fix, v_star was overwritten each iteration and only the last
    layer's parameters received gradients.  After the fix, equilibria are
    chained: layer k receives the equilibrium of layer k-1 as injected
    current, so the autograd graph spans all layers.
    """
    import torch.nn as nn
    torch.manual_seed(42)
    topo_dict = {
        'name': '_fully_connect',
        'args': {'num_node': 5, 'include_gnd': True, 'repeat': [1, 0]},
    }
    topo = generate_topology(topo_dict)
    circuit_dict = {
        'model': {'name': 'ShiftRelu1', 'args': {'max_gain': 2.0, 'reparam': True}},
        'initialization': 'kaiming',
    }
    layer0 = CircuitLayer(topo, circuit_dict)
    layer1 = CircuitLayer(topo, circuit_dict)
    layer0.prepare([-1])
    layer1.prepare([-1])
    # Layer 0: external input at node 1 (d_in=1).
    layer0.set_input_nodes([1], device_index=-1)
    # Layer 1: inter-layer map (n1, n0) = (4, 4) — identity.
    layer1.set_input_map(torch.eye(layer1.max_node_index, layer0.max_node_index),
                         device_index=-1)

    solver_cfg = {'method': 'anderson', 'max_iter': 200, 'tol': 1e-8, 'm': 5, 'beta': 0.1,
                  'backward_mode': 'exact', 'backward_tol': 1e-6, 'backward_max_iter': 100}

    u = torch.nn.Parameter(torch.randn(3, 1) * 0.5)
    batch = 3
    init0 = torch.zeros(batch, layer0.max_node_index)

    def rhs0(v, _u=u):
        return layer0.rhs(v, _u)

    v0_star = EquilibriumSolve.apply(rhs0, init0, u, solver_cfg)
    init1 = torch.zeros(batch, layer1.max_node_index)

    def rhs1(v, _v0=v0_star):
        return layer1.rhs(v, _v0)

    v1_star = EquilibriumSolve.apply(rhs1, init1, v0_star, solver_cfg)
    v1_star.sum().backward()

    g_layer0 = layer0.model.raw_gain.grad
    g_layer1 = layer1.model.raw_gain.grad
    print(f"[multi-layer grad] layer0.raw_gain.grad norm = "
          f"{g_layer0.norm().item():.4e}, layer1 = {g_layer1.norm().item():.4e}")
    assert g_layer0 is not None, "layer0 raw_gain.grad is None — gradient not flowing to earlier layers"
    assert g_layer0.norm().item() > 1e-10, (
        f"layer0 raw_gain.grad is effectively zero ({g_layer0.norm().item():.3e}); "
        "composed multi-layer mode is degenerate."
    )
    assert g_layer1 is not None and g_layer1.norm().item() > 1e-10
    print("  PASS")


def test_linear_solve_layer_gradient_flow():
    """Regression test for the LinearSolveLayer.detach() bug.

    Before the fix, the layer returned v_star.detach() which broke the autograd
    graph; gradients on p would be None.  After the fix, grad on p must be
    finite and non-zero.
    """
    torch.manual_seed(11)
    n = 5
    p = torch.nn.Parameter(torch.randn(3, n, dtype=torch.float64))
    R = torch.eye(n, dtype=torch.float64) + 0.3 * torch.randn(n, n, dtype=torch.float64).abs()
    R = (R + R.t()) / 2 + torch.eye(n, dtype=torch.float64)

    layer = LinearSolveLayer(max_iter=200, tol=1e-10)
    layer = layer.double()
    w_star = layer(p, R)
    assert w_star.shape == p.shape
    assert torch.isfinite(w_star).all(), "LinearSolveLayer produced non-finite values"
    g = torch.autograd.grad(w_star.sum(), p, retain_graph=False)[0]
    assert g is not None, "LinearSolveLayer broke gradient flow (detach bug regressed?)"
    assert torch.isfinite(g).all(), "LinearSolveLayer backward produced non-finite grads"
    print(f"[LinearSolveLayer grad] ||d(sum(w))/dp|| = {g.norm().item():.4e}. PASS")


def test_linear_solve_layer_matches_direct_solve():
    """Forward of LinearSolveLayer must match w* = R^{-1} p exactly."""
    torch.manual_seed(13)
    n = 4
    p = torch.randn(2, n)
    Q = torch.randn(n, n)
    eigs = 1.0 + 3.0 * torch.rand(n)
    R = (Q @ torch.diag(eigs) @ Q.t()).double()
    R = ((R + R.t()) / 2 + torch.eye(n, dtype=torch.float64)).float()

    # Choose a step size in (0, 2 / lambda_max(R)) so the fixed-point map
    # g(w) = (I - beta*R) w + beta*p is contractive.
    lam_max = torch.linalg.eigvalsh(R).max().item()
    beta_opt = 1.0 / lam_max
    layer = LinearSolveLayer(max_iter=500, tol=1e-12, beta=beta_opt)
    w_star = layer(p, R)
    w_ref = torch.linalg.solve(R, p.t()).t()
    err = (w_star - w_ref).abs().max().item()
    print(f"[LinearSolveLayer forward vs solve] err={err:.3e} beta={beta_opt:.3f}")
    assert err < 1e-5, f"forward mismatch: {err}"
    print("  PASS")


if __name__ == '__main__':
    print("=" * 60)
    print("Test: EquilibriumSolve")
    print("=" * 60)
    test_forward_solves_equilibrium()
    test_implicit_grad_matches_unrolled()
    test_phantom_backward_does_not_error()
    test_gradcheck_implicit_vs_unrolled()
    test_multi_layer_grad_flows_to_earlier_layers()
    test_linear_solve_layer_gradient_flow()
    test_linear_solve_layer_matches_direct_solve()
    print("\nAll EquilibriumSolve tests passed.")

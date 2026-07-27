"""Tests for CircuitLayer.rhs() — the DEQ residual function."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils.circuit_block import CircuitLayer
from utils.topology import generate_topology


def _build_linear_circuit(n=5):
    """Build a CircuitLayer with all Conductance devices (linear)."""
    topo_dict = {
        'name': '_fully_connect',
        'args': {'num_node': n, 'include_gnd': True, 'repeat': [1, 0]},
    }
    topo = generate_topology(topo_dict)
    circuit_dict = {
        'model': {'name': 'Conductance', 'args': {'max_gain': 3.0, 'reparam': True}},
        'initialization': 'ones',
    }
    layer = CircuitLayer(topo, circuit_dict)
    layer.prepare([-1])
    return layer


def test_rhs_zero_at_zero_v_no_input():
    layer = _build_linear_circuit(n=4)
    layer.set_input_nodes(None, device_index=-1)
    v = torch.zeros(1, 4)
    f = layer.rhs(v)
    assert torch.allclose(f, torch.zeros_like(f), atol=1e-6), f"f(0) nonzero: {f}"
    print(f"[rhs(0)] all zero. PASS")


def test_rhs_zero_input_unique_equilibrium():
    """For a strictly passive linear circuit, f(v) = 0 iff v = -Gamma^{-1} B^T D B * 0 = 0."""
    layer = _build_linear_circuit(n=4)
    layer.set_input_nodes(None, device_index=-1)
    # For f(v, u=0) = -A v - gamma v = 0, unique solution v*=0
    v = torch.tensor([[1.0, -2.0, 0.5, 3.0]])
    f = layer.rhs(v, u=None)
    assert not torch.allclose(f, torch.zeros_like(f)), "f(v) should be nonzero for v != 0"
    print(f"[rhs(v)] for v={v.tolist()}, f={f.tolist()}.  PASS (nonzero residual)")


def test_rhs_with_input_at_designated_node():
    """Injecting u at node 1 (label) should push f at v[0] (non-ground index)."""
    layer = _build_linear_circuit(n=4)
    layer.set_input_nodes([1], device_index=-1)
    u = torch.tensor([[0.5]])
    # max_node_index = num_node - 1 = 3 (since _fully_connect num_node=4 uses node ids 0..3)
    expected_shape = (layer.max_node_index, 1)
    assert layer.input_map.shape == expected_shape, f"bad input_map shape: {layer.input_map.shape}, expected {expected_shape}"
    assert layer.input_map[0, 0].item() == 1.0
    v = torch.zeros(1, layer.max_node_index)
    f_no_input = layer.rhs(v, u=torch.zeros_like(u))
    f_with_input = layer.rhs(v, u=u)
    diff = (f_with_input - f_no_input).abs().max().item()
    assert diff > 1e-6, f"input did not affect f: diff={diff}"
    print(f"[rhs(v, u)] effect of u on f(v=0) diff={diff:.4f}.  PASS")


def test_rhs_linear_matches_explicit():
    """For all-Conductance devices, rhs must equal -B^T D (B v) - gamma v."""
    layer = _build_linear_circuit(n=4)
    layer.set_input_nodes(None, device_index=-1)
    src = layer.src_node
    des = layer.des_node
    n = layer.max_node_index
    # Build B explicitly: B has shape (E, max_node_index + 1) with col 0 being the virtual ground.
    E = src.shape[0]
    B = torch.zeros(E, n + 1)
    arange = torch.arange(E)
    B[arange, src] = -1.0
    B[arange, des] = 1.0
    Bn = B[:, 1:]
    g_diag = layer.model.gain.detach()
    while g_diag.dim() > 1:
        g_diag = g_diag[0]
    D = g_diag.reshape(-1)
    gamma_val = layer.gamma.detach().item()

    v = torch.randn(2, n)
    rhs_implicit = layer.rhs(v, u=None)
    g_diag = layer.model.gain.detach()
    while g_diag.dim() > 1:
        g_diag = g_diag[0]
    D = g_diag.reshape(-1)
    Bv = Bn @ v.t()  # E x batch
    DBv = D.unsqueeze(1) * Bv
    BTD = -Bn.t() @ DBv  # n x batch
    rhs_explicit = (BTD - gamma_val * v.t()).t()  # batch x n
    err = (rhs_implicit - rhs_explicit).abs().max().item()
    print(f"[rhs linear match] err={err:.3e}")
    assert err < 1e-4, f"rhs does not match explicit: {err}"
    print("  PASS")


def test_gamma_is_positive():
    """gamma = softplus(raw_gamma) + 1e-2 must be > 0 always."""
    layer = _build_linear_circuit(n=4)
    g = layer.gamma.detach().item()
    assert g >= 1e-2, f"gamma below floor: {g}"
    # even after very negative raw_gamma
    layer.raw_gamma.data = torch.tensor(-50.0)
    g = layer.gamma.detach().item()
    assert g >= 1e-2 - 1e-6
    print(f"[gamma] min={1e-2:.2e}, current={g:.4e}.  PASS")


def test_rhs_grad_flows_through_gamma():
    """Gradient must flow through gamma."""
    layer = _build_linear_circuit(n=4)
    layer.set_input_nodes([1], device_index=-1)
    u = torch.tensor([[0.5]])
    # Non-zero v so that ∂(sum f)/∂gamma = -sigmoid(raw_gamma) * sum(v) is non-trivial.
    v = torch.randn(1, layer.max_node_index, requires_grad=True)
    f = layer.rhs(v, u)
    f.sum().backward()
    assert layer.raw_gamma.grad is not None
    assert layer.raw_gamma.grad.abs().item() > 0
    print(f"[rhs grad through gamma] gamma.grad={layer.raw_gamma.grad.item():.4e}.  PASS")


if __name__ == '__main__':
    print("=" * 60)
    print("Test: CircuitLayer.rhs()")
    print("=" * 60)
    test_rhs_zero_at_zero_v_no_input()
    test_rhs_zero_input_unique_equilibrium()
    test_rhs_with_input_at_designated_node()
    test_rhs_linear_matches_explicit()
    test_gamma_is_positive()
    test_rhs_grad_flows_through_gamma()
    print("\nAll rhs tests passed.")

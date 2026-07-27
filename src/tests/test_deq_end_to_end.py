"""End-to-end smoke test for the DEQ pipeline.

Builds a small CircuitNet in DEQ mode, runs a forward pass on a toy batch,
runs backward, optimizes one step, and checks that the loss decreases.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from utils.model import CircuitNet
from utils.topology import generate_topology


def build_deq_model():
    topo_structs = [
        {'name': '_fully_connect', 'args': {'num_node': 5, 'include_gnd': True, 'repeat': [2, 0]}},
    ]
    tops = [generate_topology(t) for t in topo_structs]
    sim = {'t_end': [0.0, 1.0], 'tol': 1e-3, 'min_step': 1e-5,
           'first_step': 1e-3, 'step_size': 1e-6, 'method': 'dopri5'}
    cd = {'model': {'name': 'ShiftRelu1', 'args': {'max_gain': 4.0, 'reparam': True}},
          'initialization': 'kaiming',
          'residual': None,
          'fill': None}
    sc = {'method': 'anderson', 'max_iter': 100, 'tol': 1e-6, 'm': 5, 'beta': 0.3,
          'backward_mode': 'exact', 'backward_tol': 1e-5, 'backward_max_iter': 50}
    ic = {'input_nodes': [1], 'max_gain': 4.0, 'gamma_floor': 0.1}
    return CircuitNet(
        circuit_topology=tops,
        sim_dict=sim,
        circuit_dict=cd,
        encoder=None,
        projector='first1',
        mode='deq',
        solver_cfg=sc,
        input_cfg=ic,
    )


def test_deq_forward_shape():
    torch.manual_seed(0)
    model = build_deq_model()
    model.prepare([-1])
    # d_in = 1 (input_nodes=[1]); u has shape (B, 1).
    x = torch.randn(4, 1)
    out, _ = model(x)
    print(f"[deq forward] in={x.shape} out={out.shape}")
    assert out.shape == (4, 1)
    print("  PASS")


def test_deq_backward_and_step():
    """One forward + backward + step must reduce the loss."""
    torch.manual_seed(1)
    model = build_deq_model()
    model.prepare([-1])
    # d_in = 1; u has shape (B, 1) and target has shape (B, 1).
    x = torch.randn(8, 1)
    y = torch.tanh(x)  # any scalar regression target within range
    # lr=0.001 keeps the model inside the contraction regime for the smoke test.
    opt = optim.AdamW(model.parameters(), lr=0.001)

    losses = []
    for step in range(5):
        opt.zero_grad()
        out, _ = model(x)
        loss = ((out - y) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"[deq train] losses={losses}")
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.3e} -> {losses[-1]:.3e}"
    print("  PASS")


def test_deq_solver_stats_logged():
    torch.manual_seed(2)
    model = build_deq_model()
    model.prepare([-1])
    x = torch.randn(2, 1)
    out, _ = model(x)
    stats = model.solver_stats()
    print(f"[deq solver stats] {stats}")
    assert stats is not None
    for s in stats:
        assert s['n_iter'] > 0
    print("  PASS")


if __name__ == '__main__':
    print("=" * 60)
    print("Test: DEQ end-to-end smoke")
    print("=" * 60)
    test_deq_forward_shape()
    test_deq_backward_and_step()
    test_deq_solver_stats_logged()
    print("\nAll DEQ end-to-end tests passed.")

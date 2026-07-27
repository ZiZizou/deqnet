"""Tests for device passivity reparameterization."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from utils.circuit_block import (
    ShiftRelu1, ShiftLeakyRelu1, ShiftTanh1, Conductance,
    ShiftRelu2, ShiftTanh2, Device,
)


def test_relu_gain_in_range():
    """ShiftRelu1 with reparam=True: gain in (0, max_gain)."""
    dev = ShiftRelu1(num_edge=10, max_gain=5.0, reparam=True)
    g = dev.gain
    assert (g > 0).all() and (g < 5.0).all(), f"gain out of range: min={g.min().item()}, max={g.max().item()}"
    print(f"[ShiftRelu1 gain] range=({g.min().item():.3f}, {g.max().item():.3f})")
    print("  PASS")


def test_relu_max_slope_matches_gain():
    dev = ShiftRelu1(num_edge=10, max_gain=5.0, reparam=True)
    g = dev.gain.detach()
    ms = dev.max_slope()
    assert torch.allclose(g, ms)
    print("  PASS")


def test_tanh_max_slope_matches_gain():
    dev = ShiftTanh1(num_edge=10, max_gain=5.0, reparam=True)
    g = dev.gain.detach()
    ms = dev.max_slope()
    assert torch.allclose(g, ms)
    print("  PASS")


def test_conductance_max_slope():
    dev = Conductance(num_edge=10, max_gain=8.0, reparam=True)
    g = dev.gain.detach()
    ms = dev.max_slope()
    assert torch.allclose(g, ms)
    assert (g > 0).all() and (g < 8.0).all()
    print(f"[Conductance gain] range=({g.min().item():.3f}, {g.max().item():.3f})")
    print("  PASS")


def test_device_negation_gain():
    dev = Device(num_edge=10, negation=True, activation='relu', max_gain=4.0, reparam=True)
    g = dev.gain.detach()
    assert (g > 0).all() and (g < 4.0).all()
    x = torch.randn(3, 10)
    y = dev(x, torch.zeros_like(x))
    assert y.shape == (3, 10)
    print(f"[Device(negation=True) gain] range=({g.min().item():.3f}, {g.max().item():.3f})")
    print("  PASS")


def test_relu_forward_shape():
    dev = ShiftRelu1(num_edge=8, max_gain=5.0, reparam=True)
    x = torch.randn(4, 5, 8)
    y = dev(x, torch.zeros_like(x))
    assert y.shape == (4, 5, 8)
    print("  PASS")


def test_tanh1_translation_invariant():
    """Tanh1 should depend only on (x_src - x_des), not on x_src alone."""
    dev = ShiftTanh1(num_edge=4, max_gain=3.0, reparam=True)
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    y = dev(x, torch.zeros_like(x))
    y_shifted = dev(x + 5.0, torch.zeros_like(x) + 5.0)
    diff = (y - y_shifted).abs().max().item()
    assert diff < 1e-6, f"not translation invariant: {diff}"
    print("  PASS")


def test_gain_gradients_flow():
    """Gradient must flow through the gain reparameterization."""
    dev = ShiftRelu1(num_edge=5, max_gain=2.0, reparam=True)
    x_src = torch.randn(3, 5, requires_grad=True)
    x_des = torch.randn(3, 5, requires_grad=True)
    y = dev(x_src, x_des)
    loss = y.sum()
    loss.backward()
    # raw_gain should now have nonzero grad
    assert dev.raw_gain.grad is not None
    assert dev.raw_gain.grad.abs().sum() > 0
    print(f"[gain grad] raw_gain.grad norm = {dev.raw_gain.grad.norm().item():.4f}")
    print("  PASS")


if __name__ == '__main__':
    print("=" * 60)
    print("Test: device passivity")
    print("=" * 60)
    test_relu_gain_in_range()
    test_relu_max_slope_matches_gain()
    test_tanh_max_slope_matches_gain()
    test_conductance_max_slope()
    test_device_negation_gain()
    test_relu_forward_shape()
    test_tanh1_translation_invariant()
    test_gain_gradients_flow()
    print("\nAll device tests passed.")

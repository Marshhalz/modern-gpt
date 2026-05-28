"""
Unit tests for the RoPE module (rope.py).

Tests three independent properties:
1. rotate_half / apply_rotary are mathematically correct.
2. RotaryEmbedding has no learnable parameters.
3. The relative-position property: rotating q by m and k by n gives the same
   dot product as rotating by (n-m) alone — the fundamental guarantee of RoPE.
"""

from __future__ import annotations

import torch
import pytest

from modern_gpt.rope import RotaryEmbedding, apply_rotary, rotate_half


# ---------------------------------------------------------------------------
# rotate_half
# ---------------------------------------------------------------------------

def test_rotate_half_shape():
    x   = torch.randn(3, 8)
    out = rotate_half(x)
    assert out.shape == x.shape


def test_rotate_half_values():
    """rotate_half([x1 | x2]) must equal [-x2 | x1]."""
    x        = torch.randn(2, 8)
    x1, x2   = x.chunk(2, dim=-1)
    expected = torch.cat((-x2, x1), dim=-1)
    torch.testing.assert_close(rotate_half(x), expected)


# ---------------------------------------------------------------------------
# apply_rotary
# ---------------------------------------------------------------------------

def test_apply_rotary_identity_at_zero_angle():
    """Rotating by angle 0 (cos=1, sin=0) must be the identity."""
    x    = torch.randn(4, 8)
    cos0 = torch.ones(4, 8)
    sin0 = torch.zeros(4, 8)
    torch.testing.assert_close(apply_rotary(x, cos0, sin0), x)


def test_apply_rotary_output_shape():
    x   = torch.randn(2, 6, 16)
    cos = torch.ones(6, 16)
    sin = torch.zeros(6, 16)
    out = apply_rotary(x, cos, sin)
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# RotaryEmbedding
# ---------------------------------------------------------------------------

def test_rotary_embedding_output_shapes():
    rope     = RotaryEmbedding(head_size=16, max_seq_len=64)
    cos, sin = rope(10)
    assert cos.shape == (10, 16)
    assert sin.shape == (10, 16)


def test_rotary_embedding_no_learnable_parameters():
    """RoPE tables are buffers (non-gradient) — zero trainable params."""
    rope = RotaryEmbedding(head_size=16, max_seq_len=64)
    n_params = sum(p.numel() for p in rope.parameters() if p.requires_grad)
    assert n_params == 0, f"expected 0 learnable params, got {n_params}"


def test_rotary_embedding_position_zero_is_identity():
    """cos[0] = 1 everywhere, sin[0] = 0 everywhere (angle = 0 at position 0)."""
    rope     = RotaryEmbedding(head_size=16, max_seq_len=64)
    cos, sin = rope(64)
    torch.testing.assert_close(cos[0], torch.ones(16))
    torch.testing.assert_close(sin[0], torch.zeros(16))


def test_rotary_embedding_odd_head_size_raises():
    with pytest.raises(ValueError, match="even"):
        RotaryEmbedding(head_size=7, max_seq_len=32)


# ---------------------------------------------------------------------------
# Relative-position property — the core guarantee of RoPE
# ---------------------------------------------------------------------------

def test_relative_position_property():
    """The dot product of rotated q (at pos m) and rotated k (at pos n)
    must depend only on the distance (m - n), not the absolute positions.

    Numerically: rotated_dot(5,3) == rotated_dot(10,8) == rotated_dot(20,18)
    because all three pairs have the same distance (2).
    """
    torch.manual_seed(42)
    rope     = RotaryEmbedding(head_size=16, max_seq_len=128)
    cos, sin = rope(128)

    q = torch.randn(16)
    k = torch.randn(16)

    def rotated_dot(m: int, n: int) -> float:
        qm = apply_rotary(q, cos[m], sin[m])
        kn = apply_rotary(k, cos[n], sin[n])
        return (qm @ kn).item()

    # Same distance, different absolute positions → values must match
    d_5_3   = rotated_dot(5,  3)
    d_10_8  = rotated_dot(10, 8)
    d_20_18 = rotated_dot(20, 18)

    assert abs(d_5_3 - d_10_8)  < 1e-5, f"{d_5_3:.6f} != {d_10_8:.6f}"
    assert abs(d_5_3 - d_20_18) < 1e-5, f"{d_5_3:.6f} != {d_20_18:.6f}"


def test_different_distances_give_different_values():
    """Different distances should (with overwhelming probability for random
    q/k) produce different dot products."""
    torch.manual_seed(7)
    rope     = RotaryEmbedding(head_size=16, max_seq_len=128)
    cos, sin = rope(128)
    q = torch.randn(16)
    k = torch.randn(16)

    def rotated_dot(m: int, n: int) -> float:
        return (apply_rotary(q, cos[m], sin[m]) @ apply_rotary(k, cos[n], sin[n])).item()

    d0 = rotated_dot(5, 5)   # distance 0
    d1 = rotated_dot(6, 5)   # distance 1
    d5 = rotated_dot(10, 5)  # distance 5

    # All three should differ — RoPE encodes different distances differently
    assert abs(d0 - d1) > 1e-4
    assert abs(d0 - d5) > 1e-4

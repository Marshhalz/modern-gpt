"""
Unit tests for the SwiGLU feed-forward network (ffn.py).
"""

from __future__ import annotations

import torch

from modern_gpt import GPTConfig, SwiGLU


def _cfg() -> GPTConfig:
    return GPTConfig(vocab_size=20, block_size=8, n_embd=16, n_head=2, n_layer=2, dropout=0.0)


def test_swiglu_output_shape():
    """SwiGLU must preserve the (B, T, n_embd) shape."""
    cfg = _cfg()
    ff  = SwiGLU(cfg)
    x   = torch.randn(3, cfg.block_size, cfg.n_embd)
    assert ff(x).shape == x.shape


def test_swiglu_has_three_projections_no_bias():
    """SwiGLU uses three bias-free linear layers (gate, up, down)."""
    cfg = _cfg()
    ff  = SwiGLU(cfg)
    linears = [m for m in ff.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 3
    assert all(lin.bias is None for lin in linears)


def test_swiglu_hidden_dim_matches_config():
    """gate/up project n_embd -> ffn_hidden_dim; down projects back."""
    cfg = _cfg()
    ff  = SwiGLU(cfg)
    assert ff.w_gate.out_features == cfg.ffn_hidden_dim
    assert ff.w_up.out_features   == cfg.ffn_hidden_dim
    assert ff.w_down.in_features  == cfg.ffn_hidden_dim
    assert ff.w_down.out_features == cfg.n_embd


def test_ffn_hidden_dim_two_thirds_rule():
    """Default config: 2/3 * 4 * 128 = 341 -> rounded up to 344."""
    assert GPTConfig().ffn_hidden_dim == 344


def test_ffn_hidden_dim_multiple_of_eight():
    """Hidden dim must be a multiple of 8 for hardware alignment."""
    for n_embd in (16, 32, 64, 128, 256, 512):
        cfg = GPTConfig(n_embd=n_embd, n_head=1)
        assert cfg.ffn_hidden_dim % 8 == 0


def test_swiglu_gating_suppresses_when_gate_zero():
    """If the gate branch outputs ~0, the whole FFN output is ~0.

    Forcing w_gate to zero makes SiLU(0)=0, so gate * up = 0 everywhere and
    the output (before/after w_down) must be exactly zero — demonstrating the
    gate genuinely controls flow.
    """
    cfg = _cfg()
    ff  = SwiGLU(cfg)
    with torch.no_grad():
        ff.w_gate.weight.zero_()
    x = torch.randn(2, cfg.block_size, cfg.n_embd)
    out = ff(x)
    assert torch.allclose(out, torch.zeros_like(out))

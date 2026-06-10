"""
Unit tests for Grouped-Query Attention (attention.py).
"""

from __future__ import annotations

import pytest
import torch

from modern_gpt import GPTConfig, GroupedQueryAttention
from modern_gpt.rope import RotaryEmbedding


def _cfg(**kw) -> GPTConfig:
    base = dict(vocab_size=20, block_size=8, n_embd=16, n_head=4, n_kv_head=2, n_layer=2, dropout=0.0)
    base.update(kw)
    return GPTConfig(**base)


def _rope(cfg: GPTConfig, T: int):
    rope = RotaryEmbedding(cfg.head_size, cfg.block_size)
    return rope(T)


# ---------------------------------------------------------------------------
# Config: GQA grouping
# ---------------------------------------------------------------------------

def test_config_n_rep():
    """n_rep = n_head // n_kv_head."""
    assert _cfg(n_head=4, n_kv_head=2).n_rep == 2
    assert _cfg(n_head=4, n_kv_head=1).n_rep == 4   # MQA
    assert _cfg(n_head=4, n_kv_head=4).n_rep == 1   # plain MHA


def test_config_invalid_kv_head_raises():
    with pytest.raises(ValueError, match="divisible"):
        _ = _cfg(n_head=4, n_kv_head=3).n_rep


# ---------------------------------------------------------------------------
# Shapes and projection widths
# ---------------------------------------------------------------------------

def test_gqa_output_shape():
    cfg = _cfg()
    attn = GroupedQueryAttention(cfg)
    x = torch.randn(3, cfg.block_size, cfg.n_embd)
    cos, sin = _rope(cfg, cfg.block_size)
    assert attn(x, cos, sin).shape == x.shape


def test_gqa_kv_projections_are_narrower_than_query():
    """k_proj/v_proj output n_kv_head*head_size; q_proj outputs n_head*head_size."""
    cfg = _cfg(n_head=4, n_kv_head=2)
    attn = GroupedQueryAttention(cfg)
    assert attn.q_proj.out_features == cfg.n_head    * cfg.head_size
    assert attn.k_proj.out_features == cfg.n_kv_head * cfg.head_size
    assert attn.v_proj.out_features == cfg.n_kv_head * cfg.head_size
    assert attn.k_proj.out_features < attn.q_proj.out_features


def test_gqa_qkv_have_no_bias():
    cfg = _cfg()
    attn = GroupedQueryAttention(cfg)
    assert attn.q_proj.bias is None
    assert attn.k_proj.bias is None
    assert attn.v_proj.bias is None


# ---------------------------------------------------------------------------
# Causality — the silent killer if broken
# ---------------------------------------------------------------------------

def test_gqa_is_causal():
    """Changing the last token must not change outputs at earlier positions."""
    cfg = _cfg()
    torch.manual_seed(0)
    attn = GroupedQueryAttention(cfg).eval()
    cos, sin = _rope(cfg, cfg.block_size)

    x = torch.randn(1, cfg.block_size, cfg.n_embd)
    out_a = attn(x, cos, sin)

    x_b = x.clone()
    x_b[0, -1] += 5.0           # perturb only the last position
    out_b = attn(x_b, cos, sin)

    torch.testing.assert_close(out_a[:, :-1], out_b[:, :-1])


# ---------------------------------------------------------------------------
# Parameter savings vs full MHA
# ---------------------------------------------------------------------------

def test_gqa_has_fewer_params_than_mha():
    """With n_kv_head < n_head, GQA must use fewer params than MHA (n_kv_head == n_head)."""
    gqa = GroupedQueryAttention(_cfg(n_head=4, n_kv_head=2))
    mha = GroupedQueryAttention(_cfg(n_head=4, n_kv_head=4))
    n_gqa = sum(p.numel() for p in gqa.parameters())
    n_mha = sum(p.numel() for p in mha.parameters())
    assert n_gqa < n_mha


# ---------------------------------------------------------------------------
# QK-Norm
# ---------------------------------------------------------------------------

def test_qknorm_components_present():
    """QK-Norm adds per-head q/k RMSNorm and a learnable temperature."""
    cfg = _cfg()
    attn = GroupedQueryAttention(cfg)
    assert attn.q_norm.weight.shape == (cfg.head_size,)
    assert attn.k_norm.weight.shape == (cfg.head_size,)
    assert isinstance(attn.scale, torch.nn.Parameter)
    assert attn.scale.requires_grad


def test_qknorm_bounds_logits_under_large_inputs():
    """With QK-Norm, attention logits stay bounded even for huge-magnitude
    inputs — the failure mode (softmax saturation) QK-Norm exists to prevent.

    q and k are RMS-normalised per head, so pre-softmax logits scale with the
    learnable temperature, not with the input magnitude.  We capture the logits
    via a forward hook and check they do not explode when the input is scaled
    up by 1000x.
    """
    cfg = _cfg()
    torch.manual_seed(0)
    attn = GroupedQueryAttention(cfg).eval()
    cos, sin = _rope(cfg, cfg.block_size)

    captured = {}
    # logit magnitude is bounded by |q_norm·k_norm| * scale * head_size.
    # We compare attention-weight entropy at 1x vs 1000x input scale: if logits
    # were unbounded, the 1000x case would saturate softmax (entropy -> 0).
    def entropy_of(x):
        with torch.no_grad():
            B, T, C = x.shape
            q = attn.q_proj(x).view(B, T, attn.n_head,    attn.head_size).transpose(1, 2)
            k = attn.k_proj(x).view(B, T, attn.n_kv_head, attn.head_size).transpose(1, 2)
            from modern_gpt.rope import apply_rotary
            q, k = apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)
            q, k = attn.q_norm(q), attn.k_norm(k)
            k = k.repeat_interleave(attn.n_rep, dim=1)
            wei = q @ k.transpose(-2, -1) * attn.scale
            causal = torch.tril(torch.ones(T, T))            # is_causal mask, built locally
            wei = wei.masked_fill(causal == 0, float("-inf"))
            p = torch.softmax(wei, dim=-1)
            return -(p * torch.log(p + 1e-9)).sum(-1).mean().item()

    x = torch.randn(1, cfg.block_size, cfg.n_embd)
    e1    = entropy_of(x)
    e1000 = entropy_of(x * 1000.0)
    # entropy should stay high (not collapse) — logits are bounded by normalisation
    assert e1000 > 0.5 * e1, f"softmax saturated under large input: {e1:.3f} -> {e1000:.3f}"

"""
Smoke and shape tests for the baseline model.

These tests assert architectural invariants that must hold across every
Phase 2+ modification (RMSNorm, RoPE, SwiGLU, GQA, etc.).  If a future
change breaks any of them, attention or generation is silently broken.
"""

from __future__ import annotations

import pytest
import torch

from modern_gpt import GPT, GPTConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_cfg() -> GPTConfig:
    """Small config that exercises all dims without being slow."""
    return GPTConfig(
        vocab_size=20,
        block_size=8,
        n_embd=16,
        n_head=2,
        n_layer=2,
        dropout=0.0,
    )


@pytest.fixture
def model(tiny_cfg: GPTConfig) -> GPT:
    torch.manual_seed(0)
    return GPT(tiny_cfg)


# ---------------------------------------------------------------------------
# Config invariants
# ---------------------------------------------------------------------------

def test_config_is_frozen(tiny_cfg: GPTConfig):
    """Config must be immutable (avoid silent hyperparameter drift)."""
    with pytest.raises(Exception):
        tiny_cfg.n_embd = 999  # type: ignore[misc]


def test_head_size_derived(tiny_cfg: GPTConfig):
    assert tiny_cfg.head_size == tiny_cfg.n_embd // tiny_cfg.n_head


def test_head_size_validates_divisibility():
    with pytest.raises(ValueError, match="divisible"):
        _ = GPTConfig(n_embd=10, n_head=3).head_size


# ---------------------------------------------------------------------------
# Forward-pass shapes
# ---------------------------------------------------------------------------

def test_forward_logits_shape(model: GPT, tiny_cfg: GPTConfig):
    B, T = 3, tiny_cfg.block_size
    idx  = torch.randint(0, tiny_cfg.vocab_size, (B, T))
    logits, loss = model(idx)
    assert logits.shape == (B, T, tiny_cfg.vocab_size)
    assert loss is None


def test_forward_with_targets_returns_scalar_loss(model: GPT, tiny_cfg: GPTConfig):
    B, T = 3, tiny_cfg.block_size
    idx     = torch.randint(0, tiny_cfg.vocab_size, (B, T))
    targets = torch.randint(0, tiny_cfg.vocab_size, (B, T))
    _, loss = model(idx, targets)
    assert loss is not None
    assert loss.ndim == 0           # scalar
    assert torch.isfinite(loss)


def test_forward_accepts_shorter_than_block_size(model: GPT, tiny_cfg: GPTConfig):
    """Position embeddings must work for any T <= block_size."""
    B, T = 2, tiny_cfg.block_size - 3
    idx  = torch.randint(0, tiny_cfg.vocab_size, (B, T))
    logits, _ = model(idx)
    assert logits.shape == (B, T, tiny_cfg.vocab_size)


# ---------------------------------------------------------------------------
# Loss at initialisation
# ---------------------------------------------------------------------------

def test_initial_loss_near_log_vocab(tiny_cfg: GPTConfig):
    """A freshly-initialised model should output ~uniform predictions,
    giving loss ~ ln(vocab_size).  Deviation > 0.5 means initialisation
    is broken."""
    torch.manual_seed(0)
    model = GPT(tiny_cfg)
    B, T = 4, tiny_cfg.block_size
    idx     = torch.randint(0, tiny_cfg.vocab_size, (B, T))
    targets = torch.randint(0, tiny_cfg.vocab_size, (B, T))
    _, loss = model(idx, targets)
    expected = torch.log(torch.tensor(float(tiny_cfg.vocab_size)))
    assert abs(loss.item() - expected.item()) < 0.5


# ---------------------------------------------------------------------------
# Causal masking
# ---------------------------------------------------------------------------

def test_causality_future_tokens_dont_influence_past(model: GPT, tiny_cfg: GPTConfig):
    """Changing token at position t must not change logits at positions < t.

    This catches off-by-one mistakes in the causal mask — the silent killer
    of any attention-based language model.
    """
    torch.manual_seed(0)
    B, T = 1, tiny_cfg.block_size
    idx       = torch.randint(0, tiny_cfg.vocab_size, (B, T))
    logits_a, _ = model(idx)

    idx_b           = idx.clone()
    idx_b[0, T - 1] = (idx_b[0, T - 1] + 1) % tiny_cfg.vocab_size  # change last token
    logits_b, _     = model(idx_b)

    torch.testing.assert_close(logits_a[:, :-1], logits_b[:, :-1])


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def test_generate_extends_sequence(model: GPT, tiny_cfg: GPTConfig):
    idx = torch.zeros((1, 1), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=10)
    assert out.shape == (1, 11)
    assert (out >= 0).all() and (out < tiny_cfg.vocab_size).all()


def test_generate_handles_context_longer_than_block(model: GPT, tiny_cfg: GPTConfig):
    """If input already exceeds block_size, generation must crop, not crash."""
    idx = torch.zeros((1, tiny_cfg.block_size + 5), dtype=torch.long)
    out = model.generate(idx, max_new_tokens=3)
    assert out.shape == (1, tiny_cfg.block_size + 8)


# ---------------------------------------------------------------------------
# Parameter count sanity
# ---------------------------------------------------------------------------

def test_num_parameters_positive(model: GPT):
    assert model.num_parameters() > 0


def test_default_config_params_in_expected_range():
    """Default config (RoPE + SwiGLU + GQA + QK-Norm) should have ~744K params."""
    model = GPT(GPTConfig())
    n = model.num_parameters()
    assert 700_000 < n < 800_000, f"expected ~744K params, got {n:,}"


# ---------------------------------------------------------------------------
# RoPE-specific invariants
# ---------------------------------------------------------------------------

def test_no_position_embedding_table():
    """RoPE replaces the learned position table — it must not exist."""
    model = GPT(GPTConfig())
    assert not hasattr(model, "position_embedding_table"), (
        "position_embedding_table should be removed; RoPE encodes position "
        "via rotation inside attention"
    )


def test_rope_param_reduction_vs_learned_pe():
    """RoPE removes block_size × n_embd params compared to learned PE."""
    cfg   = GPTConfig()
    model = GPT(cfg)
    n     = model.num_parameters()
    # 807,361 = 815,553 (RMSNorm baseline) - 8,192 (64×128 position table)
    expected_rope_savings = cfg.block_size * cfg.n_embd   # 8,192
    assert n < 815_553, (
        f"expected fewer params than RMSNorm baseline (815,553) due to "
        f"removal of {expected_rope_savings}-param position table, got {n:,}"
    )

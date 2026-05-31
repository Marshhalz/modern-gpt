"""
modern-gpt architecture — Phase 2.3 (SwiGLU).

Decoder-only Transformer with RMSNorm (Phase 2.1), Rotary Position
Embeddings (Phase 2.2), and a SwiGLU feed-forward network (Phase 2.3):

    tokens
      |
      token_embedding                        (learned)
      |                                      position encoded via RoPE rotation
      [ Block ] x n_layer                   inside attention, not at input
        |
        +-- RMSNorm -> MultiHeadAttention -> +   (q, k rotated by RoPE)
        |                                    \\
        +-----------------------------------+  residual
        |
        +-- RMSNorm -> SwiGLU             -> +   (gated GLU feed-forward)
        |                                    \\
        +-----------------------------------+  residual
      |
      RMSNorm (final)
      |
      lm_head: Linear(n_embd, vocab_size)
      |
      logits

See the roadmap in README.md for the planned modernisation sequence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig
from .ffn import SwiGLU
from .norm import RMSNorm
from .rope import RotaryEmbedding, apply_rotary


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class Head(nn.Module):
    """Single head of causal self-attention.

    Implements the canonical scaled dot-product attention from Vaswani et al.
    (2017), restricted to a causal (lower-triangular) mask so each position
    can only attend to itself and earlier positions.
    """

    def __init__(self, cfg: GPTConfig, head_size: int) -> None:
        super().__init__()
        self.head_size = head_size
        self.key   = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_size, bias=False)
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)),
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """`n_head` parallel attention heads, concatenated then mixed."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.heads   = nn.ModuleList(
            [Head(cfg, cfg.head_size) for _ in range(cfg.n_head)]
        )
        self.proj    = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.cat([h(x, cos, sin) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# ---------------------------------------------------------------------------
# Transformer block (pre-norm)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """Transformer block: attention then feed-forward, both with residuals.

    Uses pre-norm placement (RMSNorm before each sub-layer).  Pre-norm is
    more stable than the post-norm variant in the original 2017 paper and
    is now standard in every modern open-weight LLM.

    Normalisation is :class:`RMSNorm` (Zhang & Sennrich, 2019) and the
    feed-forward network is :class:`SwiGLU` (Shazeer, 2020) — the same choices
    as LLaMA, Mistral, Qwen, DeepSeek.  See ``benchmarks/rmsnorm.md`` and
    ``benchmarks/swiglu.md`` for the ablations against the original baseline.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.sa   = MultiHeadAttention(cfg)
        self.ffwd = SwiGLU(cfg)
        self.ln1  = RMSNorm(cfg.n_embd)
        self.ln2  = RMSNorm(cfg.n_embd)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.sa(self.ln1(x), cos, sin)
        x = x + self.ffwd(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """Decoder-only GPT for next-token prediction.

    Parameters
    ----------
    cfg : GPTConfig
        Architectural hyperparameters.

    Notes
    -----
    Weight initialisation follows GPT-2 conventions: Linear weights ~ N(0, 0.02),
    bias zero, Embedding weights ~ N(0, 0.02).  These choices materially affect
    training stability at depth.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.rope    = RotaryEmbedding(cfg.head_size, cfg.block_size)
        self.blocks  = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f    = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute logits (and loss, if targets given).

        Parameters
        ----------
        idx : LongTensor of shape (B, T)
            Input token IDs.
        targets : LongTensor of shape (B, T), optional
            Ground-truth next tokens.  When provided, the cross-entropy loss
            is returned alongside the logits.

        Returns
        -------
        logits : FloatTensor of shape (B, T, vocab_size)
        loss   : scalar tensor or None
        """
        B, T = idx.shape
        device = idx.device

        x        = self.token_embedding_table(idx)
        cos, sin = self.rope(T)
        cos      = cos.to(device)
        sin      = sin.to(device)
        for block in self.blocks:
            x = block(x, cos, sin)
        x      = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None

        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Autoregressively sample `max_new_tokens` tokens.

        The context is cropped to `block_size` tokens before each forward
        pass, matching the size of the precomputed RoPE tables.
        """
        for _ in range(max_new_tokens):
            idx_cond  = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits    = logits[:, -1, :]
            probs     = F.softmax(logits, dim=-1)
            idx_next  = torch.multinomial(probs, num_samples=1)
            idx       = torch.cat([idx, idx_next], dim=1)
        return idx

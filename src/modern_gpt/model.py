"""
modern-gpt architecture — Phase 2.6 (FlashAttention).

Decoder-only Transformer with RMSNorm (Phase 2.1), Rotary Position
Embeddings (Phase 2.2), a SwiGLU feed-forward network (Phase 2.3),
Grouped-Query Attention (Phase 2.4), QK-Norm cosine attention (Phase 2.5),
and FlashAttention via the fused SDPA kernel (Phase 2.6):

    tokens
      |
      token_embedding                        (learned)
      |                                      position encoded via RoPE rotation
      [ Block ] x n_layer                   inside attention, not at input
        |
        +-- RMSNorm -> GroupedQueryAttention -> +   (q, k rotated by RoPE)
        |                                       \\
        +--------------------------------------+  residual
        |
        +-- RMSNorm -> SwiGLU                -> +   (gated GLU feed-forward)
        |                                       \\
        +--------------------------------------+  residual
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

from .attention import GroupedQueryAttention
from .config import GPTConfig
from .ffn import SwiGLU
from .norm import RMSNorm
from .rope import RotaryEmbedding


# ---------------------------------------------------------------------------
# Transformer block (pre-norm)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """Transformer block: attention then feed-forward, both with residuals.

    Uses pre-norm placement (RMSNorm before each sub-layer).  Pre-norm is
    more stable than the post-norm variant in the original 2017 paper and
    is now standard in every modern open-weight LLM.

    Sub-layers are :class:`GroupedQueryAttention` (Ainslie et al., 2023),
    :class:`RMSNorm` (Zhang & Sennrich, 2019), and :class:`SwiGLU`
    (Shazeer, 2020) — the same choices as LLaMA, Mistral, Qwen, DeepSeek.
    See ``benchmarks/`` for the ablation behind each one.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.sa   = GroupedQueryAttention(cfg)
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

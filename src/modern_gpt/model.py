"""
modern-gpt baseline architecture.

Vanilla decoder-only Transformer (GPT-2 style):

    tokens
      |
      + token_embedding + position_embedding  (learned)
      |
      [ Block ] x n_layer
        |
        +-- LayerNorm -> MultiHeadAttention -> +
        |                                       \\
        +---------------------------------------+  residual
        |
        +-- LayerNorm -> FeedForward          -> +
        |                                       \\
        +---------------------------------------+  residual
      |
      LayerNorm (final)
      |
      lm_head: Linear(n_embd, vocab_size)
      |
      logits

This baseline (~800K params at default config) is the reference point that
every subsequent architectural change is benchmarked against.  See the
roadmap in README.md for the planned modernisation sequence.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# ---------------------------------------------------------------------------
# Feed-forward network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """Per-token MLP — the 'compute' half of a Transformer block.

    Uses the standard 4x expansion ratio from the original 2017 paper.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.ReLU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Transformer block (pre-norm)
# ---------------------------------------------------------------------------

class Block(nn.Module):
    """Transformer block: attention then feed-forward, both with residuals.

    Uses pre-norm placement (LayerNorm before each sub-layer), which is more
    stable than the post-norm variant in the original paper and is now
    standard in GPT-2, LLaMA, Mistral, and most production LLMs.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.sa   = MultiHeadAttention(cfg)
        self.ffwd = FeedForward(cfg)
        self.ln1  = nn.LayerNorm(cfg.n_embd)
        self.ln2  = nn.LayerNorm(cfg.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))
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
        self.token_embedding_table    = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.position_embedding_table = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks  = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f    = nn.LayerNorm(cfg.n_embd)
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

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x       = tok_emb + pos_emb
        x       = self.blocks(x)
        x       = self.ln_f(x)
        logits  = self.lm_head(x)

        if targets is None:
            return logits, None

        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Autoregressively sample `max_new_tokens` tokens.

        The input is cropped to the last `block_size` tokens before each
        forward pass, since the position embedding table has a fixed size.
        """
        for _ in range(max_new_tokens):
            idx_cond  = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits    = logits[:, -1, :]
            probs     = F.softmax(logits, dim=-1)
            idx_next  = torch.multinomial(probs, num_samples=1)
            idx       = torch.cat([idx, idx_next], dim=1)
        return idx

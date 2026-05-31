"""
Model configuration.

All architectural hyperparameters live in one immutable dataclass.  This
makes it trivial to swap configurations between experiments, serialise
the config alongside a checkpoint, or run hyperparameter sweeps without
touching the model code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GPTConfig:
    """Architectural hyperparameters for the modern-gpt model.

    The defaults define the **baseline** model — vanilla GPT-2 style on a
    character-level vocabulary.  Subsequent commits progressively replace
    components with modern equivalents (RMSNorm, RoPE, SwiGLU, GQA,
    FlashAttention) without changing this interface.
    """

    vocab_size: int   = 65        # Tiny Shakespeare character vocabulary
    block_size: int   = 64        # context window length (T)
    n_embd:     int   = 128       # residual stream width (C)
    n_head:     int   = 4         # number of attention heads
    n_layer:    int   = 4         # number of Transformer blocks
    dropout:    float = 0.1       # dropout probability

    @property
    def head_size(self) -> int:
        """Per-head feature width; derived, never set directly."""
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        return self.n_embd // self.n_head

    @property
    def ffn_hidden_dim(self) -> int:
        """Hidden dimension of the SwiGLU feed-forward network.

        A plain (2-matrix) FFN uses a 4x expansion.  SwiGLU adds a third
        projection (the gate), so to keep the parameter budget comparable the
        hidden dimension is scaled by 2/3 — i.e. ``2/3 * 4 * n_embd`` — then
        rounded up to a multiple of 8 for hardware alignment.  This is the
        convention used by LLaMA.

        At the default ``n_embd = 128`` this yields 344 (vs 512 for a 4x FFN).
        """
        target = int(2 / 3 * 4 * self.n_embd)
        multiple = 8
        return multiple * ((target + multiple - 1) // multiple)

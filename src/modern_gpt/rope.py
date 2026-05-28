"""
Rotary Position Embeddings (RoPE).

Replaces the learned position embedding table with a parameter-free rotation
applied directly to the query and key vectors inside attention.  This is the
position encoding used in LLaMA 1/2/3, Mistral, Qwen, DeepSeek, and almost
every modern open-weight LLM.

Key properties
--------------
* **Zero learned parameters** — cos/sin tables are a deterministic function
  of position and frequency; nothing is trained.
* **Relative position** — rotating q by m and k by n, the dot product gives
  the same result as rotating by (n − m) alone.  Absolute positions cancel,
  only distance survives.
* **Length extrapolation** — the rotation formula is defined for any integer
  position, not just those seen during training.  A learned position-embedding
  table has no row for position T+1; RoPE does.

References
----------
Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021).
    RoFormer: Enhanced Transformer with Rotary Position Embedding.
    https://arxiv.org/abs/2104.09864
"""

from __future__ import annotations

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split x into two equal halves along the last dim, return [-x2, x1].

    Implements the swap-and-negate step of the 2D rotation formula,
    vectorised across all frequency pairs simultaneously.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary position embedding to a query or key tensor.

    Vectorised form of the 2D rotation formula across all frequency pairs:

    .. math::
        x_{\\text{rot}} = x \\cdot \\cos + \\text{rotate\\_half}(x) \\cdot \\sin

    Parameters
    ----------
    x   : Tensor of shape (\\*, head_size)
    cos : Tensor of shape (T, head_size) — precomputed position cosines
    sin : Tensor of shape (T, head_size) — precomputed position sines

    cos and sin broadcast over any leading batch / head dimensions.
    """
    return x * cos + rotate_half(x) * sin


class RotaryEmbedding(nn.Module):
    r"""Precomputed RoPE cos/sin tables — no learnable parameters.

    Builds a ``(max_seq_len, head_size)`` table of cos and sin values at
    construction time and stores them as non-gradient buffers.  Each token
    position :math:`m` and frequency pair :math:`i` gets angle
    :math:`m \cdot \theta_i`, where:

    .. math::
        \theta_i = \text{base}^{-2i / \text{head\_size}},
        \quad i = 0, 1, \dots, \frac{\text{head\_size}}{2} - 1

    The logarithmically-spaced frequencies give a multi-scale representation:
    early pairs (high :math:`\theta`) rotate fast and capture fine-grained
    positional detail; late pairs (low :math:`\theta`) rotate slowly and
    capture long-range distance.

    Parameters
    ----------
    head_size   : int
        Dimension of each attention head.  Must be even.
    max_seq_len : int
        Maximum sequence length to precompute tables for.
    base        : float
        Frequency base; 10 000 follows Su et al. (2021) and the LLaMA
        convention.
    """

    def __init__(
        self,
        head_size: int,
        max_seq_len: int,
        base: float = 10_000.0,
    ) -> None:
        super().__init__()
        if head_size % 2 != 0:
            raise ValueError(f"head_size must be even for RoPE, got {head_size}")
        freqs  = 1.0 / (base ** (torch.arange(0, head_size, 2).float() / head_size))
        t      = torch.arange(max_seq_len).float()
        angles = torch.outer(t, freqs)               # (max_seq_len, head_size/2)
        emb    = torch.cat([angles, angles], dim=-1) # (max_seq_len, head_size)
        self.register_buffer("cos", emb.cos())
        self.register_buffer("sin", emb.sin())

    def forward(self, T: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cos/sin tables sliced to the current sequence length T."""
        return self.cos[:T], self.sin[:T]

    def extra_repr(self) -> str:
        return f"head_size={self.cos.shape[1]}, max_seq_len={self.cos.shape[0]}"

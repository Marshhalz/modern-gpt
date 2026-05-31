"""
Feed-forward networks.

Contains the gated SwiGLU feed-forward network used in LLaMA 1/2/3, Mistral,
Qwen, DeepSeek, and most modern open-weight LLMs.

References
----------
Shazeer, N. (2020).
    GLU Variants Improve Transformer.
    https://arxiv.org/abs/2002.05202
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


class SwiGLU(nn.Module):
    r"""SwiGLU gated feed-forward network (Shazeer, 2020).

    Replaces the original two-matrix ReLU FFN

    .. math::
        \text{FFN}(x) = W_\text{down}\,\operatorname{ReLU}(W_\text{up}\,x)

    with a three-matrix gated variant

    .. math::
        \text{SwiGLU}(x)
          = W_\text{down}\big[\operatorname{SiLU}(W_\text{gate}\,x)
            \odot (W_\text{up}\,x)\big]

    where :math:`\odot` is element-wise multiplication and
    :math:`\operatorname{SiLU}(z) = z \cdot \sigma(z)` is the smooth (Swish)
    activation.

    Two advantages over the ReLU FFN:

    * **No dead neurons.**  SiLU is smooth and non-zero for negative inputs,
      so gradients keep flowing where ReLU would zero them out.
    * **Gating.**  The gate branch learns, per hidden unit, how much of the
      content branch to pass, suppress, or amplify — a richer transformation
      than ReLU's binary cutoff.

    The hidden dimension is scaled by 2/3 relative to a 4x FFN (see
    :pyattr:`GPTConfig.ffn_hidden_dim`) so the three-matrix SwiGLU keeps a
    parameter budget comparable to the two-matrix ReLU FFN it replaces.

    Linear layers use ``bias=False``, following the LLaMA convention.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        hidden = cfg.ffn_hidden_dim
        self.w_gate  = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_up    = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w_down  = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.w_gate(x)) * self.w_up(x)
        return self.dropout(self.w_down(gated))

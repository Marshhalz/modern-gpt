"""
Normalisation layers.

This module contains the custom normalisation primitives used by the model.
Currently:
  - RMSNorm — Root Mean Square Layer Normalisation (Zhang & Sennrich, 2019),
              the normalisation used in LLaMA 1/2/3, Mistral, Mixtral, Qwen,
              DeepSeek, and most modern open-weight LLMs.

References
----------
Zhang, B. & Sennrich, R. (2019).
    Root Mean Square Layer Normalization.
    https://arxiv.org/abs/1910.07467
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    r"""Root Mean Square Layer Normalisation.

    Normalises the last dimension of the input by its RMS magnitude, then
    applies a learnable per-channel scale.  Unlike `nn.LayerNorm`, no mean
    is subtracted and no bias term is applied:

    .. math::
        \text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n} \sum_i x_i^2 + \epsilon}}
                           \cdot \gamma

    Compared to LayerNorm:
      - **No mean subtraction.**  Empirically the re-centering contributes
        negligibly to model quality (Zhang & Sennrich, 2019).
      - **No bias.**  Only the scale parameter γ is learnable.
      - **~n fewer parameters per layer** than LayerNorm.

    Performance note
    ----------------
    In pure-PyTorch eager mode this implementation is *slower* than the
    highly-optimised fused `nn.LayerNorm` kernel because it dispatches several
    individual ops.  This trade-off disappears with `torch.compile()` or by
    using PyTorch ≥ 2.4's built-in `nn.functional.rms_norm`.  We implement it
    explicitly here for clarity and to match what production LLM codebases
    (LLaMA, Mistral, etc.) write themselves.

    Parameters
    ----------
    dim : int
        Size of the feature dimension to normalise over.
    eps : float, default 1e-6
        Numerical stability constant added inside the square root.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (..., dim)
        ms      = x.pow(2).mean(-1, keepdim=True)         # mean-square along last dim
        inv_rms = torch.rsqrt(ms + self.eps)              # 1 / sqrt(ms + eps)
        return x * inv_rms * self.weight                  # normalise + scale

    def extra_repr(self) -> str:
        return f"dim={self.weight.shape[0]}, eps={self.eps}"

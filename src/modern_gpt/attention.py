"""
Attention.

Grouped-Query Attention (GQA) — a generalisation of multi-head attention in
which several query heads share a smaller number of key/value heads.  With
``n_kv_head == n_head`` it reduces to standard multi-head attention (MHA);
with ``n_kv_head == 1`` it is multi-query attention (MQA).

Why it matters: at inference the keys and values of every past token are
cached.  That KV cache — not the weights — dominates memory for long sequences
and large batches.  Reducing the number of K/V heads shrinks the cache
proportionally while keeping the (cheap, non-cached) query heads for
expressiveness.  LLaMA 2/3, Mistral, and Qwen all use GQA.

This module is also a vectorised rewrite of attention: all heads are computed
in a single batched tensor op rather than a Python loop over per-head modules.

References
----------
Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebron, F., &
    Sanghai, S. (2023). GQA: Training Generalized Multi-Query Transformer
    Models from Multi-Head Checkpoints. https://arxiv.org/abs/2305.13245
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig
from .rope import apply_rotary


class GroupedQueryAttention(nn.Module):
    r"""Causal Grouped-Query Attention with rotary position embeddings.

    ``n_head`` query heads attend, but only ``n_kv_head`` key/value heads are
    projected and stored; each K/V head is shared across ``n_rep = n_head //
    n_kv_head`` query heads via :func:`torch.Tensor.repeat_interleave`.

    The query projection keeps full width (``n_head * head_size``); the key and
    value projections are narrower (``n_kv_head * head_size``), which is where
    the parameter and KV-cache savings come from.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.n_head    = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_size = cfg.head_size
        self.n_rep     = cfg.n_rep

        self.q_proj   = nn.Linear(cfg.n_embd, cfg.n_head    * cfg.head_size, bias=False)
        self.k_proj   = nn.Linear(cfg.n_embd, cfg.n_kv_head * cfg.head_size, bias=False)
        self.v_proj   = nn.Linear(cfg.n_embd, cfg.n_kv_head * cfg.head_size, bias=False)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(cfg.block_size, cfg.block_size)),
        )
        self.attn_dropout  = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, C = x.shape

        # project, then split into heads: (B, n_heads, T, head_size)
        q = self.q_proj(x).view(B, T, self.n_head,    self.head_size).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_size).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_size).transpose(1, 2)

        # rotary position embedding on q and k (cos/sin broadcast over B and heads)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        # share each K/V head across its group of query heads
        k = k.repeat_interleave(self.n_rep, dim=1)   # (B, n_head, T, head_size)
        v = v.repeat_interleave(self.n_rep, dim=1)

        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.attn_dropout(wei)
        out = wei @ v                                 # (B, n_head, T, head_size)

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.out_proj(out))

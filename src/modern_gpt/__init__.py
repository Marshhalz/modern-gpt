"""modern-gpt — a small GPT progressively upgraded toward the LLaMA 3 architecture."""

from .attention import GroupedQueryAttention
from .config import GPTConfig
from .ffn import SwiGLU
from .model import (
    GPT,
    Block,
)
from .norm import RMSNorm
from .rope import RotaryEmbedding, apply_rotary, rotate_half

__version__ = "0.6.0"

__all__ = [
    "GPTConfig",
    "GPT",
    "Block",
    "GroupedQueryAttention",
    "SwiGLU",
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rotary",
    "rotate_half",
]

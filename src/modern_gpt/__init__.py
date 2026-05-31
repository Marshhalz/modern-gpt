"""modern-gpt — a small GPT progressively upgraded toward the LLaMA 3 architecture."""

from .config import GPTConfig
from .ffn import SwiGLU
from .model import (
    GPT,
    Block,
    Head,
    MultiHeadAttention,
)
from .norm import RMSNorm
from .rope import RotaryEmbedding, apply_rotary, rotate_half

__version__ = "0.4.0"

__all__ = [
    "GPTConfig",
    "GPT",
    "Block",
    "Head",
    "MultiHeadAttention",
    "SwiGLU",
    "RMSNorm",
    "RotaryEmbedding",
    "apply_rotary",
    "rotate_half",
]

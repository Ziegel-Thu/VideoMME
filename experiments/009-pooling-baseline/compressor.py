"""
009 Pooling Baseline 压缩模块

用 attention-weighted pooling 将每段 dense vision tokens 压缩为 K 个 tokens。
作为 cross-attention (006) 的下界 baseline。

用法:
    compressor = PoolingCompressor(K=8, dim=3584)
    compressed = compressor(dense_vision_tokens)  # (V, D) → (K, D)
"""

import torch
import torch.nn as nn


class PoolingCompressor(nn.Module):
    """Attention Pooling 压缩模块。

    每个输出 token 是 dense vision tokens 的加权平均，
    权重由 learned attention scores 决定。
    """

    def __init__(self, K=8, dim=3584, n_heads=8, n_layers=1, ffn_mult=4):
        super().__init__()
        self.K = K
        self.dim = dim
        # n_heads, n_layers, ffn_mult 保留接口兼容，但不使用
        self.score_proj = nn.Linear(dim, K)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, dense_vision):
        """
        Args:
            dense_vision: (V, D) — 单段 dense vision token embeddings

        Returns:
            compressed: (K, D) — 压缩后的 tokens
        """
        scores = self.score_proj(dense_vision)       # (V, K)
        weights = torch.softmax(scores, dim=0)       # (V, K)
        compressed = weights.T @ dense_vision        # (K, D)
        return self.out_norm(compressed)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# 保持和 006 相同的接口，train_compressor.py 直接 import
VoCoCompressor = PoolingCompressor


class InterSegmentAttention(nn.Module):
    """占位：保持 import 兼容。009 不使用段间 attention。"""

    def __init__(self, dim=3584, n_heads=8, n_layers=1, ffn_mult=4):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, tokens, segment_lengths):
        return list(torch.split(tokens, segment_lengths, dim=0))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

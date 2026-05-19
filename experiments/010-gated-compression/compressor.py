"""
010 Gated Compression 压缩模块

在 006 cross-attention compressor 基础上，输出后加 gate 层，
让模型学会对每个 compressed token 做重要性加权。

用法:
    compressor = GatedVoCoCompressor(K=8, dim=3584, n_layers=2)
    compressed = compressor(dense_vision_tokens)  # (V, D) → (K, D)
"""

import torch
import torch.nn as nn


class VoCoCompressorLayer(nn.Module):
    """单层: Cross-Attention + FFN，pre-norm。"""

    def __init__(self, dim, n_heads, ffn_mult=4):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, batch_first=False,
        )

        self.norm_ffn = nn.LayerNorm(dim)
        ffn_dim = dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, queries, kv):
        q = self.norm_q(queries)
        k = v = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(q, k, v)
        queries = queries + attn_out
        queries = queries + self.ffn(self.norm_ffn(queries))
        return queries


class GatedVoCoCompressor(nn.Module):
    """Cross-Attention 压缩 + Gate 加权。

    K 个 learnable queries 通过 cross-attention 提取信息后，
    gate 层对每个 token 做重要性加权。
    """

    def __init__(self, K=8, dim=3584, n_heads=8, n_layers=1, ffn_mult=4):
        super().__init__()
        self.K = K
        self.dim = dim
        self.n_layers = n_layers

        self.queries = nn.Parameter(torch.randn(K, dim) * 0.02)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(VoCoCompressorLayer(dim, n_heads, ffn_mult))

        self.out_norm = nn.LayerNorm(dim)

        # Gate: 每个 compressed token 一个标量权重
        self.gate = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, dense_vision):
        """
        Args:
            dense_vision: (V, D) — 单段 dense vision token embeddings

        Returns:
            compressed: (K, D) — gate 加权后的压缩 tokens
        """
        x = self.queries

        for layer in self.layers:
            x = layer(x, dense_vision)

        x = self.out_norm(x)
        g = self.gate(x)       # (K, 1)
        return x * g            # (K, D)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# 保持和 006 相同的接口
VoCoCompressor = GatedVoCoCompressor


class InterSegmentAttention(nn.Module):
    """占位：保持 import 兼容。"""

    def __init__(self, dim=3584, n_heads=8, n_layers=1, ffn_mult=4):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.layers = nn.ModuleList([
            InterSegmentAttentionLayer(dim, n_heads, ffn_mult)
            for _ in range(n_layers)
        ])
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, tokens, segment_lengths):
        x = tokens
        for layer in self.layers:
            x = layer(x)
        x = self.out_norm(x)
        return list(torch.split(x, segment_lengths, dim=0))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class InterSegmentAttentionLayer(nn.Module):
    """段间 token self-attention + FFN，pre-norm。"""

    def __init__(self, dim, n_heads, ffn_mult=4):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, batch_first=False,
        )
        self.norm_ffn = nn.LayerNorm(dim)
        ffn_dim = dim * ffn_mult
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, tokens):
        x = self.norm_attn(tokens)
        attn_out, _ = self.attn(x, x, x)
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(self.norm_ffn(tokens))
        return tokens

"""
VoCo Cross-Attention 压缩模块

将每段 ~400 个 dense vision tokens 压缩为 K 个 tokens。
架构: Perceiver / Q-Former 风格的 cross-attention。

用法:
    compressor = VoCoCompressor(K=8, dim=3584, n_layers=1)
    compressed = compressor(dense_vision_tokens)  # (V, D) → (K, D)
"""

import torch
import torch.nn as nn
import math


class VoCoCompressor(nn.Module):
    """Cross-Attention 压缩模块。

    K 个 learnable queries 通过 cross-attention 从 dense vision tokens 提取信息。
    每层: CrossAttn + FFN (pre-norm)。
    """

    def __init__(self, K=8, dim=3584, n_heads=8, n_layers=1, ffn_mult=4):
        super().__init__()
        self.K = K
        self.dim = dim
        self.n_layers = n_layers

        # Learnable queries
        self.queries = nn.Parameter(torch.randn(K, dim) * 0.02)

        # Cross-attention layers
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(VoCoCompressorLayer(dim, n_heads, ffn_mult))

        # 输出投影（可选，对齐到 LLM embedding space）
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, dense_vision):
        """
        Args:
            dense_vision: (V, D) — 单段 dense vision token embeddings

        Returns:
            compressed: (K, D) — 压缩后的 tokens
        """
        x = self.queries  # (K, D)

        for layer in self.layers:
            x = layer(x, dense_vision)

        x = self.out_norm(x)
        return x

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


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
        """
        Args:
            queries: (K, D)
            kv: (V, D)

        Returns:
            (K, D)
        """
        # Cross-Attention (pre-norm)
        q = self.norm_q(queries)
        k = v = self.norm_kv(kv)
        attn_out, _ = self.cross_attn(q, k, v)
        queries = queries + attn_out

        # FFN (pre-norm)
        queries = queries + self.ffn(self.norm_ffn(queries))
        return queries

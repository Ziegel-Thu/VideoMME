"""
011 Question-Conditioned 压缩模块

方式 C: queries = learnable + pool(Q_embed)
方式 B: 双层 cross-attention（Q→queries, queries→vision）

用法:
    compressor = ConditionedVoCoCompressor(K=8, dim=3584, n_layers=2, mode="C")
    compressed = compressor(dense_vision_tokens, q_embeds)  # (V, D), (Q_len, D) → (K, D)
"""

import torch
import torch.nn as nn


class ConditionedVoCoCompressorLayer(nn.Module):
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


class ConditionedVoCoCompressor(nn.Module):
    """Question-Conditioned Cross-Attention 压缩模块。

    mode="C": queries = learnable + linear(pool(Q))
    mode="B": 先 Q cross-attn 再 vision cross-attn
    """

    def __init__(self, K=8, dim=3584, n_heads=8, n_layers=1, ffn_mult=4, mode="C"):
        super().__init__()
        self.K = K
        self.dim = dim
        self.n_layers = n_layers
        self.mode = mode

        self.queries = nn.Parameter(torch.randn(K, dim) * 0.02)

        if mode == "C":
            self.q_proj = nn.Linear(dim, dim)
        elif mode == "B":
            self.q_cross_attn = ConditionedVoCoCompressorLayer(dim, n_heads, ffn_mult)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(ConditionedVoCoCompressorLayer(dim, n_heads, ffn_mult))

        self.out_norm = nn.LayerNorm(dim)

    def forward(self, dense_vision, q_embeds=None):
        """
        Args:
            dense_vision: (V, D)
            q_embeds: (Q_len, D) or None

        Returns:
            compressed: (K, D)
        """
        x = self.queries  # (K, D)

        if q_embeds is not None:
            if self.mode == "C":
                q_pooled = self.q_proj(q_embeds.mean(dim=0))  # (D,)
                x = x + q_pooled.unsqueeze(0)
            elif self.mode == "B":
                x = self.q_cross_attn(x, q_embeds)

        for layer in self.layers:
            x = layer(x, dense_vision)

        return self.out_norm(x)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# 保持接口兼容
VoCoCompressor = ConditionedVoCoCompressor


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
    def __init__(self, dim, n_heads, ffn_mult=4):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=False)
        self.norm_ffn = nn.LayerNorm(dim)
        ffn_dim = dim * ffn_mult
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Linear(ffn_dim, dim))

    def forward(self, tokens):
        x = self.norm_attn(tokens)
        attn_out, _ = self.attn(x, x, x)
        tokens = tokens + attn_out
        tokens = tokens + self.ffn(self.norm_ffn(tokens))
        return tokens

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


class InterSegmentAttention(nn.Module):
    """所有段压缩 token 之间的信息交互层。

    输入是已经按段压缩后的 token 拼接结果；输出按原段长度拆回 list，
    供后续仍按段计算 distillation loss。
    """

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
        """
        Args:
            tokens: (sum_i K_i, D) — 所有段的 compressed tokens 拼接
            segment_lengths: list[int] — 每段 compressed token 数

        Returns:
            list[(K_i, D)] — 交互后按段拆分的 tokens
        """
        if tokens.shape[0] != sum(segment_lengths):
            raise ValueError(
                f"tokens 长度 {tokens.shape[0]} 与 segment_lengths "
                f"之和 {sum(segment_lengths)} 不一致"
            )

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


class GatedVoCoCompressor(nn.Module):
    """Cross-Attention 压缩 + Gate 加权（010 gated 用）。"""

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
        self.gate = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())

    def forward(self, dense_vision):
        x = self.queries
        for layer in self.layers:
            x = layer(x, dense_vision)
        x = self.out_norm(x)
        g = self.gate(x)
        return x * g

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class PoolingCompressor(nn.Module):
    """Attention Pooling 压缩模块（009 pooling baseline 用）。"""

    def __init__(self, K=8, dim=3584, n_heads=8, n_layers=1, ffn_mult=4):
        super().__init__()
        self.K = K
        self.dim = dim
        self.score_proj = nn.Linear(dim, K)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, dense_vision):
        scores = self.score_proj(dense_vision)
        weights = torch.softmax(scores, dim=0)
        compressed = weights.T @ dense_vision
        return self.out_norm(compressed)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

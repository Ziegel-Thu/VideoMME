"""
012 Temporal Head — 方案 C: Cross-attention bin queries temporal head

16 个 learnable bin queries 注入 Q 信息后，
cross-attention 从所有 compressed tokens 提取时间证据。
"""

import torch
import torch.nn as nn


class BinQueryTemporalHead(nn.Module):
    """方案 C: Cross-attention bin queries temporal head。"""

    def __init__(self, dim=3584, num_bins=16, n_heads=8):
        super().__init__()
        self.dim = dim
        self.num_bins = num_bins

        self.bin_queries = nn.Parameter(torch.randn(num_bins, dim) * 0.02)
        self.q_proj = nn.Linear(dim, dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, batch_first=False,
        )
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
        )

        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, 1)

    def forward(self, compressed_flat, q_embeds):
        """
        Args:
            compressed_flat: (N_seg * K, D)
            q_embeds: (Q_len, D)

        Returns:
            bin_scores: (num_bins,) — logits
        """
        q_pooled = q_embeds.mean(dim=0)
        q_proj = self.q_proj(q_pooled)
        queries = self.bin_queries + q_proj.unsqueeze(0)

        q = self.norm_q(queries)
        kv = self.norm_kv(compressed_flat)
        attn_out, _ = self.cross_attn(q, kv, kv)
        queries = queries + attn_out

        queries = queries + self.ffn(self.norm_ffn(queries))

        out = self.out_norm(queries)
        bin_scores = self.out_proj(out).squeeze(-1)
        return bin_scores

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

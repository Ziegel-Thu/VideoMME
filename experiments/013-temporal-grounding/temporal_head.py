"""
013 Temporal Head 模块

两种方案：
- SegmentTemporalHead (方案 B): Q-conditioned per-segment 预测
- BinQueryTemporalHead (方案 C): Cross-attention bin queries

用法:
    head_b = SegmentTemporalHead(dim=3584, num_bins=16)
    head_c = BinQueryTemporalHead(dim=3584, num_bins=16)
    
    # 方案 B
    scores = head_b(compressed_segments, q_embeds)  # (num_bins,)
    
    # 方案 C
    scores = head_c(compressed_flat, q_embeds)  # (num_bins,)
"""

import torch
import torch.nn as nn


class SegmentTemporalHead(nn.Module):
    """方案 B: Q-conditioned per-segment temporal head。

    每段 compressed tokens mean pool 后，注入 Q 信息，
    per-segment 预测 evidence score，再映射到 time bins。
    """

    def __init__(self, dim=3584, num_bins=16, n_heads=8):
        super().__init__()
        self.dim = dim
        self.num_bins = num_bins

        # Q conditioning: cross-attention
        self.q_cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, batch_first=False,
        )
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        # Per-segment evidence predictor
        self.evidence_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
        )

    def forward(self, compressed_segments, q_embeds, n_segments):
        """
        Args:
            compressed_segments: list of (K, D) — 每段的 compressed tokens
            q_embeds: (Q_len, D) — question embeddings
            n_segments: int — 段数

        Returns:
            bin_scores: (num_bins,) — 每个 time bin 的 evidence score (logits)
        """
        # Step 1: 段级 pooling
        seg_features = torch.stack([seg.mean(dim=0) for seg in compressed_segments])
        # (N_seg, D)

        # Step 2: Q-conditioned cross-attention
        q = self.norm_q(seg_features)           # (N_seg, D)
        kv = self.norm_kv(q_embeds)             # (Q_len, D)
        conditioned, _ = self.q_cross_attn(q, kv, kv)
        conditioned = seg_features + conditioned  # residual, (N_seg, D)

        # Step 3: Per-segment evidence score
        seg_scores = self.evidence_mlp(conditioned).squeeze(-1)  # (N_seg,)

        # Step 4: 映射到 num_bins
        bin_scores = self._segments_to_bins(seg_scores, n_segments)
        return bin_scores

    def _segments_to_bins(self, seg_scores, n_segments):
        """将 per-segment scores 映射到固定 num_bins 个 time bins。"""
        # 每段均匀覆盖 duration，每个 bin 均匀覆盖 duration
        # 简单实现：线性插值
        if n_segments == self.num_bins:
            return seg_scores
        seg_scores_expanded = seg_scores.unsqueeze(0).unsqueeze(0)  # (1, 1, N_seg)
        bin_scores = nn.functional.interpolate(
            seg_scores_expanded, size=self.num_bins, mode='linear', align_corners=True,
        )
        return bin_scores.squeeze(0).squeeze(0)  # (num_bins,)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class BinQueryTemporalHead(nn.Module):
    """方案 C: Cross-attention bin queries temporal head。

    16 个 learnable bin queries 注入 Q 信息后，
    从所有 compressed tokens 里提取时间证据信息。
    """

    def __init__(self, dim=3584, num_bins=16, n_heads=8):
        super().__init__()
        self.dim = dim
        self.num_bins = num_bins

        # Learnable bin queries
        self.bin_queries = nn.Parameter(torch.randn(num_bins, dim) * 0.02)

        # Q injection
        self.q_proj = nn.Linear(dim, dim)

        # Cross-attention: bin queries attend to compressed tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, batch_first=False,
        )
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        # FFN
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
        )

        # Output
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, 1)

    def forward(self, compressed_flat, q_embeds):
        """
        Args:
            compressed_flat: (N_seg * K, D) — 所有段的 compressed tokens 拼接
            q_embeds: (Q_len, D) — question embeddings

        Returns:
            bin_scores: (num_bins,) — 每个 time bin 的 evidence score (logits)
        """
        # Step 1: Q injection into bin queries
        q_pooled = q_embeds.mean(dim=0)                    # (D,)
        q_proj = self.q_proj(q_pooled)                     # (D,)
        queries = self.bin_queries + q_proj.unsqueeze(0)    # (num_bins, D)

        # Step 2: Cross-attention
        q = self.norm_q(queries)                           # (num_bins, D)
        kv = self.norm_kv(compressed_flat)                 # (N_seg*K, D)
        attn_out, _ = self.cross_attn(q, kv, kv)
        queries = queries + attn_out                       # (num_bins, D)

        # Step 3: FFN
        queries = queries + self.ffn(self.norm_ffn(queries))

        # Step 4: Output
        out = self.out_norm(queries)                       # (num_bins, D)
        bin_scores = self.out_proj(out).squeeze(-1)        # (num_bins,)
        return bin_scores

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

"""
011 Temporal Head — 方案 B: Q-conditioned per-segment temporal head

每段 compressed tokens mean pool → Q conditioning (cross-attention) →
per-segment evidence score → 线性插值映射到 16 bins。
"""

import torch
import torch.nn as nn


class SegmentTemporalHead(nn.Module):
    """方案 B: Q-conditioned per-segment temporal head。"""

    def __init__(self, dim=3584, num_bins=16, n_heads=8):
        super().__init__()
        self.dim = dim
        self.num_bins = num_bins

        self.q_cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, batch_first=False,
        )
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

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
            q_embeds: (Q_len, D)
            n_segments: int

        Returns:
            bin_scores: (num_bins,) — logits
        """
        seg_features = torch.stack([seg.mean(dim=0) for seg in compressed_segments])

        q = self.norm_q(seg_features)
        kv = self.norm_kv(q_embeds)
        conditioned, _ = self.q_cross_attn(q, kv, kv)
        conditioned = seg_features + conditioned

        seg_scores = self.evidence_mlp(conditioned).squeeze(-1)

        return self._segments_to_bins(seg_scores, n_segments)

    def _segments_to_bins(self, seg_scores, n_segments):
        if n_segments == self.num_bins:
            return seg_scores
        seg_scores_expanded = seg_scores.unsqueeze(0).unsqueeze(0)
        bin_scores = nn.functional.interpolate(
            seg_scores_expanded, size=self.num_bins, mode='linear', align_corners=True,
        )
        return bin_scores.squeeze(0).squeeze(0)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

"""
Event Latent Tokens 模型核心模块

三层结构:
  1. SegmentLatentCompressor: 将每段 dense visual tokens 压缩为少量 latent tokens
  2. SegmentSelector: 根据问题选出 top-m 个最相关的段
  3. EventLatentSlots: 从 top-m 段中抽取 K 个 event latent tokens
  4. TemporalHead: 从 event latent tokens 解码时间段分布
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SegmentPositionalEncoding(nn.Module):
    """Segment-level 位置编码，让模型感知时间顺序"""

    def __init__(self, d_model: int, max_segments: int = 512):
        super().__init__()
        self.pos_embed = nn.Embedding(max_segments, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, M, D) → (B, M, D) with positional encoding added"""
        B, M, D = x.shape
        positions = torch.arange(M, device=x.device)
        return x + self.pos_embed(positions).unsqueeze(0)


class SegmentLatentCompressor(nn.Module):
    """Phase 1: 将每段 feature 压缩为 n_latent 个 latent tokens

    当输入为单个 pooled embedding (T=1) 时，退化为 MLP 投影;
    当输入为多个 patch tokens (T>1) 时，使用 cross-attention 压缩。
    """

    def __init__(self, d_model: int, n_latent: int = 2, n_heads: int = 8):
        super().__init__()
        self.n_latent = n_latent
        self.d_model = d_model

        # Learnable latent slots
        self.latent_slots = nn.Parameter(torch.randn(n_latent, d_model) * 0.02)

        # Cross-attention: latent queries attend to dense segment tokens
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

        # T=1 退化路径: 简单 MLP 投影
        self.single_token_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * n_latent),
        )

    def forward(self, segment_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            segment_features: (B, M, D) 或 (B, M, T, D)
        Returns:
            segment_latents: (B, M, n_latent, D)
        """
        if segment_features.dim() == 3:
            # T=1: 用 MLP 投影代替退化的 cross-attention
            B, M, D = segment_features.shape
            projected = self.single_token_proj(segment_features)  # (B, M, D*n_latent)
            return projected.reshape(B, M, self.n_latent, D)

        B, M, T, D = segment_features.shape
        x = segment_features.reshape(B * M, T, D)
        queries = self.latent_slots.unsqueeze(0).expand(B * M, -1, -1)
        latents, _ = self.cross_attn(queries, x, x)
        latents = self.norm(latents + queries)
        latents = self.norm2(latents + self.ffn(latents))
        return latents.reshape(B, M, self.n_latent, D)


class SegmentSelector(nn.Module):
    """Phase 2.1: 根据问题选出 top-m 个最相关段"""

    def __init__(self, d_model: int):
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.segment_proj = nn.Linear(d_model, d_model)
        self.score_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        segment_latents: torch.Tensor,
        question_embed: torch.Tensor,
        top_m: int = 8,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            segment_latents: (B, M, n_latent, D)
            question_embed: (B, D)
            top_m: 选出的段数
            mask: (B, M) — True 表示有效段，False 表示 padding

        Returns:
            selected_latents: (B, m, n_latent, D)
            selected_indices: (B, m)
            scores: (B, M)
        """
        B, M, n_latent, D = segment_latents.shape
        seg_repr = segment_latents.mean(dim=2)  # (B, M, D)

        q = self.query_proj(question_embed).unsqueeze(1).expand(-1, M, -1)
        s = self.segment_proj(seg_repr)
        combined = torch.cat([q, s], dim=-1)
        scores = self.score_head(combined).squeeze(-1)  # (B, M)

        # 将 padding 位置的分数设为 -inf
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        top_m = min(top_m, M)
        _, indices = scores.topk(top_m, dim=-1)  # (B, m)

        indices_expanded = indices.unsqueeze(2).unsqueeze(3).expand(-1, -1, n_latent, D)
        selected = torch.gather(segment_latents, 1, indices_expanded)

        return selected, indices, scores


class EventLatentSlots(nn.Module):
    """Phase 2.2: 从 top-m 段中抽取 K 个 event latent tokens"""

    def __init__(self, d_model: int, K: int = 8, n_heads: int = 8, n_layers: int = 2):
        super().__init__()
        self.K = K
        self.d_model = d_model
        self.event_slots = nn.Parameter(torch.randn(K, d_model) * 0.02)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                "norm1": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                ),
                "norm2": nn.LayerNorm(d_model),
            }))

    def forward(self, selected_latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            selected_latents: (B, m, n_latent, D)
        Returns:
            event_tokens: (B, K, D)
        """
        B, m, n_latent, D = selected_latents.shape
        kv = selected_latents.reshape(B, m * n_latent, D)
        queries = self.event_slots.unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            attn_out, _ = layer["cross_attn"](queries, kv, kv)
            queries = layer["norm1"](queries + attn_out)
            queries = layer["norm2"](queries + layer["ffn"](queries))

        return queries  # (B, K, D)


class TemporalHead(nn.Module):
    """Phase 2.3: 从 event tokens 解码 segment-level 时间分布

    使用 event-to-segment cross-attention 而非简单 dot product，
    以支持多段 grounding 和更强的表达力。
    """

    def __init__(self, d_model: int, n_heads: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.score_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        event_tokens: torch.Tensor,
        all_segment_latents: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            event_tokens: (B, K, D)
            all_segment_latents: (B, M, n_latent, D)
            mask: (B, M) — True 表示有效段

        Returns:
            segment_logits: (B, M) — 每个 segment 是证据段的 logit
        """
        B, M, n_latent, D = all_segment_latents.shape
        seg_repr = all_segment_latents.mean(dim=2)  # (B, M, D)

        # Segment queries attend to event tokens
        # key_padding_mask 对 event tokens 不需要（都有效）
        attended, _ = self.cross_attn(seg_repr, event_tokens, event_tokens)
        attended = self.norm(seg_repr + attended)  # (B, M, D)

        logits = self.score_proj(attended).squeeze(-1)  # (B, M)

        # Mask padding
        if mask is not None:
            logits = logits.masked_fill(~mask, float("-inf"))

        return logits


class EventLatentModel(nn.Module):
    """完整模型：Segment Compress → Select → Event Slots → Temporal Head"""

    def __init__(
        self,
        d_model: int = 3584,
        n_latent: int = 2,
        top_m: int = 8,
        K: int = 8,
        n_heads: int = 8,
        max_segments: int = 512,
    ):
        super().__init__()
        self.top_m = top_m
        self.d_model = d_model

        self.pos_encoding = SegmentPositionalEncoding(d_model, max_segments)
        self.compressor = SegmentLatentCompressor(d_model, n_latent, n_heads)
        self.selector = SegmentSelector(d_model)
        self.event_slots = EventLatentSlots(d_model, K, n_heads)
        self.temporal_head = TemporalHead(d_model, n_heads)

    def forward(
        self,
        segment_features: torch.Tensor,
        question_embed: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            segment_features: (B, M, D) — 每段 1 个 pooled embedding
            question_embed: (B, D) — 问题 embedding
            mask: (B, M) — True 表示有效段，None 表示全部有效

        Returns:
            dict with segment_logits, selected_indices, selector_scores,
                      event_tokens, segment_latents
        """
        # 添加位置编码
        segment_features = self.pos_encoding(segment_features)

        # Phase 1: Compress
        segment_latents = self.compressor(segment_features)

        # Phase 2.1: Select top-m
        selected_latents, selected_indices, selector_scores = self.selector(
            segment_latents, question_embed, self.top_m, mask
        )

        # Phase 2.2: Extract event tokens
        event_tokens = self.event_slots(selected_latents)

        # Phase 2.3: Temporal grounding
        segment_logits = self.temporal_head(event_tokens, segment_latents, mask)

        return {
            "segment_logits": segment_logits,
            "selected_indices": selected_indices,
            "selector_scores": selector_scores,
            "event_tokens": event_tokens,
            "segment_latents": segment_latents,
        }

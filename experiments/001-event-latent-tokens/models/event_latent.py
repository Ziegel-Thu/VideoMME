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


class SegmentLatentCompressor(nn.Module):
    """Phase 1: 将每段 dense feature 压缩为 n_latent 个 latent tokens (VoCo-style)"""

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

    def forward(self, segment_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            segment_features: (B, M, D) — M 段，每段 1 个 pooled embedding
                              或 (B, M, T, D) — M 段，每段 T 个 patch tokens
        Returns:
            segment_latents: (B, M, n_latent, D) — 每段 n_latent 个 latent tokens
        """
        if segment_features.dim() == 3:
            # 每段 1 个 embedding → 扩展为 (B, M, 1, D)
            segment_features = segment_features.unsqueeze(2)

        B, M, T, D = segment_features.shape

        # 展开为 (B*M, T, D)
        x = segment_features.reshape(B * M, T, D)

        # Latent queries: (B*M, n_latent, D)
        queries = self.latent_slots.unsqueeze(0).expand(B * M, -1, -1)

        # Cross-attention
        latents, _ = self.cross_attn(queries, x, x)
        latents = self.norm(latents + queries)
        latents = self.norm2(latents + self.ffn(latents))

        # Reshape: (B, M, n_latent, D)
        return latents.reshape(B, M, self.n_latent, D)


class SegmentSelector(nn.Module):
    """Phase 2.1: 根据问题选出 top-m 个最相关段"""

    def __init__(self, d_model: int):
        super().__init__()
        # 问题-段落相关性打分
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            segment_latents: (B, M, n_latent, D) — 每段的 latent tokens
            question_embed: (B, D) — 问题的 pooled embedding
            top_m: 选出的段数

        Returns:
            selected_latents: (B, m, n_latent, D) — 选出的段的 latent tokens
            selected_indices: (B, m) — 选出的段索引
            scores: (B, M) — 所有段的分数
        """
        B, M, n_latent, D = segment_latents.shape

        # 每段取 mean pool 作为段表示
        seg_repr = segment_latents.mean(dim=2)  # (B, M, D)

        q = self.query_proj(question_embed).unsqueeze(1).expand(-1, M, -1)  # (B, M, D)
        s = self.segment_proj(seg_repr)  # (B, M, D)

        combined = torch.cat([q, s], dim=-1)  # (B, M, 2D)
        scores = self.score_head(combined).squeeze(-1)  # (B, M)

        # Top-m selection
        top_m = min(top_m, M)
        _, indices = scores.topk(top_m, dim=-1)  # (B, m)

        # Gather selected latents
        indices_expanded = indices.unsqueeze(2).unsqueeze(3).expand(-1, -1, n_latent, D)
        selected = torch.gather(segment_latents, 1, indices_expanded)  # (B, m, n_latent, D)

        return selected, indices, scores


class EventLatentSlots(nn.Module):
    """Phase 2.2: 从 top-m 段中抽取 K 个 event latent tokens"""

    def __init__(self, d_model: int, K: int = 8, n_heads: int = 8, n_layers: int = 2):
        super().__init__()
        self.K = K
        self.d_model = d_model

        # Learnable event slots
        self.event_slots = nn.Parameter(torch.randn(K, d_model) * 0.02)

        # Multi-layer cross-attention
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
            selected_latents: (B, m, n_latent, D) — 选中段的 latent tokens

        Returns:
            event_tokens: (B, K, D) — K 个 event latent tokens
        """
        B, m, n_latent, D = selected_latents.shape

        # Flatten: (B, m * n_latent, D)
        kv = selected_latents.reshape(B, m * n_latent, D)

        # Event queries: (B, K, D)
        queries = self.event_slots.unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            attn_out, _ = layer["cross_attn"](queries, kv, kv)
            queries = layer["norm1"](queries + attn_out)
            queries = layer["norm2"](queries + layer["ffn"](queries))

        return queries  # (B, K, D)


class TemporalHead(nn.Module):
    """Phase 2.3: 从 event tokens 解码 segment-level 时间分布"""

    def __init__(self, d_model: int, max_segments: int = 256):
        super().__init__()
        self.max_segments = max_segments

        # 从 event tokens pool 出时间表示，再预测每个 segment 的概率
        self.pool = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        # 与 segment latents 做 dot product 得到分布
        self.temp_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        event_tokens: torch.Tensor,
        all_segment_latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            event_tokens: (B, K, D) — event latent tokens
            all_segment_latents: (B, M, n_latent, D) — 所有段的 latent tokens

        Returns:
            segment_probs: (B, M) — 每个 segment 是证据段的概率
        """
        # Pool event tokens: (B, D)
        event_repr = self.pool(event_tokens.mean(dim=1))  # (B, D)
        event_repr = self.temp_proj(event_repr)  # (B, D)

        # 每段取 mean: (B, M, D)
        seg_repr = all_segment_latents.mean(dim=2)

        # Dot product: (B, M)
        logits = torch.einsum("bd,bmd->bm", event_repr, seg_repr)

        return logits  # raw logits, apply sigmoid/softmax outside


class EventLatentModel(nn.Module):
    """完整模型：Segment Compress → Select → Event Slots → Temporal Head"""

    def __init__(
        self,
        d_model: int = 3584,
        n_latent: int = 2,
        top_m: int = 8,
        K: int = 8,
        n_heads: int = 8,
    ):
        super().__init__()
        self.top_m = top_m
        self.d_model = d_model

        self.compressor = SegmentLatentCompressor(d_model, n_latent, n_heads)
        self.selector = SegmentSelector(d_model)
        self.event_slots = EventLatentSlots(d_model, K, n_heads)
        self.temporal_head = TemporalHead(d_model)

    def forward(
        self,
        segment_features: torch.Tensor,
        question_embed: torch.Tensor,
    ) -> dict:
        """
        Args:
            segment_features: (B, M, D) — 每段 1 个 pooled embedding
            question_embed: (B, D) — 问题 embedding

        Returns:
            dict with:
                segment_logits: (B, M) — temporal grounding 分布
                selected_indices: (B, m) — 选中的段
                selector_scores: (B, M) — selector 分数
                event_tokens: (B, K, D) — event latent tokens
                segment_latents: (B, M, n_latent, D) — 所有段 latent
        """
        # Phase 1: Compress
        segment_latents = self.compressor(segment_features)

        # Phase 2.1: Select top-m
        selected_latents, selected_indices, selector_scores = self.selector(
            segment_latents, question_embed, self.top_m
        )

        # Phase 2.2: Extract event tokens
        event_tokens = self.event_slots(selected_latents)

        # Phase 2.3: Temporal grounding
        segment_logits = self.temporal_head(event_tokens, segment_latents)

        return {
            "segment_logits": segment_logits,
            "selected_indices": selected_indices,
            "selector_scores": selector_scores,
            "event_tokens": event_tokens,
            "segment_latents": segment_latents,
        }

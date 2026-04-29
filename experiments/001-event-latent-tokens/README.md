# 001 — Event Latent Tokens

## 目标

长视频 temporal grounding：给定长视频 + 问题，输出关键证据时间段。

核心思想：将视频压缩为 Segment Latent Tokens → 选段 → 提取 Event Latent Tokens → 解码时间分布。
训练时用 bottleneck attention 强迫答案依赖 latent token，从而更 grounded。

## 文件结构

```
001-event-latent-tokens/
├── README.md               # 本文件
├── models/
│   ├── __init__.py
│   └── event_latent.py     # 核心模型（4 个模块 + 位置编码）
└── validate_phase2.py      # Phase 2 端到端验证脚本
```

## 模型架构

```
输入: segment_features (B, M, D) + question_embed (B, D) + mask (B, M)
  │
  ├─ SegmentPositionalEncoding  — 加段级位置编码
  │
  ├─ SegmentLatentCompressor    — 每段 → n_latent 个 latent tokens
  │   T=1 时用 MLP 投影，T>1 时用 cross-attention
  │
  ├─ SegmentSelector            — 根据问题选 top-m 段
  │   padding 位置设为 -inf，topk 选段
  │
  ├─ EventLatentSlots           — K 个 learnable slots 对 top-m 段做 cross-attn
  │   2 层 cross-attention + FFN
  │
  └─ TemporalHead              — segment-to-event cross-attention → 逐段 logit
      padding 位置 mask 为 -inf
```

## 超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| d_model | 3584 | Qwen2.5-VL-7B hidden dim |
| n_latent | 2 | 每段压缩为几个 latent token |
| top_m | 8 | 选出的段数 |
| K | 8 | event latent token 数 |
| n_heads | 8 | 注意力头数 |
| max_segments | 512 | 位置编码最大段数 |

当前参数量：**~650M**（主要由 d_model=3584 导致的 FFN 参数）

## 数据

| 数据集 | 条数 | 用途 |
|--------|------|------|
| temporal_evidence.jsonl | 12,997 | 时间证据标签 (L_temp) |
| selector_trajectories.jsonl | 181,453 | 选段器监督 (L_sel) |
| features/*.pt | 5,662 | 预提取的 segment embedding |

特征格式：每个 `.pt` 文件包含 `{embeddings: (M, 3584), timestamps: [(s,e),...], segment_stride: 4.0}`

## 运行

```bash
# 环境
conda activate video

# Phase 2 验证
python validate_phase2.py

# 特征提取（已完成）
python ../../scripts/extract_features.py --max_videos 10  # 测试
python ../../scripts/extract_features.py                   # 全量
```

## 已知问题与后续计划

1. **特征提取质量**：当前用完整 VLM mean pool（混了 text token），后续应改为只取 vision token
2. **topk 不可微**：selector 无法从 temporal loss 端到端学习，需用 Seeker-173K 轨迹单独监督
3. **question embedding**：当前用伪随机 embedding，Phase 3 需接入真实文本编码
4. **参数量偏大**：可加投影层 `D=3584 → d=768` 降维

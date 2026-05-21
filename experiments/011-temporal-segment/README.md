# 011-temporal-segment: 方案 B — Q-conditioned 序列 Temporal Head

## 概述

Stage 3 temporal grounding 的方案 B。在已训好的 compressor（006 B-2L）输出上训练 temporal head，预测视频中与问题相关的证据时间段。

每段 compressed tokens mean pool 后注入 Q 信息（cross-attention），per-segment 预测 evidence score，再线性插值映射到 16 bins。

### 架构

```
视频 → Vision Encoder → Dense Tokens → Compressor (冻结) → Compressed (N_seg, K, D)
                                                                ↓
                                                     mean pool → (N_seg, D)
Question → Tokenizer → Embed → q_embeds (Q_len, D)     ↓
                                    ↓              cross-attention
                                    └──────────→  conditioned (N_seg, D)
                                                        ↓
                                              LayerNorm → MLP → (N_seg,)
                                                        ↓
                                              interpolate → (16,) bin logits
                                                        ↓
                                              BCE loss with GT bins
```

### 核心模块：SegmentTemporalHead（54.6M 参数）
- `q_cross_attn`: MultiheadAttention(3584, 8 heads) — Q conditioning
- `evidence_mlp`: LayerNorm → Linear(3584, 896) → ReLU → Linear(896, 1) — per-segment 预测
- `_segments_to_bins`: F.interpolate(linear) — N_seg → 16 bins 映射

### 与 012 方案 C 的区别
| | 011 方案 B | 012 方案 C |
|---|---|---|
| 核心思路 | per-segment 预测 → 插值到 bins | learnable bin queries attend to all tokens |
| 输入粒度 | 每段独立 pool | 全局 cross-attention |
| 时间对齐 | 隐式（插值） | 显式（attention weights 学习） |
| 参数量 | 54.6M | 70.8M |

### 优点
- 每段有独立 evidence 预测，保留时间分辨率
- 段数灵活，自动映射到固定 bins
- 参数量较小

## 数据

- **temporal_evidence.jsonl**: 12,997 条（Q + A + evidence_segments [t_s, t_e]）
- **stgr 视频**: Open-o3-Video STGR 数据集，6005 个视频（temporal_grounding + plm）
- **匹配率**: 7147/12997 (55%) 样本有对应视频
- **GT bins 统计**: 平均 4.84/16 个 positive bins，0% 全零

### Blob 路径
```
/mnt/default/bottleneck/data/parsed/temporal_evidence.jsonl
/mnt/default/bottleneck/data/stgr_videos/temporal_grounding/videos/
/mnt/default/bottleneck/data/stgr_videos/plm/videos/
/mnt/default/bottleneck/checkpoints/006_compressor/110k_B2L_ep2.pt  # 冻结 compressor
```

## 小规模测试结果（100 样本）

| lr | Epoch 1 | Epoch 3 | Epoch 5 | 趋势 |
|----|---------|---------|---------|------|
| 1e-4 | 0.939 | 0.944 | - | ❌ 震荡不收敛 |
| **1e-5** | **0.884** | **0.873** | **0.864** | **✅ 稳定下降** |

**结论**: lr=1e-5 合适，1e-4 过大导致震荡。

## 训练配置

```bash
python train_temporal.py \
  --compressor_checkpoint /path/to/110k_B2L_ep2.pt \
  --data_path /path/to/temporal_evidence.jsonl \
  --video_dirs /path/to/stgr_videos/temporal_grounding/videos,/path/to/stgr_videos/plm/videos \
  --model_path /path/to/Qwen2.5-VL-7B-Instruct \
  --head_type B --num_bins 16 --num_frames 16 --frames_per_segment 2 \
  --lr 1e-5 --epochs 10
```

## 依赖
- 006 B-2L ep2 checkpoint（冻结 compressor + inter_segment）
- temporal_evidence.jsonl（blob 上已有）
- stgr 视频数据（已上传 blob）

## 文件
```
011-temporal-segment/
├── README.md             # 本文件
├── temporal_head.py      # SegmentTemporalHead 模块
├── train_temporal.py     # 训练脚本 (--head_type B)
├── compressor.py         # VoCoCompressor（同 006，冻结加载）
├── model.py              # 视频 → dense tokens 工具
└── amlt_train.yaml       # amlt 集群训练配置
```

## 评测指标（待实现）
- tIoU (temporal Intersection over Union)
- Recall@0.5 / Recall@0.7
- mAP

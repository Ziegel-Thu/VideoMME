# 012-temporal-binquery: 方案 C — Cross-attention Bin Queries Temporal Head

## 概述

Stage 3 temporal grounding 的方案 C。16 个 learnable bin queries 注入 Q 信息后，通过 cross-attention 从所有 compressed tokens 提取时间证据。

与 011 方案 B 的关键区别：方案 B 逐段预测再插值，方案 C 用全局 attention 直接预测每个 time bin 的证据分数。

### 架构

```
视频 → Vision Encoder → Dense Tokens → Compressor (冻结) → Compressed (N_seg×K, D) 拼接
                                                                    ↓
16 learnable bin queries (16, D)                                    ↓
        ↓                                                           ↓
Question → Embed → mean pool → q_proj → (D,)                       ↓
        ↓                                                           ↓
    queries = bin_queries + q_proj         ← Q injection            ↓
        ↓                                                           ↓
    Cross-attention: Q=queries(16,D), K/V=compressed(N_seg×K, D) ───┘
        ↓  residual + FFN
        ↓  LayerNorm → Linear(D, 1) → (16,) bin logits
        ↓  BCE loss with GT bins
```

### 核心模块：BinQueryTemporalHead（70.8M 参数）
- `bin_queries`: nn.Parameter(16, 3584) — learnable bin queries
- `q_proj`: Linear(3584, 3584) — question 投影
- `cross_attn`: MultiheadAttention(3584, 8 heads) — bin queries 从 compressed tokens 提取信息
- `ffn`: Linear(3584, 896) → ReLU → Linear(896, 3584) — feed-forward
- `out_proj`: Linear(3584, 1) — 每个 bin 输出一个 evidence score

### 与 011 方案 B 的区别
| | 011 方案 B | 012 方案 C |
|---|---|---|
| 核心思路 | per-segment 预测 → 插值到 bins | learnable bin queries attend to all tokens |
| 输入粒度 | 每段独立 pool | 全局 cross-attention |
| 时间对齐 | 隐式（插值） | 显式（attention weights 学习） |
| 参数量 | 54.6M | 70.8M |

### 优点
- bin queries 通过 attention weights 自然学会对齐到正确时间位置
- Q conditioning 让 bin queries 针对问题去找证据
- 和 compressor 的 cross-attention 架构一致，设计连贯

## 数据

同 011，详见 011 README。
- **temporal_evidence.jsonl**: 12,997 条，7147 匹配视频
- **stgr 视频**: 6005 个（已上传 blob）

## 小规模测试结果（100 样本）

| lr | Epoch 1 | Epoch 2 | Epoch 3 | 趋势 |
|----|---------|---------|---------|------|
| **1e-4** | **1.084** | **0.937** | **0.928** | **✅ 明显下降** |

**结论**: lr=1e-4 可行，初始 loss 较高但快速下降。

## 训练配置

```bash
python train_temporal.py \
  --compressor_checkpoint /path/to/110k_B2L_ep2.pt \
  --data_path /path/to/temporal_evidence.jsonl \
  --video_dirs /path/to/stgr_videos/temporal_grounding/videos,/path/to/stgr_videos/plm/videos \
  --model_path /path/to/Qwen2.5-VL-7B-Instruct \
  --head_type C --num_bins 16 --num_frames 16 --frames_per_segment 2 \
  --lr 1e-4 --epochs 10
```

## 依赖
- 006 B-2L ep2 checkpoint（冻结 compressor + inter_segment）
- temporal_evidence.jsonl（blob 上已有）
- stgr 视频数据（已上传 blob）

## 文件
```
012-temporal-binquery/
├── README.md             # 本文件
├── temporal_head.py      # BinQueryTemporalHead 模块
├── train_temporal.py     # 训练脚本 (--head_type C)
├── compressor.py         # VoCoCompressor（同 006，冻结加载）
├── model.py              # 视频 → dense tokens 工具
└── amlt_train.yaml       # amlt 集群训练配置
```

## 评测指标（待实现）
- tIoU (temporal Intersection over Union)
- Recall@0.5 / Recall@0.7
- mAP

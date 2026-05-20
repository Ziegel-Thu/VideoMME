# 011-temporal-segment: 方案 B — Q-conditioned 序列 Temporal Head

## 概述

每段 compressed tokens mean pool 后注入 Q 信息，per-segment 预测 evidence score，再映射到 16 bins。

### 架构

```
Compressed Tokens (N_seg, K, D)
    ↓  mean pool → (N_seg, D)
    ↓  + q_pooled (cross-attention)
    ↓  LayerNorm → MLP → sigmoid → (N_seg,)
    ↓  interpolate → (16,) bin scores
    ↓  BCE loss with GT bins
```

### 优点
- 每段有独立 evidence 预测，保留时间分辨率
- 段数灵活，自动映射到固定 bins

## 依赖
- 006 B-2L best checkpoint（冻结 compressor）
- temporal_evidence.jsonl（blob 上已有）
- stgr 视频数据（Open-o3-Video）

## 文件
```
011-temporal-segment/
├── README.md
├── temporal_head.py      # SegmentTemporalHead
├── train_temporal.py     # 训练脚本 (--head_type B)
├── compressor.py         # VoCoCompressor（同 006，冻结）
├── model.py              # 视频处理工具
```

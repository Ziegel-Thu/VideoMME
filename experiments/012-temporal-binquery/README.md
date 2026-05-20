# 012-temporal-binquery: 方案 C — Cross-attention Bin Queries Temporal Head

## 概述

16 个 learnable bin queries 注入 Q 信息后，通过 cross-attention 从所有 compressed tokens 提取时间证据。

### 架构

```
Compressed Tokens (N_seg × K, D) 拼接
    ↓
16 bin queries (16, D) + q_pooled
    ↓  Cross-attention: Q=bins, K/V=compressed
    ↓  FFN
    ↓  Linear → sigmoid → (16,) bin scores
    ↓  BCE loss with GT bins
```

### 优点
- bin queries 通过 attention weights 自动对齐到正确时间位置
- Q conditioning 让 bin queries 针对问题去找证据
- 和 compressor 的 cross-attention 架构一致

## 依赖
- 006 B-2L best checkpoint（冻结 compressor）
- temporal_evidence.jsonl（blob 上已有）
- stgr 视频数据（Open-o3-Video）

## 文件
```
012-temporal-binquery/
├── README.md
├── temporal_head.py      # BinQueryTemporalHead
├── train_temporal.py     # 训练脚本 (--head_type C)
├── compressor.py         # VoCoCompressor（同 006，冻结）
├── model.py              # 视频处理工具
```

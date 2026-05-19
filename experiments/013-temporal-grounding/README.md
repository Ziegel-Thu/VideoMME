# 013-temporal-grounding: Temporal Head 时间证据定位

## 概述

在 compressor 输出的 compressed tokens 上训练 Temporal Head，预测视频中与问题相关的证据时间段 [t_s, t_e]。这是项目的 Stage 3，对应导师方案中的 L_temp。

### 架构

```
Compressed Tokens (K×N_seg, D)
        ↓  Pool → (D,)
        ↓  TemporalHead (MLP)
        ↓  (num_bins,) logits
        ↓  BCE loss with bin labels
```

### 与 Stage 2 的关系

- Stage 2（006）：训练 compressor 压缩视觉信息 → MCQ 准确率
- Stage 3（013）：在 compressor 基础上训练 temporal head → 时间定位
- Temporal head 的输入来自已训好的 compressor 的 hidden states

---

## 数据

| 数据集 | 条数 | 内容 |
|--------|------|------|
| temporal_evidence.jsonl | 12,997 | Q + A + evidence_segments [t_s, t_e] |
| temporal_evidence_short.jsonl | ~6K | 0-30s 子集 |
| temporal_evidence_long.jsonl | ~6K | 30-60s 子集 |

blob 路径：`/mnt/default/bottleneck/data/parsed/temporal_evidence*.jsonl`

### 已知问题
- N=4 帧时 88% 样本的 temporal labels 全零（证据被采样跳过）
- 用 16-bin 离散化，与帧数解耦
- 零长度 evidence 自动扩展 ±1s

---

## Temporal Head

```python
class TemporalHead(nn.Module):
    def __init__(self, hidden_dim=3584, num_bins=16):
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_bins),
        )
    
    def forward(self, pooled_hidden):
        return self.proj(pooled_hidden)  # (B, num_bins)
```

### 评测指标
- tIoU (temporal Intersection over Union)
- Recall@0.5
- mAP

---

## 文件结构

```
013-temporal-grounding/
├── README.md
├── plan.md
├── train_temporal.py     # Temporal head 训练（从 004 移植）
├── compressor.py         # VoCoCompressor（同 006）
├── eval_compressor.py    # MCQ 评测工具
└── model.py              # 模型工具
```

---

## 实验记录

### 002/003 A40 参照
- 002 temporal head (K=32, N=4): tIoU=0.3148, Recall@0.5=27%
- 问题：N=4 时 temporal labels 大量全零

### 013 待做（基于 006 compressor）
- 用 compressor hidden states 替代 bottleneck latent tokens 做 temporal head 输入
- 更多帧数（1fps × 30s = 30帧 → 15段），temporal labels 覆盖率更高

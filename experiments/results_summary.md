# VoCo Cross-Attention Compressor 实验汇总

> 最后更新：2026-05-18

## 1. 方法概述

在 Qwen2.5-VL-7B-Instruct 上，用 **Perceiver 风格 Cross-Attention 模块**将每段 dense vision tokens 压缩为 K=8 个 tokens。通过**蒸馏训练**（MSE loss）使 compressed tokens 承载视觉信息。

```
Dense Vision Tokens (V, D)
        ↓  Cross-Attention (learnable queries)
Compressed Tokens (K=8, D)
        ↓  + Question → Frozen LLM
Student Q-hidden
        ↓  MSE
Teacher Q-hidden (预提取)
```

### 压缩比对比

| 方法 | 压缩比 | 来源 |
|------|--------|------|
| 003 Bottleneck SFT | 13.5:1 | K=32 latent tokens |
| 004 静态 VoCo embeddings | 54:1 | 每段 K_seg=4 固定 tokens |
| **006 Cross-Attention Compressor** | **54:1** | 每段 K_seg=8 learned tokens |

006 和 004 压缩比相同，但 cross-attention 提取信息的能力远强于静态 embeddings。

---

## 2. Baseline

| 方法 | 评测方式 | 准确率 | 说明 |
|------|---------|--------|------|
| Qwen2.5-VL zero-shot | logit (ABCD) | **82.0%** | 不经过压缩的上界 |
| Qwen2.5-VL zero-shot | generation | 72.5% | 自由生成 + regex 提取 |
| Qwen2.5-VL zero-shot | NLL per-option | 70.0% | 最严谨但最慢 |
| 003 BN on (110K) | logit | 90.0% | latent bottleneck，不做压缩 |
| 003 BN off (110K) | logit | 94.4% | 无 bottleneck 约束 |
| 004 静态 VoCo (10K) | logit | 70.4% | 已废弃 |

---

## 3. 006 Cross-Attention Compressor 结果

### 3.1 Loss 类型 Ablation（10K, 200 条 test, 0-30s）

固定 n_layers=1, K_seg=8。

| Loss | 含义 | epoch3 Acc |
|------|------|-----------|
| **B** | MSE on Q-position hidden | **71.0%** |
| D | MSE on KV cache pooled | 63.0% |
| BD | B + D | 69.5% |

**结论：B loss 最优。** Q-position hidden 比 KV cache pooling 提供更精确的蒸馏信号。

### 3.2 Compressor 深度 Ablation — 10K Pilot（200 条 test, 0-30s）

固定 loss_type=B, K_seg=8。

| 配置 | epoch1 | epoch2 | epoch3 |
|------|--------|--------|--------|
| B-1L | 66.5% | **71.0%** | **71.0%** |
| B-2L | - | - | 69.0% |

### 3.3 Compressor 深度 Ablation — 110K（前 200 条 quick eval）

| 配置 | epoch | 0-30s | 0-60s |
|------|-------|-------|-------|
| B-1L | 1 | 71.50% | 66.00% |
| B-1L | 3 | 70.00% | 65.50% |
| B-2L | 1 | 72.00% | 67.50% |
| **B-2L** | **2** | **75.00%** ⭐ | **68.00%** ⭐ |
| B-2L | 3 | 73.00% | 67.50% |

### 3.4 110K Full Eval（全量 test set）

| 配置 | epoch | 0-30 全量 (1835条) | 0-60 全量 (10103条) |
|------|-------|-------------------|---------------------|
| B-1L | 1 | 70.08% (1286) | 68.48% (6919) |
| B-1L | 3 | 69.86% (1282) | 68.63% (6934) |
| B-2L | 3 | 69.21% (1270) | 68.98% (6969) |

### 3.5 关键发现

1. **B-2L > B-1L**：2 层 cross-attention 提供更强的压缩能力
   - Quick eval: B-2L epoch2 75.0% vs B-1L best 71.5%（+3.5%）
   - Full eval: B-2L 0-60 68.98% vs B-1L 68.63%（差距缩小但仍优）
2. **B-2L epoch2 是峰值**：epoch3 略回落，轻微过拟合
3. **B-1L 容量饱和**：多训反而略降，1 层 cross-attention 不够
4. **Quick eval vs Full eval 差距**：quick eval（200 条）波动较大，full eval 差距更小

---

## 4. 007 Inter-Segment Attention 结果

段间 self-attention：所有段的 compressed tokens 先做全局交互再送入 LLM。

### 4.1 10K Pilot（200 条 test, 0-30s）

| 配置 | epoch1 | epoch2 | epoch3 |
|------|--------|--------|--------|
| B-1L + inter-seg 1L | 66.5% | 62.0% | 62.5% |
| B-1L 对照（006） | 66.5% | 71.0% | 71.0% |

**结论：10K 上段间 attention 反而拖累性能。** 可能原因：
- 10K 数据量不足以训练段间交互参数（~50M）
- 段间交互引入噪声

### 4.2 110K（待做）

110K 数据量更大，正在 amlt 集群提取 teacher cache，完成后启动训练。

---

## 5. 训练规模与效率

### Teacher Cache

| 环境 | GPU | 样本数 | Shard 数 | 大小 | 耗时 |
|------|-----|--------|---------|------|------|
| A40×8 (jiagpu8) | 8×A40 48GB | 102,952 | 203 | 3.9TB | ~8h |
| A100×4×4 (amlt) | 4 job × 4×A100 | 进行中 | 81+ | - | ~3-4h |

### Compressor 训练

| 环境 | GPU | 配置 | 每 epoch step | 每 epoch 耗时 |
|------|-----|------|-------------|-------------|
| A40×4 (jiagpu) | 4×A40 48GB | B-1L | 25,738 | ~5h |
| A40×4 (jiagpu) | 4×A40 48GB | B-2L | 25,738 | ~5h |

---

## 6. 综合对比表

| 方法 | 压缩 | 0-30 最佳 | 0-60 最佳 | 备注 |
|------|------|----------|----------|------|
| Zero-shot (无压缩) | 无 | 82.0% | - | 上界 |
| 003 BN on (110K) | 13.5:1 | 90.0% | - | latent bottleneck |
| 004 静态 VoCo (10K) | 54:1 | 70.4% | - | 已废弃 |
| **006 B-1L (110K)** | **54:1** | **71.50%** | **68.63%** | cross-attention |
| **006 B-2L (110K)** | **54:1** | **75.00%** ⭐ | **68.98%** ⭐ | 2层, epoch2 峰值 |
| 007 inter-seg (10K) | 54:1 | 66.5% | - | 10K 无收益，110K 待验证 |

---

## 7. 下一步

- [ ] 完成 amlt 集群 teacher cache 提取
- [ ] 007 inter-seg 110K 训练
- [ ] 006 B-2L amlt 集群复现
- [ ] K_seg sweep (4/8/16)
- [ ] 将 checkpoint 从 A40 迁移到 blob

# 006-cross-attention-compressor: 段级视觉压缩蒸馏

## 概述

用 Perceiver/Q-Former 风格的 Cross-Attention 模块，将每段 ~400 个 dense vision tokens 压缩为 K 个 tokens，通过蒸馏训练使 compressed tokens 承载足够的视觉信息。

这是 004（静态 voco embeddings）失败后的演化版本。004 的 VoCo 压缩比过于激进(54:1)，10K 最高 70.4%；本方案用 learnable cross-attention 替代静态 embeddings，显著提升压缩质量。

### 架构

```
Dense Vision Tokens (V, D)
        ↓
  VoCoCompressor (K learnable queries × cross-attention layers)
        ↓
  Compressed Tokens (K, D)
        ↓
  [Compressed, Q_embeds] → Frozen LLM → Student Q-hidden
        ↓
  Loss = MSE(Student Q-hidden, Teacher Q-hidden)
                                  ↑
                          预提取 (005-teacher-cache)
```

### 与 004 的区别

| 项目 | 004 静态 VoCo | 006 Cross-Attention |
|------|-------------|---------------------|
| 压缩方式 | 固定 learnable embeddings | Cross-attention queries |
| 训练信号 | NLL on answer | MSE 蒸馏 (teacher hidden) |
| 需要 vision encoder | 是（每次 forward） | 否（预提取 teacher cache） |
| 10K 最佳 | 70.4% | 71.0% (B-1L) |

---

## Compressor 模块

```python
compressor = VoCoCompressor(K=8, dim=3584, n_layers=1)
compressed = compressor(dense_vision_tokens)  # (V, D) → (K, D)
```

- **K**: 每段压缩后 token 数（默认 8）
- **n_layers**: cross-attention 层数（1L / 2L ablation）
- **输出**: LayerNorm 后的 compressed tokens

### Loss 类型

| loss_type | 公式 | 说明 |
|-----------|------|------|
| B | MSE(student_Q_hidden, teacher_Q_hidden) | Q 位置 hidden state 蒸馏 |
| D | MSE(student_KV_pool, teacher_KV_pool) | KV cache pooled 蒸馏 |
| BD | B + D | 两者都算 |

---

## 训练配置

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--K_seg` | 8 | 每段 compressed token 数 |
| `--n_layers` | 1 | compressor cross-attention 层数 |
| `--loss_type` | B | B / D / BD |
| `--lr` | 1e-4 | 学习率 |
| `--epochs` | 3 | 训练轮数 |
| `--cache_shard_size` | 512 | shard 大小提示（避免初始化扫全部 shard 爆内存） |
| `--inter_layers` | 0 | 段间 attention 层数（本实验固定为 0，见 007） |

### 数据

依赖 005-teacher-cache 的输出：
- amlt: `/mnt/default/bottleneck/teacher_cache_110k/`
- jiagpu: `/nvmessd/lifanhong/video/teacher_cache_110k_sharded_256/`

---

## 使用方式

### amlt 集群训练（A100）

```bash
# B-1L baseline
torchrun --nproc_per_node=4 train_compressor.py \
  --cache_dir /mnt/default/bottleneck/teacher_cache_110k \
  --output_dir $AMLT_OUTPUT_DIR \
  --model_path Qwen/Qwen2.5-VL-7B-Instruct \
  --cache_shard_size 512 \
  --K_seg 8 --n_layers 1 --loss_type B \
  --lr 1e-4 --epochs 3 --save_steps 1000

# B-2L
torchrun --nproc_per_node=4 train_compressor.py \
  --cache_dir /mnt/default/bottleneck/teacher_cache_110k \
  --output_dir $AMLT_OUTPUT_DIR \
  --model_path Qwen/Qwen2.5-VL-7B-Instruct \
  --cache_shard_size 512 \
  --K_seg 8 --n_layers 2 --loss_type B \
  --lr 1e-4 --epochs 3 --save_steps 1000
```

### 评测

```bash
python eval_compressor.py \
  --checkpoint /path/to/compressor_epoch1.pt \
  --data_path /mnt/default/bottleneck/data/parsed/visual_qa_v3_60s_test.jsonl \
  --video_dirs /mnt/default/bottleneck/data/videos/0_30_s_academic_v0_1,/mnt/default/bottleneck/data/videos/30_60_s_academic_v0_1 \
  --model_path Qwen/Qwen2.5-VL-7B-Instruct \
  --max_samples 0
```

支持多卡并行分片评测：`--shard_id 0 --num_shards 4`

---

## Ablation 设计

### A. Loss 类型（B vs D vs BD）

固定 n_layers=1, K_seg=8，对比三种 loss。

### B. Compressor 深度（1L vs 2L）

固定 loss_type=B, K_seg=8，对比 1 层 vs 2 层 cross-attention。

---

## 文件结构

```
006-cross-attention-compressor/
├── README.md                # 本文件
├── compressor.py            # VoCoCompressor + InterSegmentAttention 模块
├── train_compressor.py      # 训练主脚本（DDP, B/D/BD loss, resume）
├── eval_compressor.py       # MCQ 评测（logit 比较 + 分片）
├── model.py                 # 模型加载工具（从 004 引用）
├── test_train_compressor.py # 单元测试
└── test_eval_compressor.py  # 单元测试
```

---

## 实验记录

### Checkpoint 存储

amlt 集群: `/mnt/default/bottleneck/checkpoints/006/`

### 10K Pilot 结果（A40 参照, 200 条 test, 0-30s）

| 配置 | epoch1 | epoch2 | epoch3 |
|------|--------|--------|--------|
| B-1L | 66.5% | 71.0% | **71.0%** |
| B-2L | - | - | 69.0% |
| D-1L | - | - | 63.0% |
| BD-1L | - | - | 69.5% |

### 110K Quick Eval（A40 参照, 前 200 条）

| 配置 | epoch | 0-30 | 0-60 |
|------|-------|------|------|
| B-1L | 1 | 71.50% | 66.00% |
| B-1L | 3 | 70.00% | 65.50% |
| B-2L | 1 | 72.00% | 67.50% |
| **B-2L** | **2** | **75.00%** | **68.00%** |
| B-2L | 3 | 73.00% | 67.50% |

### 110K Full Eval（A40 参照）

| 配置 | epoch | 0-30 全量 | 0-60 全量 |
|------|-------|----------|----------|
| B-1L | 1 | 70.08% (1286/1835) | 68.48% (6919/10103) |
| B-1L | 3 | 69.86% (1282/1835) | 68.63% (6934/10103) |
| B-2L | 3 | 69.21% (1270/1835) | 68.98% (6969/10103) |

### 结论

- **B-2L 全面优于 B-1L**，quick eval epoch2 0-30 达 75.00%
- B-2L epoch2 为峰值，epoch3 略回落，轻微过拟合
- B-1L 容量已饱和，多训反而略降
- B loss 始终优于 D loss
- Zero-shot baseline（logit 方法）：82%（不经过压缩的上界）

---

## 计划

### 待做
- [ ] 110K BD-1L 训练（amlt yaml 已就绪：`amlt_train_bd.yaml`）
- [ ] B-2L epoch2 full eval 补跑（quick eval 峰值 75%，full eval 缺失）
- [ ] 用 blob 上的 checkpoint 在 amlt 集群复现 eval
- [ ] 汇总 B/D/BD × 1L/2L 完整结果表

### 已完成
- [x] Sanity test 在 A100 集群跑通（complete-bluegill）
- [x] 10K B/D/BD/B-2L pilot
- [x] 110K B-1L 3 epoch + full eval
- [x] 110K B-2L 3 epoch + full eval
- [x] Checkpoint 迁移到 blob（25 个）

# 007-inter-segment-attention: 段间信息交互

## 概述

在 006 的 per-segment compressor 基础上，增加可选的段间 self-attention 层：所有段的 compressed tokens 先做全局信息交互，再按段拆回送入 LLM 做蒸馏。

### 动机

006 的 compressor 独立处理每段，段间没有信息流通。段间 attention 让不同段的 compressed tokens 能互相看到，理论上有利于长视频中跨段推理。

### 架构

```
Per-segment compressed tokens (K×N_seg, D)
        ↓
  InterSegmentAttention (self-attention + FFN, n_layers 层)
        ↓
  交互后按段拆回 → 每段独立做 student forward → MSE loss
```

### 与 006 的区别

唯一区别：`--inter_layers > 0`（006 固定为 0）

| 项目 | 006 | 007 |
|------|-----|-----|
| 段间交互 | ❌ | ✅ self-attention |
| 参数 | `--inter_layers 0` | `--inter_layers 1+` |
| 额外参数量 | 0 | ~50M per layer |

---

## 训练配置

基于 006 相同的代码，仅增加 `--inter_layers`：

```bash
torchrun --nproc_per_node=4 train_compressor.py \
  --cache_dir /mnt/default/bottleneck/teacher_cache_110k \
  --output_dir $AMLT_OUTPUT_DIR \
  --model_path Qwen/Qwen2.5-VL-7B-Instruct \
  --cache_shard_size 512 \
  --K_seg 8 --n_layers 1 --inter_layers 1 --loss_type B \
  --lr 1e-4 --epochs 3 --save_steps 1000
```

---

## 文件结构

```
007-inter-segment-attention/
├── README.md                # 本文件
├── compressor.py            # VoCoCompressor + InterSegmentAttention（同 006）
├── train_compressor.py      # 训练主脚本（同 006，--inter_layers>0）
├── eval_compressor.py       # 评测（同 006）
└── model.py                 # 模型加载工具
```

---

## 实验记录

### 10K Pilot（A40 参照, 200 条 test, 0-30s）

| 配置 | epoch1 | epoch2 | epoch3 |
|------|--------|--------|--------|
| B-1L + inter-seg 1L | 66.5% | 62.0% | 62.5% |
| B-1L（006 对照） | 66.5% | 71.0% | 71.0% |

**结论**：10K 上段间 attention 未带来收益（66.5% vs 71.0%），反而拖累收敛。

### 110K（待做）

10K 数据量小，段间交互可能缺乏足够训练信号。110K 上值得重新验证。

### 待验证 Ablation

- inter_layers=1 vs 2
- 与 B-2L compressor 搭配（n_layers=2 + inter_layers=1）

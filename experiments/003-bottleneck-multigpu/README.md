# 003-bottleneck-multigpu: 多卡 DDP 训练

## 概述

从 002 的单卡训练升级为多卡 DDP，支持更多帧数和更大等效 batch size。

### 相对 002 的改进

| 项目 | 002 (单卡) | 003 (多卡) |
|------|-----------|-----------|
| GPU | 1× A100 80GB | 4× A100 (可调) |
| 帧数 | N=4 (固定) | N=8/16 (可配置) |
| Batch size | 1 | 等效 4×grad_accum |
| Gradient checkpointing | 无 | 支持 |
| 数据划分 | train/val | train/val/test |
| 代码结构 | 单文件 | 模块化 (model/data/train) |
| Hook 安全 | 无保护 | try/finally 防泄漏 |
| Hidden state 获取 | output_hidden_states=True (全层) | Hook 只捕获最后层 latent 位置 |

---

## 文件结构

```
003-bottleneck-multigpu/
├── README.md           # 本文件
├── model.py            # 模型加载、mask 构造、checkpoint 工具
├── data.py             # Dataset、collate_fn、TemporalHead
├── train_ddp.py        # DDP 训练主脚本
└── amlt.yaml           # 集群提交配置
```

### model.py 核心函数

| 函数 | 功能 |
|------|------|
| `setup_model_and_tokenizer(K, device, gradient_checkpointing)` | 加载模型，DDP wrap 前调用 |
| `build_bottleneck_mask(input_ids, latent_token_ids)` | LIVR-style 4D attention mask |
| `install_bottleneck_hooks(layers, mask_4d)` | 在 decoder layers 上注册 mask hook |
| `get_language_model_layers(model)` | 获取 decoder layers（兼容 DDP） |
| `save_checkpoint(model, ...)` / `load_checkpoint(model, ...)` | 自动 unwrap DDP |

### data.py 核心组件

| 组件 | 功能 |
|------|------|
| `VideoQADataset` | 加载 jsonl + 视频索引，__getitem__ 返回原始 dict |
| `collate_fn` | 视频采帧 → processor → 构造 answer-only labels |
| `TemporalHead` | MLP (D→D/2→num_bins)，从 latent hidden 预测时间段 |

---

## 使用方式

### 本地单卡（调试）

```bash
python train_ddp.py \
    --data_path /path/to/visual_qa_v2.jsonl \
    --video_dirs /path/to/videos/stgr,/path/to/videos/llava-video \
    --output_dir outputs_debug \
    --K 32 --num_frames 4 --epochs 1 \
    --bottleneck --gradient_checkpointing \
    --max_samples 10 --overfit
```

### 本地多卡

```bash
torchrun --nproc_per_node=4 train_ddp.py \
    --data_path /path/to/visual_qa_v2.jsonl \
    --video_dirs /path/to/videos/stgr,/path/to/videos/llava-video \
    --output_dir outputs_4gpu \
    --K 32 --num_frames 8 --epochs 10 --lr 2e-5 \
    --bottleneck --gradient_checkpointing \
    --grad_accum 2
```

### amlt 集群提交

```bash
amlt run amlt.yaml stage2-exp -d "Stage 2 K=32 8帧 4卡训练"
```

amlt.yaml 中预配置了两个 job:
- `stage2-k32-8f`: 8 帧，保守配置
- `stage2-k32-16f`: 16 帧，更多视觉信息

---

## 参数说明

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--num_frames` | 8 | 每视频采样帧数，需配合 `--gradient_checkpointing` |
| `--K` | 32 | latent token 数量 |
| `--grad_accum` | 2 | 梯度累积步数，等效 bs = GPU数 × grad_accum |
| `--lr` | 2e-5 | 学习率 |
| `--bottleneck` | off | 启用 LIVR bottleneck mask |
| `--gradient_checkpointing` | off | 启用梯度检查点（N≥8 时必须） |
| `--n_val` | 500 | 验证集大小上限 |
| `--n_test` | 1000 | 测试集大小上限 |
| `--resume_from` | None | 从 checkpoint 继续训练 |

### 帧数与显存估计（per GPU, A100 80GB）

| 帧数 | 无 GC | 有 GC | 建议 |
|------|------|------|------|
| N=4 | ~73GB | ~45GB | 单卡可跑 |
| N=8 | OOM | ~55GB | ✅ 推荐 |
| N=16 | OOM | ~70GB | 可尝试 |
| N=32 | OOM | ~80GB+ | 可能 OOM |

---

## 数据划分

数据使用固定 seed=42 划分为 train / val / test:
- **test set** 的 indices 保存到 `output_dir/test_indices.json`，供后续评测复现
- val/test 大小可通过 `--n_val` / `--n_test` 调整

---

## 从 002 迁移

从 002 的 best_model.pt 继续训练:

```bash
--resume_from /path/to/002/outputs_k32_full/best_model.pt
```

Checkpoint 格式完全兼容（LoRA state_dict + latent embeddings）。

---

## DDP 实现要点

1. **Hook-based mask 注入**: 每个 rank 独立构造 mask 并注入各自的 batch，通过 `try/finally` 确保 hook 不泄漏
2. **Embedding grad hook**: 所有 rank 独立注册，在 DDP all-reduce 前将非 latent 行梯度归零
3. **Hidden state 捕获**: 用 forward hook 只捕获最后层 latent 位置（不使用 `output_hidden_states=True`，节省显存）
4. **Gradient accumulation**: 用 `model.no_sync()` 跳过中间步的 all-reduce
5. **Checkpoint**: 只 rank 0 保存，保存时自动 unwrap DDP

### 绝对不能改的地方（继承自 002）

1. `attn_implementation="eager"` — SDPA 会忽略自定义 mask
2. Bottleneck mask 的 LIVR 规则 — 非 vision/latent token 不能看 vision
3. Token 顺序 `[vision][question][latent][answer]`
4. Embedding grad hook — 只更新 latent 行
5. Labels 构造 — 只在 assistant 回复部分算 loss

---

## Blob 存储位置

同 002，数据在 Azure Blob Storage:
```
存储账户: shuwangmain
容器:     zhengshurui
根路径:   video_project/
```

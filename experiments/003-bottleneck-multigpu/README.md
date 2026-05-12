# 003-bottleneck-multigpu: 多卡 DDP 训练

## 概述

从 002 的单卡训练升级为多卡 DDP，支持更多帧数和更大等效 batch size。
当前阶段：Stage 2 Bottleneck SFT（MCQ 数据训 L_ans）。

### 训练路线全景

```
Stage 1: VoCo Attention Distillation（004-voco-distillation）       ← 独立实验，测试中
Stage 2: Bottleneck SFT（MCQ 数据训 L_ans）                          ← 集群训练中
Stage 3: Temporal Head（temporal 数据训 L_temp）                      ← 代码已写好
Stage 4: 长视频 VoCo 压缩 + Segment Selector                        ← 后续
```

### 相对 002 的改进

| 项目 | 002 (单卡) | 003 (多卡) |
|------|-----------|-----------|
| GPU | 1× A100 80GB | 4× A100 (可调) |
| 帧数 | N=4 (固定) | N=4/12 (可配置) |
| Batch size | 1 | 等效 GPU数×grad_accum |
| Gradient checkpointing | 无 | 支持 |
| 数据划分 | train/val | train/val/test |
| 数据量 | 19K (47%重复) | 110K (0-60s, 去重) |
| 代码结构 | 单文件 | 模块化 (model/data/train) |
| Hook 安全 | 无保护 | try/finally 防泄漏 |
| LR scheduler | 无 | 可选 cosine |
| Checkpoint | epoch 级 | epoch + step 级 (防抢占) |

---

## 文件结构

```
003-bottleneck-multigpu/
├── README.md              # 本文件
├── SETUP_GUIDE.md         # 外部机器部署指南
├── model.py               # 模型加载、mask 构造、checkpoint 工具
├── data.py                # Dataset、collate_fn
├── train_ddp.py           # Stage 2: Bottleneck SFT 训练
├── train_temporal.py      # Stage 3: Temporal Head 训练
├── train_distill.py       # Stage 1: VoCo-style 蒸馏训练
├── sanity_check.py        # Bottleneck 有效性验证
├── eval_mcq.py            # MCQ 准确率评测
├── amlt.yaml              # 集群提交配置
└── .amltignore            # amlt 上传排除规则
```

---

## 实验记录

### 综合对比表（500 条 test set, 0-30s）

| 模型 | 数据量 | BN/VoCo | Epoch | Test Acc | Test Loss | 备注 |
|------|--------|---------|-------|----------|-----------|------|
| 原始 Qwen2.5-VL (zero-shot) | - | - | - | 36.60% | - | baseline |
| 003 BN on | 20K | BN K=32 N=4 | 1 | 81.50%* | 0.1429 | *200条 |
| 003 BN on | 20K | BN K=32 N=4 | 2 | 83.50%* | 0.1639 | *200条 |
| 003 BN on | 20K | BN K=32 N=4 | 3 | 82.00%* | 0.2985 | *200条, 过拟合 |
| 003 BN off | 20K | 无BN K=32 N=4 | 1 | 92.50%* | 0.0539 | *200条 |
| **003 BN on** | **110K** | **BN K=32 N=4** | **1** | **90.00%** | **0.0913** | ★ |
| 003 BN on | 110K | BN K=32 N=4 | 2 | 89.00% | 0.0980 | |
| 003 BN on | 110K | BN K=32 N=4 | 3 | 89.80% | 0.1074 | |
| 003 BN on | 110K | BN K=32 N=4 | 4 | 88.20% | 0.1616 | |
| **003 BN off** | **110K** | **无BN K=32 N=4** | **1** | **94.00%** | **0.0504** | |
| 003 BN off | 110K | 无BN K=32 N=4 | 2 | **94.40%** | 0.0573 | ★ |
| 003 BN on | 110K | BN K=48 N=12 | 1 | 89.00% | 0.1044 | |
| 003 BN off | 110K | 无BN K=48 N=12 | 1 | 91.80% | 0.0768 | |
| 004 VoCo 分段 | 10K | VoCo K_seg=4 1fps | 1 | 70.40% | 0.3760 | |
| 004 VoCo 分段 | 10K | VoCo K_seg=4 1fps | 2 | 66.00% | 0.4432 | 过拟合 |

### Sanity Check 对比

| 指标 | BN on 20K ep1 | BN off 20K ep1 | BN on 110K ep1 |
|------|--------------|----------------|----------------|
| normal | 0.0731 | 0.0754 | 0.0709 |
| zero_no_bn | 2.1843 | 4.0705 | 1.6816 |
| bn_normal | 0.3281 | 0.3345 | 0.1300 |
| bn_zero | 0.5688 | 1.3871 | 1.0183 |
| **Zero Δ (BN)** | **+0.24** | **+1.05** | **+0.89** |

### 本地 20K BN on 详细 (K=32, N=4, 单卡)

数据: visual_qa_v3_short_train.jsonl, 20K 条 (0-30s)

| Epoch | Train Loss | Val Loss | Test Acc (200条) |
|-------|-----------|----------|-----------------|
| 1 | 0.1745 | **0.1294** ★ | 81.50% |
| 2 | 0.1068 | 0.1723 | **83.50%** |
| 3 | 0.0588 | 0.2607 | 82.00% |

### 集群 2K 验证 (K=48, N=12, 2×A100)

数据: 2K 条 (0-30s), 3 epoch

| Epoch | Train Loss | Val Loss |
|-------|-----------|----------|
| 3 | 0.0767 | 0.3001 |

best_val_loss = 0.1831。代码跑通，DDP + gradient checkpointing 工作正常。

### 集群实验进度

最后更新: 2026-05-12 02:45

#### Stage 2 Bottleneck SFT (003)

| 实验 | BN | N | K | Cosine | 数据 | SLA | 状态 | best_val_loss | Eval |
|------|-----|---|---|--------|------|-----|------|---------------|------|
| stage2-60s-n4 | ✅ | 4 | 32 | ❌ | 110K | STD | ✅ pass (5ep) | 0.1012 | 90.0% |
| stage2-nobn-n4 | ❌ | 4 | 32 | ❌ | 110K | STD | ✅ pass (5ep) | 0.0877 | 94.0%/94.4% |
| stage2-60s | ✅ | 12 | 48 | ❌ | 110K | STD | 🛑 cancelled (Ep2 63%) | 0.1044 | 89.0% |
| stage2-no-bn-fair | ❌ | 12 | 48 | ❌ | 110K | STD | 🛑 cancelled (Ep2 63%) | 0.0768 | 91.8% |
| stage2-60s-cosine | ✅ | 12 | 48 | ✅ | 110K | BSC | 🛑 cancelled (Ep1 17%) | - | - |
| stage2-no-bn-cos | ❌ | 12 | 48 | ❌ | 110K | BSC | 🛑 cancelled (Ep1 27%) | - | - |

#### VoCo 分段压缩 (004)

| 实验 | 版本 | 数据 | 状态 | 结果 |
|------|------|------|------|------|
| voco-single-200 | 单forward | 200×3ep | ✅ pass | val_loss 0.30→0.25→0.21 |
| voco-single-10k | 单forward | 10K×5ep | ✅ pass | ep1 70.4%, ep2 66.0% (过拟合) |
| voco-concat-v3 | 拼接(detach) | 200×3ep | ✅ pass (DDP修复) | train 1.52→0.57→0.28 |
| voco-110k-v4 | 单forward | 110K | ❌ SIGABRT (ep1 8%) | loss 卡 3.4375 不降 |
| voco-concat-110k | 拼接版 | 110K | 🟢 running (ep1 31%) | 正常训练中 |

#### 核心结论

- **BN On/Off**: BN on 90% vs BN off 94%（gap ~4%），bottleneck 牺牲准确率换 grounding 能力
- **N=4 vs N=12**: N=12 反而略差（BN: 89% vs 90%, NoBN: 91.8% vs 94%），可能训练不充分或帧数多反而引入噪声
- **VoCo vs BN**: VoCo 10K 70.4% 远低于 BN 110K 90%，压缩比过于激进(54:1 vs 13.5:1)，且数据量不足

---

## 数据

### MCQ 数据 (Stage 2)

来源: LLaVA-Video-178K (academic + nextqa) + Open-o3-Video STGR

| 数据集 | Train | Val | Test | 来源 |
|--------|-------|-----|------|------|
| 全量 v3 | 152,257 | 7,558 | 15,355 | activitynet, charades, nextqa, youcook2, ego4d |
| 0-30s 短视频 | 38,845 | 928 | 1,835 | 同上，过滤 0_30_s |
| 0-60s | 110,026 | 5,102 | 10,103 | 同上，过滤 0_30_s + 30_60_s |

按视频级别划分（seed=42），无重复，无跨 split 泄漏。

解析脚本: `scripts/parse_mcq_data.py`

### Temporal 数据 (Stage 3)

- temporal_evidence.jsonl: 12,997 条（有 evidence_segments 标注）
- Fixed 16-bin 方案，和帧数解耦
- 零长度 evidence 自动扩展 ±1s

### Blob 存储

```
shuwangmain / amulet / bottleneck/data/
├── visual_qa_v3_short_{train,val,test}.jsonl    # 0-30s
├── visual_qa_v3_60s_{train,val,test}.jsonl      # 0-60s
└── videos/
    ├── 0_30_s_academic_v0_1/                    # 12,139 mp4, 36GB
    └── 30_60_s_academic_v0_1/                   # 10,503 mp4, 46GB
```

---

## 使用方式

### Stage 2: Bottleneck SFT

```bash
# 单卡
python train_ddp.py \
  --data_path <data>/visual_qa_v3_short_train.jsonl \
  --video_dirs <videos>/0_30_s_academic_v0_1 \
  --output_dir outputs \
  --K 32 --num_frames 4 --epochs 5 --lr 2e-5 \
  --bottleneck --grad_accum 4 --save_steps 200

# 多卡
torchrun --nproc_per_node=4 train_ddp.py \
  --data_path <data>/visual_qa_v3_60s_train.jsonl \
  --video_dirs <videos>/0_30_s_academic_v0_1,<videos>/30_60_s_academic_v0_1 \
  --output_dir outputs \
  --K 48 --num_frames 12 --epochs 5 --lr 2e-5 \
  --bottleneck --gradient_checkpointing --grad_accum 4 --save_steps 500
```

### Stage 3: Temporal Head

```bash
python train_temporal.py \
  --stage2_checkpoint outputs/best_model.pt \
  --data_path <data>/temporal_evidence.jsonl \
  --video_dirs <videos> \
  --output_dir outputs_temporal \
  --K 32 --num_frames 8 --num_bins 16 --epochs 10 \
  --pos_weight_auto
```

### Sanity Check (第 1 epoch 后必跑)

```bash
python sanity_check.py \
  --checkpoint outputs/checkpoint_epoch1.pt \
  --data_path <data>/visual_qa_v3_short_val.jsonl \
  --video_dirs <videos> \
  --K 32 --num_frames 4 --max_samples 30
```

### MCQ 评测

```bash
python eval_mcq.py \
  --checkpoint outputs/best_model.pt \
  --data_path <data>/visual_qa_v3_short_test.jsonl \
  --video_dirs <videos> \
  --K 32 --num_frames 4 --max_samples 200 --bottleneck
```

---

## 参数说明

| 参数 | 默认 | 说明 |
|------|------|------|
| --K | 48 | latent token 数量 |
| --num_frames | 12 | 每视频采样帧数 |
| --grad_accum | 2 | 梯度累积步数 |
| --lr | 2e-5 | 学习率 |
| --bottleneck | off | **核心参数**，启用 LIVR mask |
| --gradient_checkpointing | off | N≥8 时必须 |
| --cosine_lr | off | cosine warmup+decay |
| --save_steps | 100 | step 级 checkpoint |
| --resume_from | None | 从 checkpoint 继续 |
| --max_samples | None | 限制数据量（调试） |

---

## 绝对不能改的

1. `attn_implementation="eager"` — SDPA 忽略自定义 mask
2. Bottleneck mask 的 LIVR 规则 — 非 vision/latent 不能看 vision
3. Token 顺序 `[vision][question][latent][answer]`
4. Embedding grad hook — 只更新 latent 行
5. Labels — 只在 assistant 回复部分算 loss

---

## 集群资源备忘

- VC: `msrresrchbasicvc` / workspace: `msraairwsws`
- 4×A100 Standard: 可排上（~5分钟）
- 8×A100: 只有 Basic，排队困难
- Premium: 无配额
- SKU: 必须指定 `80G4-A100-NvLink`（不能用 `G4`）
- code.local_dir: 用 `$CONFIG_DIR`（避免上传无关文件）

# 004-voco-distillation: VoCo-style 视频 Token 压缩蒸馏

## 概述

独立训练 SegmentCompressor 模块，将每帧 ~370 个 dense vision tokens 压缩成 4-16 个 segment tokens。使用 KL 蒸馏让压缩后的 tokens 保留原始视觉信息。

训好的 compressor 后续接入 003 的 bottleneck pipeline，使长视频（N=30+帧）可训练。

## 位置

```
训练路线:
Stage 1: VoCo Distillation（本实验）  ← 当前
Stage 2: Bottleneck SFT（003）
Stage 3: Temporal Head（003）
Stage 4: 长视频端到端（后续）
```

## 架构

```
视频 → Vision Encoder → dense tokens (370×N帧)
                            ↓ 按帧切分
                    SegmentCompressor (Perceiver-style cross-attention)
                            ↓
                    segment tokens (tpf×N帧, tpf=4/8/16)
                            ↓ 替换 dense tokens
                    送入 LLM → answer logits
```

### SegmentCompressor

- 可学习 segment queries (tokens_per_frame 个)
- Cross-attention: queries attend to dense tokens
- FFN + residual + LayerNorm
- 压缩比: 370 → tpf (46:1 当 tpf=8)

### 蒸馏方式

- **Teacher**: dense video tokens → 冻结 LLM → answer logits
- **Student**: compressed segment tokens → 冻结 LLM → answer logits
- **Loss**: KL(teacher_logits, student_logits) 在 answer token 位置
- **温度**: T=2.0（可调）
- 不依赖 bottleneck/LoRA/latent tokens

## 文件结构

```
004-voco-distillation/
├── README.md            # 本文件
├── train_distill.py     # 蒸馏训练脚本
└── amlt.yaml            # 集群配置（待添加）
```

## 使用方式

```bash
# 小规模测试
python train_distill.py \
  --data_path <data>/visual_qa_v3_short_train.jsonl \
  --video_dirs <videos>/0_30_s_academic_v0_1 \
  --output_dir outputs_distill \
  --tokens_per_frame 8 --num_frames 4 --epochs 3 \
  --max_samples 100

# 全量训练
torchrun --nproc_per_node=4 train_distill.py \
  --data_path <data>/visual_qa_v3_short_train.jsonl \
  --video_dirs <videos>/0_30_s_academic_v0_1 \
  --output_dir outputs_distill \
  --tokens_per_frame 8 --num_frames 4 --epochs 5 --lr 1e-4
```

## 参数说明

| 参数 | 默认 | 说明 |
|------|------|------|
| --tokens_per_frame | 8 | 每帧压缩后 token 数 |
| --num_frames | 4 | 每视频采样帧数 |
| --temperature | 2.0 | KL 蒸馏温度 |
| --lr | 1e-4 | Compressor 学习率 |

## 实验记录

### 集群测试

| 实验 | tpf | 数据 | 状态 |
|------|-----|------|------|
| distill-test | 8 | 100 条 × 3 epoch | 测试中 |

## 后续计划

1. 小规模验证 loss 下降 → compressor 能学
2. tokens_per_frame ablation: 4/8/16
3. 全量 0-30s 训练
4. 接入 003 bottleneck pipeline（替代 dense tokens 进 LLM）

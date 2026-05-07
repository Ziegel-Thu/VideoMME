# Bottleneck SFT 训练 — 外部机器部署指南

## 项目概述

在 Qwen2.5-VL-7B 上训练 LIVR-style latent visual bottleneck，用于 grounded video reasoning。

核心思路：在 attention mask 中加入瓶颈约束，K 个可学习 latent tokens 成为 answer tokens 获取视觉信息的唯一通道，迫使模型将关键视觉信息压缩进 latent tokens。

当前阶段：**Stage 2 Bottleneck SFT**，只训 L_ans（MCQ answer NLL loss）。

---

## 1. 获取代码

```bash
git clone git@github.com:Ziegel-Thu/VideoMME.git video
cd video
```

训练代码在 `experiments/003-bottleneck-multigpu/`：
- `model.py` — 模型加载、bottleneck mask 构造、checkpoint 工具
- `data.py` — Dataset、collate_fn
- `train_ddp.py` — DDP 训练主脚本（单卡/多卡兼容）
- `sanity_check.py` — bottleneck 有效性验证

---

## 2. 环境安装

```bash
conda create -n video python=3.11 -y
conda activate video
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.57.6 peft accelerate qwen-vl-utils decord pillow huggingface_hub
```

硬件要求：
- 单卡：A100 80GB（N=4帧, K=32, ~73GB 显存）
- 多卡：2-8× A100/H100，配合 gradient checkpointing 可用更多帧

---

## 3. 准备数据

### 3.1 下载视频

视频来自 LLaVA-Video-178K 数据集的 0-30s 学术子集（约 12,000 个 mp4，~36GB）。

```bash
# 方法 1：用 huggingface-cli 下载（推荐）
huggingface-cli download lmms-lab/LLaVA-Video-178K \
  --repo-type dataset \
  --include "0_30_s_academic_v0_1/**/*.mp4" \
  --local-dir data/llava-video/

# 方法 2：如果网络受限，可以只下载部分子来源
# activitynet 和 charades 加起来约 8000 个视频，足够训练
huggingface-cli download lmms-lab/LLaVA-Video-178K \
  --repo-type dataset \
  --include "0_30_s_academic_v0_1/academic_source/activitynet/*" \
            "0_30_s_academic_v0_1/academic_source/Charades/*" \
  --local-dir data/llava-video/
```

下载完成后确认视频数量：
```bash
find data/llava-video/0_30_s_academic_v0_1/ -name "*.mp4" | wc -l
# 预期：约 12,000 个
```

### 3.2 生成标注 jsonl

解析脚本会自动从 HuggingFace 下载 MCQ 标注 JSON，然后解析、去重、按视频级别划分 train/val/test。

```bash
python scripts/parse_mcq_data.py
```

这会生成：
- `data/parsed/visual_qa_v3_train.jsonl` — 全量训练集（~152K 条）
- `data/parsed/visual_qa_v3_val.jsonl` — 验证集（~7.5K 条）
- `data/parsed/visual_qa_v3_test.jsonl` — 测试集（~15K 条）

### 3.3 过滤出 0-30s 短视频子集

全量数据包含 0s-3min 的视频，我们先只用 0-30s 的：

```bash
python -c "
import json
for split in ['train', 'val', 'test']:
    inp = f'data/parsed/visual_qa_v3_{split}.jsonl'
    out = f'data/parsed/visual_qa_v3_short_{split}.jsonl'
    kept = 0
    with open(inp) as fi, open(out, 'w') as fo:
        for line in fi:
            if '0_30_s' in json.loads(line)['video_path']:
                fo.write(line)
                kept += 1
    print(f'{split}: {kept} 条')
"
```

预期输出：train ~38K / val ~900 / test ~1800。

注意：jsonl 中 `video_path` 是原始绝对路径，但训练代码通过 `--video_dirs` 参数按文件名（basename）匹配视频，不依赖 jsonl 中的绝对路径。

---

## 4. 训练

### 4.1 单卡训练

```bash
cd experiments/003-bottleneck-multigpu/

python train_ddp.py \
  --data_path ../../data/parsed/visual_qa_v3_short_train.jsonl \
  --video_dirs ../../data/llava-video/0_30_s_academic_v0_1 \
  --output_dir outputs \
  --K 32 --num_frames 4 --epochs 5 --lr 2e-5 \
  --bottleneck --grad_accum 4 --save_steps 200 \
  2>&1 | tee log_train.txt
```

### 4.2 多卡 DDP 训练

```bash
cd experiments/003-bottleneck-multigpu/

torchrun --nproc_per_node=<GPU数> train_ddp.py \
  --data_path ../../data/parsed/visual_qa_v3_short_train.jsonl \
  --video_dirs ../../data/llava-video/0_30_s_academic_v0_1 \
  --output_dir outputs \
  --K 48 --num_frames 12 --epochs 5 --lr 2e-5 \
  --bottleneck --gradient_checkpointing --grad_accum 2 --save_steps 100 \
  2>&1 | tee log_train.txt
```

### 参数说明

| 参数 | 说明 | 单卡推荐 | 多卡推荐 |
|------|------|---------|---------|
| `--K` | latent token 数量 | 32 | 48 |
| `--num_frames` | 每视频采样帧数 | 4 | 12 |
| `--grad_accum` | 梯度累积步数 | 4 | 2 |
| `--gradient_checkpointing` | 节省显存 | 不需要(N=4) | 必须(N≥8) |
| `--bottleneck` | LIVR bottleneck mask | **必须开** | **必须开** |
| `--save_steps` | 每 N 步保存 checkpoint | 200 | 100 |
| `--max_samples` | 限制数据量(调试用) | 按需 | 按需 |
| `--resume_from` | 从 checkpoint 继续 | checkpoint 路径 | checkpoint 路径 |

等效 batch size = GPU 数 × grad_accum。推荐 4-16。

### 4.3 先用小数据验证

建议先跑 100 条 × 1 epoch 确认没有 OOM / 报错：

```bash
python train_ddp.py \
  --data_path ../../data/parsed/visual_qa_v3_short_train.jsonl \
  --video_dirs ../../data/llava-video/0_30_s_academic_v0_1 \
  --output_dir outputs_test \
  --K 32 --num_frames 4 --epochs 1 --lr 2e-5 \
  --bottleneck --grad_accum 4 --max_samples 100
```

---

## 5. 第 1 epoch 后必须跑 Sanity Check

验证 bottleneck 是否真的让 latent tokens 承载了视觉信息：

```bash
python sanity_check.py \
  --checkpoint outputs/checkpoint_epoch1.pt \
  --data_path ../../data/parsed/visual_qa_v3_short_val.jsonl \
  --video_dirs ../../data/llava-video/0_30_s_academic_v0_1 \
  --K 32 --num_frames 4 --max_samples 30
```

预期输出：
```
  Zero Δ (BN):    +0.XX  ✅ PASS    (> 0.1 说明 bottleneck 生效)
```

如果 FAIL，说明模型没有依赖视觉信息，训练有问题，不要继续跑。

---

## 6. 绝对不能改的（改了会静默失效）

1. **`attn_implementation="eager"`** — 已硬编码在 model.py 中。SDPA 会忽略自定义 4D mask，导致 bottleneck 完全无效但训练看起来正常。
2. **`--bottleneck` 必须开** — 这是核心机制。
3. **Token 顺序 `[vision][question][latent][answer]`** — latent 必须在 question 之后，这样它能同时看到视觉和问题。
4. **Embedding grad hook** — 只更新 latent token 的 embedding，其他行梯度归零。已在 model.py 中自动处理。
5. **Labels 构造** — 只在 assistant 回复部分算 loss。已在 data.py 中自动处理。

---

## 7. 输出

训练会在 `--output_dir` 下保存：
- `checkpoint_epoch{N}.pt` — 每 epoch 的 checkpoint
- `best_model.pt` — 最佳 val_loss 的 checkpoint
- `latest_step.pt` — 最近一次 step checkpoint（防中断丢进度）
- `test_indices.json` — test set 的数据 indices（复现评测用）

Checkpoint 格式：
```python
{
    "lora_state_dict": {...},        # LoRA 权重
    "latent_embeddings": {tid: emb}, # K 个 latent token embedding
    "epoch": int,
    "val_loss": float,
    "step": int,                     # step checkpoint 才有
}
```

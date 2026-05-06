# 002-latent-bottleneck: Latent Visual Bottleneck for Grounded Video Reasoning

## 项目目标

在 Qwen2.5-VL-7B 上实现 LIVR-style 的 **latent visual bottleneck**，用于 grounded long video reasoning。

- **输入**：长视频 + 问题
- **输出**：答案 + 证据时间段（temporal grounding）
- **核心 claim**：bottleneck 迫使 K 个 latent tokens 承载关键视觉信息，answer tokens 只能通过 latent 间接获取 vision → latent 的 hidden states 天然编码了"哪些视觉内容被选中"，可直接接 temporal head 做 grounding

### 训练路线（导师方案）

```
Stage 1: VoCo 压缩（可跳过）
Stage 2: Bottleneck SFT — MCQ 数据训 L_ans（NLL），学会通过 latent 回答问题  ← 当前阶段
Stage 3: Temporal Head — temporal 数据训 L_temp（BCE），从 latent hidden state 预测时间段
Stage 4: 长视频 VoCo 压缩 + Segment Selector（未开始）
```

---

## 架构

```
Token 顺序: [system] [vision] [question] [latent_0..K-1] [answer]

                          Attention Mask 规则 (LIVR-style)
                    ┌─────────────────────────────────────┐
                    │  vision  → 正常 causal               │
                    │  latent  → 正常 causal (能看 vision)  │
                    │  question → ✘ 不能看 vision           │  ← 瓶颈!
                    │  answer   → ✘ 不能看 vision           │  ← 瓶颈!
                    │  question/answer → ✔ 能看 latent+text │
                    └─────────────────────────────────────┘

 ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
 │ <vision> │──▶│<question>│──▶│ <latent> │──▶│ <answer> │
 │ N帧视频  │   │  MCQ题目 │   │  K=32    │   │  选项    │
 └──────────┘   └──────────┘   └──────────┘   └──────────┘
      ↑                              ↑              │
   冻结 VE                     可学习 embed      NLL Loss (L_ans)
                               (grad hook)
                                     │
                                     ▼
                              ┌──────────────┐
                              │ Temporal Head │ → L_temp (BCE)
                              │ MLP: D→D/2→B │    per-bin sigmoid
                              └──────────────┘
```

### 关键实现细节

| 项目 | 细节 |
|------|------|
| Base model | `Qwen/Qwen2.5-VL-7B-Instruct` |
| Attention | **必须用 `eager`**，SDPA 会忽略自定义 4D mask（见"踩坑记录"） |
| Latent tokens | K=32 个 `<latent_0>` ~ `<latent_31>`，加到 tokenizer 的 special tokens |
| LoRA | r=16, alpha=32, target=q/k/v/o/gate/up/down_proj, dropout=0.05 |
| Embedding | 整个 embedding 层 requires_grad=True，但 grad hook 只保留 latent 行的梯度 |
| Vision encoder | 完全冻结 |
| Mask 注入 | 通过 `register_forward_pre_hook` 在每个 decoder layer 替换 `kwargs["attention_mask"]` |
| PEFT 路径 | `model.base_model.model.model.language_model.layers` |
| 帧数 | N=4（N=8 在 A100 80GB 上 OOM） |
| Hidden dim | 3584（Qwen2.5-VL-7B 的 d_model） |

### SDPA vs Eager 踩坑（最关键的发现）

- Qwen2.5-VL 的 SDPA attention 在内部使用 `is_causal=True` 调用 `F.scaled_dot_product_attention`
- 这会让 PyTorch **完全忽略**传入的 `attn_mask` 参数，走 flash attention 的 causal kernel
- 通过 hook 注入的 4D mask 看起来被传进去了，但实际上**没有生效**
- **验证方法**：打印 attention weights，eager 下 answer→vision 权重 = 0.000000，SDPA 下 > 0
- **结论**：使用自定义 4D mask 必须 `attn_implementation="eager"`

---

## 文件结构

```
002-latent-bottleneck/
├── README.md                # 本文件
├── mvp.py                   # 核心模块：模型加载、bottleneck mask、视频输入构造
├── train_video.py           # Stage 2+3 训练：多帧视频 + L_ans + 可选 L_temp
├── train_mvp.py             # 早期单帧训练（已弃用，保留兼容）
├── train_temporal_only.py   # Stage 3 冻 LoRA 只训 temporal head
├── eval_mcq.py              # MCQ 准确率评测
├── eval_temporal.py         # 时间段 grounding 评测 (tIoU, Recall)
├── sanity_video.py          # 多帧 sanity check (zero/shuffled vision)
├── auto_sanity.py           # 单帧 sanity check
└── outputs_k32_full/        # 当前最佳 checkpoint 目录
    ├── best_model.pt        # val_loss=0.0665
    ├── checkpoint_epoch{1,2,3}.pt
    └── ...
```

### mvp.py 核心函数

| 函数 | 功能 |
|------|------|
| `setup_model_and_tokenizer(K, lora_r)` | 加载 Qwen2.5-VL + K latent tokens + LoRA + 冻 VE |
| `build_bottleneck_mask(input_ids, latent_token_ids)` | LIVR-style 4D mask：非 vision/latent → vision 被 block |
| `build_bottleneck_mask_answer_only(...)` | 消融版：只 block answer → vision，question 仍可看 |
| `build_training_input_video(processor, tokenizer, latent_tokens, video_path, question, answer, num_frames)` | 多帧视频输入构造 |

### train_video.py 训练逻辑

- `VideoQADataset`: 加载 jsonl，从视频用 decord 均匀采 N 帧
- `collate_fn_video`: batch 化 + 构造 labels（只在 assistant 回复部分计算 NLL）
- `TemporalHead`: MLP (3584 → 1792 → num_bins)，sigmoid 输出，BCE loss
- Hook-based mask 注入：每个 batch 的 forward 前挂 hook，forward 后删
- Optimizer: AdamW, lr=2e-5, weight_decay=0.01
- Val split: `random_split(seed=42)`，n_val = min(500, len//10)

### Checkpoint 格式

```python
{
    "lora_state_dict": model.state_dict(),     # LoRA 权重
    "latent_embeddings": {token_id: embedding}, # K 个 latent token 的 embedding
    "epoch": int,
    "val_loss": float,
}
```

加载方法：
```python
model.load_state_dict(ckpt["lora_state_dict"], strict=False)
embed = model.get_input_embeddings()
for tid, emb in ckpt["latent_embeddings"].items():
    embed.weight.data[tid] = emb.to(embed.weight.device)
```

---

## 环境依赖

```bash
conda create -n video python=3.11
conda activate video
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.57.6 peft accelerate
pip install qwen-vl-utils decord pillow
```

硬件：NVIDIA A100 80GB（N=4 帧约用 73GB 显存）

---

## 运行方式

### Stage 2: Bottleneck SFT（MCQ 数据）

```bash
python train_video.py \
    --data_path <visual_qa_v2.jsonl 路径> \
    --output_dir outputs_k32_full \
    --K 32 --num_frames 4 --epochs 10 --lr 2e-5 \
    --bottleneck --mask_mode livr \
    --batch_size 1 \
    2>&1 | tee log_k32_full.txt
```

### Stage 3: Temporal Head（冻 LoRA，只训 MLP）

```bash
python train_temporal_only.py \
    --data_path <temporal_evidence_filtered.jsonl 路径> \
    --bottleneck_ckpt outputs_k32_full/best_model.pt \
    --output_dir outputs_temporal \
    --K 32 --num_frames 4 --epochs 10 --lr 1e-4 \
    --batch_size 4 \
    2>&1 | tee log_temporal.txt
```

### Sanity Check（训练后必跑）

```bash
python sanity_video.py \
    --checkpoint outputs_k32_full/best_model.pt \
    --data_path <visual_qa_v2.jsonl 路径> \
    --K 32 --num_frames 4 --num_samples 50
```

预期：`BN zero Δ > 0.1` 表示 bottleneck 生效。

### MCQ 评测

```bash
python eval_mcq.py \
    --checkpoints outputs_k32_full/checkpoint_epoch{1,2,3}.pt \
    --data_path <visual_qa_v2.jsonl 路径> \
    --K 32 --max_samples 200 --bottleneck
```

---

## 实验记录

### 单帧实验

| 实验 | 数据 | Epochs | Mask | 数据类型 | BN Zero Δ | 结果 |
|------|------|--------|------|---------|-----------|------|
| temporal QA (SDPA) | 7.7K | 5 | LIVR | temporal QA | +0.001 | ❌ SDPA mask 没生效 |
| MCQ (SDPA) | 19K | 1 | LIVR | MCQ | +0.001 | ❌ SDPA mask 没生效 |
| **MCQ v2 (eager)** | **19K** | **3** | **LIVR** | **纯 MCQ** | **+0.35** | **✅ PASS** |
| MCQ v2 (eager) | 19K | 3 | answer-only | 纯 MCQ | +0.14 | ✅ PASS（LIVR 更强） |

### Multi-frame K sweep (1K × 5ep, N=4帧, LIVR mask)

| K | 压缩比 (vision:latent) | val_loss (best) | BN Zero Δ | 结果 |
|---|------------------------|-----------------|-----------|------|
| 8 | 185:1 | 0.2085 | +0.16 | ✅ PASS |
| 64 | 23:1 | 0.2165 | +0.77 | ✅ PASS |

**结论**: K 越大 bottleneck 效果越强。选择 K=32 做全量训练。

### K=32 全量训练 (19K MCQ × 3ep, LIVR mask, eager)

| Epoch | val_loss | 状态 |
|-------|----------|------|
| 1 | 0.1261 | |
| 2 | 0.1171 | |
| 3 | 0.0665 | ★ best（仍在下降） |

Sanity check: BN zero Δ = +0.11 ✅

### MCQ 评测结果 (K=32, val split 200条, seed=42)

| Checkpoint | Val Accuracy | Val Loss |
|------------|-------------|----------|
| epoch1 | 94.50% (189/200) | 0.0975 |
| epoch2 | 96.00% (192/200) | 0.0897 |
| epoch3 | **98.50%** (197/200) | 0.0320 |

### Temporal Head 评测 (K=32, 862 filtered samples × 10ep)

| 指标 | 值 |
|------|-----|
| 平均 tIoU | 0.3148 |
| Recall@0.3 | 44.5% |
| Recall@0.5 | 27.0% |
| Recall@0.7 | 12.0% |

### Temporal 数据在 N=4 帧下的问题

temporal_evidence.jsonl 有 12,997 条数据，但 N=4 帧均匀采样时，**88% 的样本所有帧都不落在 evidence 时间段内**（evidence 太短，帧间距太大），导致 temporal labels 全为零。过滤后只剩 **862 条**可用数据。

**原因**：evidence 时间跨度通常只有几秒（如 19s-25s），但 4 帧均匀采样的间距可达 10-30s，大概率跳过 evidence。

**解决方向**：
1. 改用 fixed-bin 方案——把视频时间轴均分 16 个 bin，预测每个 bin 的 evidence 概率（不依赖采样帧位置）
2. 增加帧数 N（但 N=8 在 A100 80G 上 OOM，需要 gradient checkpointing 或更大显存）
3. 在集群上用更大显存/多卡跑更多帧

---

## 踩坑记录

1. **SDPA 忽略自定义 mask** — 浪费 2 天，必须用 eager attention
2. **temporal QA 数据无视觉依赖** — 模型不看图也能猜时间段，sanity 不过
3. **latent tokens 放在 question 之前** — causal mask 下 latent 看不到 question，信息不足 → 改为 `[vision][question][latent][answer]`
4. **训练时 labels 要手动构造** — 只在 assistant 回复部分算 NLL，之前的全设 -100
5. **generate() + bottleneck hook** — KV cache 下 mask 尺寸不匹配，改用 logits 判断准确率
6. **N=4 帧下 temporal 数据 88% 标签全零** — 只能训 862 条，效果有限

---

## Blob 存储位置

数据已上传至 Azure Blob Storage，供集群训练使用。

```
存储账户: shuwangmain
容器:     zhengshurui
根路径:   video_project/

video_project/
├── code/                                    # 所有 Python 代码
│   ├── mvp.py                               # 核心模块
│   ├── train_video.py                       # Stage 2+3 训练脚本
│   ├── train_temporal_only.py               # Stage 3 temporal head 训练
│   ├── train_mvp.py                         # 早期单帧训练（兼容保留）
│   ├── eval_mcq.py                          # MCQ 评测
│   ├── eval_temporal.py                     # Temporal grounding 评测
│   ├── sanity_video.py                      # 多帧 sanity check
│   ├── sanity_check.py                      # 单帧 sanity check
│   ├── auto_sanity.py                       # 自动 sanity
│   └── eval_mvp.py                          # 早期评测
│
├── data/
│   ├── parsed/
│   │   ├── visual_qa_v2.jsonl               # 19,237 条 MCQ（Stage 2 训练）
│   │   ├── temporal_evidence.jsonl          # 12,997 条 temporal 标注（Stage 3）
│   │   ├── temporal_evidence_filtered.jsonl # 862 条（4帧过滤后）
│   │   ├── mixed_mcq_temporal.jsonl         # 26,470 条混合数据
│   │   ├── selector_trajectories.jsonl      # 181,453 条 selector 轨迹（Stage 4）
│   │   ├── visual_qa.jsonl                  # 早期版本
│   │   ├── temporal_evidence_long.jsonl     # 长证据子集
│   │   └── temporal_evidence_short.jsonl    # 短证据子集
│   │
│   └── videos/
│       ├── stgr/                            # Open-o3-Video STGR 视频 (40,404 文件, ~45GB)
│       └── llava-video/                     # LLaVA-Video 学术视频 (30,467 文件, ~234GB)
│
├── checkpoints/
│   └── k32_full/outputs_k32_full/
│       ├── best_model.pt                    # val_loss=0.0665 ★ 当前最佳
│       ├── checkpoint_epoch1.pt             # val_loss=0.1261
│       ├── checkpoint_epoch2.pt             # val_loss=0.1171
│       └── checkpoint_epoch3.pt             # val_loss=0.0665
│
└── CLAUDE.md                                # 项目约定
```

### 数据格式

**visual_qa_v2.jsonl**（MCQ，Stage 2 训练用）：
```json
{
  "video_path": "stgr/xxx.mp4",
  "question": "What does the person do after ...?\nOptions:\nA. ...\nB. ...\nC. ...\nD. ...",
  "answer": "B"
}
```

**temporal_evidence.jsonl**（Temporal，Stage 3 训练用）：
```json
{
  "video_path": "stgr/xxx.mp4",
  "question": "...",
  "answer": "...",
  "evidence_segments": [[19.0, 25.0], [45.0, 52.0]]
}
```

---

## 给其他 Agent 的上下文 Prompt

如果你是一个新的 AI agent，需要在集群上继续这个项目的训练，以下是你需要知道的一切：

### 项目概述

这是一个 **Latent Visual Bottleneck for Grounded Video Reasoning** 研究项目。核心思路是在 Qwen2.5-VL-7B 的 attention 中加入 bottleneck mask：让 K=32 个可学习的 latent tokens 成为 answer tokens 获取视觉信息的唯一通道。这样 latent tokens 被迫学习压缩和选择关键视觉信息，其 hidden states 天然编码了"选了什么"，可以直接接 temporal head 做时间段 grounding。

### 当前进度

- **Stage 2 (Bottleneck SFT) ✅ 已完成 3 epoch**：19K MCQ 数据，K=32, N=4帧，LIVR mask，eager attention。val_loss 从 0.1261 → 0.0665（仍在下降），val MCQ accuracy 98.5%。Sanity check 通过（BN zero Δ=+0.11）。
- **Stage 3 (Temporal Head) ⚠️ 初步完成但数据不足**：只有 862 条有效 temporal 数据（4帧采样下 88% 标签全零），tIoU=0.31。需要改用 fixed-bin 方案或增加帧数。

### 下一步计划

1. **Stage 2 继续训练 10 epoch**：val_loss 仍在下降，需要更多 epoch。从 `best_model.pt` (epoch3) 继续。
2. **Stage 3 改进 temporal head**：用 fixed 16-bin 方案替代 per-frame 分类，让所有 12,997 条 temporal 数据都可用。

### 集群训练命令

**Stage 2 继续训练**（最优先）：
```bash
# 假设 blob 挂载到 /mnt/blob/video_project
cd /mnt/blob/video_project/code

python train_video.py \
    --data_path /mnt/blob/video_project/data/parsed/visual_qa_v2.jsonl \
    --output_dir /mnt/blob/video_project/checkpoints/k32_10ep \
    --K 32 \
    --num_frames 4 \
    --epochs 10 \
    --lr 2e-5 \
    --bottleneck \
    --mask_mode livr \
    --batch_size 1 \
    2>&1 | tee /mnt/blob/video_project/log_k32_10ep.txt
```

注意：train_video.py 中视频路径需要适配。jsonl 中 `video_path` 字段是相对路径如 `stgr/xxx.mp4`，代码通过 `video_index` 字典自动匹配——需确保视频目录作为参数传入或在代码中硬编码为 blob 挂载路径。

### 绝对不能改的地方

1. `attn_implementation="eager"` — 改成 sdpa 会导致 mask 静默失效
2. Bottleneck mask 的 LIVR 规则 — 非 vision/latent token 不能看 vision
3. Token 顺序 `[vision][question][latent][answer]` — latent 必须在 question 之后
4. Embedding grad hook — 只更新 latent 行，其他行梯度归零
5. Labels 构造 — 只在 assistant 回复部分算 loss（之前全设 -100）

### 环境要求

- Python 3.11, PyTorch 2.6.0+cu124, transformers 4.57.6, peft, decord, qwen-vl-utils
- GPU 显存 ≥ 73GB（N=4 帧, K=32, batch_size=1）
- 推荐 A100 80GB

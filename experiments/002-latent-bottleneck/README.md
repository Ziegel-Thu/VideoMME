# 002-latent-bottleneck: Latent Visual Bottleneck MVP

在 Qwen2.5-VL-7B 上实现 LIVR-style 的 latent visual bottleneck，验证 bottleneck attention mask 是否能迫使 latent tokens 承载关键视觉信息。

---

## 架构示意

```
                          Attention Mask 规则
                    ┌─────────────────────────────┐
                    │  谁 → 能看谁                │
                    │                             │
 ┌──────────┐      │  vision  → vision + text     │   正常 causal
 │  Vision   │      │  latent  → vision + text     │   正常 causal
 │  Tokens   │      │  answer  → latent + text     │   ← 瓶颈! 看不到 vision
 │ (Qwen2.5  │      │                             │
 │  VL enc)  │      └─────────────────────────────┘
 └────┬──────┘
      │
      ▼
 ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
 │ <vision> │──▶│ <latent> │──▶│  <text>  │──▶│ <answer> │
 │ tokens   │   │ ×8 slots │   │ question │   │ tokens   │
 └──────────┘   └──────────┘   └──────────┘   └──────────┘
       ↑              ↑              ↑              │
       │              │              │              ▼
    冻结 VE       可学习 embed     LoRA 微调      NLL Loss
                  (grad hook)

 Bottleneck OFF: answer 可以直接 attend 到 vision（标准 causal）
 Bottleneck ON:  answer 只能通过 latent tokens 间接获取视觉信息
```

---

## 文件结构

```
002-latent-bottleneck/
├── README.md           # 本文件
├── mvp.py              # 核心模块：模型加载、bottleneck mask、数据构造、smoke test
├── train_mvp.py        # 训练脚本：支持 overfit test 和 bn-on/off ablation
└── log_bn_on.txt       # bottleneck-on 训练日志
```

### mvp.py 核心函数

| 函数 | 功能 |
|------|------|
| `setup_model_and_tokenizer()` | 加载 Qwen2.5-VL，扩展 tokenizer 加 K 个 latent tokens，配 LoRA，冻结 VE |
| `build_bottleneck_mask()` | 构造 4D bottleneck attention mask（向量化实现） |
| `build_training_input()` | 构造单条训练样本输入序列 |
| `smoke_test()` | 快速验证 forward pass 正确性 |

### train_mvp.py 训练逻辑

- `ImageQADataset`: 加载 temporal_evidence.jsonl，从视频中提取关键帧
- Hook-based mask 注入: 在每个 attention layer 用 `register_forward_pre_hook` 替换 mask
- Labels: 仅 assistant 回答部分计算 NLL loss

---

## 运行方式

### 前置依赖

```bash
pip install torch transformers peft qwen-vl-utils decord pillow
```

### 1. Smoke Test（快速验证 forward pass）

```bash
cd experiments/002-latent-bottleneck
python mvp.py
```

预期输出：加载模型 → 构造 mask → forward 成功 → `Smoke Test PASSED ✅`

### 2. Overfit Test（验证训练 pipeline 正确性）

```bash
python train_mvp.py --overfit --epochs 50 --bottleneck
```

预期：loss 从 ~2.9 降到 ~0.0，证明 bottleneck mask 不妨碍梯度回传。

### 3. 完整训练 — Bottleneck ON

```bash
# 建议在 tmux 中运行（命名: exp-002-train）
python train_mvp.py \
    --bottleneck \
    --epochs 5 \
    --K 8 \
    --lr 2e-5 \
    --evidence_path /home/v-shuzheng/video/data/parsed/temporal_evidence.jsonl \
    --video_dir /home/v-shuzheng/video/data/open-o3-video/videos/stgr \
    2>&1 | tee log_bn_on.txt
```

### 4. 完整训练 — Bottleneck OFF（对照组）

```bash
python train_mvp.py \
    --epochs 5 \
    --K 8 \
    --lr 2e-5 \
    --evidence_path /home/v-shuzheng/video/data/parsed/temporal_evidence.jsonl \
    --video_dir /home/v-shuzheng/video/data/open-o3-video/videos/stgr \
    2>&1 | tee log_bn_off.txt
```

---

## 当前结果

### 第一轮（temporal QA 数据）❌ 失败

| 指标 | bn-on | bn-off |
|------|-------|--------|
| Epoch 1 val_loss | 1.1086 | 1.1078 |
| Sanity: normal | 0.2873 | — |
| Sanity: blank vision | 0.2879 | — |
| Sanity: shuffled | 0.2873 | — |

**结论**：blank/shuffled ≈ normal → 模型没在用视觉信息，全靠语言先验。
**原因**：temporal QA 数据（Q=事件描述, A=时间段）不需要看图就能猜答案。

### 第二轮（MCQ + temporal-spatial 数据）🔄 进行中

| 配置 | 数据 | Epochs | 状态 |
|------|------|--------|------|
| bn-on | 7758 visual_qa | 3 | 训练中 |
| bn-off | 7758 visual_qa | 3 | 训练中 |

数据切换为必须看图才能答对的 MCQ + temporal-spatial QA。
Epoch 1 完成后自动跑 sanity check（tmux: auto-sanity）。

---

## 关键设计决策

1. **SDPA + hook 注入 mask**：通过 `register_forward_pre_hook` 在每层 attention 替换 mask
2. **Latent token embedding 只更新 latent 行**：grad hook 把非 latent 行梯度置零
3. **Vision encoder 完全冻结**：只训练 LoRA + latent embedding
4. **关键帧采样**：从视频中间帧提取单帧，MVP 阶段用图片近似视频
5. **数据必须视觉依赖**：temporal QA 不行，MCQ/visual QA 才行（教训 2）

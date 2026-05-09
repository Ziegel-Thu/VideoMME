# 005-feature-cache-distillation: 基于离线特征+缓存的高效蒸馏

## 概述

针对 1fps 高帧数（N=30）蒸馏的训练成本问题，采用"两次离线 → 一次轻量训练"的方案：

```
Stage 0 (003): 1fps vision encoder 特征 → 缓存 video_embeds
Stage 0.5 (本实验前置): teacher answer logits → 缓存
Stage 1 训练: compressor 蒸馏（轻量）
```

## 为什么需要

直接做 N=30 蒸馏的成本：
- Teacher forward (dense, 3400 tokens, eager attention): ~50GB, 5-10s/步
- Student forward: 5-8GB, 0.5s/步
- 38K × 5 epoch / 4 卡 ≈ 100 小时

如果 teacher logits 离线缓存：
- 训练时**只跑 student**（5-8GB, 0.5s/步）
- 总时间从 100 小时降到约 **5-10 小时**
- compressor 质量不变（loss 还是 KL）

## 流程

### Stage A: Teacher logits 缓存（一次性，离线）

输入：每个视频的 vision_embeds（已由 003 提取）+ MCQ 数据
输出：每个 (video, question) 对的 teacher answer logits

```python
for video, question, answer in dataset:
    inputs = build_input(video_embeds, question, answer)
    logits = teacher.forward(inputs).logits[answer_positions]
    save(logits)  # ~K_answer × vocab_size × 2 bytes
```

每条样本 logits 约 5 × 152K × 2 ≈ 1.5MB，38K 样本 ≈ 60GB（可接受）。

### Stage B: Compressor 蒸馏训练（轻量）

```python
for video_embeds, cached_teacher_logits, question, answer in dataset:
    compressed = compressor(video_embeds)
    student_logits = student.forward(compressed, question, answer).logits[answer_positions]
    loss = KL(student_logits || cached_teacher_logits)
    loss.backward()
```

只需要 student forward + 加载缓存，速度快 10x+。

## 文件结构

```
005-feature-cache-distillation/
├── README.md
├── cache_teacher.py     # Stage A: 缓存 teacher logits
├── train_distill.py     # Stage B: 用缓存训蒸馏
└── amlt.yaml
```

## 待依赖

- 003 的 1fps 特征提取（正在跑）
- 等特征提完后开始

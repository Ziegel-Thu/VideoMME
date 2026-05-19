# 012-adapter-compression: Adapter 式轻量压缩

## 概述

不用独立的 compressor 模块，而是在 LLM 的前几层插入 adapter，让 LLM 自己学会在 forward 过程中压缩视觉 tokens。类似 LLaMA-Adapter 的做法。

### 动机

006 的 cross-attention compressor 是一个外部模块，和 LLM 的 representation space 可能不完全对齐。如果让 LLM 内部层自己做压缩，compressed tokens 天然在 LLM 的 hidden space 中，可能减少信息损失。

### 架构

```
Dense Vision Tokens (V, D) + Learnable Compress Tokens (K, D)
        ↓  LLM 前 N 层 forward（带 adapter）
        ↓  取 Compress Token 位置的 hidden states
Compressed Tokens (K, D)
        ↓  LLM 剩余层 forward + Question
        ↓  MSE with teacher
```

### 与 006 的区别

| 项目 | 006 Cross-Attention | 012 Adapter |
|------|---------------------|-------------|
| 压缩模块 | 独立 Perceiver | LLM 内部 adapter |
| 参数 | compressor 自己的权重 | adapter 层 + compress tokens |
| 对齐 | 需要蒸馏对齐 | 天然在 LLM hidden space |
| 实现复杂度 | 低 | 中高（需要改 LLM forward） |

---

## 计划

### 待做
- [ ] 调研 LLaMA-Adapter / TokenLearner 实现细节
- [ ] 设计 adapter 注入方案：在 LLM 哪几层插入，compress tokens 如何交互
- [ ] 实现 AdapterCompressor
- [ ] 110K 训练 K=8, epochs=3
- [ ] Full eval 对比 006

### 设计思路

**方案 A：前缀式 compress tokens**
```python
# 在 LLM 输入前拼接 K 个 learnable tokens
# forward 前 N 层后，取这 K 个位置的 hidden
input = [compress_tokens(K), vision_tokens(V)]
hidden = LLM_layers[:N](input)
compressed = hidden[:K]  # (K, D)
```

**方案 B：Adapter 层注入**
```python
# 在 LLM 前 N 层的每层后插入轻量 adapter
# adapter 做 cross-attention: compress_queries attend to layer hidden
for i, layer in enumerate(LLM.layers[:N]):
    hidden = layer(hidden)
    if i < N:
        compress_tokens = adapter[i](compress_tokens, hidden)
compressed = compress_tokens  # (K, D)
```

### 风险
- 需要修改 LLM forward 逻辑，侵入性强
- 前 N 层 forward 需要过完整 V tokens，显存可能偏大
- 和 gradient checkpointing 的兼容性需要验证

### 参考
- LLaMA-Adapter: Efficient Fine-tuning of LLaMA
- TokenLearner: What Can 8 Learned Tokens Do for Images and Videos?
- Perceiver IO: A General Architecture for Structured Inputs & Outputs

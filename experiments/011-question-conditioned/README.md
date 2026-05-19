# 011-question-conditioned: Question-Conditioned 压缩

## 概述

在 006 cross-attention compressor 的基础上，将 question 信息注入压缩过程，使压缩针对具体问题选择性保留视觉信息。

### 动机

当前 006 的 compressor 是 question-agnostic：先压缩再拼问题，压缩时不知道要回答什么。如果压缩时就知道问题，可以更精准地保留相关视觉信息。

### 架构（方式 C：加法注入，最简单）

```
Question Embeddings (Q_len, D)
        ↓  pool → (D,)
        ↓
Learnable Queries (K, D) + pool(Q)  ← 注入问题信息
        ↓  Cross-Attention with Dense Vision
Compressed Tokens (K, D)
        ↓  + Question → Frozen LLM → MSE with teacher
```

### 三种实现方式

| 方式 | 做法 | 复杂度 | 优先级 |
|------|------|--------|--------|
| **C（加法）** | `queries = learnable + pool(Q_embed)` | 最低（一行） | 先试 |
| **B（双层 cross-attn）** | 先 Q→queries cross-attn，再 queries→vision cross-attn | 中等 | 其次 |
| **A（拼接 query）** | Q 和 learnable queries 拼在一起做 cross-attn | 低 | 备选 |

### 与 006 的区别

| 项目 | 006 | 011 |
|------|-----|-----|
| 压缩方式 | question-agnostic | question-conditioned |
| Cache 复用 | ✅ 同视频不同问题可复用 | ❌ 不同问题需重新压缩 |
| 蒸馏训练代价 | 无区别 | 无区别（每样本本来就一个 Q） |

### Trade-off

- 优势：压缩更精准，理论上精度更高
- 劣势：推理时不能复用 cache，同视频多问题场景效率低
- 当前阶段（蒸馏训练 + MCQ eval）无额外代价

---

## 计划

### 待做
- [ ] 实现方式 C：ConditionedVoCoCompressor（加法注入）
- [ ] 110K 训练 B-2L + Q-conditioned, K=8, epochs=3
- [ ] Full eval 0-30 + 0-60
- [ ] 与 006 B-2L 对比
- [ ] 如果 C 有收益，再实现方式 B（双层 cross-attn）

### 设计思路（方式 C）
```python
class ConditionedVoCoCompressor(VoCoCompressor):
    def __init__(self, K=8, dim=3584, **kwargs):
        super().__init__(K=K, dim=dim, **kwargs)
        self.q_pool = nn.Linear(dim, dim)
    
    def forward(self, dense_vision, q_embeds=None):
        x = self.queries  # (K, D)
        if q_embeds is not None:
            q_pooled = self.q_pool(q_embeds.mean(dim=0))  # (D,)
            x = x + q_pooled.unsqueeze(0)                 # (K, D)
        for layer in self.layers:
            x = layer(x, dense_vision)
        return self.out_norm(x)
```

### 设计思路（方式 B）
```python
class DualCrossAttnCompressor(nn.Module):
    def __init__(self, K=8, dim=3584):
        self.queries = nn.Parameter(torch.randn(K, dim) * 0.02)
        self.q_cross_attn = CrossAttnLayer(dim)   # queries attend to Q
        self.v_cross_attn = CrossAttnLayer(dim)   # queries attend to vision
    
    def forward(self, dense_vision, q_embeds):
        x = self.q_cross_attn(self.queries, q_embeds)   # 注入问题
        x = self.v_cross_attn(x, dense_vision)          # 提取视觉
        return x
```

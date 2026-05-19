# 010-gated-compression: Gate 加权压缩

## 概述

在 006 cross-attention compressor 输出后加 gate 层，让模型学会对每个 compressed token 做重要性加权。不同 token 负责不同类型视觉信息，gate 抑制不重要的 token。

### 架构

```
Dense Vision Tokens (V, D)
        ↓  Cross-Attention (同 006)
Compressed Tokens (K, D)
        ↓  Gate: sigmoid(linear(compressed))  ← 新增
Gated Compressed (K, D)
        ↓  + Question → Frozen LLM → MSE with teacher
```

### 与 006 的区别

| 项目 | 006 | 010 |
|------|-----|-----|
| 压缩 | cross-attention | cross-attention + gate |
| 额外参数 | 0 | ~7K（一个 linear + sigmoid） |
| 动机 | K 个 token 等权 | 学习 token 重要性分配 |

---

## 计划

### 待做
- [ ] 实现 GatedVoCoCompressor（在 VoCoCompressor 基础上加 gate）
- [ ] 110K 训练 B-2L + gate, K=8, epochs=3
- [ ] Full eval 0-30 + 0-60
- [ ] 与 006 B-2L 对比，验证 gate 是否带来收益
- [ ] 分析 gate 值分布：是否有 token 被持续抑制

### 设计思路
```python
class GatedVoCoCompressor(VoCoCompressor):
    def __init__(self, K=8, dim=3584, **kwargs):
        super().__init__(K=K, dim=dim, **kwargs)
        self.gate = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, dense_vision):
        compressed = super().forward(dense_vision)  # (K, D)
        gate = self.gate(compressed)                # (K, 1)
        return compressed * gate                    # (K, D)
```

### 可选变体
- **Per-dim gate**：`nn.Linear(dim, dim)` → 每个维度独立 gate
- **Shared gate across segments**：所有段共享 gate 参数 vs 段独立
- **Hard gate**：topk 选择而非 soft sigmoid

# 009-pooling-baseline: Attention Pooling 压缩

## 概述

用 weighted average pooling 替代 cross-attention 作为压缩方式的下界 baseline。验证 cross-attention（006）到底贡献了多少。

### 架构

```
Dense Vision Tokens (V, D)
        ↓  Learned attention scores → weighted average
Compressed Tokens (K, D)
        ↓  + Question → Frozen LLM → MSE with teacher
```

### 与 006 的区别

| 项目 | 006 Cross-Attention | 009 Pooling |
|------|---------------------|-------------|
| 压缩方式 | Learnable queries + cross-attn | Attention score + weighted avg |
| 参数量 | ~50M/layer | ~7K（一个 linear） |
| 表达力 | 强 | 弱（下界 baseline） |

---

## 计划

### 待做
- [ ] 实现 PoolingCompressor（替代 VoCoCompressor）
- [ ] 110K 训练 B loss, K=8, epochs=3
- [ ] Full eval 0-30 + 0-60
- [ ] 与 006 B-2L 对比，量化 cross-attention 的贡献

### 设计思路
```python
class PoolingCompressor(nn.Module):
    def __init__(self, K=8, dim=3584):
        self.score_proj = nn.Linear(dim, K)  # (V, D) → (V, K)
        self.out_norm = nn.LayerNorm(dim)
    
    def forward(self, dense_vision):
        scores = self.score_proj(dense_vision)  # (V, K)
        weights = softmax(scores, dim=0)        # (V, K)
        compressed = weights.T @ dense_vision   # (K, D)
        return self.out_norm(compressed)
```

# 010 Gated Compression 计划

## 目标
在 cross-attention compressor 输出后加 gate，验证 token 重要性加权是否带来收益。

## 步骤
1. [ ] 实现 GatedVoCoCompressor（继承 VoCoCompressor + gate）
   - `gate = sigmoid(linear(compressed))` → `compressed * gate`
2. [ ] 修改 train_compressor.py 支持 `--compressor_type gated`
3. [ ] 本地 sanity test
4. [ ] 110K 训练 B-2L + gate, K=8, epochs=3
5. [ ] Full eval 0-30 + 0-60
6. [ ] 与 006 B-2L 对比
7. [ ] 分析 gate 值分布（哪些 token 被抑制）

## 可选变体（根据初步结果决定）
- Per-dim gate：`nn.Linear(dim, dim)`
- Hard gate：topk 选择

## 依赖
- teacher cache 完成（005）
- 006 B-2L 作为对照

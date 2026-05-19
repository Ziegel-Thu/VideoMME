# 011 Question-Conditioned Compression 计划

## 目标
将 question 信息注入压缩过程，验证 question-aware 压缩是否优于 question-agnostic（006）。

## 步骤
1. [ ] 实现方式 C：ConditionedVoCoCompressor（加法注入）
   - `queries = learnable + pool(Q_embed)`
   - 改动最小，一个 linear 层
2. [ ] 修改 train_compressor.py：传 q_embeds 给 compressor
3. [ ] 修改 eval_compressor.py：同上
4. [ ] 本地 sanity test
5. [ ] 110K 训练 B-2L + Q-conditioned, K=8, epochs=3
6. [ ] Full eval 0-30 + 0-60
7. [ ] 与 006 B-2L 对比
8. [ ] 如果方式 C 有收益，实现方式 B（双层 cross-attn）

## 注意事项
- 蒸馏训练阶段无额外代价（每样本本来就有一个 Q）
- 推理时不能复用 cache（不同问题需重新压缩）
- teacher cache 中已包含 q_embeds，不需要重新提取

## 依赖
- teacher cache 完成（005）
- 006 B-2L 作为对照

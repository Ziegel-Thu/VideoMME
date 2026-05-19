# 012 Adapter Compression 计划

## 目标
在 LLM 内部用 adapter 做压缩，让 compressed tokens 天然对齐 LLM hidden space。

## 步骤
1. [ ] 调研 LLaMA-Adapter / TokenLearner 实现细节
2. [ ] 确定方案：前缀式 or adapter 层注入
3. [ ] 实现 AdapterCompressor
4. [ ] 验证与 gradient checkpointing 兼容性
5. [ ] 本地 sanity test
6. [ ] 110K 训练 K=8, epochs=3
7. [ ] Full eval 对比 006

## 风险
- 侵入性强：需要修改 LLM forward 逻辑
- 前 N 层需要过完整 V tokens，显存可能偏大
- gradient checkpointing 兼容性未知

## 参考论文
- LLaMA-Adapter
- TokenLearner
- Perceiver IO

## 依赖
- teacher cache 完成（005）
- 006 B-2L 作为对照
- 优先级最低，等 009-011 结果出来后再决定是否启动

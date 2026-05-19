# 009 Pooling Baseline 计划

## 目标
用 attention pooling 作为压缩下界 baseline，量化 cross-attention 的贡献。

## 步骤
1. [ ] 实现 PoolingCompressor（替代 VoCoCompressor）
   - `score_proj = nn.Linear(dim, K)` → softmax → weighted avg
   - 参数量 ~7K vs cross-attention ~50M
2. [ ] 修改 train_compressor.py 支持 `--compressor_type pool`
3. [ ] 本地 sanity test（4 样本 1 epoch）
4. [ ] 110K 训练 B loss, K=8, epochs=3
5. [ ] Full eval 0-30 + 0-60
6. [ ] 与 006 B-2L 对比出结论

## 依赖
- teacher cache 完成（005）

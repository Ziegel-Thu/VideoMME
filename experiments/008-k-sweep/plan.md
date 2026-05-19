# 008 K-sweep 计划

## 目标
扫描 K_seg=2/4/8/16/32，找到压缩率和精度的最佳平衡点。

## 配置
- 固定：B loss, 2L, lr=1e-4, epochs=3
- 变量：K_seg = 2 / 4 / 8 / 16 / 32
- amlt yaml: `amlt_k_sweep.yaml`（5 个 job 并行）

## 待做
- [ ] 等 teacher cache 补完
- [ ] 提交 5 个 K 值训练
- [ ] 每个 K 值取 best epoch
- [ ] Full eval 0-30 + 0-60
- [ ] 画 K vs Accuracy 曲线
- [ ] 确定后续默认 K 值

## 预期
- K↑ → 精度↑ 压缩比↓
- K=32 ≈ 003 BN 压缩比（12.5:1 vs 13.5:1）
- K=2 极端压缩，看下界

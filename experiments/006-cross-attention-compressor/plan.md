# 006 Cross-Attention Compressor 计划

## 目标
验证 cross-attention compressor 的 loss 类型和深度 ablation。

## 已完成
- [x] 10K B/D/BD/B-2L pilot
- [x] 110K B-1L 3 epoch + full eval
- [x] 110K B-2L 3 epoch + full eval
- [x] Sanity test 在 A100 集群跑通
- [x] Checkpoint 迁移到 blob（25 个）

## 进行中
- [ ] 110K BD-1L 训练（yaml: `amlt_train_bd.yaml`，等 cache 完成）

## 待做
- [ ] B-2L epoch1/2 full eval 补跑
- [ ] 用 blob checkpoint 在 amlt 集群 eval
- [ ] 汇总 B/D/BD × 1L/2L 完整结果表
- [ ] eval amlt yaml 编写

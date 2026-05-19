# 007 Inter-Segment Attention 计划

## 目标
验证段间 self-attention 在 110K 数据上是否有收益（10K 上无收益）。

## 已完成
- [x] 10K pilot：B-1L + inter-seg 1L → 66.5%（无收益）
- [x] Checkpoint 迁移到 blob
- [x] amlt yaml 就绪（`amlt_train.yaml`）

## 待做
- [ ] 等 teacher cache p2/p3 补完
- [ ] 提交 110K B-1L + inter_layers=1 训练
- [ ] 每 epoch 做 quick eval（200 条）
- [ ] 最终 full eval 0-30 + 0-60
- [ ] 与 006 B-1L 对照，出结论

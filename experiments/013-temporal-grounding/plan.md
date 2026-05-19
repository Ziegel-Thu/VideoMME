# 013 Temporal Grounding 计划

## 目标
在 compressor 输出上训练 temporal head，预测证据时间段。

## 步骤
1. [ ] 适配 train_temporal.py：输入改为 compressor hidden states（非 voco hidden）
2. [ ] 确认 temporal_evidence.jsonl 在 blob 上可用
3. [ ] 加载 006 B-2L best checkpoint 作为 frozen compressor
4. [ ] 本地 sanity test（少量样本验证 loss 能降）
5. [ ] 110K temporal 训练（12K temporal 数据）
6. [ ] 评测 tIoU / Recall@0.5

## 依赖
- 006 B-2L best checkpoint（已在 blob）
- temporal_evidence.jsonl（需确认 blob 路径）
- teacher cache 不需要（temporal head 直接用视频）

## 注意事项
- N=4 帧时 88% temporal labels 全零 → 需要更多帧
- 用 1fps 采样，0-60s 视频最多 60 帧 / 30 段
- 16-bin 离散化与帧数解耦
- 可能需要调 pos_weight 平衡正负样本

## 参照
- 002 tIoU=0.3148, Recall@0.5=27%（K=32, N=4, bottleneck）

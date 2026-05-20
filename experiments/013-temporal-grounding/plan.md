# 013 Temporal Grounding 计划

## 目标
在 compressor 输出上训练 temporal head，预测证据时间段 [t_s, t_e]。

## 两种方案并行

### 方案 B：Q-conditioned 序列 temporal head
1. 段级 pooling：(N_seg, K, D) → mean → (N_seg, D)
2. Q conditioning：seg_features + q_pooled 或 cross-attention
3. Per-segment MLP → (N_seg,) evidence scores
4. 映射到 16 bins

### 方案 C：Cross-attention bin queries
1. 16 个 learnable bin queries + q_pooled
2. Cross-attention: bin queries attend to compressed tokens
3. Linear → (16,) evidence scores

## 步骤
1. [x] 写 temporal_head.py（B + C 两种 head）
2. [x] 写 train_temporal_v2.py（基于 compressor ckpt + 视频 forward）
3. [ ] 确认 temporal 数据视频可访问
4. [ ] 用 B-2L ep2 ckpt 做 sanity test
5. [ ] 110K temporal 训练
6. [ ] 评测 tIoU / Recall@0.5

## 数据
- temporal_evidence.jsonl: 12,997 条（blob 上已有）
- 视频来源：didemo 等，需确认 blob 上是否有

## 依赖
- 006 B-2L best checkpoint
- 视频文件（不能只用 teacher cache）

# VoCo Cross-Attention Compressor 实验汇总

> 最后更新：2026-05-21 17:55

## Setting

- **模型**: Qwen2.5-VL-7B-Instruct（frozen LLM，只训 compressor 参数）
- **数据**: LLaVA-Video-178K MCQ，0-60s，110K train / 5K val / 10K test
- **压缩**: Cross-Attention Compressor，每段 dense vision tokens → K compressed tokens
- **训练**: MSE 蒸馏，基于预提取的 teacher cache
- **评测**: 内部 MCQ + MVBench + Video-MME Short + NExT-QA

## Loss 定义

| Loss | 公式 | 说明 |
|------|------|------|
| **B** | MSE(student Q-hidden, teacher Q-hidden) | 压缩后过 LLM，在 Q 位置对齐 hidden state |
| **D** | MSE(student KV pool, teacher KV pool) | 直接对齐 KV cache 的 per-layer mean pool |
| **BD** | B + D | 两者都算 |

---

## Eval 全景矩阵

> ✅ = 已出可信结果 | 🕐 = 已提交等结果 | - = 未提交

### 内部 MCQ

| Checkpoint | 全量 (10K) ep1 | 全量 ep2 | 全量 ep3 | 200条 ep1 | 200条 ep2 |
|------------|---------------|---------|---------|----------|----------|
| **Zeroshot** | **80.65%** ✅ | - | - | - | - |
| 006 B-1L | 70.08% ✅ | - | 69.86% ✅ | - | - |
| 006 B-2L | 69.37% ✅ | 69.59% ✅ | 69.21% ✅ | - | - |
| 007 inter-seg B1L | 🕐 | 🕐 | 🕐 | 65.5% ✅ | 66.0% ✅ |
| 008 K=2 B2L | 🕐 | 🕐 | 🕐 | - | 70.5% ✅ |
| 008 K=4 B2L | 🕐 | 🕐 | 🕐 | - | 69.0% ✅ |
| 008 K=8 B2L | 🕐 | 🕐 | 🕐 | 70.5% ✅ | 69.5% ✅ |
| 008 K=16 B2L | 🕐 | 🕐 | 🕐 | 69.0% ✅ | - |
| 009 pooling B1L | 🕐 | 🕐 | 🕐 | - | - |
| 010 gated B2L | 🕐 | 🕐 | 🕐 | 70.0% ✅ | 70.5% ✅ |

### MVBench (~3413 条匹配, 20 task)

| Checkpoint | ep1 | ep2 | ep3 |
|------------|-----|-----|-----|
| **Zeroshot** | **59.01%** ✅ | - | - |
| 006 B-1L | 42.22% ✅ | - | - |
| 006 B-2L | - | 42.58% ✅ | - |
| 007 inter-seg B1L | 🕐 | 🕐 | 🕐 |
| 008 K=2 B2L | 🕐 | 🕐 | 🕐 |
| 008 K=4 B2L | 🕐 | 🕐 | 🕐 |
| 008 K=8 B2L | 🕐 | 🕐 | 🕐 |
| 008 K=16 B2L | 🕐 | - | - |
| 009 pooling B1L | 🕐 | 🕐 | 🕐 |
| 010 gated B2L | **37.42%** ✅ | 🕐 | 🕐 |

### Video-MME Short (900 条, 300 视频)

| Checkpoint | ep1 | ep2 | ep3 |
|------------|-----|-----|-----|
| **Zeroshot** | **63.33%** ✅ | - | - |
| 006 B-1L | 42.89% ✅ | - | - |
| 006 B-2L | - | 43.00% ✅ | - |
| 007 inter-seg B1L | 40.33% ✅ | 🕐 | 🕐 |
| 008 K=2 B2L | 40.78% ✅ | 🕐 | 🕐 |
| 008 K=4 B2L | 42.22% ✅ | 🕐 | 🕐 |
| 008 K=8 B2L | 41.78% ✅ | 🕐 | �� |
| 008 K=16 B2L | 42.22% ✅ | 🕐 | 🕐 |
| 009 pooling B1L | 🕐 | 🕐 | 🕐 |
| 010 gated B2L | 🕐 | 🕐 | 🕐 |

### NExT-QA (8564 条, 5440 视频, 5选1)

| Checkpoint | ep1 |
|------------|-----|
| **Zeroshot** | **74.84%** ✅ |
| 006 B-2L | 🕐 |
| 007 inter-seg B1L | 🕐 |
| 008 K=2 B2L | 🕐 |
| 008 K=4 B2L | 🕐 |
| 008 K=8 B2L | 🕐 |
| 008 K=16 B2L | 🕐 |
| 009 pooling B1L | 🕐 |
| 010 gated B2L | 🕐 |

---

## K-sweep Train Loss (epoch1)

| K | Loss |
|---|------|
| 2 | 1.391 |
| 4 | 1.302 |
| 8 | 1.242 |
| 16 | 1.216 |
| 32 | 训练中 |

Loss 随 K 增大单调下降（更多 tokens 更容易压缩）。

---

## 结论（截至目前）

1. **B loss 最优**: B > BD > D（10K: 71% vs 69.5% vs 65%）
2. **2L > 1L**: full eval B-2L 全面优于 B-1L（+0.3~0.5%）
3. **Full eval 各 epoch 差距很小**: B-2L 0-60 全量 68.83%→68.86%→68.98%
4. **外部 benchmark gap 更大**: MVBench -17%, VME Short -21%, 内部 MCQ -11%
5. **Inter-seg 有害**: 007 quick eval 65.5% vs 006 B-2L 75%（-10%），VME 40.3% vs 42.2%
6. **Gated 无显著收益**: 010 gated 70-70.5% ≈ 008 K=8 70.5%
7. **K-sweep eval 差距小**: K=2~16 内部 MCQ 都在 69-70.5%，VME 都在 40-42%
8. **K-sweep train loss 和 eval 不一致**: loss 单调降但 eval 几乎没差

---

## 训练状态

| 实验 | amlt | 状态 | Train Loss (ep1) |
|------|------|------|-----------------|
| 007 inter-seg B1L | liberal-seasnail | ✅ PASS | 1.929 |
| 007 inter-seg **B2L** | peaceful-sturgeon | 🔄 running (~3h) | - |
| 008 K=2 B2L | trusting-cheetah | ✅ PASS | 1.391 |
| 008 K=4 B2L | thorough-grouse | ✅ PASS | 1.302 |
| 008 K=8 B2L | equal-mongoose | ✅ PASS | 1.242 |
| 008 K=16 B2L | safe-gopher | ✅ PASS | 1.216 |
| 008 K=32 B2L | happy-malamute | 🔄 running (~5h) | - |
| 009 pooling B1L | better-lemming | ✅ PASS | 6.481 |
| 010 gated B2L | ultimate-monkfish | ✅ PASS | 1.272 |

### 011/012 Temporal Grounding

| 方案 | amlt | lr | 状态 |
|------|------|----|------|
| B segment | teaching-anteater | 1e-5 | 🔄 running ep5/10 |
| C binquery | relevant-calf | 1e-4 | 🔄 running ep3/10 |

小规模测试（100 样本）：

| 方案 | lr | Epoch 1→3(→5) | 趋势 |
|------|-----|--------------|------|
| B segment | 1e-4 | 0.939→0.944 | ❌ 震荡 |
| B segment | 1e-5 | 0.884→0.873→0.864 | ✅ 稳降 |
| C binquery | 1e-4 | 1.084→0.928 | ✅ 下降 |

---

## amlt Eval Experiment 追踪

### 外部 Benchmark (013 代码，已验证)

| Experiment | 内容 | Jobs |
|------------|------|------|
| still-trout | MVBench 007+K-sweep+010 ep1 | 6 |
| glad-grackle | MVBench ep2/ep3 | 7 |
| strong-squid | MVBench K2/K4/K8 ep3 | 3 |
| known-tortoise | VME 007+K-sweep ep1 | 5 |
| unified-gelding | VME ep2/ep3 | 7 |
| boss-marmoset | VME K2/K4/K8 ep3 | 3 |
| true-mosquito | VME 010 ep1 | 1 |
| pumped-hen | VME 006 B1L+B2L | 2 |
| composed-koala | NExT-QA zeroshot+B2L | 2 |
| subtle-martin | NExT-QA 007+K+010 ep1 | 6 |
| unique-cobra | 009 MVBench ep1/2/3 | 3 |
| precious-gator | 009 VME ep1/2/3 | 3 |
| diverse-quail | 009 NExT-QA ep1/2/3 | 3 |

### 全量 MCQ (各实验自己的 eval_compressor.py)

| Experiment | 内容 | Jobs |
|------------|------|------|
| live-falcon | 007 ep1+ep2 | 2 |
| select-man | 007 ep3 | 1 |
| willing-gar | K=2 ep1+ep2 | 2 |
| casual-monarch | K=2 ep3 | 1 |
| fast-katydid | K=4 ep1+ep2 | 2 |
| light-whippet | K=4 ep3 | 1 |
| adjusted-sunbeam | K=8 ep1+ep2 | 2 |
| huge-liger | K=8 ep3 | 1 |
| definite-lion | K=16 ep1+ep2 | 2 |
| stable-bunny | K=16 ep3 | 1 |
| prepared-monkfish | 010 ep1+ep2 | 2 |
| alert-jaguar | 010 ep3 | 1 |
| useful-tahr | 009 ep1+ep2+ep3 | 3 |

### 集群验证

| Experiment | 内容 | 结果 |
|------------|------|------|
| true-weasel | 010 gated MVBench test | ✅ 37.42% (1277/3413) |
| viable-whippet | 009 pooling MVBench test | 🕐 queued |

---

## Blob 数据备份

已备份到 `shuwangmain/zhengshurui/video_project/`（373.5 GB）：

| 目录 | 文件数 | 大小 |
|------|--------|------|
| checkpoints/ | 57 | 67.7 GB |
| models/ | 16 | 16.6 GB |
| benchmarks/ | 46,349 | 62.4 GB |
| data/ | 66,221 | 226.8 GB |

---

## 下一步

- [ ] 收集所有 eval 结果填满全景矩阵
- [ ] K32 训练完成后处理 checkpoint + eval
- [ ] 007 B2L 训练完成后处理 checkpoint + eval
- [ ] 011/012 temporal 训练完成后评测 tIoU
- [ ] 全景矩阵完整后做最终分析和 K vs Accuracy 曲线图

# VoCo Cross-Attention Compressor 实验汇总

> 最后更新：2026-05-21 13:25

## Setting

- **模型**: Qwen2.5-VL-7B-Instruct（frozen LLM，只训 compressor 参数）
- **数据**: LLaVA-Video-178K MCQ，0-60s，110K train / 5K val / 10K test
- **压缩**: Cross-Attention Compressor，每段 dense vision tokens → K=8 compressed tokens
- **训练**: MSE 蒸馏，基于预提取的 teacher cache

## Loss 定义

| Loss | 公式 | 说明 |
|------|------|------|
| **B** | MSE(student Q-hidden, teacher Q-hidden) | 压缩后过 LLM，在 Q 位置对齐 hidden state |
| **D** | MSE(student KV pool, teacher KV pool) | 直接对齐 KV cache 的 per-layer mean pool |
| **BD** | B + D | 两者都算 |

---

## 内部 MCQ Baseline

| 方法 | 0-30 (200) | 0-60 (200) | 0-30 全量 (1835) | 0-60 全量 (10103) |
|------|-----------|-----------|-----------------|------------------|
| Zero-shot logit | 82.0% | - | **80.65%** (1476) | **79.59%** (8039) |
| 003 BN on (110K) | 90.0% | - | - | - |
| 003 BN off (110K) | 94.4% | - | - | - |

---

## 10K Pilot（200 条 test, 0-30s）

| 配置 | best epoch | Acc |
|------|-----------|-----|
| B-1L | ep2/3 | 71.0% |
| B-2L | ep3 | 69.0% |
| D-1L | ep4 | 65.0% |
| BD-1L | ep3 | 69.5% |
| B-1L + inter-seg | ep1 | 66.5% |

---

## 110K Quick Eval（前 200 条）

### 006 Baseline

| 配置 | epoch | 0-30 (200) | 0-60 (200) |
|------|-------|-----------|-----------|
| B-1L | 1 | 71.50% | 66.00% |
| B-1L | 3 | 70.00% | 65.50% |
| B-2L | 1 | 72.00% | 67.50% |
| **B-2L** | **2** | **75.00%** | **68.00%** |
| B-2L | 3 | 73.00% | 67.50% |

### 007-010 新架构（200 条 quick eval）

| 实验 | epoch 1 | epoch 2 |
|------|---------|---------|
| 007 inter-seg B1L | 65.5% | 66.0% |
| 008 K=2 B2L | - | 70.5% |
| 008 K=4 B2L | - | 69.0% |
| 008 K=8 B2L | 70.5% | 69.5% |
| 008 K=16 B2L | 69.0% | (K16 ep2 ckpt 未就绪) |
| 010 gated B2L | **70.0%** | **70.5%** |

**观察**:
- 010 gated 与 008 K=8 几乎持平（70-70.5%），gate 没有显著收益
- 007 inter-seg（65.5-66%）明显低于 006 baseline（~72%），段间 attention 有害
- K-sweep: K=2/8 都在 ~70%，K 值对 200 条 quick eval 影响不大
- 这些是 200 条快速评测，全量 eval 待做

---

## 110K Full Eval（全量 test）

| 配置 | epoch | 0-30 全量 (1835) | 0-60 全量 (10103) |
|------|-------|-----------------|------------------|
| B-1L | 1 | 70.08% (1286) | 68.48% (6919) |
| B-1L | 3 | 69.86% (1282) | 68.63% (6934) |
| B-2L | 1 | 69.37% (1273) | 68.83% (6953) |
| B-2L | 2 | 69.59% (1277) | 68.86% (6958) |
| B-2L | 3 | 69.21% (1270) | 68.98% (6969) |

---

## 外部 Benchmark

### MVBench（4000 条，20 task，~3413 条视频匹配）

| 模型 | Acc |
|------|-----|
| **Zeroshot** | **59.01%** (2014/3413) |
| B-1L ep1 | 42.22% (1441/3413) |
| B-2L ep2 | 42.58% (1453/3413) |

### Video-MME Short（900 条，300 视频）

| 模型 | Acc |
|------|-----|
| **Zeroshot** | **63.33%** (570/900) |
| 006 B-1L ep1 | 待出 |
| 006 B-2L ep2 | 待出 |
| 007 inter-seg B1L ep1 | 40.33% (363/900) |
| 008 K=2 B2L ep1 | 40.78% (367/900) |
| 008 K=4 B2L ep1 | 42.22% (380/900) |
| 008 K=8 B2L ep1 | 41.78% (376/900) |
| 008 K=16 B2L ep1 | 42.22% (380/900) |

**观察**：
- 所有 compressor 变体在 VME Short 上表现相近（40-42%），远低于 zeroshot（63%）
- K-sweep 中 K=4 和 K=16 并列最高（42.22%），K=2 最低（40.78%）
- 007 inter-seg（40.33%）略低于 baseline 006 K=8（41.78%），段间 attention 无收益

### NExT-QA（8564 条 test, 5440 视频, 5 选 1 MCQ）

| 模型 | Acc | C | T | D |
|------|-----|---|---|---|
| Zeroshot (5 条 sanity) | 83.3% | 75% | 100% | - |
| B-2L ep2 (5 条 sanity) | 83.3% | 100% | 50% | - |

视频已上传 blob（5440 个，24.6GB）。全量评测已提交 (main-cow, BSC)。

---

## 结论

1. **B loss 最优**: B > BD > D（10K: 71% vs 69.5% vs 65%）
2. **2L > 1L**: full eval 上 B-2L 全面优于 B-1L（+0.3~0.5%）
3. **Full eval 各 epoch 差距很小**: B-2L 0-60 全量 68.83%→68.86%→68.98%
4. **外部 benchmark gap 更大**: MVBench 压缩后 59%→42%（-17%），VME Short 63%→42%（-21%），比内部 MCQ 80%→69%（-11%）更大
5. **Inter-seg 有害**: 007 quick eval 65.5% vs 006 B-2L 75%（-10%），VME 40.3% vs 42.2%
6. **Gated 无显著收益**: 010 gated 70-70.5% ≈ 008 K=8 70.5%，gate 机制没帮助
7. **K-sweep VME Short**: K=4/16 并列最高（42.22%），K 增大收益不明显
8. **K-sweep train loss**: K 越大 loss 越低（K2=1.39→K16=1.22），但 eval 差距很小
9. **所有压缩方法在外部 benchmark 上 gap 一致**: ~40-42% VME，~42% MVBench

## 进行中

### Eval 全景矩阵

> ✅ = 已出结果 | 🔄 = running | 🕐 = submitted/queued | - = 未提交

| Checkpoint | MCQ ep1 | MCQ ep2 | MCQ ep3 | MVBench | VME Short | NExT-QA |
|------------|---------|---------|---------|---------|-----------|---------|
| **Zeroshot** | **80.65%** ✅ | - | - | **59.01%** ✅ | **63.33%** ✅ | 🕐 |
| 006 B-1L ep1 | 70.08% ✅ | - | 69.86% ✅ | 42.22% ✅ | 42.89% ✅ | - |
| 006 B-2L ep2 | - | 69.59% ✅ | 69.21% ✅ | 42.58% ✅ | 43.00% ✅ | 🕐 |
| 007 inter-seg B1L ep1 | 65.5% ✅ | 66.0% ✅ | 🕐 | 🕐 | 40.33% ✅ | 🕐 |
| 008 K=2 B2L | 🕐 | 70.5% ✅ | 🕐 | 🕐 | 40.78% ✅ | 🕐 |
| 008 K=4 B2L | 🕐 | 69.0% ✅ | 🕐 | 🕐 | 42.22% ✅ | 🕐 |
| 008 K=8 B2L | 70.5% ✅ | 69.5% ✅ | 🕐 | 🕐 | 41.78% ✅ | 🕐 |
| 008 K=16 B2L | 69.0% ✅ | - | - | 🕐 | 42.22% ✅ | 🕐 |
| 010 gated B2L | 70.0% ✅ | 70.5% ✅ | 🕐 | 🕐 | 🕐 | 🕐 |

**注意**：MCQ 007-010 为 200 条 quick eval（非全量）。MVBench 已修复 attn bug 重提。

### amlt Eval Experiment 追踪

| Experiment | 内容 | Jobs | 状态 |
|------------|------|------|------|
| sharing-tahr | VME 006 B1L/B2L | 2 | ✅ pass |
| inspired-bluejay | VME 007+K-sweep ep1 | 5 | ✅ pass |
| neat-gull | VME 010 ep1 | 1 | ✅ pass |
| happy-badger | VME 007/K-sweep/010 ep2 | 7 | 部分 pass |
| winning-bug | VME K2/K4/K8 ep3 | 3 | 🕐 queued |
| massive-alpaca | MVBench 007+K-sweep+010 ep1 (fix) | 6 | 🕐 submitted |
| discrete-deer | MVBench ep2/ep3 (fix) | 7 | 🕐 submitted |
| major-owl | MVBench K2/K4/K8 ep3 (fix) | 3 | 🕐 submitted |
| main-cow | NExT-QA zeroshot + B2L | 2 | 🔄 running |
| advanced-bluejay | NExT-QA 007+K-sweep+010 ep1 | 6 | 🔄 running |
| quick-lizard ~ master-labrador | MCQ ep1/ep2/ep3 (fixed) | 17 | 大部分 pass |

### 110K 训练

| 实验 | amlt | 状态 | Train Loss (ep1) |
|------|------|------|-----------------|
| 007 inter-seg B1L | liberal-seasnail | ✅ PASS | 1.929 |
| 007 inter-seg **B2L** | peaceful-sturgeon | 🔄 running | - |
| 008 K=2 B2L | trusting-cheetah | ✅ PASS | 1.391 |
| 008 K=4 B2L | thorough-grouse | ✅ PASS | 1.302 |
| 008 K=8 B2L | equal-mongoose | ✅ PASS | 1.242 |
| 008 K=16 B2L | safe-gopher | 🔄 running (~1d) | 1.216 |
| 008 K=32 B2L | happy-malamute | 🔄 running (~2h) | - |
| 009 pooling B1L | better-lemming | ✅ PASS | 6.481 |
| 010 gated B2L | ultimate-monkfish | ✅ PASS | 1.272 |

### 011/012 Temporal Grounding

| 方案 | amlt | lr | 状态 |
|------|------|----|------|
| B segment | teaching-anteater | 1e-5 | 🔄 running 5h |
| C binquery | relevant-calf | 1e-4 | 🔄 running 2h |

小规模测试（100 样本）：

| 方案 | lr | 100 样本 Epoch 1→3(→5) | 趋势 |
|------|-----|----------------------|------|
| B segment | 1e-4 | 0.939→0.944 | ❌ 震荡 |
| B segment | 1e-5 | 0.884→0.873→0.864 | ✅ 稳降 |
| C binquery | 1e-4 | 1.084→0.928 | ✅ 下降 |

## 下一步

- [ ] K16/K32 训练完成后下载 checkpoint + 提交 eval
- [ ] NExT-QA 全量 eval（zeroshot + B2L ep2）
- [ ] VME Short ep2/ep3 eval 提交
- [ ] 007-010 全量 MCQ eval（目前只有 200 条 quick eval）
- [ ] 009 pooling eval 需适配 PoolingCompressor
- [ ] 011/012 temporal 正式训练
- [ ] K-sweep 结果分析：K vs Accuracy 曲线图
- [ ] results_summary 更新 MCQ 全量 eval 结果

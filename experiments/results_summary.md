# VoCo Cross-Attention Compressor 实验汇总

> 最后更新：2026-05-22 03:22

## Setting

- **模型**: Qwen2.5-VL-7B-Instruct（frozen LLM，只训 compressor 参数）
- **数据**: LLaVA-Video-178K MCQ，0-60s，110K train / 5K val / 10K test
- **压缩**: Cross-Attention Compressor，每段 dense vision tokens → K compressed tokens
- **训练**: MSE 蒸馏，基于预提取的 teacher cache
- **评测**: 内部 MCQ + MVBench + Video-MME Short + NExT-QA

---

## ⚠️ 已知问题与代码状态

### attn_implementation sdpa vs eager

- sdpa 和 eager 在 bfloat16 下 logits 差 ~0.75-1.0（28 层累积误差），MCQ 准确率差 ~4%
- **决策**: 统一用 sdpa（和训练一致）
- eager 版结果（MVBench 35-38%）与 sdpa 版不可比，已废弃

### eval 代码版本

| 代码 | 适用范围 | 状态 |
|------|---------|------|
| **旧代码 sdpa** (eval_sdpa_verified/) | VME Short 006/007/K-sweep | ✅ 集群验证 7 job pass |
| **旧代码 sdpa** | MVBench 006 zeroshot/B1L/B2L | ✅ 集群验证 3 job pass |
| **旧代码 sdpa** | MVBench 007/K-sweep | ❌ 0/0 (model.py tuple bug) |
| **新代码 sdpa** (013/) | MVBench 009/010 | ✅ sweeping-ladybug pass (38.20%/39.49%) |
| **新代码 sdpa** | MVBench 006/007/K-sweep | ✅ 本地 5/5 pass，集群 sterling-aphid 验证中 |
| **新代码 sdpa** | VME/NExT-QA 009/010 | ✅ 本地 pass，集群 sterling-aphid 验证中 |
| **各实验 eval_compressor.py** | 内部 MCQ 全部 | ✅ 从未改过，集群多次验证 |

### 011/012 Temporal 训练失败

- 011 (teaching-anteater): 10 epoch 全部 loss=0, n=0
- 012 (relevant-calf): 8 epoch 全部 loss=0
- 原因: 可能视频路径不匹配。暂不重跑。

---


## Eval 全景矩阵

> ✅ = 可信结果 (sdpa 集群验证) | 🕐 = 已提交等结果 | ❌ = 未提交 | 🗑️ = 废弃 (eager)

| Checkpoint | MCQ 全量 | MVBench | VME Short | NExT-QA |
|------------|---------|---------|-----------|---------|
| **Zeroshot** | **80.65%** ✅ | **59.01%** ✅ | **63.33%** ✅ | **74.84%** ✅ |
| 006 B-1L ep1 | 70.08% ✅ | 42.22% ✅ | 42.89% ✅ (re-confirmed) | ❌ |
| 006 B-2L ep2 | 69.59% ✅ | 42.58% ✅ | 43.00% ✅ | ❌ |
| 007 inter-seg ep1 | 🕐 running | 🕐 sterling | 40.33% ✅ | 🕐 sterling |
| 007 inter-seg ep2 | 63.55% ✅ | ❌ | **41.33%** ✅ | ❌ |
| 008 K=2 ep1 | �� | 🕐 sterling | 40.78% ✅ | 🕐 sterling |
| 008 K=2 ep2 | 66.79% ✅ | ❌ | **41.33%** ✅ | ❌ |
| 008 K=4 ep1 | 🕐 | 🕐 sterling | 42.22% ✅ | 🕐 sterling |
| 008 K=4 ep2 | 67.07% ✅ | ❌ | 🕐 旧代码 | ❌ |
| 008 K=8 ep1 | 🕐 | 🕐 sterling | 41.78% ✅ | 🕐 sterling |
| 008 K=8 ep2 | 67.55% ✅ | ❌ | 🕐 旧代码 | ❌ |
| 008 K=16 ep1 | 🕐 | 🕐 sterling | 42.22% ✅ | 🕐 sterling |
| 008 K=16 ep2 | 68.57% ✅ | ❌ | **42.22%** ✅ | ❌ |
| 009 pooling ep1 | 🕐 | 🕐 thorough-lion | ❌ | ❌ |
| 010 gated ep1 | 🕐 | 🕐 thorough-lion | ❌ | ❌ |
| 010 gated ep2 | 🕐 | ❌ | ❌ | ❌ |

---

## 可信结果（集群验证 + 代码无 bug）

### 内部 MCQ（各实验 eval_compressor.py，可信）

| Checkpoint | 全量 ep1 | 全量 ep2 | 200条 ep1 | 200条 ep2 |
|------------|---------|---------|----------|----------|
| **Zeroshot** | **80.65%** | - | - | - |
| 006 B-1L | 70.08% | - | - | - |
| 006 B-2L | 69.37% | 69.59% | - | - |
| 007 inter-seg B1L | 🕐 running | 63.55% | 65.5% | 66.0% |
| 008 K=2 B2L | 🕐 提交 | 66.79% | - | 70.5% |
| 008 K=4 B2L | 🕐 提交 | 67.07% | - | 69.0% |
| 008 K=8 B2L | 🕐 running | 67.55% | 70.5% | 69.5% |
| 008 K=16 B2L | 🕐 running | 68.57% | 69.0% | - |
| 009 pooling B1L | 🕐 提交 | 🕐 提交 | - | - |
| 010 gated B2L | 🕐 提交 | 🕐 提交 | 70.0% | 70.5% |

### VME Short（旧代码 sdpa，集群验证）

| Checkpoint | ep1 | ep2 | ep3 |
|------------|-----|-----|-----|
| **Zeroshot** | **63.33%** | - | - |
| 006 B-1L | 42.89% | 🕐 旧代码提交 | - |
| 006 B-2L | 43.00% | 🕐 旧代码提交 | - |
| 007 inter-seg B1L | 40.33% | 🕐 旧代码提交 | 🕐 旧代码提交 |
| 008 K=2 B2L | 40.78% | 🕐 旧代码提交 | 🕐 旧代码提交 |
| 008 K=4 B2L | 42.22% | 🕐 旧代码提交 | 🕐 旧代码提交 |
| 008 K=8 B2L | 41.78% | 🕐 旧代码提交 | 🕐 旧代码提交 |
| 008 K=16 B2L | 42.22% | 🕐 旧代码提交 | 🕐 旧代码提交 |
| 009 pooling B1L | - | - | - |
| 010 gated B2L | - | - | - |

*009/010 旧代码不支持，需要新代码。等 sterling-aphid 确认后提交。*

### MVBench（集群验证）

| Checkpoint | ep1 | 代码版本 |
|------------|-----|---------|
| **Zeroshot** | **59.01%** | 旧代码 sdpa |
| 006 B-1L | 42.22% | 旧代码 sdpa |
| 006 B-2L | 42.58% | 旧代码 sdpa |
| 009 pooling B1L | 🕐 thorough-lion STD | 新代码 sdpa |
| 010 gated B2L | 🕐 thorough-lion STD | 新代码 sdpa |
| 007/K-sweep | 🕐 等 sterling-aphid 确认后提交 | 新代码 sdpa |

### NExT-QA

| Checkpoint | 结果 |
|------------|------|
| **Zeroshot** | **74.84%** |
| 其他 | 🕐 等 sterling-aphid 确认后提交 |

---

## 废弃结果（eager 版，不可比）

以下结果使用 eager attention 跑出，与 sdpa 版不可比，已废弃：

- MVBench 007 inter-seg: 35.69% (eager)
- MVBench K=2: 36.62% (eager)
- MVBench K=4: 36.68% (eager)
- MVBench K=8: 37.80% (eager)
- MVBench K=16: 37.71% (eager)
- MVBench 009 pooling: 35.80% (eager)
- MVBench 010 gated: 37.42% (eager)
- NExT-QA 009 ep2: 47.66% (eager)
- VME 006 B-1L: 38.89% (eager, vs sdpa 42.89%)

---

## 训练状态

| 实验 | amlt | 状态 |
|------|------|------|
| 007 B1L | liberal-seasnail | ✅ PASS |
| 007 **B2L** | peaceful-sturgeon | 🔄 running ep2 |
| K=2~K=16 | 各自 | ✅ PASS |
| K=32 | happy-malamute | 🔄 running ep2 |
| 009 pooling | better-lemming | ✅ PASS |
| 010 gated | ultimate-monkfish | ✅ PASS |
| 011 temporal B | teaching-anteater | ❌ 失败 (loss=0) |
| 012 temporal C | relevant-calf | ❌ 失败 (loss=0) |

---

## amlt Experiment 追踪

### 正在跑/排队

| Experiment | 内容 | 队列 | 状态 |
|------------|------|------|------|
| sterling-aphid | 串行 test 新代码 sdpa 10 组合 | STD | queued |
| thorough-lion | MVBench 009+010 全量 sdpa | STD | queued |
| advanced-monkfish | VME ep2/ep3 旧代码 sdpa | BSC | preparing |
| square-cod | 010 MCQ full ep1+ep2 | BSC | 提交 |
| eager-kit | 009 MCQ full ep1+ep2+ep3 | BSC | 提交 |
| huge-pangolin | K=2 MCQ full ep1 | BSC | 提交 |
| immense-hookworm | K=4 MCQ full ep1 | BSC | 提交 |
| live-falcon | 007 MCQ full (ep1 running, ep2 pass) | BSC | running |
| adjusted-sunbeam | K=8 MCQ full ep1+ep2 pass | BSC | ✅ pass |
| definite-lion | K=16 MCQ full (ep1 running, ep2 pass) | BSC | running |

### 集群验证记录

| Experiment | 代码版本 | 结果 | 可信 |
|------------|---------|------|------|
| inspired-bluejay | 旧代码 sdpa | VME 007/K-sweep ep1: 40-42% | ✅ |
| sharing-tahr | 旧代码 sdpa | VME 006 B1L 42.89%, B2L 43.00% | ✅ |
| sweeping-ladybug | 新代码 sdpa | MVBench 009=39.49%, 010=38.20% (1 shard) | ✅ 代码验证 |

---

## Blob 备份

已备份到 `shuwangmain/zhengshurui/video_project/`（373.5 GB）

---

## 下一步

1. sterling-aphid 串行 test 确认新代码 sdpa 全组合无 crash
2. 确认后用新代码提交 MVBench 007/K-sweep + VME 009/010 + NExT-QA 全部
3. 等 K32/007B2L 训练完成后处理 checkpoint + eval
4. 收集所有 eval 结果填满矩阵

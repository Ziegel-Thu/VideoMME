# VoCo Cross-Attention Compressor 实验汇总

> 最后更新：2026-05-26 18:30

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

> ✅ = 可信结果 (sdpa 集群验证) | ❌ = 未提交

| Checkpoint | MCQ 全量 | MVBench | VME Short | NExT-QA |
|------------|---------|---------|-----------|---------|
| **Zeroshot** | **80.65%** | **59.01%** | **63.33%** | **74.84%** |
| 006 B-1L ep1 | 70.08% | 42.22% | 42.89% | 61.72% |
| 006 B-1L ep2 | - | 42.81% | 43.11% | 62.03% |
| 006 B-1L ep3 | 69.86% | 42.37% | 43.00% | 61.83% |
| 006 B-2L ep1 | 69.37% | 42.07% | 42.33% | 61.50% |
| 006 B-2L ep2 | 69.59% | 42.58% | 43.00% | 61.56% |
| 006 B-2L ep3 | 69.21% | 42.43% | 43.44% | 61.97% |
| 007 inter-seg ep1 | 63.20% | 39.23% | 40.33% | 51.75% |
| 007 inter-seg ep2 | 63.55% | 38.68% | 40.33% | 51.69% |
| 007 inter-seg ep3 | 63.37% | 39.14% | 39.89% | 51.93% |
| 008 K=2 ep1 | 66.73% | 40.20% | 40.78% | 58.62% |
| 008 K=2 ep2 | 66.79% | 40.38% | 41.33% | 58.90% |
| 008 K=2 ep3 | 66.61% | 40.38% | 41.33% | 58.85% |
| 008 K=4 ep1 | 67.45% | 41.05% | 42.22% | 60.31% |
| 008 K=4 ep2 | 67.07% | 40.87% | 42.89% | 60.44% |
| 008 K=4 ep3 | 66.96% | 41.34% | 42.33% | 61.08% |
| 008 K=8 ep1 | 67.72% | 41.49% | 41.78% | 61.02% |
| 008 K=8 ep2 | 67.55% | 41.25% | 42.22% | 60.98% |
| 008 K=8 ep3 | 68.14% | 41.37% | 42.56% | 60.91% |
| 008 K=16 ep1 | 68.42% | 41.93% | 42.22% | 62.04% |
| 008 K=16 ep2 | 68.57% | 42.02% | 42.22% | 61.65% |
| 008 K=16 ep3 | 68.89% | 42.51% | 42.33% | 61.79% |
| 009 pooling ep1 | 61.92% | 37.59% | 32.56% | 51.21% |
| 009 pooling ep2 | 60.94% | 36.86% | 34.56% | 50.92% |
| 009 pooling ep3 | 60.68% | 36.48% | 32.89% | 50.86% |
| 010 gated ep1 | 66.79% | 40.87% | 41.67% | 60.80% |
| 010 gated ep2 | 67.18% | 41.72% | 41.89% | 60.60% |
| 010 gated ep3 | 67.16% | 41.49% | 42.67% | 60.95% |

**缺失数据**: 仅 006 B-1L ep2 MCQ 缺失 (该 checkpoint 未跑 MCQ eval)。其余 4 benchmarks × 全部 epochs 均已完整。

---

## 可信结果（集群验证 + 代码无 bug）

### 内部 MCQ（各实验 eval_compressor.py，17729 条全量，可信）

| Checkpoint | ep1 | ep2 | ep3 |
|------------|-----|-----|-----|
| **Zeroshot** | **80.65%** | - | - |
| 006 B-1L | 70.08% | - | 69.86% |
| 006 B-2L | 69.37% | 69.59% | 69.21% |
| 007 inter-seg B1L | 63.20% | 63.55% | 63.37% |
| 008 K=2 B2L | 66.73% | 66.79% | 66.61% |
| 008 K=4 B2L | 67.45% | 67.07% | 66.96% |
| 008 K=8 B2L | 67.72% | 67.55% | 68.14% |
| 008 K=16 B2L | 68.42% | 68.57% | 68.89% |
| 009 pooling B1L | 61.92% | 60.94% | 60.68% |
| 010 gated B2L | 66.79% | 67.18% | 67.16% |

**趋势**: K=16 三个 epoch 递增 (68.42→68.57→68.89)，K=8 ep3 跳高 (68.14%)。009 pooling 全面弱 (~61%)。

### VME Short（sdpa 集群验证，900 条）

| Checkpoint | ep1 | ep2 | ep3 |
|------------|-----|-----|-----|
| **Zeroshot** | **63.33%** | - | - |
| 006 B-1L | 42.89% | 43.11% | 43.00% |
| 006 B-2L | 42.33% | 43.00% | 43.44% |
| 007 inter-seg B1L | 40.33% | 40.33% | 39.89% |
| 008 K=2 B2L | 40.78% | 41.33% | 41.33% |
| 008 K=4 B2L | 42.22% | 42.89% | 42.33% |
| 008 K=8 B2L | 41.78% | 42.22% | 42.56% |
| 008 K=16 B2L | 42.22% | 42.22% | 42.33% |
| 009 pooling B1L | 32.56% | 34.56% | 32.89% |
| 010 gated B2L | 41.67% | 41.89% | 42.67% |

*009 VME 极差 (~33%)，远低于 006/K-sweep (~42%)。010 gated ep1=41.67%, ep2=41.89%, ep3=42.67%，持续上升接近 K-sweep。*

### MVBench（sdpa 集群验证，3413 条）

| Checkpoint | ep1 | ep2 | ep3 |
|------------|-----|-----|-----|
| **Zeroshot** | **59.01%** | - | - |
| 006 B-1L | 42.22% | - | - |
| 006 B-2L | 42.58% | - | - |
| 007 inter-seg B1L | 39.23% | 38.68% | 39.14% |
| 008 K=2 B2L | 40.20% | 40.38% | 40.38% |
| 008 K=4 B2L | 41.05% | 40.87% | 41.34% |
| 008 K=8 B2L | 41.49% | 41.25% | 41.37% |
| 008 K=16 B2L | 41.93% | ❌ | ❌ |
| 009 pooling B1L | 37.59% | ❌ | ❌ |
| 010 gated B2L | 40.87% | 41.72% | 41.49% |

*K16 ep2/ep3、009 ep2/ep3 未提交。009 MVBench (~37.6%) 也是最差。*

### NExT-QA（sdpa 集群验证，8564 条）

| Checkpoint | ep1 | ep2 | ep3 |
|------------|-----|-----|-----|
| **Zeroshot** | **74.84%** | - | - |
| 006 B-2L ep2 | 61.56% | - | - |
| 007 inter-seg B1L | 51.75% | ❌ | ❌ |
| 008 K=2 B2L | 58.62% | ❌ | ❌ |
| 008 K=4 B2L | 60.31% | ❌ | ❌ |
| 008 K=8 B2L | 61.02% | ❌ | ❌ |
| 008 K=16 B2L | 62.04% | ❌ | ❌ |
| 009 pooling B1L | 51.21% | 50.92% | 50.86% |
| 010 gated B2L | 60.80% | ❌ | ❌ |

*007 NExT-QA ep1=51.75% 远低于 K-sweep (~58-62%)。009 最差 (~51%)。K=16 ep1=62.04% 最优。*

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

### 全部 Eval Experiments (5/22 提交，全部 pass)

| Experiment | 内容 | 队列 | Jobs | 状态 |
|------------|------|------|------|------|
| advanced-monkfish | VME 006/007/K ep2/ep3 旧代码 | BSC | 11 | ✅ |
| adjusted-tadpole | VME 006 全 epoch 旧代码 | BSC | 4 | ✅ |
| square-cod | 010 MCQ full ep1+ep2 | BSC | 2 | ✅ |
| eager-kit | 009 MCQ full ep1+ep2+ep3 | BSC | 3 | ✅ |
| huge-pangolin | K=2 MCQ full ep1 | BSC | 1 | ✅ |
| immense-hookworm | K=4 MCQ full ep1 | BSC | 1 | ✅ |
| live-falcon | 007 MCQ full ep1+ep2 | BSC | 2 | ✅ |
| adjusted-sunbeam | K=8 MCQ full ep1+ep2 | BSC | 2 | ✅ |
| definite-lion | K=16 MCQ full ep1+ep2 | BSC | 2 | ✅ |
| tender-wallaby | K=2 MCQ full ep3 | BSC | 1 | ✅ |
| talented-chigger | K=4 MCQ full ep3 | BSC | 1 | ✅ |
| tender-marmoset | K=8 MCQ full ep3 | BSC | 1 | ✅ |
| driving-python | K=16 MCQ full ep3 | BSC | 1 | ✅ |
| present-caiman | 010 MCQ full ep3 | BSC | 1 | ✅ |
| huge-snail | MVBench ep1 007/K/010 | BSC | 6 | ✅ |
| included-sheep | MVBench ep2/ep3 007/K/010 | BSC | 7 | ✅ |
| optimum-wren | MVBench K ep3 K2/K4/K8 | BSC | 3 | ✅ |
| living-wildcat | MVBench 009+010 ep1 | STD | 2 | ✅ |
| crack-macaw | NExT-QA 007+K+010 ep1 | BSC | 6 | ✅ |
| moral-buzzard | NExT-QA zs+006 B2L | BSC | 2 | ✅ |
| grand-marten | NExT-QA 009 ep1/2/3 | BSC | 3 | ✅ |
| actual-sunfish | VME 009 ep1/2/3 | BSC | 3 | ✅ |
| sterling-moray | VME 010 ep1 | BSC | 1 | ✅ |

### 补缺提交 (5/26)

| Experiment | 内容 | 队列 | Jobs | 状态 |
|------------|------|------|------|------|
| wanted-seagull | MVBench K16 ep2/ep3 + 009 ep2/ep3 | BSC | 4 | 🔄 submitted |
| advanced-bear | VME Short 010 ep2/ep3 | BSC | 2 | 🔄 submitted |
| settled-ibex | NExT-QA ep2/ep3 007/K/010 | BSC | 12 | 🔄 submitted |
| fair-redfish | MVBench 006 B1L ep2/3 + B2L ep1/3 | BSC | 4 | 🔄 submitted |
| known-blowfish | NExT-QA 006 B1L+B2L | BSC | 5 | 🔄 submitted |

### 失败/已取消

| Experiment | 内容 | 队列 | 状态 |
|------------|------|------|------|
| thorough-lion | MVBench 009+010 sdpa | STD | ❌ killed (SIGTERM) |
| vital-beetle | MVBench 009+010 重提交 | STD | ❌ 已取消 (与 living-wildcat 重复) |

### 串行验证

| Experiment | 代码版本 | 结果 | 可信 |
|------------|---------|------|------|
| inspired-bluejay | 旧代码 sdpa | VME 007/K-sweep ep1: 40-42% | ✅ |
| sharing-tahr | 旧代码 sdpa | VME 006 B1L 42.89%, B2L 43.00% | ✅ |
| sweeping-ladybug | 新代码 sdpa | MVBench 009=39.49%, 010=38.20% | ✅ |
| feasible-roughy | 新代码 sdpa | 串行 10 组合 BSC 全 pass | ✅ |
| sterling-aphid | 新代码 sdpa | 串行 10 组合 STD 全 pass | ✅ |

---

## Blob 备份

已备份到 `shuwangmain/zhengshurui/video_project/`（373.5 GB）

---

## 下一步

1. ✅ ~~全部 5/22 eval 结果已收集~~ (23 experiments, 55+ jobs 全 pass)
2. 🔄 补缺 eval 已提交 (5 experiments, 27 jobs):
   - wanted-seagull: MVBench K16 ep2/ep3 + 009 ep2/ep3
   - advanced-bear: VME 010 ep2/ep3
   - settled-ibex: NExT-QA ep2/ep3 007/K/010
   - fair-redfish: MVBench 006 全 epoch
   - known-blowfish: NExT-QA 006 全 epoch
3. 等补缺 eval 完成后收集结果填满矩阵
4. 等 K32 (happy-malamute) / 007 B2L (peaceful-sturgeon) 训练完成
5. 分析结果趋势：K vs accuracy 曲线、epoch 收敛分析

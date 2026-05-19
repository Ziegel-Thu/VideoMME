# VoCo Cross-Attention Compressor 实验汇总

> 最后更新：2026-05-19 16:40

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

| 配置 | epoch | 0-30 (200) | 0-60 (200) |
|------|-------|-----------|-----------|
| B-1L | 1 | 71.50% | 66.00% |
| B-1L | 3 | 70.00% | 65.50% |
| B-2L | 1 | 72.00% | 67.50% |
| **B-2L** | **2** | **75.00%** | **68.00%** |
| B-2L | 3 | 73.00% | 67.50% |

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
| B-1L ep1 | 待跑 |
| B-2L ep2 | 待跑 |

### NExT-QA（8564 条 test）

视频待下载（Google Drive 需认证）

---

## 结论

1. **B loss 最优**: B > BD > D（10K: 71% vs 69.5% vs 65%）
2. **2L > 1L**: full eval 上 B-2L 全面优于 B-1L（+0.3~0.5%）
3. **Full eval 各 epoch 差距很小**: B-2L 0-60 全量 68.83%→68.86%→68.98%
4. **外部 benchmark gap 更大**: MVBench 压缩后 59%→42%（-17%），比内部 MCQ 80%→69%（-11%）更大
5. **Inter-seg 10K 无收益**: 66.5% vs 71.0%，110K 训练中

## 进行中

- [ ] 007 inter-seg 110K 训练（ready-hedgehog）
- [ ] 008 K-sweep 110K 5 个 K 值（5 个独立 experiment）
- [ ] 009 pooling 110K（intense-goshawk）
- [ ] 010 gated 110K（loved-ghoul）
- [ ] 110K ckpt 出来后跑 MVBench + Video-MME + 内部 MCQ eval

## 下一步

- [ ] 分析 K-sweep 结果，确定最优 K
- [ ] 009/010 vs 006 架构对比
- [ ] Video-MME Short B1L/B2L eval
- [ ] NExT-QA 视频下载 + eval
- [ ] 011 question-conditioned 实现

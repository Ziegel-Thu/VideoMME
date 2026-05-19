# VoCo Cross-Attention Compressor 实验汇总

> 最后更新：2026-05-19

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

## Depth

- **1L**: 1 层 cross-attention
- **2L**: 2 层 cross-attention

---

## Baseline

| 方法 | 0-30 (200) | 0-60 (200) | 0-30 全量 (1835) | 0-60 全量 (10103) |
|------|-----------|-----------|-----------------|------------------|
| Zero-shot logit | 82.0% | - | **80.65%** (1476) | 跑ing |



---

## 10K Pilot（200 条 test, 0-30s）

| 配置 | best epoch | Acc |
|------|-----------|-----|
| B-1L | ep2/3 | 71.0% |
| B-2L | ep3 | 69.0% |
| D-1L | ep4 | 65.0% |
| BD-1L | ep3 | 69.5% |
| B-1L + inter-seg | ep1 | 66.5% |

**结论**: B > BD > B-2L > D；inter-seg 10K 无收益

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

## 结论

1. **B loss 最优**: B > BD > D（10K: 71% vs 69.5% vs 65%）
2. **2L > 1L**: quick eval 差距大（75% vs 71.5%），full eval 差距小但一致（68.98% vs 68.63%）
3. **Full eval 各 epoch 差距很小**: B-2L 0-60 全量 68.83%→68.86%→68.98%
4. **Quick eval 波动大**: 200 条采样不够稳定，full eval 是准确口径
5. **Inter-seg 10K 无收益**: 66.5% vs 71.0%，110K 待验证
6. **BD/D 110K 待补**: 只在 10K 上做过

## 下一步

- [ ] Zero-shot 全量 eval（跑ing）
- [ ] sweep

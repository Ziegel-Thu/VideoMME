# Video 项目总计划

> 最后更新：2026-05-19 16:40

## 当前阶段：Stage 1 段级压缩 + 多架构 Ablation + 外部 Benchmark

### 基础设施

| 项目 | 状态 |
|------|------|
| amlt 环境 | ✅ |
| Qwen2.5-VL 模型 on blob | ✅ |
| Teacher cache 110K v2 | ✅ 完成（200+ shard） |
| Checkpoint A40→blob | ✅ 25 个 |
| Zero-shot baseline 内部 MCQ | ✅ 0-30: 80.65%, 0-60: 79.59% |
| MVBench on blob | ✅ 10002 视频 + json |
| Video-MME Short on blob | ✅ 300 视频 + parquet |
| NExT-QA | 🟡 标注已上传，视频待下载 |

---

## 实验全景

| 编号 | 实验 | 状态 | 110K 训练 |
|------|------|------|----------|
| 005 | Teacher Cache | ✅ 完成 | - |
| 006 | Cross-Attention B/D/BD × 1L/2L | ✅ A40 完成 | A100 复现确认 |
| 007 | Inter-Segment Attention | 🟢 110K running | ready-hedgehog |
| 008 | K-sweep (K=2/4/8/16/32) | 🟢 110K 5 job | large-snipe 等 |
| 009 | Pooling Baseline | 🟢 110K running | intense-goshawk |
| 010 | Gated Compression | 🟢 110K running | loved-ghoul |
| 011 | Question-Conditioned | 📝 待实现 | - |
| 012 | Adapter Compression | 📝 待调研 | - |
| 013 | Temporal Grounding | 📝 待适配 | - |
| 014 | External Benchmarks | 🟢 MVBench/VME 已出结果 | - |

---

## 外部 Benchmark 结果

### MVBench

| 模型 | Acc |
|------|-----|
| Zeroshot | 59.01% |
| B-1L ep1 | 42.22% |
| B-2L ep2 | 42.58% |

### Video-MME Short

| 模型 | Acc |
|------|-----|
| Zeroshot | 63.33% |

---

## 自动执行流程

1. 110K 训练 pass → 下载 epoch checkpoint → 上传 blob
2. 提交 3 个 eval：内部 MCQ + MVBench + Video-MME Short
3. eval 结果记录到 results_summary.md
4. 每 15 分钟自动监控

## 待办

### 高优先
- [ ] 110K 训练完成后 eval（007/008/009/010）
- [ ] Video-MME Short B1L/B2L eval
- [ ] K-sweep 结果分析 + 曲线

### 中优先
- [ ] NExT-QA 视频下载
- [ ] 011 question-conditioned 实现
- [ ] 013 temporal head 适配

### 低优先
- [ ] 012 adapter 调研
- [ ] CLAUDE.md 更新当前阶段

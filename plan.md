# Video 项目总计划

> 最后更新：2026-05-19

## 当前阶段：Stage 1 段级压缩 + Ablation

### 基础设施

| 项目 | 状态 |
|------|------|
| amlt 环境 | ✅ |
| Qwen2.5-VL 模型 on blob | ✅ `/mnt/default/bottleneck/models/Qwen2.5-VL-7B-Instruct/` |
| Teacher cache 110K (v2) | 🟡 p2c/p3c 补完中 (quick-jackal) |
| Checkpoint A40→blob | ✅ 25 个 |
| Zero-shot baseline | ✅ 0-30: 80.65%, 0-60: 79.59% |
| MVBench on blob | ✅ 17GB |

---

## 实验执行顺序

### 阶段 A：cache 完成后立即执行（自动）

cache 完成标志：quick-jackal p2c + p3c 都 pass 且日志有"提取完成"。

完成后按顺序执行：

#### 1. 007 Inter-Segment Attention
```
smoketest → 跑 amlt_train.yaml --max_samples 32 --epochs 1
10K      → 跑 amlt_train.yaml --max_samples 10000 --epochs 3
110K     → 跑 amlt_train.yaml 全量
```

#### 2. 008 K-sweep (K=2/4/8/16/32)
```
smoketest → 每个 K 跑 32 样本 1 epoch
110K     → 跑 amlt_k_sweep.yaml 全量（5 个 job 并行）
```

#### 3. 009 Pooling Baseline
```
smoketest → 32 样本 1 epoch
110K     → 跑 amlt_train.yaml 全量
```

#### 4. 010 Gated Compression
```
smoketest → 32 样本 1 epoch
110K     → 跑 amlt_train.yaml 全量
```

### 阶段 B：阶段 A 结果出来后

#### 5. 008 结果分析
- 画 K vs Accuracy 曲线
- 确定后续默认 K 值

#### 6. 009/010 vs 006 对比
- pooling 下界 vs cross-attention vs gated
- 量化各架构贡献

#### 7. 014 External Benchmarks
- MVBench / NExT-QA / Video-MME Short
- Zero-shot + B-2L best 对比

### 阶段 C：架构探索

#### 8. 011 Question-Conditioned（待讨论后启动）
#### 9. 012 Adapter Compression（优先级最低）
#### 10. 013 Temporal Grounding（Stage 3）

---

## 自动执行规则

1. **每个实验按 smoketest → 10K → 110K 递进**
   - smoketest 失败 → 修 bug，不往下走
   - 10K 跑通 → 提交 110K（10K 不需要特别好的结果，只要 loss 在降）
   - 110K 完成 → full eval 0-30 + 0-60

2. **提交前检查清单**（CLAUDE.md 第 7 条）
   - amlt yaml `--dump` 验证通过
   - model_path 用 blob 路径
   - prefetch_workers=0
   - 确认 code_dir 内文件齐全（compressor.py, model.py 等）

3. **结果记录**
   - 每个实验完成后更新对应 README.md + plan.md
   - 更新 results_summary.md
   - git commit + push

4. **失败处理**
   - SIGKILL → 检查 OOM 原因，降资源重试
   - import error → 检查依赖，修代码重提
   - preempt → 带 --resume 重提（如果支持）
   - 不自作主张 cancel 正常运行的 job

---

## Blob 路径汇总

```
/mnt/default/bottleneck/
├── data/parsed/                     # MCQ + temporal jsonl
├── data/videos/                     # 0-30s + 30-60s 视频
├── models/Qwen2.5-VL-7B-Instruct/  # frozen LLM
├── teacher_cache_110k_v2/           # teacher shard cache
├── checkpoints/006_compressor/      # A40 checkpoint
├── benchmarks/mvbench/              # MVBench 数据
└── benchmarks/nextqa/               # (待下载)
```

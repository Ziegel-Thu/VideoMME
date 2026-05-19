# 014 External Benchmarks 计划

## 目标
在 MVBench / NExT-QA / Video-MME Short 上评测 compressor 泛化能力。

## 步骤

### Phase 1: 数据准备
1. [ ] 下载 MVBench（HuggingFace，视频+标注）
2. [ ] 下载 NExT-QA（Google Drive 视频 + GitHub 标注）
3. [ ] 下载 Video-MME（HuggingFace，过滤 ≤60s）
4. [ ] 上传视频到 blob `bottleneck/benchmarks/{mvbench,nextqa,videomme_short}/`
5. [ ] 解析标注格式，统一为 jsonl

### Phase 2: 评测脚本
6. [ ] 写 eval_mvbench.py（适配 3-4 选 + 20 task 分类输出）
7. [ ] 写 eval_nextqa.py（适配 5 选 + C/T/D 分类输出）
8. [ ] 写 eval_videomme_short.py（适配 4 选 + 6 domain 分类输出）
9. [ ] 支持 --checkpoint 参数：有则用 compressor eval，无则 zero-shot

### Phase 3: 跑评测
10. [ ] Zero-shot baseline 三个 benchmark
11. [ ] 006 B-2L best 三个 benchmark
12. [ ] 汇总对比表

## 依赖
- 006 B-2L checkpoint（已在 blob）
- Qwen2.5-VL 模型（已在 blob）

## 优先级
MVBench > Video-MME Short > NExT-QA

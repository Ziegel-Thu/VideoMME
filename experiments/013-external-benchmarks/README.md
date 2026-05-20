# 014-external-benchmarks: 外部 Benchmark 评测

## 概述

用外部公开 benchmark 验证 compressor 的泛化能力，不只在自己的 MCQ 数据上评。

## Benchmark 列表

### 1. MVBench
- **来源**: OpenGVLab, CVPR 2024 Highlight
- **条数**: 4,000（20 task × 200）
- **视频时长**: ~16s 平均，全在 0-60s 内
- **选项**: 3-4 选
- **下载**: `huggingface.co/datasets/OpenGVLab/MVBench`
- **认可度**: 极高，token 压缩论文标配（VideoChat2, FastV, VideoLLaMA 等）

### 2. NExT-QA
- **来源**: CVPR 2021
- **条数**: 8,564（test），47K（全量）
- **视频时长**: ~39-44s 平均，大部分在 0-60s
- **选项**: 5 选
- **问题类型**: Causal (C), Temporal (T), Descriptive (D)
- **下载**: Google Drive（视频）+ GitHub `doc-doc/NExT-QA`（标注）
- **认可度**: 高，SeViLA, VideoChat2 等广泛使用

### 3. Video-MME Short（≤60s 过滤）
- **来源**: CVPR 2025，业界标准
- **条数**: Short split 900 条，过滤 ≤60s 后约 400-500 条（需确认）
- **视频时长**: Short split 11s-2min，avg 82.5s
- **选项**: 4 选（ABCD）
- **下载**: `huggingface.co/datasets/lmms-lab/Video-MME`
- **认可度**: 最高，GPT-4/Gemini 评测标准

---

## 评测方案

所有 benchmark 统一用 logit 方法（和我们内部 MCQ 口径一致）：
1. Zero-shot Qwen2.5-VL baseline
2. 006 B-2L best checkpoint（压缩后）
3. 对比压缩损失

---

## 文件结构

```
014-external-benchmarks/
├── README.md
├── plan.md
├── scripts/
│   ├── download_mvbench.py
│   ├── download_nextqa.py
│   ├── download_videomme.py
│   ├── eval_mvbench.py
│   ├── eval_nextqa.py
│   └── eval_videomme_short.py
└── data/                        # gitignore, 下载到 blob
```

---

## 实验记录

（待填）

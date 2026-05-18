# 005-teacher-cache: Teacher 特征预提取

## 概述

为 cross-attention compressor（006）提供训练数据。预提取 Qwen2.5-VL-7B-Instruct 的 dense vision embeddings 和 teacher Q-position hidden states，保存为 shard 格式，避免训练时重复过 vision encoder。

### 输入 → 输出

```
输入: 视频 + MCQ 问题 (visual_qa_v3_60s_train.jsonl, 110K 条)
      ↓ Qwen2.5-VL forward (frozen)
输出: teacher_shard_rank{R}_{N}.pt (每 shard 512 条)
```

### 每条样本包含

```python
{
    "segments": [tensor(V_i, D), ...],           # 每段 dense vision embeddings (bf16)
    "q_embeds": tensor(Q_len, D),                # question text embeddings (bf16)
    "teacher_q_hidden": [tensor(Q_len, D), ...], # teacher Q-position hidden per segment (bf16)
    "video_path": str,
    "question": str,
    "answer": str,
    "n_segments": int,
    "global_idx": int,                           # 原始 jsonl 行号，用于 resume
}
```

---

## Shard Cache 规范

- **只允许 shard 格式**：目录内必须是 `teacher_shard_*.pt`
- **训练/评测只允许从本地高速存储读取**
  - amlt 集群：`/mnt/default/bottleneck/teacher_cache_110k/`（blob mount）
  - jiagpu（A40 集群）：`/nvmessd/lifanhong/video/teacher_cache_110k_sharded_256/`（本地 SSD）
- **旧的按样本 `.pt` 小文件 cache 已废弃**

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--shard_size` | 512 | 每 shard 样本数 |
| `--fps` | 1.0 | 视频采样帧率 |
| `--frames_per_segment` | 2 | 每段帧数 |
| `--max_frames` | 30 | 单视频最大帧数 |
| `--prefetch_workers` | 0 | 后台视频解码线程数（amlt 集群建议 0，避免 OOM） |
| `--resume` | - | 扫已有 shard 跳过已完成样本 |

---

## 使用方式

### amlt 集群（A100, 推荐）

分 4 个 job 并行提取，每 job 4×A100：

```bash
amlt run amlt_extract_teacher.yaml -d "110K teacher cache extract"
```

详见 `amlt_extract_teacher.yaml`，4 个 job 按 `--start_idx / --end_idx` 分片。

**注意**：
- `--prefetch_workers 0`（amlt blob mount 下 prefetch>0 会因 page cache 积累触发 OOM）
- 模型从 HuggingFace 自动下载（`Qwen/Qwen2.5-VL-7B-Instruct`）
- 输出到 `/mnt/default/bottleneck/teacher_cache_110k/`

### jiagpu（A40 集群）

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  torchrun --nproc_per_node=8 extract_teacher.py \
  --data_path /nvmessd/lifanhong/video/parsed/visual_qa_v3_0_60_train.jsonl \
  --video_dirs /nvmessd/lifanhong/video/llava-video/0_30_s_academic_v0_1,/nvmessd/lifanhong/video/llava-video/30_60_s_academic_v0_1 \
  --output_dir /nvmessd/lifanhong/video/teacher_cache_110k_sharded_256 \
  --model_path /nvmessd/lifanhong/.cache/modelscope/Qwen/Qwen2___5-VL-7B-Instruct \
  --shard_size 512 --prefetch_workers 0 --resume
```

### 旧 cache 重打包

```bash
python pack_teacher_cache.py \
  --input_dir /path/to/old_single_file_cache \
  --output_dir /path/to/new_shard_cache \
  --shard_size 1000
```

---

## 文件结构

```
005-teacher-cache/
├── README.md                    # 本文件
├── extract_teacher.py           # 预提取主脚本（DDP）
├── pack_teacher_cache.py        # 旧 cache → shard 重打包
├── model.py                     # 模型加载 + vision embed 工具（从 004 引用）
├── amlt_extract_teacher.yaml    # amlt 集群 4-job 并行配置
├── test_extract_teacher.py      # 单元测试
└── test_pack_teacher_cache.py   # 单元测试
```

---

## 实验记录

### amlt 集群提取（A100, 2026-05-18）

| 项目 | 数值 |
|------|------|
| 实验名 | expert-arachnid |
| SKU | 80G4-A100-NvLink × 4 jobs |
| prefetch_workers | 0 |
| 数据 | visual_qa_v3_60s_train.jsonl (110K) |
| 输出 | `/mnt/default/bottleneck/teacher_cache_110k/` |
| 状态 | 运行中 |

### A40 集群提取参照（jiagpu8, 2026-05-17）

| 项目 | 数值 |
|------|------|
| GPU | 8× A40 48GB |
| 最终可训练样本数 | 102,952 |
| shard 数 | 203 |
| cache 大小 | 3.9TB |
| 错误 | 176（视频 decode 错误，已跳过） |

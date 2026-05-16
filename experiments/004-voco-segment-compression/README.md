# 004-voco-segment-compression: VoCo 风格分段压缩

## 概述

完全照搬 VoCo-LLaMA 的设计：
- 视频按 2-4s 切分成段
- 每段独立 forward：`[vision_t, voco_t×K_seg]`，attention mask 让 vision 后的 text 不能跨过 voco 看 vision
- 拼接所有段的 voco activations + question + answer
- **标准 SFT NLL loss**（Cross Entropy），不用 KL 蒸馏

## 设计

### Token 顺序（VoCo 风格 - 方案 B）

```
[vision_seg1] [voco_1×K] [vision_seg2] [voco_2×K] ... [vision_segN] [voco_N×K] [Q] [A]
                                                                                ↑
                                                                       这里 Q/A 之前所有 vision
                                                                       通过 voco 屏障被压缩
```

**Q 在所有 voco 之后** → latent (voco) 是 question-agnostic，可以缓存复用。

### Attention Mask

对每段 voco 起到"屏障"作用：
- voco 之前的 vision tokens 看不到 voco 之后的内容
- voco 之后的 token（其他段 vision、Q、A）不能直接看 voco 之前的 vision，只能通过 voco 看

### Training Loss

标准 NLL on answer tokens：
```python
loss = CrossEntropy(logits[answer_positions], answer_token_ids)
```

不需要 teacher-student KL。

### 关键参数

- 段长度: 2-4 秒
- 每段 voco 数: K_seg = 4-8
- 视频帧率: 1fps
- 30s 视频 → 8-15 段 → ~32-120 voco tokens

## 与 003 (bottleneck SFT) 的关系

003 是单段 + question 在 latent 之前的版本。004 是分段 + question 在 latent 之后的 VoCo 风格。

| 项目 | 003 | 004 |
|------|-----|-----|
| 分段 | ❌ 整体一段 | ✅ 2-4s 一段 |
| Q 位置 | 在 latent 之前 (conditioned) | 在 latent 之后 (agnostic) |
| Loss | NLL on answer | NLL on answer |
| 适合长视频 | ❌（序列 OOM） | ✅（每段独立 forward） |

## 实现要点

1. **Attention mask 实现**: 参考 VoCo-LLaMA `make_voco_mask_llava`
2. **分段 forward + KV cache 拼接**: 每段独立处理后 concatenate
3. **Q 位置**: 把 Q 拼到所有 voco caches 之后
4. **A 位置**: 标准最后位置
5. **Temporal Head (Stage 3)**: question-aware，从 voco hidden states + Q 解码时间段

## Shard Cache 规范（当前默认流程）

- **teacher cache 只允许 shard 格式**：目录内必须是 `teacher_shard_*.pt`
- **训练/评测只允许从本地 SSD 读取**：路径必须在 `/nvmessd/...`
- **禁止直接从 NFS 读取 teacher cache**：`train_compressor.py` 会显式拒绝 `/beegfs_hdd/...`
- **旧的按样本 `.pt` 小文件 cache 已废弃**：不要再用 `glob + open` 方式读上万个小文件

### 生成 shard cache

直接提取为 shard：

```bash
python extract_teacher.py \
  --output_dir /nvmessd/lifanhong/video/teacher_cache_10k_sharded_v2 \
  --shard_size 1000
```

把已有 SSD 小文件 cache 打包成 shard：

```bash
python pack_teacher_cache.py \
  --input_dir /nvmessd/lifanhong/video/teacher_cache_10k \
  --output_dir /nvmessd/lifanhong/video/teacher_cache_10k_sharded_v2 \
  --shard_size 1000
```

### shard 训练命令

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_compressor.py \
  --cache_dir /nvmessd/lifanhong/video/teacher_cache_10k_sharded_v2 \
  --output_dir /nvmessd/lifanhong/video/outputs_compressor_1L \
  --model_path /nvmessd/lifanhong/.cache/modelscope/Qwen/Qwen2___5-VL-7B-Instruct \
  --n_layers 1 \
  --K_seg 8 \
  --loss_type B
```

## 0-60 数据准备状态（jiagpu8 SSD）

`visual_qa_v3_0_60_{train,val,test}.jsonl` 里的样本数是 **MCQ 题目数**，不是视频数。当前 train split：

| split | MCQ | 去重视频 | 组成 |
|------|-----:|---------:|------|
| train | 110,026 | 6,426 | 30_60 academic: 71,181；0_30 academic: 38,845 |
| val | 5,102 | 379 | 30_60 academic: 4,174；0_30 academic: 928 |
| test | 10,103 | 747 | 30_60 academic: 8,268；0_30 academic: 1,835 |

### 视频目录

必须都在本地 SSD，不能从 NFS 直接读视频或 teacher cache：

```bash
/nvmessd/lifanhong/video/llava-video/0_30_s_academic_v0_1
/nvmessd/lifanhong/video/llava-video/30_60_s_academic_v0_1
```

当前已确认：

- `30_60_s_academic_v0_1`: mirror 上有 10 个 tar.gz，已下载到 `/nvmessd/lifanhong/video/_downloads/30_60_s_academic_v0_1`，正在解压到正式视频目录
- `0_30_s_academic_v0_1`: 本地当前只有部分子集（约 1,756 个视频），mirror 上完整 academic 子集是 8 个 tar.gz，仍需补齐

### mirror 下载来源

Hugging Face 官方域名在当前环境不可达；使用 `hf-mirror.com`：

```bash
curl -L "https://hf-mirror.com/api/datasets/lmms-lab/LLaVA-Video-178K/tree/main/30_60_s_academic_v0_1" |
  jq -r '.[] | select(.path | test("videos_[0-9]+\\.tar\\.gz$")) |
  "https://hf-mirror.com/datasets/lmms-lab/LLaVA-Video-178K/resolve/main/\(.path)"'
```

### 覆盖率检查

```bash
python3 - <<'PY'
import json
from pathlib import Path

base = Path("/nvmessd/lifanhong/video")
video_dirs = [
    base / "llava-video/0_30_s_academic_v0_1",
    base / "llava-video/30_60_s_academic_v0_1",
]
video_index = {
    p.name
    for d in video_dirs if d.is_dir()
    for p in d.rglob("*")
    if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v"}
}
for split in ["train", "val", "test"]:
    path = base / f"parsed/visual_qa_v3_0_60_{split}.jsonl"
    vids = set()
    with path.open() as f:
        for line in f:
            item = json.loads(line)
            vids.add(Path(item["video_path"]).name)
    resolved = vids & video_index
    print(split, "unique_videos", len(vids), "resolved", len(resolved), "missing", len(vids - resolved))
PY
```

110K teacher shard cache 只有在上述视频覆盖率接近完整后再提取；推荐 `--shard_size 256`，当前 DDP 写法会产生约 **432** 个 shard 文件。

## 文件结构

```
004-voco-segment-compression/
├── README.md
├── model.py            # 模型加载 + voco token 注入（支持 use_lora=True/False）
├── data.py             # Dataset + 分段 collate
├── train.py            # 单 forward 版（mask 隔离段间 vision）
├── train_concat.py     # 拼接版（KV cache concat，省计算量）+ LoRA
├── train_voco_only.py  # VoCo-only 训练（冻结 LLM，无 LoRA，只训 voco_embeds）
├── eval_mcq.py         # MCQ 评测（logit / generation / NLL 三种方法）
├── eval_voco.py        # VoCo checkpoint 评测
└── amlt.yaml
```

## 实验记录

### Pilot 结果

| 实验 | 版本 | 数据 | 结果 |
|------|------|------|------|
| 本地 5 条 | 单forward | 5×1ep | ✅ loss 4.03 |
| 本地 10 条 | 拼接版 | 10×3ep | ✅ loss 3.83→1.11→0.42 |
| voco-single-200 (集群) | 单forward 4卡 | 200×3ep | ✅ val_loss 0.30→0.25→0.21 |
| voco-concat-v3 (集群) | 拼接版 4卡 | 200×3ep | ✅ pass (DDP修复后) train 1.52→0.57→0.28 |

### 10K 评测结果 (500 条 test set)

| Epoch | Test Acc | Test Loss |
|-------|----------|-----------|
| 1 | 70.40% | 0.3760 |
| 2 | 66.00% | 0.4432 (过拟合) |

### 110K 集群训练

| 实验 | 版本 | 状态 | 结果 |
|------|------|------|------|
| voco-110k-v4 | 单forward | ❌ SIGABRT (ep1 8%) | loss 卡 3.4375 不降，rank 1 被 kill |
| voco-concat-110k | 拼接版 | 🟢 running (ep1 31%) | 正常训练中，loss 在降 |

### 技术发现

1. **VoCo 原论文用 NLL loss**（不是 KL 蒸馏），是标准 SFT + VoCo mask
2. **单 forward + mask**: 稳定可靠，可用 gradient checkpointing
3. **拼接版**: 计算量小但不能用 gradient checkpointing (和 use_cache 冲突)
4. **KV cache 保留计算图**: 梯度可反传到 voco_embeds
5. **拼接版 DDP**: 需手动 all_reduce，避免 UnboundLocalError + NCCL 同步问题
6. **NCCL timeout**: 混合长度数据需设 timeout=2h（init_process_group）
7. **VoCo 压缩比 54:1** vs 003 BN 13.5:1，信息损失大
8. **teacher cache 必须 shard 化**：上万个小 `.pt` 文件会带来严重 metadata 压力，尤其不能放大到 NFS
9. **Compressor 必须只用压缩 token 做评测**：不能再把 dense vision 混回 answer-time context

## 已完成

- [x] VoCo 风格 attention mask（向量化实现）
- [x] 单 forward 版 + DDP
- [x] 拼接版 + detach vision 省显存
- [x] 拼接版 DDP crash 修复（UnboundLocalError + NCCL 同步）
- [x] Pilot 验证 loss 收敛
- [x] 10K 训练 + 评测
- [x] 110K 拼接版集群训练中
- [x] model.py 支持 use_lora=False（冻结 LLM）
- [x] VoCo-only 训练脚本 (train_voco_only.py)
- [x] MCQ 评测脚本 (eval_mcq.py) — logit / generation / NLL 三种方法

## 待办

- [ ] 110K 评测（等拼接版完成）
- [ ] 单 forward 110K 崩溃排查
- [ ] 评测 vs 003 (单段 bottleneck SFT)
- [ ] Stage 3: question-aware temporal head
- [ ] VoCo-only ablation 完整训练 + 评测
- [ ] VoCo-only vs VoCo+LoRA 对比

## Cross-Attention Compressor（有效版本）

静态 `voco_embeds` baseline 已废弃；当前有效路线是 **Cross-Attention compressor**：

- `compressor.py`: learnable queries + cross-attention
- `train_compressor.py`: 支持 `loss_type=B|D|BD`
- `eval_compressor.py`: 评测时只使用 compressed tokens，不回注 dense vision

### Compressor 10K / 200 条 test 结果

以下结果均在**相同条件**下评测：`--video_dirs=0_30_s_academic_v0_1`，`--max_samples 200`，`visual_qa_v3_0_60_test.jsonl`。

| 配置 | epoch1 | epoch2 | epoch3 | epoch4 | epoch5 |
|------|--------|--------|--------|--------|--------|
| B-1L | 66.5% | 71.0% | **71.0%** | 67.5% | 68.5% |
| B-2L | - | - | 69.0% | - | - |
| D-1L | - | - | 63.0% | 65.0% | 62.5% |
| BD-1L | - | - | 69.5% | - | - |

结论：
- **B-1L 在 epoch3 达到峰值 71.0%，epoch4/5 略降到 67.5-68.5%**，说明 10K 数据已饱和，继续训有轻微过拟合
- **D-1L 基本持平**（63.0% → 65.0% → 62.5%），D loss 效果始终不如 B
- **最佳配置仍是 B-1L epoch3**

---

## A40 集群实验（jiagpu8, 8×A40 48GB）

### 环境

- **机器**: jiagpu8
- **GPU**: 8× NVIDIA A40 48GB
- **存储**: NFS (代码) + SSD `/nvmessd/` (数据/视频)
- **conda**: `video` (python 3.11, torch 2.6.0+cu124, transformers 4.57.6, peft 0.19.1)
- **数据**: SSD 上 visual_qa_v3_0_60 (110K train / 5K val / 10K test), 1755 个视频（tar part 2 解压）

### VoCo-only Ablation

**目的**: 验证纯 VoCo 压缩能力。冻结 LLM（无 LoRA），只训练 voco_embeds (K_seg=8, 28K 参数)。
如果无 LoRA 也能收敛，说明 VoCo 压缩本身 work，LLM 不需要额外适配。

**与 train_concat.py 的区别**:

| 项目 | train_concat (VoCo+LoRA) | train_voco_only (VoCo-only) |
|------|--------------------------|----------------------------|
| 可训练参数 | voco_embeds + LoRA (~20M) | voco_embeds only (28K) |
| K_seg 默认 | 4 | 8 |
| LR 默认 | 2e-5 | 1e-3 |
| 预提特征 | 支持 | 暂不支持 |
| 显存占用 | 较高（LoRA 梯度+优化器） | 较低（只有 voco_embeds 梯度） |

### Zero-shot Baseline 重测

**目的**: 用更合理的评测方法（per-option logit 比较 + 自由生成）重测 Qwen2.5-VL zero-shot 准确率。

**旧评测 (eval_baseline.py)**: next-token argmax 全词表 → 36.6%（偏低，受格式影响）
**新评测 (eval_mcq.py)**: 三种方法对比
- logit: ABCD 四个 token 的 logit 比较（一次 forward）
- gen: 自由生成 + 正则提取答案
- nll: per-option 完整 NLL（最严谨，慢 4 倍）

### 实验进度

| 实验 | tmux | GPU | 状态 | 结果 |
|------|------|-----|------|------|
| zero-shot 评测 (200条) | eval-baseline | 卡 0 | ✅ 完成 | logit=82%, gen=72.5%, nll=70% |
| VoCo-only overfit (10条) | voco-overfit | 卡 1 | ✅ 完成 | loss 8.65→0.01, 33GB显存 |
| VoCo-only 10K (K_seg=8) | voco-10k | 卡 1 | 🟢 训练中 | ~3.5s/step, ~9h/epoch |

### 关键发现

1. **旧评测低估 baseline**：旧方法 36.6% → 新 logit 方法 82%（差距来自全词表 argmax vs ABCD token 比较）
2. **VoCo-only 能 overfit**：28K 参数（冻结 LLM）loss 8.65→0.01，说明纯 voco 压缩可行
3. **A40 显存足够**：overfit 用 33GB / 48GB，有余量
4. **模型需从 modelscope 下载**：jiagpu8 无法访问 huggingface.co，用 modelscope 镜像
5. **模型路径**: `/nvmessd/lifanhong/.cache/modelscope/Qwen/Qwen2___5-VL-7B-Instruct`

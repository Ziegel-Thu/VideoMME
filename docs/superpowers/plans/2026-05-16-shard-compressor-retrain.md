# Shard Cache Compressor Retrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 teacher cache 改成 shard 格式并基于本地 SSD shard 继续训练 B-1L compressor。

**Architecture:** 保留现有 teacher 内容，但不再使用按样本 `.pt` 小文件。`extract_teacher.py` 直接输出 shard，`train_compressor.py` 只接受本地 SSD shard cache 并拒绝 NFS 路径，`pack_teacher_cache.py` 负责把旧 SSD 小文件一次性重打包为 shard。训练从 `compressor_epoch3.pt` 续到更高 epoch。

**Tech Stack:** Python, PyTorch, unittest, tmux, Qwen2.5-VL, 本地 SSD `/nvmessd`

---

### Task 1: 固化 shard-only 缓存接口

**Files:**
- Modify: `experiments/004-voco-segment-compression/train_compressor.py`
- Modify: `experiments/004-voco-segment-compression/extract_teacher.py`
- Create: `experiments/004-voco-segment-compression/pack_teacher_cache.py`
- Test: `experiments/004-voco-segment-compression/test_train_compressor.py`
- Test: `experiments/004-voco-segment-compression/test_extract_teacher.py`
- Test: `experiments/004-voco-segment-compression/test_pack_teacher_cache.py`

- [ ] **Step 1: 写失败测试，要求训练拒绝 NFS 和非 shard cache**

```python
with self.assertRaises(ValueError):
    validate_cache_dir("/beegfs_hdd/.../teacher_cache_10k")

with self.assertRaises(ValueError):
    TeacherCacheDataset(tmpdir_with_single_pt_files)
```

- [ ] **Step 2: 运行测试确认失败**

Run:
```bash
cd experiments/004-voco-segment-compression
/beegfs_hdd/data/nfs_share/users/lifanhong/nishome/miniconda3/envs/video/bin/python -m unittest test_train_compressor.py
```

Expected: `ImportError` / `ValueError not raised`

- [ ] **Step 3: 写最小实现**

```python
def validate_cache_dir(cache_dir):
    real_path = os.path.realpath(cache_dir)
    if real_path.startswith("/beegfs_hdd/"):
        raise ValueError("禁止直接从 NFS 读取 teacher cache")
    return real_path

shard_files = sorted(glob.glob(os.path.join(self.cache_dir, "teacher_shard_*.pt")))
if not shard_files:
    raise ValueError("请先生成 shard cache")
```

- [ ] **Step 4: 为提取端增加 ShardWriter**

```python
class ShardWriter:
    def add(self, sample):
        self.buffer.append(sample)
        if len(self.buffer) >= self.shard_size:
            self.flush()
```

- [ ] **Step 5: 运行测试确认通过**

Run:
```bash
cd experiments/004-voco-segment-compression
/beegfs_hdd/data/nfs_share/users/lifanhong/nishome/miniconda3/envs/video/bin/python -m unittest test_train_compressor.py test_extract_teacher.py test_pack_teacher_cache.py
```

Expected: `Ran 5 tests ... OK`

- [ ] **Step 6: Commit**

```bash
git add experiments/004-voco-segment-compression/train_compressor.py \
        experiments/004-voco-segment-compression/extract_teacher.py \
        experiments/004-voco-segment-compression/pack_teacher_cache.py \
        experiments/004-voco-segment-compression/test_train_compressor.py \
        experiments/004-voco-segment-compression/test_extract_teacher.py \
        experiments/004-voco-segment-compression/test_pack_teacher_cache.py
git commit -m "改为 shard-only teacher cache 流程"
```

### Task 2: 用 SSD 上旧 cache 重打 shard

**Files:**
- Use: `experiments/004-voco-segment-compression/pack_teacher_cache.py`

- [ ] **Step 1: 在目标机器本地 SSD 上创建新 shard 目录**

```bash
mkdir -p /nvmessd/lifanhong/video/teacher_cache_10k_sharded_v2
```

- [ ] **Step 2: 运行打包脚本**

```bash
cd experiments/004-voco-segment-compression
/beegfs_hdd/data/nfs_share/users/lifanhong/nishome/miniconda3/envs/video/bin/python pack_teacher_cache.py \
  --input_dir /nvmessd/lifanhong/video/teacher_cache_10k \
  --output_dir /nvmessd/lifanhong/video/teacher_cache_10k_sharded_v2 \
  --shard_size 1000
```

Expected: 输出约 `10 个 shard`

- [ ] **Step 3: 核对 shard 数量**

Run:
```bash
ls /nvmessd/lifanhong/video/teacher_cache_10k_sharded_v2/teacher_shard_*.pt | wc -l
```

Expected: `10`

- [ ] **Step 4: Commit（如脚本有改动才需要）**

```bash
git status --short
```

Expected: 只有文档或代码改动，没有运行产物被纳入 git

### Task 3: 基于 shard 继续训练 B-1L

**Files:**
- Use: `experiments/004-voco-segment-compression/train_compressor.py`
- Output: `/nvmessd/lifanhong/video/outputs_compressor_1L_continue`

- [ ] **Step 1: 确认 checkpoint 与 shard 在本地 SSD**

```bash
ls /nvmessd/lifanhong/video/outputs_compressor_1L/compressor_epoch3.pt
ls /nvmessd/lifanhong/video/teacher_cache_10k_sharded_v2/teacher_shard_000.pt
```

- [ ] **Step 2: 用 tmux 启动续训**

```bash
tmux new-session -d -s comp-1L-cont \
  "cd /beegfs_hdd/data/nfs_share/users/lifanhong/nishome/video/experiments/004-voco-segment-compression && \
   CUDA_VISIBLE_DEVICES=2,3,4,5 \
   /beegfs_hdd/data/nfs_share/users/lifanhong/nishome/miniconda3/envs/video/bin/torchrun \
   --nproc_per_node=4 --master_port=29503 train_compressor.py \
   --cache_dir /nvmessd/lifanhong/video/teacher_cache_10k_sharded_v2 \
   --output_dir /nvmessd/lifanhong/video/outputs_compressor_1L_continue \
   --model_path /nvmessd/lifanhong/.cache/modelscope/Qwen/Qwen2___5-VL-7B-Instruct \
   --n_layers 1 --K_seg 8 --lr 1e-4 --epochs 5 --save_steps 500 \
   --resume_from /nvmessd/lifanhong/video/outputs_compressor_1L/compressor_epoch3.pt \
   --loss_type B 2>&1 | tee /nvmessd/lifanhong/video/log_compressor_1L_continue.txt"
```

- [ ] **Step 3: 检查启动日志**

Run:
```bash
tail -20 /nvmessd/lifanhong/video/log_compressor_1L_continue.txt
```

Expected: 出现 `start_epoch=3` 和 `Epoch 4/5`

### Task 4: 更新实验文档

**Files:**
- Modify: `experiments/004-voco-segment-compression/README.md`

- [ ] **Step 1: 写入 shard cache 规范**

```md
## Shard Cache 规范

- teacher cache 只允许 shard 格式：`teacher_shard_*.pt`
- 训练只允许从本地 SSD `/nvmessd/...` 读取
- 禁止直接从 NFS 训练
- 旧的按样本 `.pt` 小文件目录废弃，不再使用
```

- [ ] **Step 2: 写入当前最佳结果**

```md
### Compressor 结果（200 test）

| 配置 | Acc |
|------|-----|
| B-1L | 71.0% |
| B-2L | 69.0% |
| D-1L | 63.0% |
| BD-1L | 69.5% |
```

- [ ] **Step 3: Commit**

```bash
git add experiments/004-voco-segment-compression/README.md docs/superpowers/plans/2026-05-16-shard-compressor-retrain.md
git commit -m "记录 shard cache 流程与 compressor 结果"
```

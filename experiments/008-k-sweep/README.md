# 008-k-sweep: 压缩 Token 数量扫描

## 概述

固定最佳配置（B loss, 2L cross-attention），扫描每段压缩 token 数量 K_seg，找到压缩率和精度的最佳平衡点。

### 实验设计

| K_seg | 每段 compressed tokens | 压缩比（~400 dense → K） | 说明 |
|-------|----------------------|------------------------|------|
| 2 | 2 | 200:1 | 极端压缩 |
| 4 | 4 | 100:1 | |
| 8 | 8 | 50:1 | 当前默认值 |
| 16 | 16 | 25:1 | |
| 32 | 32 | 12.5:1 | 接近 003 BN 的压缩比 |

### 固定参数

- loss_type: B
- n_layers: 2
- inter_layers: 0
- lr: 1e-4
- epochs: 3
- 数据: 110K teacher cache

---

## 使用方式

```bash
# 例: K=4
torchrun --nproc_per_node=4 train_compressor.py \
  --cache_dir /mnt/default/bottleneck/teacher_cache_110k_v2 \
  --output_dir $AMLT_OUTPUT_DIR \
  --model_path /mnt/default/bottleneck/models/Qwen2.5-VL-7B-Instruct \
  --cache_shard_size 512 \
  --K_seg 4 --n_layers 2 --loss_type B \
  --lr 1e-4 --epochs 3 --save_steps 1000
```

---

## 文件结构

```
008-k-sweep/
├── README.md              # 本文件
├── compressor.py           # VoCoCompressor（同 006）
├── train_compressor.py     # 训练脚本（同 006）
├── eval_compressor.py      # 评测脚本（同 006）
├── model.py                # 模型工具（同 006）
└── amlt_k_sweep.yaml       # 5 个 K 值并行训练
```

---

## 实验记录

（待填）

---

## 计划

### 待做
- [ ] 等 teacher cache p2/p3 补完后提交 5 个 K 值训练（amlt yaml 已就绪：`amlt_k_sweep.yaml`）
- [ ] 每个 K 值跑 3 epoch，取 best epoch
- [ ] Full eval 0-30 + 0-60
- [ ] 画 K vs Accuracy 曲线，找最优压缩率
- [ ] 结合 K-sweep 结果决定后续实验的默认 K 值

### 预期
- K 增大 → 精度提升但压缩比降低
- K=32 应该接近 003 BN（压缩比 12.5:1 vs 13.5:1），验证 cross-attention 是否能匹配 bottleneck
- K=2 是极端压缩，看下界在哪

# 014-liteframe: LiteFrame Student Encoder

基于 [LiteFrame (arXiv:2605.17260)](https://arxiv.org/abs/2605.17260)，
为 Qwen2.5-VL-7B-Instruct 实现轻量级视频编码器蒸馏。

## 方法

- **Student**: ViT-Base (12L, 768D, 12H) ≈ 88M 参数
- **Teacher**: Qwen2.5-VL ViT (32L, 1280D, 16H) 冻结
- **压缩**: 4帧 448×448 → 4096 tokens → 256 tokens (16× 压缩)
  - 第4层后 stride-[2,2,2] DW Conv3D
  - 第8层后 stride-[2,1,1] DW Conv3D
- **Loss**: MSE(student, temporal_avg_pool(teacher)) + 3σ outlier clipping

## 文件

| 文件 | 说明 |
|------|------|
| `liteframe_encoder.py` | Student ViT-Base 架构 |
| `train_ctd.py` | CTD 蒸馏训练 (支持 DDP + 合成数据) |
| `test_sanity.py` | 形状/梯度/bf16 正确性验证 |

## 快速开始

```bash
# 合成数据测试
python test_sanity.py
python train_ctd.py --synthetic --steps 100

# 真实训练 (4×A100)
torchrun --nproc_per_node=4 train_ctd.py \
    --video_dir /path/to/videos \
    --steps 10000 --batch_size 2
```

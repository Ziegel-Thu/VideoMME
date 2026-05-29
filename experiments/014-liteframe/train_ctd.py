"""
LiteFrame Compressed Token Distillation (CTD) 训练

蒸馏 Qwen2.5-VL ViT → LiteFrame student encoder。
Teacher ViT 冻结，只训练 student。
Loss: MSE(student_output, temporal_pool(teacher_output))。

用法:
  # 合成数据测试
  python train_ctd.py --synthetic --steps 100

  # 单卡训练
  python train_ctd.py --video_dir /path/to/videos --steps 10000

  # 多卡 DDP
  torchrun --nproc_per_node=4 train_ctd.py \
      --video_dir /path/to/videos --steps 10000
"""

import os
import sys
import glob
import math
import json
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from liteframe_encoder import create_liteframe_base


# ============================================================
# 常量
# ============================================================

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
IMAGE_SIZE = 448
PATCH_SIZE = 14
TEMPORAL_PATCH_SIZE = 2  # Qwen2.5-VL teacher 的时间 patch 大小


# ============================================================
# 数据集
# ============================================================

class VideoClipDataset(Dataset):
    """从视频文件中采样 4 帧 clip 的数据集。"""

    def __init__(self, video_paths, num_frames=4, image_size=448):
        self.video_paths = video_paths
        self.num_frames = num_frames
        self.image_size = image_size
        self.mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
        self.std = torch.tensor(CLIP_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        import decord
        decord.bridge.set_bridge("torch")

        path = self.video_paths[idx]
        try:
            vr = decord.VideoReader(path, num_threads=1)
            total = len(vr)
            if total < self.num_frames:
                # 帧数不足，重复最后一帧
                indices = list(range(total)) + [total - 1] * (self.num_frames - total)
            else:
                # 随机起点，连续采样
                start = random.randint(0, total - self.num_frames)
                indices = list(range(start, start + self.num_frames))

            frames = vr.get_batch(indices)  # (T, H, W, C) uint8
            frames = frames.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)

            # Resize to image_size
            frames = F.interpolate(
                frames, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )

            # Normalize
            frames = (frames - self.mean) / self.std  # (T, C, H, W)
            return frames  # (T, C, H, W)

        except Exception as e:
            # 出错时返回随机噪声
            print(f"  [WARN] 加载视频失败 {path}: {e}", file=sys.stderr)
            return torch.randn(self.num_frames, 3, self.image_size, self.image_size)


class SyntheticDataset(Dataset):
    """合成随机数据，用于快速测试。"""

    def __init__(self, size=1000, num_frames=4, image_size=448):
        self.size = size
        self.num_frames = num_frames
        self.image_size = image_size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.randn(self.num_frames, 3, self.image_size, self.image_size)


# ============================================================
# Teacher 特征提取
# ============================================================

def frames_to_teacher_input(frames, device):
    """将归一化帧转换为 Qwen2.5-VL ViT 的输入格式。

    Args:
        frames: (B, T, C, H, W) 归一化视频帧
        device: 目标设备
    Returns:
        pixel_values: (total_patches, C, tps, ps, ps) — ViT patch 输入
        grid_thw: (B, 3) — 每个视频的 grid 尺寸
    """
    B, T, C, H, W = frames.shape
    ps, tps = PATCH_SIZE, TEMPORAL_PATCH_SIZE
    T_grid = T // tps
    H_grid = H // ps
    W_grid = W // ps

    # (B, T, C, H, W) → (B, C, T, H, W)
    x = frames.permute(0, 2, 1, 3, 4)
    # → (B, C, T_grid, tps, H_grid, ps, W_grid, ps)
    x = x.view(B, C, T_grid, tps, H_grid, ps, W_grid, ps)
    # → (B, T_grid, H_grid, W_grid, C, tps, ps, ps)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
    # → (total_patches, C, tps, ps, ps)
    x = x.reshape(-1, C, tps, ps, ps)

    grid_thw = torch.tensor(
        [[T_grid, H_grid, W_grid]] * B, dtype=torch.long,
    )
    return x.to(device), grid_thw.to(device)


def extract_teacher_features(teacher_vit, frames, device):
    """提取 teacher ViT post-merge 特征 + temporal pool。

    Args:
        teacher_vit: Qwen2.5-VL 的 vision transformer (冻结)
        frames: (B, T, C, H, W) 归一化视频帧
        device: 目标设备
    Returns:
        target: (B, H_merged * W_merged, D_out) — CTD 监督目标
    """
    B, T = frames.shape[:2]
    pixel_values, grid_thw = frames_to_teacher_input(frames, device)

    with torch.no_grad():
        # teacher_vit forward: (total_patches, C, tps, ps, ps) → (total_merged, D_out)
        teacher_out = teacher_vit(
            pixel_values.to(dtype=torch.bfloat16), grid_thw=grid_thw,
        )

    # teacher_out: (B * T_merged * H_merged * W_merged, D_out)
    T_merged = T // TEMPORAL_PATCH_SIZE
    spatial_merge = teacher_vit.spatial_merge_size
    H_merged = IMAGE_SIZE // PATCH_SIZE // spatial_merge
    W_merged = IMAGE_SIZE // PATCH_SIZE // spatial_merge
    tokens_per_video = T_merged * H_merged * W_merged
    D_out = teacher_out.shape[-1]

    teacher_out = teacher_out.view(B, T_merged, H_merged * W_merged, D_out)

    # Temporal average pooling → (B, H_merged * W_merged, D_out)
    target = teacher_out.mean(dim=1)
    return target


# ============================================================
# Loss
# ============================================================

def clipped_mse_loss(pred, target, clip_sigma=3.0):
    """MSE with outlier clipping (论文 Appendix A)。"""
    diff = pred - target
    if clip_sigma > 0:
        with torch.no_grad():
            std = diff.std()
            if std > 0:
                mask = (diff.abs() < clip_sigma * std).float()
            else:
                mask = torch.ones_like(diff)
        diff = diff * mask
    return (diff ** 2).mean()


# ============================================================
# 工具函数
# ============================================================

def get_video_paths(video_dir=None, video_list=None):
    """获取视频文件路径列表。"""
    paths = []
    if video_dir:
        # 支持逗号分隔的多目录
        for d in video_dir.split(","):
            d = d.strip()
            if not d:
                continue
            for ext in ["*.mp4", "*.avi", "*.mkv", "*.webm", "*.mov"]:
                paths.extend(glob.glob(os.path.join(d, "**", ext), recursive=True))
    if video_list:
        with open(video_list) as f:
            for line in f:
                line = line.strip()
                if line and os.path.exists(line):
                    paths.append(line)
    return sorted(set(paths))


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Cosine schedule with linear warmup。"""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def log(msg, is_main=True):
    if is_main:
        print(msg, flush=True)


# ============================================================
# 主训练函数
# ============================================================

def train(args):
    # DDP 初始化
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = rank == 0

    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    log("=" * 60, is_main)
    log("LiteFrame CTD 训练", is_main)
    log(f"  world_size: {world_size}, rank: {rank}", is_main)
    log(f"  steps: {args.steps}, batch_size: {args.batch_size}", is_main)
    log(f"  lr: {args.lr}, warmup_ratio: {args.warmup_ratio}", is_main)
    log(f"  num_frames: {args.num_frames}", is_main)
    log(f"  model_path: {args.model_path}", is_main)
    log("=" * 60, is_main)

    # ---- 加载 Teacher ViT ----
    log("加载 Teacher ViT...", is_main)
    if args.synthetic:
        teacher_vit = None
        log("  [合成模式] 跳过 teacher 加载", is_main)
    else:
        from transformers import Qwen2_5_VLForConditionalGeneration
        full_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, device_map="cpu",
        )
        teacher_vit = full_model.visual.to(device)
        teacher_vit.eval()
        for p in teacher_vit.parameters():
            p.requires_grad = False
        n_teacher = sum(p.numel() for p in teacher_vit.parameters())
        del full_model
        torch.cuda.empty_cache()
        log(f"  Teacher ViT: {n_teacher:,} 参数", is_main)

    # ---- 创建 Student ----
    log("创建 Student encoder...", is_main)
    student = create_liteframe_base(output_dim=args.output_dim).to(
        device, dtype=torch.bfloat16,
    )
    n_student = student.num_params()
    log(f"  Student: {n_student:,} 参数", is_main)

    if world_size > 1:
        student = DDP(student, device_ids=[local_rank])

    # ---- 数据集 ----
    if args.synthetic:
        dataset = SyntheticDataset(
            size=max(args.steps * args.batch_size, 10000),
            num_frames=args.num_frames,
        )
        log(f"  合成数据集: {len(dataset)} 样本", is_main)
    else:
        video_paths = get_video_paths(args.video_dir, args.video_list)
        if not video_paths:
            raise ValueError("没有找到视频文件，请指定 --video_dir 或 --video_list")
        dataset = VideoClipDataset(
            video_paths, num_frames=args.num_frames, image_size=IMAGE_SIZE,
        )
        log(f"  视频数据集: {len(dataset)} 个视频", is_main)

    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=sampler, shuffle=(sampler is None),
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True,
    )

    # ---- 优化器 + 调度器 ----
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=args.lr, weight_decay=0.01,
    )
    warmup_steps = int(args.steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, args.steps)

    # ---- 从 checkpoint 恢复 ----
    global_step = 0
    if args.resume_from and os.path.exists(args.resume_from):
        ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        student_state = ckpt["student"]
        if world_size > 1:
            student.module.load_state_dict(student_state)
        else:
            student.load_state_dict(student_state)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        global_step = ckpt.get("step", 0)
        log(f"  从 {args.resume_from} 恢复, step={global_step}", is_main)

    # ---- 训练循环 ----
    os.makedirs(args.output_dir, exist_ok=True)
    student.train()
    running_loss = 0.0
    epoch = 0

    log(f"\n开始训练 (target: {args.steps} steps)...", is_main)

    while global_step < args.steps:
        if sampler:
            sampler.set_epoch(epoch)

        for batch in loader:
            if global_step >= args.steps:
                break

            # batch: (B, T, C, H, W)
            frames = batch.to(device, dtype=torch.bfloat16)

            # Student forward: (B, T, C, H, W) → (B, C, T, H, W) → student
            student_input = frames.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W)
            student_out, grid = student(student_input)  # (B, N, output_dim)

            # Teacher target
            if teacher_vit is not None:
                target = extract_teacher_features(
                    teacher_vit, frames, device,
                )  # (B, 256, 3584)
            else:
                # 合成模式: 用随机 target
                target = torch.randn_like(student_out)

            # CTD Loss
            loss = clipped_mse_loss(student_out, target.detach(), clip_sigma=3.0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            global_step += 1
            running_loss += loss.item()

            # 日志
            if is_main and global_step % args.log_interval == 0:
                avg_loss = running_loss / args.log_interval
                lr = scheduler.get_last_lr()[0]
                log(f"  [step {global_step}/{args.steps}] "
                    f"loss={avg_loss:.6f} lr={lr:.2e}")
                running_loss = 0.0

            # 保存 checkpoint
            if is_main and global_step % args.save_steps == 0:
                save_path = os.path.join(
                    args.output_dir, f"checkpoint_{global_step}.pt",
                )
                student_model = student.module if world_size > 1 else student
                torch.save({
                    "student": student_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": global_step,
                    "args": vars(args),
                }, save_path)
                log(f"  ✓ 保存 checkpoint: {save_path}", is_main)

        epoch += 1

    # 最终保存
    if is_main:
        save_path = os.path.join(args.output_dir, "checkpoint_final.pt")
        student_model = student.module if world_size > 1 else student
        torch.save({
            "student": student_model.state_dict(),
            "step": global_step,
            "args": vars(args),
        }, save_path)
        log(f"\n✓ 训练完成! 最终 checkpoint: {save_path}", is_main)

    if world_size > 1:
        dist.destroy_process_group()


# ============================================================
# 入口
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="LiteFrame CTD 训练")
    # 数据
    p.add_argument("--video_dir", type=str, default=None)
    p.add_argument("--video_list", type=str, default=None)
    p.add_argument("--synthetic", action="store_true", help="使用合成数据测试")
    p.add_argument("--num_frames", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    # 模型
    p.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--output_dim", type=int, default=3584)
    # 训练
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=4e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=1000)
    # 输出
    p.add_argument("--output_dir", type=str, default="outputs_liteframe_ctd")
    p.add_argument("--resume_from", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())

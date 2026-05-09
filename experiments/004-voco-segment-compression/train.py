"""
004 训练: VoCo-style 分段压缩 SFT (单 forward 版本)

简化版: 不做 KV cache 拼接，一次 forward 跑整个序列。
mask 实现"段间隔离 + Q/A bottleneck"。

用法:
  # 单卡
  python train.py --data_path ... --video_dirs ... --output_dir ...

  # 多卡 DDP
  torchrun --nproc_per_node=4 train.py --data_path ... --video_dirs ... --output_dir ...
"""

import os
import sys
import json
import argparse

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from model import setup_voco_model
from data import VoCoVideoDataset, voco_collate


def setup_distributed():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return local_rank, world_size
    return 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(msg):
    if is_main():
        print(msg)


def reduce_mean(value, world_size):
    if world_size <= 1:
        return value
    t = torch.tensor(value, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item() / world_size


def train(args):
    local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    log("=" * 60)
    log("004 VoCo 分段压缩训练")
    log(f"  GPU 数量: {world_size}")
    log(f"  K_seg: {args.K_seg}, frames/seg: {args.frames_per_segment}")
    log(f"  fps: {args.fps}, max_frames: {args.max_frames}")
    log("=" * 60)

    # 模型
    model, processor, tokenizer = setup_voco_model(
        K_seg=args.K_seg,
        lora_r=args.lora_r,
        device=device,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    if world_size > 1:
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=False)

    # 数据
    video_dirs = args.video_dirs.split(",")
    dataset = VoCoVideoDataset(
        args.data_path, video_dirs, max_samples=args.max_samples,
    )

    n_val = min(100, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_sampler = DistributedSampler(train_set, shuffle=True) if world_size > 1 else None

    raw_model = model.module if hasattr(model, "module") else model

    def collate(batch):
        return voco_collate(
            batch, raw_model, processor, tokenizer,
            fps=args.fps,
            frames_per_segment=args.frames_per_segment,
            max_frames=args.max_frames,
        )

    train_loader = DataLoader(
        train_set, batch_size=1, sampler=train_sampler,
        shuffle=(train_sampler is None), collate_fn=collate,
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, collate_fn=collate,
    )

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    if is_main():
        os.makedirs(args.output_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)

        total_loss = 0
        n = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}",
                    disable=not is_main())
        for batch in pbar:
            if batch is None:
                continue

            try:
                _, loss, _ = model(
                    inputs_embeds=batch["inputs_embeds"],
                    attention_mask=batch["attention_mask"],
                    voco_4d_mask=batch["voco_4d_mask"],
                    labels=batch["labels"],
                )

                if loss is None or torch.isnan(loss):
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

                total_loss += loss.item()
                n += 1
                if is_main():
                    pbar.set_postfix(loss=f"{loss.item():.4f}",
                                     segs=batch["n_segments"])

            except Exception as e:
                if is_main() and n < 3:
                    print(f"  [错误] {e}")
                continue

        avg_loss = reduce_mean(total_loss / max(n, 1), world_size)
        log(f"  Epoch {epoch + 1}: train_loss={avg_loss:.4f}")

        # Val
        model.eval()
        val_loss = 0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                try:
                    _, loss, _ = model(
                        inputs_embeds=batch["inputs_embeds"],
                        attention_mask=batch["attention_mask"],
                        voco_4d_mask=batch["voco_4d_mask"],
                        labels=batch["labels"],
                    )
                    if loss is not None and not torch.isnan(loss):
                        val_loss += loss.item()
                        vn += 1
                except Exception:
                    continue

        avg_val = reduce_mean(val_loss / max(vn, 1), world_size)
        log(f"  Epoch {epoch + 1}: val_loss={avg_val:.4f}")

        # Save (rank 0 only)
        if is_main():
            ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch + 1}.pt")
            rm = model.module if hasattr(model, "module") else model
            torch.save({
                "voco_embeds": rm.voco_embeds.detach().cpu(),
                "lora_state_dict": {
                    k: v.cpu() for k, v in rm.base.state_dict().items()
                    if "lora_" in k
                },
                "epoch": epoch + 1,
                "val_loss": avg_val,
                "K_seg": args.K_seg,
            }, ckpt_path)
            log(f"  saved → {ckpt_path}")

            if avg_val < best_val:
                best_val = avg_val
                best_path = os.path.join(args.output_dir, "best_model.pt")
                torch.save({
                    "voco_embeds": rm.voco_embeds.detach().cpu(),
                    "lora_state_dict": {
                        k: v.cpu() for k, v in rm.base.state_dict().items()
                        if "lora_" in k
                    },
                    "epoch": epoch + 1,
                    "val_loss": avg_val,
                    "K_seg": args.K_seg,
                }, best_path)
                log(f"  ★ New best (val_loss={avg_val:.4f})")

    log(f"\nDone. best_val_loss={best_val:.4f}")
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--video_dirs", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--K_seg", type=int, default=4)
    parser.add_argument("--frames_per_segment", type=int, default=2)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max_frames", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    args = parser.parse_args()
    train(args)


"""
004 VoCo-only 训练：冻结 LLM（无 LoRA），只训练 voco_embeds

验证纯 VoCo 压缩能力：K_seg 个可学习 token 能否在冻结 LLM 下
通过 KV cache 传递足够的视觉信息来回答问题。

基于 train_concat.py 精简，去掉 LoRA 相关逻辑。

用法:
  # 单卡 overfit 测试
  python train_voco_only.py \
      --data_path /path/to/data.jsonl \
      --video_dirs /path/to/videos \
      --output_dir outputs_voco_only \
      --max_samples 10 --K_seg 8

  # 多卡
  torchrun --nproc_per_node=8 train_voco_only.py \
      --data_path /path/to/data.jsonl \
      --video_dirs /path/to/videos \
      --output_dir outputs_voco_only \
      --K_seg 8 --epochs 3
"""

import os
import sys
import json
import argparse
import traceback

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers.cache_utils import DynamicCache

from model import setup_voco_model, get_video_embeds, split_into_segments
from data import VoCoVideoDataset, encode_text


# ============================================================
# 分段 forward + KV cache 拼接（复用 train_concat 逻辑）
# ============================================================

def get_inner_model(base):
    """获取最内层 Qwen2_5_VLForConditionalGeneration（解 PEFT/DDP）。"""
    m = base.module if hasattr(base, "module") else base
    try:
        from peft import PeftModel
        if isinstance(m, PeftModel):
            return m.base_model.model
    except ImportError:
        pass
    return m


def forward_segment(inner, vision_embeds, voco_embeds, position_offset=0):
    """跑一段的 forward: [vision_t, voco_t]，返回 voco 位置的 KV cache。

    vision 部分 detach（不需要梯度），只保留 voco 的梯度。
    """
    K = voco_embeds.shape[0]
    vision_detached = vision_embeds.detach()

    inputs_embeds = torch.cat([vision_detached, voco_embeds], dim=0).unsqueeze(0)
    seg_len = inputs_embeds.shape[1]
    attention_mask = torch.ones(1, seg_len, dtype=torch.long,
                                device=vision_embeds.device)
    pos_1d = torch.arange(
        position_offset, position_offset + seg_len,
        device=vision_embeds.device,
    )
    position_ids = pos_1d.view(1, 1, -1).expand(3, 1, -1)

    cache = DynamicCache()
    out = inner.model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
    )

    # 提取 voco 位置（最后 K 个）的 KV cache
    voco_cache = []
    pkv = out.past_key_values
    for layer_idx in range(len(pkv)):
        k, v = pkv[layer_idx]
        voco_cache.append((k[:, :, -K:, :], v[:, :, -K:, :]))

    return voco_cache


def concat_voco_caches(seg_caches):
    """把多段的 voco cache 拼接成一个完整 cache。"""
    cache = DynamicCache()
    n_layers = len(seg_caches[0])

    for layer_idx in range(n_layers):
        k_list = [seg_caches[s][layer_idx][0] for s in range(len(seg_caches))]
        v_list = [seg_caches[s][layer_idx][1] for s in range(len(seg_caches))]
        cache.update(
            torch.cat(k_list, dim=2),
            torch.cat(v_list, dim=2),
            layer_idx,
        )

    total_K = seg_caches[0][0][0].shape[2] * len(seg_caches)
    return cache, total_K


def voco_concat_forward(model, vision_embeds, tokens_per_frame, frames_per_segment,
                        q_embeds, a_embeds=None):
    """完整的拼接 forward：分段 → 提取 voco cache → 拼接 → 最终 forward。"""
    raw = model.module if hasattr(model, "module") else model
    inner = get_inner_model(raw.base)
    K = raw.K_seg
    voco_per_seg = raw.voco_embeds

    segments = split_into_segments(vision_embeds, tokens_per_frame, frames_per_segment)

    seg_caches = []
    position_offset = 0
    for seg in segments:
        seg_cache = forward_segment(inner, seg, voco_per_seg, position_offset=position_offset)
        seg_caches.append(seg_cache)
        position_offset += seg.shape[0] + K

    full_cache, total_voco = concat_voco_caches(seg_caches)

    if a_embeds is not None:
        text_embeds = torch.cat([q_embeds, a_embeds], dim=0).unsqueeze(0)
    else:
        text_embeds = q_embeds.unsqueeze(0)

    text_len = text_embeds.shape[1]
    full_attn_mask = torch.ones(1, total_voco + text_len,
                                dtype=torch.long, device=text_embeds.device)

    text_pos = torch.arange(
        position_offset, position_offset + text_len, device=text_embeds.device,
    )
    text_position_ids = text_pos.view(1, 1, -1).expand(3, 1, -1)

    out = inner.model(
        inputs_embeds=text_embeds,
        attention_mask=full_attn_mask,
        position_ids=text_position_ids,
        past_key_values=full_cache,
        use_cache=True,
    )
    hidden = out[0]
    logits = inner.lm_head(hidden)

    return logits, hidden, total_voco


# ============================================================
# Collate（简化版，无 LoRA/预提特征）
# ============================================================

def voco_only_collate(batch, voco_model, processor, tokenizer,
                      fps=1.0, frames_per_segment=2, max_frames=30):
    """准备 vision_embeds + Q/A embeds，batch_size=1。"""
    assert len(batch) == 1
    item = batch[0]

    try:
        raw = voco_model.module if hasattr(voco_model, "module") else voco_model
        base = raw.base
        device = raw.voco_embeds.device
        dtype = raw.voco_embeds.dtype

        import decord
        from qwen_vl_utils import process_vision_info
        from data import extract_frames
        import uuid as _uuid
        import tempfile as _tmpfile

        vr = decord.VideoReader(item["_resolved_video"])
        duration = len(vr) / vr.get_avg_fps()

        n_frames = max(frames_per_segment, min(int(duration * fps), max_frames))
        n_frames = (n_frames // frames_per_segment) * frames_per_segment
        if n_frames == 0:
            n_frames = frames_per_segment

        frames, _, _ = extract_frames(item["_resolved_video"], n_frames)

        uid = f"{os.getpid()}_{_uuid.uuid4().hex[:8]}"
        tmp_dir = _tmpfile.gettempdir()
        tmp_paths = []
        for i, img in enumerate(frames):
            p = os.path.join(tmp_dir, f"_vf_{uid}_{i}.jpg")
            img.save(p)
            tmp_paths.append(p)

        messages = [{"role": "user", "content": [
            {"type": "video", "video": tmp_paths, "fps": fps},
            {"type": "text", "text": "x"},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info(messages)
        proc_inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                                return_tensors="pt")

        for p in tmp_paths:
            try:
                os.remove(p)
            except OSError:
                pass

        pv = proc_inputs.get("pixel_values_videos")
        vg = proc_inputs.get("video_grid_thw")
        if pv is None:
            return None

        video_embeds, tokens_per_frame, _ = get_video_embeds(base, pv, vg, device)
        video_embeds = video_embeds.to(dtype)

        # Q/A embeds
        inner = get_inner_model(base)
        embed_layer = inner.get_input_embeddings()

        q_text = f"<|im_start|>user\n{item['question']}<|im_end|>\n<|im_start|>assistant\n"
        a_text = f"{item['answer']}<|im_end|>"

        q_embeds, _ = encode_text(tokenizer, embed_layer, q_text, device, dtype)
        a_embeds, a_ids = encode_text(tokenizer, embed_layer, a_text, device, dtype)

        return {
            "video_embeds": video_embeds,
            "tokens_per_frame": tokens_per_frame,
            "frames_per_segment": frames_per_segment,
            "q_embeds": q_embeds,
            "a_embeds": a_embeds,
            "a_ids": a_ids,
        }
    except Exception as e:
        print(f"  [collate 错误] {item.get('_resolved_video', '?')}: {e}\n"
              f"{traceback.format_exc()}")
        return None


# ============================================================
# 分布式工具
# ============================================================

def setup_distributed():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        import datetime as _dt
        dist.init_process_group("nccl", timeout=_dt.timedelta(hours=2))
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


# ============================================================
# Checkpoint（只保存 voco_embeds，无 LoRA）
# ============================================================

def save_checkpoint(raw_model, optimizer, epoch, step, args, output_dir, tag="step"):
    ckpt_path = os.path.join(output_dir, f"checkpoint_{tag}.pt")
    torch.save({
        "voco_embeds": raw_model.voco_embeds.detach().cpu(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "K_seg": args.K_seg,
    }, ckpt_path)
    return ckpt_path


def load_resume_checkpoint(model, optimizer, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.voco_embeds.data.copy_(ckpt["voco_embeds"])
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"], ckpt["step"]


# ============================================================
# 训练
# ============================================================

def train(args):
    local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    log("=" * 60)
    log("004 VoCo-only 训练（无 LoRA，只训 voco_embeds）")
    log(f"  GPU 数量: {world_size}")
    log(f"  K_seg: {args.K_seg}, frames/seg: {args.frames_per_segment}")
    log(f"  fps: {args.fps}, max_frames: {args.max_frames}")
    log(f"  lr: {args.lr}, epochs: {args.epochs}")
    log(f"  可训练参数: {args.K_seg} × 3584 = {args.K_seg * 3584:,}")
    log(f"  save_steps: {args.save_steps}")
    log("=" * 60)

    model, processor, tokenizer = setup_voco_model(
        model_name=args.model_path,
        K_seg=args.K_seg,
        device=device,
        use_lora=False,
        attn_implementation="eager",
    )

    # 确认只有 voco_embeds 可训练
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    log(f"  可训练参数列表:")
    for n, p in trainable:
        log(f"    {n}: {p.shape}")
    assert len(trainable) == 1 and "voco_embeds" in trainable[0][0], \
        f"应该只有 voco_embeds 可训练，但发现: {[n for n, _ in trainable]}"

    video_dirs = args.video_dirs.split(",")
    dataset = VoCoVideoDataset(
        args.data_path, video_dirs, max_samples=args.max_samples,
    )

    if len(dataset) == 0:
        log("⚠️ 数据集为空，退出")
        return

    n_val = min(50, max(1, len(dataset) // 10))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    log(f"  数据: train={n_train}, val={n_val}")

    raw_model = model

    def collate(batch):
        return voco_only_collate(
            batch, raw_model, processor, tokenizer,
            fps=args.fps,
            frames_per_segment=args.frames_per_segment,
            max_frames=args.max_frames,
        )

    train_sampler = DistributedSampler(train_set, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_set, batch_size=1, sampler=train_sampler,
        shuffle=(train_sampler is None), collate_fn=collate,
    )

    # 只有 voco_embeds
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    if is_main():
        os.makedirs(args.output_dir, exist_ok=True)

    # Resume
    start_epoch = 0
    start_step = 0
    if args.resume_from:
        log(f"从 {args.resume_from} 恢复...")
        start_epoch, start_step = load_resume_checkpoint(
            raw_model, optimizer, args.resume_from,
        )
        log(f"  恢复到 epoch={start_epoch}, step={start_step}")
        if world_size > 1:
            for p in model.parameters():
                dist.broadcast(p.data, src=0)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)

        total_loss = 0
        n = 0
        global_step = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}",
                    disable=not is_main())
        for batch in pbar:
            global_step += 1

            if epoch == start_epoch and global_step <= start_step:
                continue

            if batch is None:
                if world_size > 1:
                    dummy = torch.zeros(1, device=device)
                    dist.all_reduce(dummy)
                continue

            loss = None
            try:
                logits, _, total_voco = voco_concat_forward(
                    model,
                    vision_embeds=batch["video_embeds"],
                    tokens_per_frame=batch["tokens_per_frame"],
                    frames_per_segment=batch["frames_per_segment"],
                    q_embeds=batch["q_embeds"],
                    a_embeds=batch["a_embeds"],
                )

                q_len = batch["q_embeds"].shape[0]
                a_len = batch["a_embeds"].shape[0]
                a_ids = batch["a_ids"]

                shift_logits = logits[0, q_len - 1: q_len - 1 + a_len, :]
                shift_labels = a_ids.to(device)

                if shift_logits.shape[0] != shift_labels.shape[0]:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
                else:
                    loss = nn.functional.cross_entropy(
                        shift_logits.float(), shift_labels,
                    )

                if torch.isnan(loss):
                    loss = torch.tensor(0.0, device=device, requires_grad=True)

            except RuntimeError as e:
                if "out of memory" in str(e):
                    if is_main():
                        print(f"  [OOM] 跳过该样本，清理显存")
                    torch.cuda.empty_cache()
                else:
                    if is_main():
                        print(f"  [错误] {e}")
                loss = None

            except Exception as e:
                if is_main():
                    print(f"  [错误] {e}")
                loss = None

            # 梯度同步
            if loss is not None and loss.requires_grad:
                optimizer.zero_grad()
                loss.backward()

                if world_size > 1:
                    for p in params:
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                            p.grad.div_(world_size)

                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

                real_loss = loss.item()
                if real_loss > 0:
                    total_loss += real_loss
                    n += 1
                if is_main():
                    pbar.set_postfix(loss=f"{real_loss:.4f}", voco=total_voco)
            else:
                if world_size > 1:
                    for p in params:
                        if p.grad is not None:
                            p.grad.zero_()
                    dummy = torch.zeros(1, device=device)
                    dist.all_reduce(dummy)

            # Step checkpoint
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                if world_size > 1:
                    dist.barrier()
                if is_main():
                    p = save_checkpoint(
                        raw_model, optimizer, epoch, global_step,
                        args, args.output_dir, tag=f"e{epoch+1}_s{global_step}",
                    )
                    log(f"  step checkpoint → {p}")
                if world_size > 1:
                    dist.barrier()

        avg_loss = total_loss / max(n, 1)
        log(f"  Epoch {epoch + 1}: train_loss={avg_loss:.4f} (n={n})")

        # Epoch checkpoint
        if world_size > 1:
            dist.barrier()

        if is_main():
            ckpt_path = os.path.join(
                args.output_dir, f"checkpoint_epoch{epoch + 1}.pt",
            )
            torch.save({
                "voco_embeds": raw_model.voco_embeds.detach().cpu(),
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "K_seg": args.K_seg,
            }, ckpt_path)
            log(f"  epoch checkpoint → {ckpt_path}")

            save_checkpoint(
                raw_model, optimizer, epoch + 1, 0,
                args, args.output_dir, tag="latest",
            )

        if world_size > 1:
            dist.barrier()

        start_step = 0

    log(f"\n训练完成.")
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoCo-only 训练（无 LoRA）")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--video_dirs", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--K_seg", type=int, default=8,
                        help="每段 voco token 数（默认 8，28K 可训练参数）")
    parser.add_argument("--frames_per_segment", type=int, default=2)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max_frames", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="学习率（无 LoRA 参数少，默认 1e-3）")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--model_path", type=str,
                        default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="模型路径（HF name 或本地路径）")
    args = parser.parse_args()
    train(args)

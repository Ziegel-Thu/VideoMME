"""
004 拼接版 (KV cache concat): 真正分段 forward

每段独立 forward → 提取该段 voco 位置的 KV cache → 拼接所有段 → 最后一次 forward 跑 [all_voco_cache, Q, A]

优势:
- 每段 forward 序列短（vision_t + voco_t ≈ 200+4 = 204 tokens）
- 最终 forward 短（n_seg × K_seg + Q + A ≈ 60+50 = 110 tokens）
- 总计算量 << 单 forward 版

劣势:
- 实现复杂（KV cache 拼接、position_ids 处理）
- 不同段的 KV cache 拼起来 position_ids 要保持一致

用法:
  python train_concat.py --data_path ... --video_dirs ... --output_dir ...
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


def get_inner_model(base):
    """获取最内层 Qwen2_5_VLForConditionalGeneration（解 PEFT/DDP）。"""
    m = base.module if hasattr(base, "module") else base
    from peft import PeftModel
    if isinstance(m, PeftModel):
        return m.base_model.model
    return m


def forward_segment(inner, vision_embeds, voco_embeds, position_offset=0):
    """跑一段的 forward: [vision_t, voco_t]，返回 voco 位置的 KV cache。

    优化：vision 部分不保留计算图（detach），只保留 voco 的梯度。
    这样每段 forward 的显存 = vision forward (无梯度) + voco KV cache (有梯度)。

    Args:
        inner: Qwen2_5_VLForConditionalGeneration
        vision_embeds: (V_t, D) 该段 vision tokens
        voco_embeds: (K, D) 该段 voco tokens（需要梯度）
        position_offset: 段在最终序列中的起始位置

    Returns:
        voco_cache: list of (K_proj, V_proj) 每层的 voco 部分 KV cache（保留梯度）
    """
    K = voco_embeds.shape[0]

    # Vision 部分不需要梯度（冻结的 vision encoder 输出）
    vision_detached = vision_embeds.detach()

    inputs_embeds = torch.cat([vision_detached, voco_embeds], dim=0).unsqueeze(0)
    seg_len = inputs_embeds.shape[1]
    attention_mask = torch.ones(1, seg_len, dtype=torch.long,
                                device=vision_embeds.device)
    # Qwen2.5-VL mRoPE 要求 position_ids 形状为 (3, batch, seq_len)
    # 对非视觉 token，3 个分量（temporal, height, width）取相同值
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
    n_layers = len(pkv)
    for layer_idx in range(n_layers):
        k, v = pkv[layer_idx]
        # 只保留 voco 部分（最后 K 个），vision 部分丢弃
        voco_cache.append((k[:, :, -K:, :], v[:, :, -K:, :]))

    return voco_cache


def concat_voco_caches(seg_caches):
    """把多段的 voco cache 拼接成一个完整 cache。

    Args:
        seg_caches: list of seg_cache, 每个 seg_cache 是 list of (k, v) per layer

    Returns:
        cache: DynamicCache containing all voco K/V
        total_voco_len: int
    """
    cache = DynamicCache()
    n_layers = len(seg_caches[0])
    K_per_seg = seg_caches[0][0][0].shape[2]
    total_K = K_per_seg * len(seg_caches)

    for layer_idx in range(n_layers):
        k_list = [seg_caches[s][layer_idx][0] for s in range(len(seg_caches))]
        v_list = [seg_caches[s][layer_idx][1] for s in range(len(seg_caches))]
        cache.update(
            torch.cat(k_list, dim=2),
            torch.cat(v_list, dim=2),
            layer_idx,
        )

    return cache, total_K


def voco_concat_forward(model, vision_embeds, tokens_per_frame, frames_per_segment,
                        q_embeds, a_embeds=None):
    """完整的拼接 forward 流程。

    1. 切段
    2. 每段独立 forward 提取 voco cache
    3. 拼接所有 voco cache
    4. 最后一次 forward 跑 [Q, A]，past_key_values 是拼接的 voco cache

    Returns:
        logits, hidden_states (Q+A 部分)
    """
    # 解包 DDP（如果 model 被 DDP 包装，需要通过 .module 访问属性）
    raw = model.module if hasattr(model, "module") else model
    inner = get_inner_model(raw.base)
    K = raw.K_seg
    voco_per_seg = raw.voco_embeds  # (K, D)

    # 切段
    segments = split_into_segments(vision_embeds, tokens_per_frame, frames_per_segment)
    n_segments = len(segments)

    # 每段独立 forward 提取 voco cache（累加 position_offset 保证位置递增）
    seg_caches = []
    position_offset = 0
    for seg in segments:
        seg_cache = forward_segment(inner, seg, voco_per_seg, position_offset=position_offset)
        seg_caches.append(seg_cache)
        # 该段 vision + voco 占的位置总数
        position_offset += seg.shape[0] + K

    # 拼接 voco caches
    full_cache, total_voco = concat_voco_caches(seg_caches)

    # 最后一次 forward: 输入是 [Q, A]，past_kv 是 voco caches
    if a_embeds is not None:
        text_embeds = torch.cat([q_embeds, a_embeds], dim=0).unsqueeze(0)
    else:
        text_embeds = q_embeds.unsqueeze(0)

    text_len = text_embeds.shape[1]
    # attention_mask: (1, total_voco + text_len) — 标准 padding mask 全 1
    full_attn_mask = torch.ones(1, total_voco + text_len,
                                dtype=torch.long, device=text_embeds.device)

    # position_ids: text 部分接在所有段（vision+voco）之后
    text_pos = torch.arange(
        position_offset, position_offset + text_len, device=text_embeds.device,
    )
    # mRoPE: (3, 1, text_len)
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


def voco_concat_collate(batch, voco_model, processor, tokenizer,
                        fps=1.0, frames_per_segment=2, max_frames=30,
                        feature_dir=None):
    """拼接版 collate: 准备 vision_embeds + Q/A embeds，不提前拼成大序列。

    如果 feature_dir 不为 None，优先从预提特征加载（跳过 vision encoder）。
    """
    assert len(batch) == 1
    item = batch[0]

    try:
        # 解包 DDP（collate 应使用 unwrapped model）
        raw = voco_model.module if hasattr(voco_model, "module") else voco_model
        base = raw.base
        device = raw.voco_embeds.device
        dtype = raw.voco_embeds.dtype

        video_embeds = None
        tokens_per_frame = None

        # 尝试从预提特征加载
        if feature_dir:
            vname = os.path.basename(item["_resolved_video"])
            # 特征文件命名: {video_name}_1fps.pt
            feat_name = vname.replace(".mp4", "") + "_1fps.pt"
            feat_path = os.path.join(feature_dir, feat_name)
            if os.path.exists(feat_path):
                feat = torch.load(feat_path, map_location=device)
                video_embeds = feat["video_embeds"].to(device=device, dtype=dtype)
                tokens_per_frame = feat["tokens_per_frame"]

        # 没有预提特征则走 vision encoder
        if video_embeds is None:
            import uuid as _uuid, tempfile as _tmpfile
            import decord
            from qwen_vl_utils import process_vision_info
            from data import extract_frames

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
        m = base.module if hasattr(base, "module") else base
        from peft import PeftModel
        if isinstance(m, PeftModel):
            embed_layer = m.base_model.model.get_input_embeddings()
        else:
            embed_layer = m.get_input_embeddings()

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


def setup_distributed():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        import datetime as _dt; dist.init_process_group("nccl", timeout=_dt.timedelta(hours=2))
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


def save_checkpoint(raw_model, optimizer, epoch, step, args, output_dir, tag="step"):
    """保存 checkpoint（voco_embeds + LoRA + optimizer）。"""
    ckpt_path = os.path.join(output_dir, f"checkpoint_{tag}.pt")
    torch.save({
        "voco_embeds": raw_model.voco_embeds.detach().cpu(),
        "lora_state_dict": {
            k: v.cpu() for k, v in raw_model.base.state_dict().items()
            if "lora_" in k
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "K_seg": args.K_seg,
    }, ckpt_path)
    return ckpt_path


def load_resume_checkpoint(model, optimizer, ckpt_path):
    """从 step checkpoint 恢复训练。"""
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 恢复 voco_embeds
    model.voco_embeds.data.copy_(ckpt["voco_embeds"])

    # 恢复 LoRA weights
    m = model.base.module if hasattr(model.base, "module") else model.base
    m.load_state_dict(ckpt["lora_state_dict"], strict=False)

    # 恢复 optimizer
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    return ckpt["epoch"], ckpt["step"]


def train(args):
    local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    log("=" * 60)
    log("004 VoCo 拼接版训练")
    log(f"  GPU 数量: {world_size}")
    log(f"  K_seg: {args.K_seg}, frames/seg: {args.frames_per_segment}")
    log(f"  save_steps: {args.save_steps}")
    if args.feature_dir:
        log(f"  预提特征: {args.feature_dir}")
    log("=" * 60)

    # 拼接版不注入 4D mask，可以用 SDPA 加速
    lora_targets = args.lora_targets.split(",") if args.lora_targets else None
    model, processor, tokenizer = setup_voco_model(
        K_seg=args.K_seg,
        lora_r=args.lora_r,
        lora_targets=lora_targets,
        device=device,
        gradient_checkpointing=False,  # 拼接版不能用 grad ckpt (和 use_cache 冲突)
        attn_implementation="sdpa",
    )

    video_dirs = args.video_dirs.split(",")
    dataset = VoCoVideoDataset(
        args.data_path, video_dirs, max_samples=args.max_samples,
    )

    n_val = min(50, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    raw_model = model

    def collate(batch):
        return voco_concat_collate(
            batch, raw_model, processor, tokenizer,
            fps=args.fps,
            frames_per_segment=args.frames_per_segment,
            max_frames=args.max_frames,
            feature_dir=args.feature_dir,
        )

    train_sampler = DistributedSampler(train_set, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_set, batch_size=1, sampler=train_sampler,
        shuffle=(train_sampler is None), collate_fn=collate,
    )

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
        # 同步参数到所有 rank
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

            # 跳过已完成的 steps（resume 时）
            if epoch == start_epoch and global_step <= start_step:
                continue

            if batch is None:
                continue

            try:
                logits, _, total_voco = voco_concat_forward(
                    model,
                    vision_embeds=batch["video_embeds"],
                    tokens_per_frame=batch["tokens_per_frame"],
                    frames_per_segment=batch["frames_per_segment"],
                    q_embeds=batch["q_embeds"],
                    a_embeds=batch["a_embeds"],
                )

                # logits 是 [Q, A] 部分；构造 labels（只算 A 的 loss）
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

                optimizer.zero_grad()
                loss.backward()

                # 多卡手动同步梯度
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

            except Exception as e:
                if is_main():
                    print(f"  [错误] {e}")
                pass

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
        log(f"  Epoch {epoch + 1}: train_loss={avg_loss:.4f}")

        # Epoch checkpoint
        if world_size > 1:
            dist.barrier()

        if is_main():
            # epoch checkpoint（不含 optimizer，用于 eval）
            ckpt_path = os.path.join(
                args.output_dir, f"checkpoint_epoch{epoch + 1}.pt",
            )
            torch.save({
                "voco_embeds": raw_model.voco_embeds.detach().cpu(),
                "lora_state_dict": {
                    k: v.cpu() for k, v in raw_model.base.state_dict().items()
                    if "lora_" in k
                },
                "epoch": epoch + 1,
                "val_loss": avg_loss,
                "K_seg": args.K_seg,
            }, ckpt_path)
            log(f"  epoch checkpoint → {ckpt_path}")

            # 同时保存含 optimizer 的 resume checkpoint
            save_checkpoint(
                raw_model, optimizer, epoch + 1, 0,
                args, args.output_dir, tag="latest",
            )

        if world_size > 1:
            dist.barrier()

        # 重置 start_step（只在第一个 resume epoch 跳 step）
        start_step = 0

    log(f"\nDone.")
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
    parser.add_argument("--lora_targets", type=str, default=None,
                        help="LoRA target modules，逗号分隔 (默认全部 7 个)")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="每 N steps 保存 checkpoint（防抢占）")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="从 step checkpoint 恢复训练")
    parser.add_argument("--feature_dir", type=str, default=None,
                        help="预提特征目录（跳过 vision encoder）")
    args = parser.parse_args()
    train(args)

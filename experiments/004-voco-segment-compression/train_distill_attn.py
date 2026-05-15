"""
方案 B：Attention Output 匹配蒸馏

逐段独立训练 voco_embeds。Teacher 和 Student 都拼接 Q text，
比较 Q 在 attend 到 vision/voco 之后的 hidden state 表示。

Teacher: [vision_seg (V), Q (~50)] → decoder → Q 位置 hidden (50, D)
Student: [voco_embeds (K), Q (~50)] → decoder → Q 位置 hidden (50, D)
Loss = MSE(student_Q_hidden, teacher_Q_hidden)

相比方案 A，这里引入 Q 作为 probe，衡量 voco tokens 能否让 Q
获得与 dense vision tokens 等价的 attend 结果。

用法:
  python train_distill_attn.py \
      --data_path /path/to/data.jsonl \
      --video_dirs /path/to/videos \
      --output_dir outputs_distill_attn \
      --K_seg 8 --epochs 3
"""

import os
import sys
import argparse
import traceback
import uuid
import tempfile

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from model import setup_voco_model, get_video_embeds, split_into_segments
from data import VoCoVideoDataset, extract_frames, encode_text


# ============================================================
# 工具函数
# ============================================================

def get_inner(base):
    """获取最内层 Qwen2_5_VLForConditionalGeneration（解 PEFT/DDP）。"""
    m = base.module if hasattr(base, 'module') else base
    try:
        from peft import PeftModel
        if isinstance(m, PeftModel):
            return m.base_model.model
    except ImportError:
        pass
    return m


# ============================================================
# Collate: 视频采帧 → vision encoder → 切段 + Q embeds
# ============================================================

def distill_collate(batch, model, processor, tokenizer,
                    fps=1.0, frames_per_segment=2, max_frames=30):
    """准备 segments 列表 + Q embeds，batch_size=1。"""
    assert len(batch) == 1
    item = batch[0]

    try:
        base = model.base
        device = model.voco_embeds.device
        dtype = model.voco_embeds.dtype

        import decord
        from qwen_vl_utils import process_vision_info

        vr = decord.VideoReader(item["_resolved_video"])
        duration = len(vr) / vr.get_avg_fps()

        n_frames = max(frames_per_segment, min(int(duration * fps), max_frames))
        n_frames = (n_frames // frames_per_segment) * frames_per_segment
        if n_frames == 0:
            n_frames = frames_per_segment

        frames, _, _ = extract_frames(item["_resolved_video"], n_frames)

        # processor 需要临时图片文件
        uid = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
        tmp_dir = tempfile.gettempdir()
        tmp_paths = []
        for i, img in enumerate(frames):
            p = os.path.join(tmp_dir, f"_vf_{uid}_{i}.jpg")
            img.save(p)
            tmp_paths.append(p)

        messages = [{"role": "user", "content": [
            {"type": "video", "video": tmp_paths, "fps": fps},
            {"type": "text", "text": "x"},
        ]}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        proc_inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            return_tensors="pt",
        )

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

        segments = split_into_segments(
            video_embeds, tokens_per_frame, frames_per_segment,
        )

        # Q embeds（只需要 question，不需要 answer）
        inner = get_inner(base)
        embed_layer = inner.get_input_embeddings()
        q_text = (f"<|im_start|>user\n{item['question']}"
                  f"<|im_end|>\n<|im_start|>assistant\n")
        q_ids = tokenizer.encode(
            q_text, add_special_tokens=False, return_tensors="pt",
        ).to(device)
        q_embeds = embed_layer(q_ids).squeeze(0).to(dtype)  # (Q_len, D)

        return {"segments": segments, "q_embeds": q_embeds}

    except Exception as e:
        print(f"  [collate 错误] {item.get('_resolved_video', '?')}: {e}")
        return None


# ============================================================
# 逐段蒸馏 loss
# ============================================================

def distill_attn_loss(inner, vision_seg, voco_embeds, q_embeds, device):
    """对单段计算 attention output 匹配 loss。

    Teacher: [vision_seg (V), Q (Q_len)] → decoder → Q 位置 hidden
    Student: [voco_embeds (K), Q (Q_len)] → decoder → Q 位置 hidden
    Loss = MSE(student_Q_hidden, teacher_Q_hidden)
    """
    V = vision_seg.shape[0]
    K = voco_embeds.shape[0]
    Q_len = q_embeds.shape[0]
    q_detached = q_embeds.detach()

    # --- Teacher (no_grad) ---
    with torch.no_grad():
        t_input = torch.cat(
            [vision_seg.detach(), q_detached], dim=0,
        ).unsqueeze(0)  # (1, V+Q_len, D)
        t_len = V + Q_len
        t_pos = torch.arange(t_len, device=device)
        t_pos_ids = t_pos.view(1, 1, -1).expand(3, 1, -1)
        t_attn = torch.ones(1, t_len, dtype=torch.long, device=device)

        t_out = inner.model(
            inputs_embeds=t_input,
            attention_mask=t_attn,
            position_ids=t_pos_ids,
            use_cache=False,
        )
        t_hidden = t_out[0].squeeze(0)     # (V+Q_len, D)
        t_q_hidden = t_hidden[-Q_len:]     # (Q_len, D)

    # --- Student (有 grad，只有 voco_embeds 有梯度) ---
    s_input = torch.cat(
        [voco_embeds, q_detached], dim=0,
    ).unsqueeze(0)  # (1, K+Q_len, D)
    s_len = K + Q_len
    s_pos = torch.arange(s_len, device=device)
    s_pos_ids = s_pos.view(1, 1, -1).expand(3, 1, -1)
    s_attn = torch.ones(1, s_len, dtype=torch.long, device=device)

    s_out = inner.model(
        inputs_embeds=s_input,
        attention_mask=s_attn,
        position_ids=s_pos_ids,
        use_cache=False,
    )
    s_hidden = s_out[0].squeeze(0)     # (K+Q_len, D)
    s_q_hidden = s_hidden[-Q_len:]     # (Q_len, D)

    loss = nn.functional.mse_loss(s_q_hidden, t_q_hidden)
    return loss


# ============================================================
# 训练循环
# ============================================================

def train(args):
    # 分布式初始化
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        import datetime as _dt
        dist.init_process_group("nccl", timeout=_dt.timedelta(hours=2))
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0
        world_size = 1

    device = torch.device(f"cuda:{local_rank}")
    is_main = (local_rank == 0)

    def log(msg):
        if is_main:
            print(msg)

    log("=" * 60)
    log("方案 B: Attention Output 匹配蒸馏")
    log(f"  GPU 数量: {world_size}")
    log(f"  K_seg: {args.K_seg}, frames/seg: {args.frames_per_segment}")
    log(f"  fps: {args.fps}, max_frames: {args.max_frames}")
    log(f"  lr: {args.lr}, epochs: {args.epochs}")
    log(f"  save_steps: {args.save_steps}")
    log(f"  model: {args.model_path}")
    log("=" * 60)

    # 加载模型（冻结 LLM，只训 voco_embeds）
    model, processor, tokenizer = setup_voco_model(
        model_name=args.model_path,
        K_seg=args.K_seg,
        device=device,
        use_lora=False,
        attn_implementation="sdpa",
    )

    # 确认可训练参数
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    log(f"  可训练参数:")
    for n, p in trainable:
        log(f"    {n}: {p.shape}")
    assert len(trainable) == 1 and "voco_embeds" in trainable[0][0]

    # 多卡同步 voco_embeds 初始值
    if world_size > 1:
        dist.broadcast(model.voco_embeds.data, src=0)

    base = model.base
    inner = get_inner(base)

    # 数据集
    video_dirs = args.video_dirs.split(",")
    dataset = VoCoVideoDataset(
        args.data_path, video_dirs, max_samples=args.max_samples,
    )
    if len(dataset) == 0:
        log("⚠️ 数据集为空，退出")
        return

    def collate(batch):
        return distill_collate(
            batch, model, processor, tokenizer,
            fps=args.fps,
            frames_per_segment=args.frames_per_segment,
            max_frames=args.max_frames,
        )

    train_sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    loader = DataLoader(dataset, batch_size=1,
                        sampler=train_sampler,
                        shuffle=(train_sampler is None),
                        collate_fn=collate)

    # 优化器（只有 voco_embeds）
    optimizer = torch.optim.AdamW(
        [model.voco_embeds], lr=args.lr, weight_decay=0.01,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0
    start_epoch = 0

    # 从 checkpoint 恢复
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu")
        model.voco_embeds.data.copy_(ckpt["voco_embeds"].to(device))
        start_epoch = ckpt.get("epoch", 0)
        log(f"  从 {args.resume_from} 恢复，start_epoch={start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_n = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}",
                    disable=not is_main)
        for batch_data in pbar:
            if batch_data is None:
                continue

            segments = batch_data["segments"]
            q_embeds = batch_data["q_embeds"]
            global_step += 1

            try:
                optimizer.zero_grad()
                sample_loss = 0.0
                n_segs = 0

                # 逐段计算 loss 并 backward（释放每段计算图）
                for seg in segments:
                    seg_loss = distill_attn_loss(
                        inner, seg, model.voco_embeds, q_embeds, device,
                    )
                    seg_loss.backward()
                    sample_loss += seg_loss.item()
                    n_segs += 1

                # 梯度同步：所有 rank 的 voco_embeds 梯度求平均
                if world_size > 1 and model.voco_embeds.grad is not None:
                    dist.all_reduce(model.voco_embeds.grad, op=dist.ReduceOp.SUM)
                    model.voco_embeds.grad.div_(world_size)

                torch.nn.utils.clip_grad_norm_([model.voco_embeds], 1.0)
                optimizer.step()

                avg_seg_loss = sample_loss / max(n_segs, 1)
                epoch_loss += sample_loss
                epoch_n += n_segs

                if is_main:
                    pbar.set_postfix(
                        loss=f"{avg_seg_loss:.6f}", segs=n_segs, step=global_step,
                    )
                    print(f"  [step {global_step}] loss={avg_seg_loss:.6f} "
                          f"(n_segs={n_segs})")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    if is_main:
                        print(f"  [OOM] step {global_step}，跳过该样本")
                    torch.cuda.empty_cache()
                else:
                    if is_main:
                        print(f"  [错误] step {global_step}: {e}")
                continue

            except Exception as e:
                if is_main:
                    print(f"  [错误] step {global_step}: {e}")
                continue

            # 定期保存（只 rank 0）
            if args.save_steps > 0 and global_step % args.save_steps == 0 and is_main:
                ckpt_path = os.path.join(
                    args.output_dir,
                    f"voco_embeds_e{epoch+1}_s{global_step}.pt",
                )
                torch.save({
                    "voco_embeds": model.voco_embeds.detach().cpu(),
                    "epoch": epoch + 1,
                    "step": global_step,
                    "K_seg": args.K_seg,
                }, ckpt_path)
                print(f"  💾 checkpoint → {ckpt_path}")

        # Epoch 结束
        avg_epoch_loss = epoch_loss / max(epoch_n, 1)
        log(f"  Epoch {epoch + 1} 完成: avg_loss={avg_epoch_loss:.6f} "
            f"(total_segs={epoch_n})")

        if is_main:
            ckpt_path = os.path.join(
                args.output_dir, f"voco_embeds_epoch{epoch+1}.pt",
            )
            torch.save({
                "voco_embeds": model.voco_embeds.detach().cpu(),
                "epoch": epoch + 1,
                "train_loss": avg_epoch_loss,
                "K_seg": args.K_seg,
            }, ckpt_path)
            print(f"  💾 epoch checkpoint → {ckpt_path}")
        if dist.is_initialized():
            dist.barrier()

    log("\n训练完成.")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="方案 B: Attention Output 匹配蒸馏",
    )
    parser.add_argument("--data_path", required=True, help="jsonl 数据路径")
    parser.add_argument("--video_dirs", required=True, help="视频目录，逗号分隔")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--K_seg", type=int, default=8,
                        help="每段 voco token 数")
    parser.add_argument("--frames_per_segment", type=int, default=2)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max_frames", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--model_path", type=str,
                        default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="模型路径")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="从 checkpoint 恢复训练")
    args = parser.parse_args()
    train(args)

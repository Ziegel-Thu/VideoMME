"""
方案 D：KV Cache 匹配蒸馏

逐段独立训练 voco_embeds。在每层 decoder 的 KV cache 层面，
用 MSE 匹配 student 的 KV 与 teacher 经均值池化后的 KV。

Teacher: vision_seg (V) → decoder (use_cache) → 每层 KV (n_heads, V, head_dim)
Student: voco_embeds (K) → decoder (use_cache) → 每层 KV (n_heads, K, head_dim)
对每层: Loss += MSE(student_K, pool(teacher_K)) + MSE(student_V, pool(teacher_V))
Total Loss = sum over all layers

Teacher 的 V 个 KV 用 adaptive avg pool 压缩到 K 个，再和 student 匹配。

用法:
  python train_distill_kv.py \
      --data_path /path/to/data.jsonl \
      --video_dirs /path/to/videos \
      --output_dir outputs_distill_kv \
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import setup_voco_model, get_video_embeds, split_into_segments
from data import VoCoVideoDataset, extract_frames


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


def mean_pool_kv(tensor, target_len):
    """均值池化 KV cache 的序列维度。

    Args:
        tensor: (B, H, S, D) — KV cache
        target_len: 目标序列长度（K_seg）
    Returns:
        (B, H, target_len, D)
    """
    B, H, S, D = tensor.shape
    # adaptive_avg_pool1d 需要 (N, C, L) 格式
    t = tensor.reshape(B * H, S, D).permute(0, 2, 1)  # (B*H, D, S)
    t = F.adaptive_avg_pool1d(t, target_len)            # (B*H, D, target_len)
    return t.permute(0, 2, 1).reshape(B, H, target_len, D)


# ============================================================
# Collate: 视频采帧 → vision encoder → 切段
# ============================================================

def distill_collate(batch, model, processor, tokenizer,
                    fps=1.0, frames_per_segment=2, max_frames=30):
    """准备 segments 列表，batch_size=1。"""
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

        return {"segments": segments}

    except Exception as e:
        print(f"  [collate 错误] {item.get('_resolved_video', '?')}: {e}")
        return None


# ============================================================
# 逐段蒸馏 loss
# ============================================================

def distill_kv_loss(inner, vision_seg, voco_embeds, device):
    """对单段计算 KV cache 匹配 loss。

    Teacher: vision_seg (V) → decoder (use_cache) → 每层 KV
    Student: voco_embeds (K) → decoder (use_cache) → 每层 KV
    对每层: MSE(student_K, pool(teacher_K)) + MSE(student_V, pool(teacher_V))
    """
    V = vision_seg.shape[0]
    K = voco_embeds.shape[0]

    # --- Teacher (no_grad) ---
    with torch.no_grad():
        t_embeds = vision_seg.detach().unsqueeze(0)  # (1, V, D)
        t_pos = torch.arange(V, device=device)
        t_pos_ids = t_pos.view(1, 1, -1).expand(3, 1, -1)
        t_attn = torch.ones(1, V, dtype=torch.long, device=device)

        t_out = inner.model(
            inputs_embeds=t_embeds,
            attention_mask=t_attn,
            position_ids=t_pos_ids,
            use_cache=True,
        )
        t_pkv = t_out.past_key_values  # DynamicCache

    # --- Student (有 grad) ---
    s_embeds = voco_embeds.unsqueeze(0)  # (1, K, D)
    s_pos = torch.arange(K, device=device)
    s_pos_ids = s_pos.view(1, 1, -1).expand(3, 1, -1)
    s_attn = torch.ones(1, K, dtype=torch.long, device=device)

    s_out = inner.model(
        inputs_embeds=s_embeds,
        attention_mask=s_attn,
        position_ids=s_pos_ids,
        use_cache=True,
    )
    s_pkv = s_out.past_key_values  # DynamicCache

    # 逐层匹配 KV cache
    n_layers = len(s_pkv)
    total_loss = torch.tensor(0.0, device=device)

    for layer_idx in range(n_layers):
        t_k, t_v = t_pkv[layer_idx]  # (B, H, V, head_dim)
        s_k, s_v = s_pkv[layer_idx]  # (B, H, K, head_dim)

        # Teacher KV 均值池化到 K 个
        t_k_pooled = mean_pool_kv(t_k, K)  # (B, H, K, head_dim)
        t_v_pooled = mean_pool_kv(t_v, K)

        total_loss = total_loss + F.mse_loss(s_k, t_k_pooled.detach())
        total_loss = total_loss + F.mse_loss(s_v, t_v_pooled.detach())

    return total_loss


# ============================================================
# 训练循环
# ============================================================

def train(args):
    device = torch.device("cuda")

    print("=" * 60)
    print("方案 D: KV Cache 匹配蒸馏")
    print(f"  K_seg: {args.K_seg}, frames/seg: {args.frames_per_segment}")
    print(f"  fps: {args.fps}, max_frames: {args.max_frames}")
    print(f"  lr: {args.lr}, epochs: {args.epochs}")
    print(f"  save_steps: {args.save_steps}")
    print(f"  model: {args.model_path}")
    print("=" * 60)

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
    print(f"  可训练参数:")
    for n, p in trainable:
        print(f"    {n}: {p.shape}")
    assert len(trainable) == 1 and "voco_embeds" in trainable[0][0]

    base = model.base
    inner = get_inner(base)

    # 数据集
    video_dirs = args.video_dirs.split(",")
    dataset = VoCoVideoDataset(
        args.data_path, video_dirs, max_samples=args.max_samples,
    )
    if len(dataset) == 0:
        print("⚠️ 数据集为空，退出")
        return

    def collate(batch):
        return distill_collate(
            batch, model, processor, tokenizer,
            fps=args.fps,
            frames_per_segment=args.frames_per_segment,
            max_frames=args.max_frames,
        )

    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate)

    # 优化器（只有 voco_embeds）
    optimizer = torch.optim.AdamW(
        [model.voco_embeds], lr=args.lr, weight_decay=0.01,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_n = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch_data in pbar:
            if batch_data is None:
                continue

            segments = batch_data["segments"]
            global_step += 1

            try:
                optimizer.zero_grad()
                sample_loss = 0.0
                n_segs = 0

                # 逐段计算 loss 并 backward（释放每段计算图）
                for seg in segments:
                    seg_loss = distill_kv_loss(
                        inner, seg, model.voco_embeds, device,
                    )
                    seg_loss.backward()
                    sample_loss += seg_loss.item()
                    n_segs += 1

                torch.nn.utils.clip_grad_norm_([model.voco_embeds], 1.0)
                optimizer.step()

                avg_seg_loss = sample_loss / max(n_segs, 1)
                epoch_loss += sample_loss
                epoch_n += n_segs

                pbar.set_postfix(
                    loss=f"{avg_seg_loss:.4f}", segs=n_segs, step=global_step,
                )
                print(f"  [step {global_step}] loss={avg_seg_loss:.4f} "
                      f"(n_segs={n_segs})")

            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  [OOM] step {global_step}，跳过该样本")
                    torch.cuda.empty_cache()
                else:
                    print(f"  [错误] step {global_step}: {e}")
                continue

            except Exception as e:
                print(f"  [错误] step {global_step}: {e}")
                continue

            # 定期保存
            if args.save_steps > 0 and global_step % args.save_steps == 0:
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
        print(f"  Epoch {epoch + 1} 完成: avg_loss={avg_epoch_loss:.4f} "
              f"(total_segs={epoch_n})")

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

    print("\n训练完成.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="方案 D: KV Cache 匹配蒸馏",
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
    args = parser.parse_args()
    train(args)

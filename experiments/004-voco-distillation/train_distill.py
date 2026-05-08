"""
VoCo-style Attention Distillation: 视频 token 压缩

独立模块，不依赖 bottleneck/LoRA/latent tokens。
目标：教 SegmentCompressor 把 dense vision tokens 压成少量 segment tokens，且不丢信息。

蒸馏方式：
  Teacher: dense video tokens → LLM → answer logits
  Student: compressed segment tokens → LLM → answer logits
  Loss: KL(teacher_logits, student_logits) 在 answer 位置

用法:
  # 单卡
  python train_distill.py \
    --data_path ../../data/parsed/visual_qa_v3_60s_train.jsonl \
    --video_dirs ../../data/llava-video/0_30_s_academic_v0_1 \
    --output_dir outputs_distill \
    --tokens_per_frame 8 --num_frames 4 --epochs 5

  # 多卡
  torchrun --nproc_per_node=4 train_distill.py ...
"""

import os
import json
import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "../003-bottleneck-multigpu"))
from data import VideoQADataset, extract_frames

# Qwen2.5-VL special token IDs
VISION_START_ID = 151652
VISION_END_ID = 151653
VIDEO_PAD_ID = 151656


# ============================================================
# Segment Compressor
# ============================================================

class SegmentCompressor(nn.Module):
    """把每帧的 dense vision tokens 压缩成少量 segment tokens。

    使用 cross-attention：可学习的 segment queries attend 到 dense tokens。
    Perceiver-style：cross-attn + FFN + residual + norm。
    """

    def __init__(self, hidden_dim=3584, tokens_per_frame=8, num_heads=8):
        super().__init__()
        self.tokens_per_frame = tokens_per_frame
        self.hidden_dim = hidden_dim

        # 可学习 segment queries
        self.segment_queries = nn.Parameter(
            torch.randn(tokens_per_frame, hidden_dim) * 0.02
        )

        # Cross-attention
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_kv = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True,
        )

        # FFN
        self.norm_ffn = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, dense_tokens):
        """
        Args:
            dense_tokens: (B, L, D) — 一帧/一段的 dense vision tokens

        Returns:
            compressed: (B, tokens_per_frame, D) — 压缩后的 segment tokens
        """
        B = dense_tokens.shape[0]
        queries = self.segment_queries.to(
            dtype=dense_tokens.dtype, device=dense_tokens.device,
        )
        queries = queries.unsqueeze(0).expand(B, -1, -1)

        # Cross-attention + residual
        q = self.norm_q(queries)
        kv = self.norm_kv(dense_tokens)
        attn_out, _ = self.cross_attn(query=q, key=kv, value=kv)
        x = queries + attn_out

        # FFN + residual
        x = x + self.ffn(self.norm_ffn(x))

        return x

    def compress_video(self, video_embeds, tokens_per_frame_list):
        """按帧切分 dense video embeddings 并压缩。

        Args:
            video_embeds: (total_tokens, D) — vision encoder 的完整输出
            tokens_per_frame_list: list[int] — 每帧的 token 数量

        Returns:
            compressed: (1, total_compressed_tokens, D)
        """
        segments = []
        offset = 0
        for n_tokens in tokens_per_frame_list:
            frame_tokens = video_embeds[offset:offset + n_tokens].unsqueeze(0)
            compressed = self.forward(frame_tokens)  # (1, tpf, D)
            segments.append(compressed.squeeze(0))
            offset += n_tokens

        # 拼接所有帧的 segment tokens
        return torch.cat(segments, dim=0).unsqueeze(0)  # (1, N*tpf, D)


# ============================================================
# 构造压缩后的模型输入
# ============================================================

def build_compressed_inputs(
    model, processor, tokenizer, compressor,
    frames, question, answer, latent_tokens=None,
):
    """构造两个版本的输入: teacher (dense) 和 student (compressed)。

    Returns:
        teacher_inputs: dict — 正常 dense 输入
        student_inputs: dict — 压缩后输入 (inputs_embeds)
        answer_mask: (L,) bool — answer token 位置
    """
    from qwen_vl_utils import process_vision_info
    import tempfile, uuid

    # 临时保存帧
    uid = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    tmp_dir = tempfile.gettempdir()
    tmp_paths = []
    for i, img in enumerate(frames):
        p = os.path.join(tmp_dir, f"_vframe_{uid}_{i}.jpg")
        img.save(p)
        tmp_paths.append(p)

    latent_str = "".join(latent_tokens) if latent_tokens else ""
    question_text = f"{question}\n{latent_str}" if latent_str else question

    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": tmp_paths, "fps": 1.0},
            {"type": "text", "text": question_text},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": answer},
        ]},
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    teacher_inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        return_tensors="pt", padding=True,
    )

    # 清理临时文件
    for p in tmp_paths:
        try:
            os.remove(p)
        except OSError:
            pass

    # --- 给 teacher 构造 answer-only labels ---
    t_input_ids = teacher_inputs["input_ids"][0]
    t_labels = torch.full_like(t_input_ids, -100)
    t_ids_list = t_input_ids.tolist()
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_starts = [i for i, t in enumerate(t_ids_list) if t == im_start_id]
    if im_starts:
        ast = im_starts[-1]
        assistant_prefix = tokenizer.encode("assistant\n", add_special_tokens=False)
        content_start = ast + 1 + len(assistant_prefix)
        t_labels[content_start:] = t_input_ids[content_start:]
    teacher_inputs["labels"] = t_labels.unsqueeze(0)

    # --- 构造 student inputs ---
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    # 获取 dense video embeddings
    m = model.module if hasattr(model, "module") else model
    # 兼容 PEFT 和非 PEFT 模型
    from peft import PeftModel
    if isinstance(m, PeftModel):
        inner_model = m.base_model.model.model  # PEFT wrapped
    else:
        inner_model = m.model  # 原始 Qwen2_5_VLForConditionalGeneration

    pixel_values_videos = teacher_inputs.get("pixel_values_videos")
    video_grid_thw = teacher_inputs.get("video_grid_thw")

    if pixel_values_videos is None:
        return teacher_inputs, teacher_inputs, None

    pv = pixel_values_videos.to(device, dtype=dtype)
    vg = video_grid_thw.to(device)

    with torch.no_grad():
        video_embeds_list = inner_model.get_video_features(pv, vg)
        video_embeds = torch.cat(video_embeds_list, dim=0)  # (total_tokens, D)

    # 每帧 token 数量（从 video_grid_thw 推算）
    # video_grid_thw: (num_videos, 3) → (T, H, W)
    t, h, w = vg[0].tolist()
    tokens_per_frame_count = (h // 2) * (w // 2)  # Qwen2.5-VL 做了 2x2 merge
    n_frames = t
    tokens_per_frame_list = [tokens_per_frame_count] * n_frames

    # 压缩
    compressed = compressor.compress_video(
        video_embeds.to(dtype), tokens_per_frame_list,
    )  # (1, N*tpf, D)

    # 构造新的 input_ids（减少 video_pad 数量）
    input_ids = teacher_inputs["input_ids"][0]
    ids_list = input_ids.tolist()

    # 找 video_pad 区间
    pad_positions = [i for i, t in enumerate(ids_list) if t == VIDEO_PAD_ID]
    if not pad_positions:
        return teacher_inputs, teacher_inputs, None

    pad_start = pad_positions[0]
    pad_end = pad_positions[-1] + 1
    n_compressed = compressed.shape[1]

    # 新 input_ids: 前缀 + n_compressed 个 video_pad + 后缀
    new_ids = (
        ids_list[:pad_start]
        + [VIDEO_PAD_ID] * n_compressed
        + ids_list[pad_end:]
    )
    new_input_ids = torch.tensor([new_ids], device=device)

    # 构造 inputs_embeds
    embed_layer = inner_model.get_input_embeddings()
    text_embeds = embed_layer(new_input_ids)  # (1, new_L, D)

    # 用 compressed tokens 替换 video_pad 位置
    student_embeds = text_embeds.clone()
    student_embeds[0, pad_start:pad_start + n_compressed] = compressed[0]

    # Labels: 只在 answer 部分
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    labels = torch.full_like(new_input_ids, -100)
    im_starts = [i for i, t in enumerate(new_ids) if t == im_start_id]
    if im_starts:
        ast = im_starts[-1]
        assistant_prefix = tokenizer.encode("assistant\n", add_special_tokens=False)
        content_start = ast + 1 + len(assistant_prefix)
        labels[0, content_start:] = new_input_ids[0, content_start:]

    student_inputs = {
        "inputs_embeds": student_embeds.to(dtype),
        "labels": labels,
        "attention_mask": torch.ones_like(new_input_ids),
    }

    # Answer mask（对齐 teacher/student 的 answer 位置用于 KL）
    answer_mask = labels[0] != -100

    return teacher_inputs, student_inputs, answer_mask


# ============================================================
# 分布式工具
# ============================================================

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


# ============================================================
# 训练
# ============================================================

def train(args):
    local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    log("=" * 60)
    log("VoCo-style Distillation 训练")
    log(f"  tokens_per_frame: {args.tokens_per_frame}")
    log(f"  num_frames: {args.num_frames}")
    log(f"  压缩比: ~370 → {args.tokens_per_frame} ({370 // args.tokens_per_frame}:1)")
    log("=" * 60)

    # --- 加载模型（冻结，不加 LoRA）---
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    model.config.use_cache = False
    model.eval()

    for p in model.parameters():
        p.requires_grad = False

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    tokenizer = processor.tokenizer

    # --- Compressor ---
    compressor = SegmentCompressor(
        hidden_dim=3584,
        tokens_per_frame=args.tokens_per_frame,
        num_heads=8,
    ).to(device, dtype=torch.bfloat16)

    log(f"Compressor params: {sum(p.numel() for p in compressor.parameters()):,}")

    if world_size > 1:
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        compressor = DDP(compressor, device_ids=[local_rank])

    # --- 数据 ---
    video_dirs = args.video_dirs.split(",")
    dataset = VideoQADataset(
        args.data_path, video_dirs, max_samples=args.max_samples,
    )

    n_val = min(500, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    log(f"数据: train={n_train}, val={n_val}")

    train_sampler = DistributedSampler(train_set, shuffle=True) if world_size > 1 else None

    def collate(batch):
        return batch[0]  # 返回原始 dict

    train_loader = DataLoader(
        train_set, batch_size=1, sampler=train_sampler,
        shuffle=(train_sampler is None), collate_fn=collate,
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, collate_fn=collate,
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        compressor.parameters(), lr=args.lr, weight_decay=0.01,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    best_val = float("inf")

    # --- 训练循环 ---
    for epoch in range(args.epochs):
        compressor.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)

        total_loss = 0.0
        n_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}",
                    disable=not is_main())

        for item in pbar:
            try:
                frames, timestamps, duration = extract_frames(
                    item["_resolved_video"], args.num_frames,
                )

                teacher_inputs, student_inputs, answer_mask = \
                    build_compressed_inputs(
                        model, processor, tokenizer, compressor,
                        frames, item["question"], item["answer"],
                    )

                if answer_mask is None or not answer_mask.any():
                    continue

                # Teacher forward（frozen，不需要梯度）
                t_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                           for k, v in teacher_inputs.items()
                           if k not in ("_frame_timestamps", "_video_duration")}
                with torch.no_grad():
                    teacher_out = model(**t_batch)

                # Student forward（compressor 需要梯度）
                s_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                           for k, v in student_inputs.items()}
                # 绕过 visual scatter：直接用 inputs_embeds，通过内部 LLM
                m_inner = model.module if hasattr(model, "module") else model
                student_out = m_inner.model(
                    inputs_embeds=s_batch["inputs_embeds"],
                    attention_mask=s_batch.get("attention_mask"),
                )
                student_hidden = student_out[0]
                student_logits = m_inner.lm_head(student_hidden)

                # KL divergence 在 answer 位置
                # 需要对齐 teacher/student 的 answer token 位置
                t_labels = teacher_inputs.get("labels")
                if t_labels is not None:
                    t_labels = t_labels.to(device)
                    t_answer_mask = t_labels[0] != -100
                    t_answer_logits = teacher_out.logits[0, t_answer_mask]
                else:
                    continue

                s_answer_mask = answer_mask.to(device)
                s_answer_logits = student_logits[0, s_answer_mask]

                # 截断到相同长度（answer token 数应相同）
                min_len = min(t_answer_logits.shape[0], s_answer_logits.shape[0])
                if min_len == 0:
                    continue

                t_logits = t_answer_logits[:min_len].float()
                s_logits = s_answer_logits[:min_len].float()

                # KL(teacher || student)
                loss = F.kl_div(
                    F.log_softmax(s_logits / args.temperature, dim=-1),
                    F.softmax(t_logits / args.temperature, dim=-1),
                    reduction="batchmean",
                ) * (args.temperature ** 2)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(compressor.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                n_steps += 1
                if is_main():
                    pbar.set_postfix(loss=f"{loss.item():.4f}")

            except Exception as e:
                if n_steps < 3:
                    log(f"  [错误] {e}")
                continue

        avg_loss = total_loss / max(n_steps, 1)
        log(f"  Epoch {epoch + 1}: train_loss={avg_loss:.4f}")

        # --- Save ---
        if is_main():
            comp = compressor.module if hasattr(compressor, "module") else compressor
            torch.save({
                "state_dict": comp.state_dict(),
                "tokens_per_frame": args.tokens_per_frame,
                "epoch": epoch + 1,
                "train_loss": avg_loss,
            }, os.path.join(args.output_dir, f"compressor_epoch{epoch + 1}.pt"))

            if avg_loss < best_val:
                best_val = avg_loss
                torch.save({
                    "state_dict": comp.state_dict(),
                    "tokens_per_frame": args.tokens_per_frame,
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                }, os.path.join(args.output_dir, "compressor_best.pt"))
                log(f"  ★ New best (loss={avg_loss:.4f})")

    log(f"\n蒸馏完成. best_loss={best_val:.4f}")
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoCo-style 蒸馏训练")

    parser.add_argument("--data_path", required=True)
    parser.add_argument("--video_dirs", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None)

    parser.add_argument("--tokens_per_frame", type=int, default=8,
                        help="每帧压缩后的 token 数量")
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="KL 蒸馏温度")

    args = parser.parse_args()
    train(args)

"""
预提取 teacher 特征：vision embeddings + teacher Q-hidden + question embeddings

保存格式 (per sample):
{
    "segments": [tensor(V_i, D), ...],           # 每段 dense vision embeddings (bf16)
    "q_embeds": tensor(Q_len, D),                # question text embeddings (bf16)
    "teacher_q_hidden": [tensor(Q_len, D), ...], # teacher Q-position hidden per segment (bf16)
    "video_path": str,
    "question": str,
    "n_segments": int,
}

用法:
  CUDA_VISIBLE_DEVICES=0,1,...,7 torchrun --nproc_per_node=8 extract_teacher.py \
    --data_path /path/to/train.jsonl \
    --video_dirs /path/to/videos \
    --output_dir /nvmessd/lifanhong/video/teacher_cache_10k \
    --model_path /path/to/Qwen2.5-VL-7B-Instruct
"""

import os
import json
import glob
import uuid
import tempfile
import argparse
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import decord

from model import get_video_embeds, split_into_segments


class ShardWriter:
    """按 shard_size 将样本聚合保存，避免一条一个小文件。"""

    def __init__(self, output_dir, shard_size=1000, rank=0):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.rank = rank
        self.buffer = []
        self.shard_idx = 0
        os.makedirs(output_dir, exist_ok=True)

    def add(self, sample):
        self.buffer.append(sample)
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return None
        shard_path = os.path.join(
            self.output_dir,
            f"teacher_shard_rank{self.rank}_{self.shard_idx:03d}.pt",
        )
        torch.save(self.buffer, shard_path)
        self.buffer = []
        self.shard_idx += 1
        return shard_path

    def close(self):
        return self.flush()


def extract_frames(video_path, num_frames):
    """从视频均匀采样 num_frames 帧。"""
    vr = decord.VideoReader(video_path)
    total = len(vr)
    fps = vr.get_avg_fps()
    indices = [min(int(i * total / num_frames), total - 1)
               for i in range(num_frames)]
    frames = [Image.fromarray(vr[idx].asnumpy()) for idx in indices]
    duration = total / fps
    return frames, duration


def get_inner(base_model):
    """获取 Qwen 内层模型（绕过 PeftModel 包装）。"""
    m = base_model.module if hasattr(base_model, "module") else base_model
    from peft import PeftModel
    if isinstance(m, PeftModel):
        return m.base_model.model
    return m


def teacher_forward_segment(inner, vision_seg, q_embeds, device):
    """单段 teacher forward → Q 位置 hidden states。

    输入: [vision_seg (V), Q (Q_len)]
    输出: teacher Q-position hidden (Q_len, D)
    """
    V = vision_seg.shape[0]
    Q_len = q_embeds.shape[0]

    t_input = torch.cat([vision_seg, q_embeds], dim=0).unsqueeze(0)
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
    t_hidden = t_out[0].squeeze(0)  # (V+Q_len, D)
    return t_hidden[-Q_len:]         # (Q_len, D)


def main(args):
    # DDP 初始化
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    is_main = (rank == 0)

    if is_main:
        print(f"=== Teacher 特征提取 ===")
        print(f"  world_size: {world_size}")
        print(f"  output_dir: {args.output_dir}")

    # 加载模型（不需要 LoRA，只要 frozen base）
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    base.eval()
    for p in base.parameters():
        p.requires_grad = False

    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer
    inner = get_inner(base)
    embed_layer = inner.get_input_embeddings()
    dtype = torch.bfloat16

    # 加载数据（不用 DataLoader，直接按 rank 分配）
    video_dirs = args.video_dirs.split(",")
    video_index = {}
    for vdir in video_dirs:
        if not os.path.isdir(vdir):
            continue
        for f in glob.glob(os.path.join(vdir, "**", "*.mp4"), recursive=True):
            video_index[os.path.basename(f)] = f

    samples = []
    with open(args.data_path) as f:
        for line in f:
            item = json.loads(line.strip())
            vname = os.path.basename(item["video_path"])
            if vname in video_index:
                item["_resolved_video"] = video_index[vname]
                samples.append(item)

    if args.skip_samples:
        samples = samples[args.skip_samples:]
    if args.max_samples:
        samples = samples[:args.max_samples]

    if is_main:
        print(f"  总样本数: {len(samples)}")

    os.makedirs(args.output_dir, exist_ok=True)

    writer = ShardWriter(args.output_dir, shard_size=args.shard_size, rank=rank)

    # 按 rank 分配样本
    my_indices = list(range(rank, len(samples), world_size))

    # ---- Prefetch: 后台线程预解码视频 + processor 预处理 ----
    from qwen_vl_utils import process_vision_info

    def prefetch_one(idx):
        """在 CPU 线程中完成视频解码和 processor 预处理，返回 GPU forward 所需输入。"""
        item = samples[idx]
        try:
            vr = decord.VideoReader(item["_resolved_video"])
            duration = len(vr) / vr.get_avg_fps()
            n_frames = max(args.frames_per_segment,
                          min(int(duration * args.fps), args.max_frames))
            n_frames = (n_frames // args.frames_per_segment) * args.frames_per_segment
            if n_frames == 0:
                n_frames = args.frames_per_segment

            frames, _ = extract_frames(item["_resolved_video"], n_frames)

            uid = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
            tmp_dir = tempfile.gettempdir()
            tmp_paths = []
            for i, img in enumerate(frames):
                p = os.path.join(tmp_dir, f"_vf_{uid}_{i}.jpg")
                img.save(p)
                tmp_paths.append(p)

            messages = [{"role": "user", "content": [
                {"type": "video", "video": tmp_paths, "fps": args.fps},
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

            return idx, proc_inputs, None
        except Exception as e:
            return idx, None, e

    n_success = 0
    n_skip = 0
    n_error = 0

    prefetch_workers = args.prefetch_workers
    pbar = tqdm(total=len(my_indices), desc=f"[rank {rank}]", disable=not is_main)

    def _gpu_forward_and_save(idx, proc_inputs):
        """GPU forward + 保存，供 prefetch 和串行路径共用。"""
        nonlocal n_success, n_error
        item = samples[idx]
        try:
            pv = proc_inputs.get("pixel_values_videos")
            vg = proc_inputs.get("video_grid_thw")
            if pv is None:
                n_error += 1
                return

            with torch.no_grad():
                video_embeds, tokens_per_frame, _ = get_video_embeds(
                    base, pv, vg, device,
                )
                video_embeds = video_embeds.to(dtype)

                segments = split_into_segments(
                    video_embeds, tokens_per_frame, args.frames_per_segment,
                )

                q_text = (f"<|im_start|>user\n{item['question']}"
                          f"<|im_end|>\n<|im_start|>assistant\n")
                q_ids = tokenizer.encode(
                    q_text, add_special_tokens=False, return_tensors="pt",
                ).to(device)
                q_embeds = embed_layer(q_ids).squeeze(0).to(dtype)

                teacher_q_hidden = []
                for seg in segments:
                    t_hidden = teacher_forward_segment(
                        inner, seg, q_embeds, device,
                    )
                    teacher_q_hidden.append(t_hidden.cpu())

            save_data = {
                "segments": [s.cpu() for s in segments],
                "q_embeds": q_embeds.cpu(),
                "teacher_q_hidden": teacher_q_hidden,
                "video_path": item.get("video_path", ""),
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "n_segments": len(segments),
            }
            writer.add(save_data)
            n_success += 1

            if is_main:
                pbar.set_postfix(ok=n_success, skip=n_skip, err=n_error)

        except Exception as e:
            n_error += 1
            if is_main:
                print(f"  [错误] idx={idx}: {e}")

    if prefetch_workers > 0:
        # ---- Prefetch 模式 ----
        with ThreadPoolExecutor(max_workers=prefetch_workers) as executor:
            futures = []
            submit_ptr = 0
            max_inflight = prefetch_workers * 2
            while submit_ptr < len(my_indices) and submit_ptr < max_inflight:
                futures.append(executor.submit(prefetch_one, my_indices[submit_ptr]))
                submit_ptr += 1

            for future in futures:
                idx, proc_inputs, err = future.result()

                if submit_ptr < len(my_indices):
                    futures.append(executor.submit(prefetch_one, my_indices[submit_ptr]))
                    submit_ptr += 1

                if err is not None or proc_inputs is None:
                    n_error += 1
                    if is_main and err:
                        print(f"  [错误] idx={idx}: {err}")
                    pbar.update(1)
                    continue

                _gpu_forward_and_save(idx, proc_inputs)
                pbar.update(1)

                if n_success % 50 == 0:
                    torch.cuda.empty_cache()
    else:
        # ---- 串行模式（prefetch_workers=0）----
        for idx in my_indices:
            idx_result, proc_inputs, err = prefetch_one(idx)
            if err is not None or proc_inputs is None:
                n_error += 1
                if is_main and err:
                    print(f"  [错误] idx={idx}: {err}")
                pbar.update(1)
                continue

            _gpu_forward_and_save(idx, proc_inputs)
            pbar.update(1)

            if n_success % 50 == 0:
                torch.cuda.empty_cache()

    if is_main:
        print(f"\n=== 提取完成 ===")
        print(f"  成功: {n_success}, 跳过: {n_skip}, 错误: {n_error}")

    writer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="预提取 teacher 特征（vision embeds + teacher Q-hidden）",
    )
    parser.add_argument("--data_path", required=True, help="jsonl 数据路径")
    parser.add_argument("--video_dirs", required=True, help="视频目录，逗号分隔")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--model_path", type=str,
                        default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--skip_samples", type=int, default=None,
                        help="跳过前 N 条样本，用于多机分段提取")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--frames_per_segment", type=int, default=2)
    parser.add_argument("--max_frames", type=int, default=30)
    parser.add_argument("--shard_size", type=int, default=1000)
    parser.add_argument("--prefetch_workers", type=int, default=4,
                        help="后台视频解码线程数，减少 GPU 空等时间")
    args = parser.parse_args()
    main(args)

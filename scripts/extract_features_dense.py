"""
Phase 1 完整版: 提取 dense patch tokens（非 mean pool）

每段视频采样 1 帧，保留 vision encoder 输出的所有 patch tokens，
用于后续 cross-attention 压缩为 latent tokens。

输出格式: {video_name}.pt → {
    "patch_tokens": (M, T, D),  # M 段，每段 T 个 patch tokens
    "timestamps": [(start, end), ...],
    "segment_stride": float,
}
"""

import os
import sys
import json
import glob
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import decord
decord.bridge.set_bridge("torch")


def load_video_frames(video_path: str, segment_stride: float = 4.0):
    """切段并每段取中间帧"""
    vr = decord.VideoReader(video_path)
    fps = vr.get_avg_fps()
    total_frames = len(vr)

    segment_frames = int(segment_stride * fps)
    if segment_frames < 1:
        segment_frames = 1

    frames = []
    timestamps = []
    for start in range(0, total_frames, segment_frames):
        end = min(start + segment_frames, total_frames)
        mid = (start + end) // 2
        frame = vr[mid]  # (H, W, C)
        frames.append(frame)
        timestamps.append((start / fps, end / fps))

    if not frames:
        return None, None

    frames = torch.stack(frames)  # (M, H, W, C)
    return frames, timestamps


def extract_vision_tokens(frames, model, processor, device):
    """
    只通过 vision encoder 提取 patch tokens，不走 LLM decoder。
    返回每帧的 vision token sequence。
    """
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    all_patch_tokens = []

    for i in range(len(frames)):
        frame_np = frames[i].numpy()
        img = Image.fromarray(frame_np)

        # 构造 Qwen2.5-VL 的消息格式
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Describe."},
            ],
        }]

        # 用 processor 处理
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1]  # (1, seq_len, D)

            # 找到 vision token 的位置：
            # Qwen2.5-VL 用 image_grid_thw 来标记视觉 token 范围
            # 简单方法：找 input_ids 中 vision placeholder token 的位置
            input_ids = inputs["input_ids"][0]

            # vision token 的 ID 范围（Qwen2.5-VL 用 151652-151655 做 vision placeholder）
            vision_mask = (input_ids >= 151652) & (input_ids <= 151655)
            vision_positions = vision_mask.nonzero(as_tuple=True)[0]

            if len(vision_positions) > 0:
                vision_hidden = hidden[0, vision_positions, :]  # (T, D)
                all_patch_tokens.append(vision_hidden.cpu().half())
            else:
                # fallback: 取所有 hidden 的均值作为单 token
                all_patch_tokens.append(hidden[0].mean(dim=0, keepdim=True).cpu().half())

    return all_patch_tokens  # list of (T_i, D), T_i 可能不同


def pad_patch_tokens(patch_tokens_list, max_T=None):
    """将不等长的 patch token sequences pad 到统一长度"""
    if max_T is None:
        max_T = max(t.shape[0] for t in patch_tokens_list)

    D = patch_tokens_list[0].shape[-1]
    M = len(patch_tokens_list)

    padded = torch.zeros(M, max_T, D, dtype=patch_tokens_list[0].dtype)
    lengths = []

    for i, tokens in enumerate(patch_tokens_list):
        T = min(tokens.shape[0], max_T)
        padded[i, :T] = tokens[:T]
        lengths.append(T)

    return padded, lengths  # (M, max_T, D), [T_1, T_2, ...]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment_stride", type=float, default=4.0)
    parser.add_argument("--output_dir", type=str,
                        default="/home/v-shuzheng/video/data/features_dense")
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--max_patch_tokens", type=int, default=256,
                        help="每段最多保留多少个 patch token")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 建视频索引
    STGR_ROOT = "/home/v-shuzheng/video/data/open-o3-video/videos/stgr"
    video_index = {}
    for mp4 in glob.glob(os.path.join(STGR_ROOT, "**/*.mp4"), recursive=True):
        video_index[os.path.basename(mp4)] = mp4

    # 从 temporal_evidence 提取需要处理的视频列表
    needed_videos = set()
    with open("/home/v-shuzheng/video/data/parsed/temporal_evidence.jsonl") as f:
        for line in f:
            d = json.loads(line)
            basename = os.path.basename(d["video_path"])
            if basename in video_index:
                needed_videos.add(basename)

    video_list = sorted(needed_videos)
    video_list = [v for v in video_list if not (output_dir / f"{v}.pt").exists()]
    print(f"需要提取: {len(video_list)}")

    if args.max_videos:
        video_list = video_list[:args.max_videos]
        print(f"本批处理: {len(video_list)}")

    if not video_list:
        print("无需处理")
        return

    # 加载模型
    print("加载 Qwen2.5-VL-7B-Instruct...")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    print("模型加载完成")

    success, fail = 0, 0
    for video_name in tqdm(video_list, desc="提取 dense patch tokens"):
        video_path = video_index[video_name]
        out_path = output_dir / f"{video_name}.pt"

        try:
            frames, timestamps = load_video_frames(video_path, args.segment_stride)
            if frames is None:
                fail += 1
                continue

            patch_tokens_list = extract_vision_tokens(frames, model, processor, device)
            padded, lengths = pad_patch_tokens(patch_tokens_list, args.max_patch_tokens)

            torch.save({
                "patch_tokens": padded,       # (M, max_T, D)
                "patch_lengths": lengths,     # [T_1, T_2, ...]
                "timestamps": timestamps,
                "video_path": video_path,
                "segment_stride": args.segment_stride,
            }, out_path)

            success += 1

        except Exception as e:
            print(f"  Error {video_name}: {e}")
            fail += 1

    print(f"\n完成: success={success}, fail={fail}")


if __name__ == "__main__":
    main()

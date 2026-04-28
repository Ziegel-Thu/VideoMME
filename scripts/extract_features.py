"""
Phase 0.1: 用 Qwen2.5-VL 的 vision encoder 提取 segment-level features

对每个视频:
  1. 按 segment_stride 秒切段
  2. 每段均匀采样 1 帧
  3. 过 vision encoder 得到 pooled embedding
  4. 保存为 .pt 文件: {video_basename}.pt → shape (M, D)
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
    """切段并每段取 1 帧中间帧"""
    vr = decord.VideoReader(video_path)
    fps = vr.get_avg_fps()
    total_frames = len(vr)
    duration = total_frames / fps

    segment_frames = int(segment_stride * fps)
    if segment_frames < 1:
        segment_frames = 1

    frames = []
    timestamps = []
    for start in range(0, total_frames, segment_frames):
        end = min(start + segment_frames, total_frames)
        mid = (start + end) // 2
        frame = vr[mid]  # (H, W, C) torch tensor
        frames.append(frame)
        timestamps.append((start / fps, end / fps))

    if not frames:
        return None, None

    # Stack: (M, H, W, C)
    frames = torch.stack(frames)
    return frames, timestamps


def extract_features_simple(frames: torch.Tensor, model, processor, device):
    """用 Qwen2.5-VL 的 vision encoder 提取每帧特征，返回 pooled embeddings"""
    from PIL import Image

    embeddings = []
    for i in range(len(frames)):
        # 转 PIL
        frame_np = frames[i].numpy()
        img = Image.fromarray(frame_np)

        # 用 processor 处理
        inputs = processor(
            images=[img],
            text="<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe.<|im_end|>",
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # 取最后一层 hidden state 的 mean pool 作为 segment embedding
            hidden = outputs.hidden_states[-1]  # (1, seq_len, D)
            # 只取 vision token 部分的均值
            emb = hidden.mean(dim=1).squeeze(0).cpu()  # (D,)
            embeddings.append(emb)

    return torch.stack(embeddings)  # (M, D)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment_stride", type=float, default=4.0)
    parser.add_argument("--output_dir", type=str,
                        default="/home/v-shuzheng/video/data/features")
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--batch_start", type=int, default=0)
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
    print(f"需要提取特征的视频: {len(video_list)}")

    # 跳过已处理的
    video_list = [v for v in video_list if not (output_dir / f"{v}.pt").exists()]
    print(f"跳过已处理后: {len(video_list)}")

    if args.max_videos:
        video_list = video_list[args.batch_start:args.batch_start + args.max_videos]
        print(f"本批处理: {len(video_list)}")

    if not video_list:
        print("无需处理，退出")
        return

    # 加载模型
    print("加载 Qwen2.5-VL-7B-Instruct...")
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    print("模型加载完成")

    # 提取特征
    success, fail = 0, 0
    for video_name in tqdm(video_list, desc="提取特征"):
        video_path = video_index[video_name]
        out_path = output_dir / f"{video_name}.pt"

        try:
            frames, timestamps = load_video_frames(video_path, args.segment_stride)
            if frames is None:
                fail += 1
                continue

            embeddings = extract_features_simple(frames, model, processor, device)

            torch.save({
                "embeddings": embeddings,  # (M, D)
                "timestamps": timestamps,  # [(start, end), ...]
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

"""
Stage 0: 离线预提视频特征

用 Qwen2.5-VL 的 vision encoder 提取 dense vision tokens，保存为 .pt 文件。
训练时直接加载 tensor，跳过视频解码和 vision encoder forward。

用法:
  # 单卡
  python extract_features.py \
    --data_path ../../data/parsed/visual_qa_v3_short_train.jsonl \
    --video_dirs ../../data/llava-video/0_30_s_academic_v0_1 \
    --output_dir ../../data/features/0_30_s \
    --num_frames 4

  # 多卡加速
  torchrun --nproc_per_node=4 extract_features.py \
    --data_path ... --video_dirs ... --output_dir ... --num_frames 12

输出格式:
  output_dir/{video_basename}_{num_frames}f.pt = {
      "video_embeds": tensor (total_tokens, D),  # dense vision tokens
      "tokens_per_frame": int,                    # 每帧 token 数
      "num_frames": int,
      "duration": float,
      "frame_timestamps": list[float],
      "video_grid_thw": tensor,
  }
"""

import os
import sys
import json
import glob
import argparse
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "../003-bottleneck-multigpu"))
from data import extract_frames


def setup_distributed():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return local_rank, world_size
    return 0, 1


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(msg):
    if is_main():
        print(msg)


def extract_video_features(model, processor, video_path, fps, device,
                           max_frames=60):
    """提取一个视频的 dense vision tokens。

    Args:
        fps: 采样帧率（如 1.0 = 每秒 1 帧）
        max_frames: 最大帧数上限

    Returns:
        dict with video_embeds, tokens_per_frame, etc.
        or None if failed.
    """
    import tempfile, uuid
    import decord

    # 根据视频时长和 fps 计算帧数
    try:
        vr = decord.VideoReader(video_path)
        duration = len(vr) / vr.get_avg_fps()
    except Exception:
        return None

    num_frames = max(1, min(int(duration * fps), max_frames))
    frames, timestamps, duration = extract_frames(video_path, num_frames)

    # 保存临时帧
    uid = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    tmp_dir = tempfile.gettempdir()
    tmp_paths = []
    for i, img in enumerate(frames):
        p = os.path.join(tmp_dir, f"_vf_{uid}_{i}.jpg")
        img.save(p)
        tmp_paths.append(p)

    # 用 processor 构造输入（只需要视频部分）
    messages = [{"role": "user", "content": [
        {"type": "video", "video": tmp_paths, "fps": 1.0},
        {"type": "text", "text": "describe"},
    ]}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        return_tensors="pt",
    )

    # 清理临时文件
    for p in tmp_paths:
        try:
            os.remove(p)
        except OSError:
            pass

    pixel_values_videos = inputs.get("pixel_values_videos")
    video_grid_thw = inputs.get("video_grid_thw")

    if pixel_values_videos is None:
        return None

    pv = pixel_values_videos.to(device, dtype=torch.bfloat16)
    vg = video_grid_thw.to(device)

    with torch.no_grad():
        video_embeds_list = model.model.get_video_features(pv, vg)
        video_embeds = torch.cat(video_embeds_list, dim=0)  # (total_tokens, D)

    # 每帧 token 数
    t, h, w = vg[0].tolist()
    tokens_per_frame = video_embeds.shape[0] // t

    return {
        "video_embeds": video_embeds.cpu(),
        "tokens_per_frame": tokens_per_frame,
        "num_frames": num_frames,
        "duration": duration,
        "frame_timestamps": timestamps,
        "video_grid_thw": vg.cpu(),
    }


def main(args):
    local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    log("=" * 60)
    log("Stage 0: 视频特征提取")
    log(f"  fps: {args.fps}")
    log(f"  max_frames: {args.max_frames}")
    log(f"  GPU 数量: {world_size}")
    log("=" * 60)

    # 加载模型（只需要 vision encoder）
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    # 建立视频索引
    video_dirs = args.video_dirs.split(",")
    video_index = {}
    for vdir in video_dirs:
        if not os.path.isdir(vdir):
            continue
        for f in glob.glob(os.path.join(vdir, "**", "*.mp4"), recursive=True):
            video_index[os.path.basename(f)] = f

    # 加载数据，收集所有需要处理的视频
    videos_to_process = set()
    with open(args.data_path) as f:
        for line in f:
            item = json.loads(line.strip())
            vname = os.path.basename(item["video_path"])
            if vname in video_index:
                videos_to_process.add(vname)

    videos_list = sorted(videos_to_process)
    log(f"需要提取: {len(videos_list)} 个视频")

    # 输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 分配给各 rank
    my_videos = videos_list[local_rank::world_size]
    log(f"  Rank {local_rank}: 处理 {len(my_videos)} 个视频")

    extracted = 0
    skipped = 0
    errors = 0

    for vname in tqdm(my_videos, desc=f"Rank {local_rank}",
                      disable=not is_main()):
        out_name = f"{Path(vname).stem}_1fps.pt"
        out_path = os.path.join(args.output_dir, out_name)

        # 跳过已提取的
        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            result = extract_video_features(
                model, processor, video_index[vname],
                args.fps, device, max_frames=args.max_frames,
            )
            if result is not None:
                torch.save(result, out_path)
                extracted += 1
            else:
                errors += 1
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [错误] {vname}: {e}")

    log(f"\n完成: 提取 {extracted}, 跳过 {skipped}, 失败 {errors}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="视频特征提取")
    parser.add_argument("--data_path", required=True,
                        help="jsonl 数据路径（决定提取哪些视频）")
    parser.add_argument("--video_dirs", required=True,
                        help="视频目录，逗号分隔")
    parser.add_argument("--output_dir", required=True,
                        help="特征输出目录")
    parser.add_argument("--fps", type=float, default=1.0,
                        help="采样帧率（每秒几帧）")
    parser.add_argument("--max_frames", type=int, default=60,
                        help="单视频最大帧数上限")
    args = parser.parse_args()
    main(args)

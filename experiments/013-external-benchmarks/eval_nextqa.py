"""
NExT-QA 评测脚本

支持 zero-shot 和 compressor 两种模式。
NExT-QA 是 5 选 MCQ，问题类型分 Causal (C), Temporal (T), Descriptive (D)。

用法:
  python eval_nextqa.py \
    --video_dir /mnt/default/bottleneck/benchmarks/nextqa/videos \
    --csv_path /mnt/default/bottleneck/benchmarks/nextqa/test.csv \
    --model_path /mnt/default/bottleneck/models/Qwen2.5-VL-7B-Instruct
"""

import os
import json
import csv
import argparse
import uuid
import tempfile

import torch
from tqdm import tqdm
from PIL import Image
import decord


def extract_frames(video_path, num_frames):
    try:
        vr = decord.VideoReader(video_path)
        total = len(vr)
        indices = [min(int(i * total / num_frames), total - 1) for i in range(num_frames)]
        frames = [Image.fromarray(vr[idx].asnumpy()) for idx in indices]
        return frames
    except Exception:
        return None


def eval_logit(model, processor, tokenizer, frames, question, candidates, device):
    from qwen_vl_utils import process_vision_info

    options = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(candidates)])
    prompt = (
        f"Select the best answer to the following multiple-choice question "
        f"based on the video. Respond with only the letter (A, B, C, D, E) "
        f"of the correct option.\n{question}\n{options}\nThe best answer is:"
    )

    uid = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    tmp_dir = tempfile.gettempdir()
    tmp_paths = []
    for i, img in enumerate(frames):
        p = os.path.join(tmp_dir, f"_nq_{uid}_{i}.jpg")
        img.save(p)
        tmp_paths.append(p)

    messages = [{"role": "user", "content": [
        {"type": "video", "video": tmp_paths, "fps": 1.0},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt").to(device)

    for p in tmp_paths:
        try: os.remove(p)
        except OSError: pass

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]

    option_ids = {chr(65+i): tokenizer.encode(chr(65+i), add_special_tokens=False)[0] for i in range(5)}
    option_logits = {k: logits[v].item() for k, v in option_ids.items()}
    return max(option_logits, key=option_logits.get)


def main(args):
    device = torch.device("cuda")

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="eager",
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer

    # 加载数据
    with open(args.csv_path) as f:
        reader = csv.DictReader(f)
        data = list(reader)

    if args.num_shards > 1:
        data = data[args.shard_id::args.num_shards]

    # 建视频索引
    import glob
    video_index = {}
    for f in glob.glob(os.path.join(args.video_dir, "**", "*.mp4"), recursive=True):
        vid = os.path.splitext(os.path.basename(f))[0]
        video_index[vid] = f

    stats = {"C": {"correct": 0, "total": 0}, "T": {"correct": 0, "total": 0}, "D": {"correct": 0, "total": 0}}
    total_correct = 0
    total_count = 0

    for item in tqdm(data, desc="NExT-QA", disable=(args.shard_id != 0)):
        vid = item["video"]
        if vid not in video_index:
            continue

        frames = extract_frames(video_index[vid], args.num_frames)
        if frames is None:
            continue

        try:
            candidates = [item[f"a{i}"] for i in range(5)]
            gt_idx = int(item["answer"])
            gt_letter = chr(65 + gt_idx)
            qtype = item["type"][0]  # C/T/D

            pred = eval_logit(model, processor, tokenizer, frames, item["question"], candidates, device)

            if pred == gt_letter:
                total_correct += 1
                if qtype in stats:
                    stats[qtype]["correct"] += 1
            total_count += 1
            if qtype in stats:
                stats[qtype]["total"] += 1

        except Exception:
            continue

    total_acc = total_correct / max(total_count, 1) * 100
    print(f"\nNExT-QA 总准确率: {total_acc:.2f}% ({total_correct}/{total_count})")
    for qtype in ["C", "T", "D"]:
        s = stats[qtype]
        acc = s["correct"] / max(s["total"], 1) * 100
        print(f"  {qtype}: {acc:.2f}% ({s['correct']}/{s['total']})")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"total_acc": total_acc, "total_correct": total_correct,
                        "total_count": total_count, "stats": stats,
                        "shard_id": args.shard_id, "num_shards": args.num_shards}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NExT-QA 评测")
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    main(args)

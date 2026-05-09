"""
Zero-shot baseline: 原始 Qwen2.5-VL-7B 不加 LoRA 直接跑 MCQ

用法:
  python eval_baseline.py \
    --data_path <data>/visual_qa_v3_short_test.jsonl \
    --video_dirs <videos> \
    --max_samples 500
"""

import os
import sys
import json
import argparse
import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from data import extract_frames


def eval_baseline(args):
    device = torch.device("cuda")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    tokenizer = processor.tokenizer

    # 加载数据
    import glob
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

    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"数据: {len(samples)} 条")

    from qwen_vl_utils import process_vision_info
    import tempfile, uuid

    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    correct = 0
    total = 0
    errors = 0

    for item in tqdm(samples, desc="Baseline 评测"):
        try:
            frames, _, _ = extract_frames(item["_resolved_video"], args.num_frames)

            uid = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
            tmp_paths = []
            for i, img in enumerate(frames):
                p = os.path.join(tempfile.gettempdir(), f"_vframe_{uid}_{i}.jpg")
                img.save(p)
                tmp_paths.append(p)

            messages = [
                {"role": "user", "content": [
                    {"type": "video", "video": tmp_paths, "fps": 1.0},
                    {"type": "text", "text": item["question"]},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": item["answer"]},
                ]},
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                return_tensors="pt", padding=True,
            )

            for p in tmp_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass

            # 构造 labels
            input_ids = inputs["input_ids"][0]
            labels = torch.full_like(input_ids, -100)
            ids_list = input_ids.tolist()
            im_starts = [i for i, t in enumerate(ids_list) if t == im_start_id]
            if im_starts:
                ast = im_starts[-1]
                assistant_prefix = tokenizer.encode("assistant\n", add_special_tokens=False)
                content_start = ast + 1 + len(assistant_prefix)
                labels[content_start:] = input_ids[content_start:]

            inputs["labels"] = labels.unsqueeze(0)
            inputs = {k: v.to(device) for k, v in inputs.items()
                      if isinstance(v, torch.Tensor)}

            with torch.no_grad():
                out = model(**inputs)

            # 判断准确率
            ans_pos = None
            for i, t in enumerate(ids_list):
                if labels[i] != -100:
                    ans_pos = i
                    break

            if ans_pos is None or ans_pos >= len(ids_list):
                errors += 1
                continue

            pred_token = out.logits[0, ans_pos - 1, :].argmax().item()
            gt_token = ids_list[ans_pos]

            if pred_token == gt_token:
                correct += 1
            total += 1

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  [错误] {e}")
            continue

    acc = correct / max(total, 1)
    print(f"\n{'=' * 50}")
    print(f"Zero-shot Baseline (原始 Qwen2.5-VL-7B)")
    print(f"{'=' * 50}")
    print(f"  准确率: {acc:.2%} ({correct}/{total})")
    if errors > 0:
        print(f"  跳过 {errors} 条出错样本")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--video_dirs", required=True)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=500)
    args = parser.parse_args()
    eval_baseline(args)

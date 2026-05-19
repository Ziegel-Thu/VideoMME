"""
MCQ 评测：三种方法对比

方法 1 (logit): 一次 forward，比较 A/B/C/D 四个 token 在 answer 位置的 logit
方法 2 (gen):   自由生成回答，正则提取答案字母
方法 3 (nll):   每个选项单独算完整 NLL（慢，但最严谨）

默认同时跑方法 1+2（快），--include_nll 额外跑方法 3。

用法:
  python eval_mcq.py \
      --data_path /path/to/test.jsonl \
      --video_dirs /path/to/videos \
      --max_samples 500 --num_frames 8

  # 也跑方法 3（慢 4 倍）
  python eval_mcq.py ... --include_nll
"""

import os
import sys
import re
import json
import glob
import argparse
import uuid
import tempfile

import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import decord
from PIL import Image


def extract_frames(video_path, num_frames):
    """从视频均匀采样 num_frames 帧。"""
    try:
        vr = decord.VideoReader(video_path)
        total = len(vr)
        fps = vr.get_avg_fps()
        indices = [min(int(i * total / num_frames), total - 1)
                   for i in range(num_frames)]
        frames = [Image.fromarray(vr[idx].asnumpy()) for idx in indices]
        timestamps = [idx / fps for idx in indices]
        duration = total / fps
        return frames, timestamps, duration
    except Exception as e:
        print(f"  ⚠️ 视频读取失败 {video_path}: {e}，用黑帧替代")
        frames = [Image.new("RGB", (224, 224), (0, 0, 0))] * num_frames
        return frames, [0.0] * num_frames, 1.0


def prepare_video_inputs(processor, frames, question, answer_text=None, fps=1.0):
    """构造模型输入。answer_text=None 时用 generation 模式。"""
    from qwen_vl_utils import process_vision_info

    uid = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    tmp_dir = tempfile.gettempdir()
    tmp_paths = []
    for i, img in enumerate(frames):
        p = os.path.join(tmp_dir, f"_eval_{uid}_{i}.jpg")
        img.save(p)
        tmp_paths.append(p)

    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": tmp_paths, "fps": fps},
            {"type": "text", "text": question},
        ]},
    ]
    if answer_text is not None:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": answer_text}]}
        )

    text = processor.apply_chat_template(
        messages, tokenize=False,
        add_generation_prompt=(answer_text is None),
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

    return inputs


def find_answer_position(input_ids, tokenizer):
    """找到 assistant 回复内容的起始位置。"""
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    ids_list = input_ids.tolist()
    im_starts = [i for i, t in enumerate(ids_list) if t == im_start_id]
    if not im_starts:
        return None
    ast = im_starts[-1]
    assistant_prefix = tokenizer.encode("assistant\n", add_special_tokens=False)
    return ast + 1 + len(assistant_prefix)


# ============================================================
# 方法 1: Logit 比较（一次 forward）
# ============================================================

def eval_logit(model, processor, tokenizer, frames, question, gt_answer, device, fps=1.0):
    """一次 forward，比较 A/B/C/D 在 answer 位置的 logit。"""
    inputs = prepare_video_inputs(processor, frames, question, answer_text=gt_answer, fps=fps)
    input_ids = inputs["input_ids"][0]
    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    ans_pos = find_answer_position(input_ids, tokenizer)
    if ans_pos is None or ans_pos >= len(input_ids):
        return None

    with torch.no_grad():
        out = model(**inputs)

    # 取 answer 起始位置前一个 token 的 logits（预测 answer 第一个 token）
    logits = out.logits[0, ans_pos - 1, :].float()

    # 比较 A/B/C/D 四个 token 的概率
    option_tokens = {}
    for letter in "ABCD":
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if ids:
            option_tokens[letter] = ids[0]

    if not option_tokens:
        return None

    option_logits = {letter: logits[tid].item() for letter, tid in option_tokens.items()}
    pred = max(option_logits, key=option_logits.get)

    return {"pred": pred, "logits": option_logits}


# ============================================================
# 方法 2: 自由生成 + 正则解析
# ============================================================

def extract_answer_letter(text):
    """从生成文本中提取答案字母。"""
    text = text.strip()
    # 直接就是一个字母
    if text in "ABCD":
        return text
    # "A" / "A." / "A)" / "(A)" 开头
    m = re.match(r'^\(?([A-D])\)?[.):]*', text)
    if m:
        return m.group(1)
    # "The answer is A" / "answer: B" 等
    m = re.search(r'(?:answer|option)\s*(?:is|:)?\s*\(?([A-D])\)?', text, re.IGNORECASE)
    if m:
        return m.group(1)
    # 最后一个出现的 A/B/C/D
    m = re.findall(r'\b([A-D])\b', text)
    if m:
        return m[-1]
    return None


def eval_generation(model, processor, tokenizer, frames, question, device,
                    fps=1.0, max_new_tokens=50):
    """自由生成，正则提取答案。"""
    inputs = prepare_video_inputs(processor, frames, question, answer_text=None, fps=fps)
    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
        )

    # 只取生成的部分
    input_len = inputs["input_ids"].shape[1]
    gen_text = tokenizer.decode(gen_ids[0, input_len:], skip_special_tokens=True)

    pred = extract_answer_letter(gen_text)
    return {"pred": pred, "generated_text": gen_text}


# ============================================================
# 方法 3: Per-option NLL（每个选项单独 forward）
# ============================================================

def eval_nll(model, processor, tokenizer, frames, question, device, fps=1.0):
    """对 A/B/C/D 每个选项算 NLL，选最低的。"""
    option_losses = {}
    for letter in "ABCD":
        inputs = prepare_video_inputs(processor, frames, question,
                                       answer_text=letter, fps=fps)
        input_ids = inputs["input_ids"][0]
        labels = torch.full_like(input_ids, -100)

        ans_pos = find_answer_position(input_ids, tokenizer)
        if ans_pos is not None:
            labels[ans_pos:] = input_ids[ans_pos:]

        inputs["labels"] = labels.unsqueeze(0)
        inputs = {k: v.to(device) for k, v in inputs.items()
                  if isinstance(v, torch.Tensor)}

        with torch.no_grad():
            out = model(**inputs)
        option_losses[letter] = out.loss.item()

    pred = min(option_losses, key=option_losses.get)
    return {"pred": pred, "losses": option_losses}


# ============================================================
# 主评测循环
# ============================================================

def main(args):
    device = torch.device("cuda")

    print("加载模型...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer

    # 建立视频索引
    video_dirs = args.video_dirs.split(",")
    video_index = {}
    for vdir in video_dirs:
        if not os.path.isdir(vdir):
            continue
        for f in glob.glob(os.path.join(vdir, "**", "*.mp4"), recursive=True):
            video_index[os.path.basename(f)] = f

    # 加载数据
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
    if args.num_shards > 1:
        samples = samples[args.shard_id::args.num_shards]
    print(f"评测数据: {len(samples)} 条 (shard {args.shard_id}/{args.num_shards})")
    print(f"方法: logit + gen{' + nll' if args.include_nll else ''}")
    print(f"帧数: {args.num_frames}")

    # 统计
    stats = {
        "logit": {"correct": 0, "total": 0, "errors": 0},
        "gen": {"correct": 0, "total": 0, "errors": 0},
    }
    if args.include_nll:
        stats["nll"] = {"correct": 0, "total": 0, "errors": 0}

    results = []

    for item in tqdm(samples, desc="评测"):
        try:
            frames, _, _ = extract_frames(item["_resolved_video"], args.num_frames)
            question = item["question"]
            gt = item["answer"].strip()
            result = {"gt": gt}

            # 方法 1: logit
            r1 = eval_logit(model, processor, tokenizer, frames, question, gt, device,
                            fps=args.fps)
            if r1 and r1["pred"]:
                result["logit_pred"] = r1["pred"]
                result["logit_scores"] = r1["logits"]
                stats["logit"]["total"] += 1
                if r1["pred"] == gt:
                    stats["logit"]["correct"] += 1
            else:
                stats["logit"]["errors"] += 1

            # 方法 2: generation
            r2 = eval_generation(model, processor, tokenizer, frames, question, device,
                                 fps=args.fps)
            if r2 and r2["pred"]:
                result["gen_pred"] = r2["pred"]
                result["gen_text"] = r2["generated_text"]
                stats["gen"]["total"] += 1
                if r2["pred"] == gt:
                    stats["gen"]["correct"] += 1
            else:
                result["gen_text"] = r2["generated_text"] if r2 else ""
                stats["gen"]["errors"] += 1

            # 方法 3: NLL（可选）
            if args.include_nll:
                r3 = eval_nll(model, processor, tokenizer, frames, question, device,
                              fps=args.fps)
                if r3 and r3["pred"]:
                    result["nll_pred"] = r3["pred"]
                    result["nll_losses"] = r3["losses"]
                    stats["nll"]["total"] += 1
                    if r3["pred"] == gt:
                        stats["nll"]["correct"] += 1
                else:
                    stats["nll"]["errors"] += 1

            results.append(result)

        except Exception as e:
            for s in stats.values():
                s["errors"] += 1
            if sum(s["errors"] for s in stats.values()) // len(stats) <= 5:
                print(f"  [错误] {e}")

    # 打印结果
    print(f"\n{'=' * 60}")
    print(f"Zero-shot Qwen2.5-VL-7B MCQ 评测 ({len(samples)} 条, {args.num_frames} 帧)")
    print(f"{'=' * 60}")
    for method, s in stats.items():
        acc = s["correct"] / max(s["total"], 1)
        print(f"  {method:6s}: {acc:.2%} ({s['correct']}/{s['total']})"
              f"  errors={s['errors']}")

    # 方法一致性分析
    if len(results) > 0:
        agree = sum(1 for r in results
                    if r.get("logit_pred") and r.get("gen_pred")
                    and r["logit_pred"] == r["gen_pred"])
        both = sum(1 for r in results
                   if r.get("logit_pred") and r.get("gen_pred"))
        if both > 0:
            print(f"\n  logit vs gen 一致率: {agree/both:.2%} ({agree}/{both})")

    # 保存结果
    if args.output:
        output_data = {
            "config": {
                "num_frames": args.num_frames,
                "fps": args.fps,
                "max_samples": args.max_samples,
                "include_nll": args.include_nll,
            },
            "stats": stats,
            "results": results[:100],  # 只保存前 100 条详细结果
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n  详细结果 → {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCQ 评测（logit + generation + NLL）")
    parser.add_argument("--data_path", required=True, help="测试数据 jsonl")
    parser.add_argument("--video_dirs", required=True, help="视频目录，逗号分隔")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--include_nll", action="store_true",
                        help="额外跑 per-option NLL 方法（慢 4 倍）")
    parser.add_argument("--output", type=str, default=None,
                        help="保存结果 JSON")
    parser.add_argument("--model_path", type=str,
                        default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="模型路径（HF name 或本地路径）")
    parser.add_argument("--shard_id", type=int, default=0, help="当前分片 ID（从 0 开始）")
    parser.add_argument("--num_shards", type=int, default=1, help="总分片数；1 表示不分片")
    args = parser.parse_args()
    main(args)

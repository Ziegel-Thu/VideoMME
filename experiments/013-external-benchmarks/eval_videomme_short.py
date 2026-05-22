"""
Video-MME Short 评测脚本

过滤 Short split（duration=="short"），支持 zero-shot 评测。

用法:
  python eval_videomme_short.py \
    --video_dir /mnt/default/bottleneck/benchmarks/videomme/videos \
    --parquet_path /mnt/default/bottleneck/benchmarks/videomme/test.parquet \
    --model_path /mnt/default/bottleneck/models/Qwen2.5-VL-7B-Instruct
"""

import os
import json
import argparse
import uuid
import tempfile
import re

import torch
from tqdm import tqdm
from PIL import Image
import decord
import pandas as pd


def extract_frames(video_path, num_frames):
    try:
        vr = decord.VideoReader(video_path)
        total = len(vr)
        indices = [min(int(i * total / num_frames), total - 1) for i in range(num_frames)]
        frames = [Image.fromarray(vr[idx].asnumpy()) for idx in indices]
        return frames
    except Exception:
        return None


def parse_options(options_list):
    """解析 Video-MME 的 options 列表，返回 [(letter, text), ...]"""
    parsed = []
    for opt in options_list:
        m = re.match(r"([A-D])\.\s*(.*)", opt)
        if m:
            parsed.append((m.group(1), m.group(2)))
    return parsed


def eval_logit(model, processor, tokenizer, frames, question, options, device):
    from qwen_vl_utils import process_vision_info

    options_text = "\n".join(options)
    prompt = (
        f"Select the best answer to the following multiple-choice question "
        f"based on the video. Respond with only the letter (A, B, C, or D) "
        f"of the correct option.\n{question}\n{options_text}\nThe best answer is:"
    )

    uid = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    tmp_dir = tempfile.gettempdir()
    tmp_paths = []
    for i, img in enumerate(frames):
        p = os.path.join(tmp_dir, f"_vm_{uid}_{i}.jpg")
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

    option_ids = {c: tokenizer.encode(c, add_special_tokens=False)[0] for c in "ABCD"}
    option_logits = {k: logits[v].item() for k, v in option_ids.items()}
    return max(option_logits, key=option_logits.get)


def main(args):
    device = torch.device("cuda")

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa" if args.checkpoint else "eager",
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer

    # 加载 compressor
    compressor = None
    inter_segment = None
    if args.checkpoint:
        from compressor import VoCoCompressor, InterSegmentAttention
        from model import get_video_embeds, split_into_segments
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        K_seg = ckpt.get("K_seg", 8)
        n_layers = ckpt.get("n_layers", 1)
        inter_layers = ckpt.get("inter_layers", 0) or 0
        print(f"Compressor: K={K_seg}, n_layers={n_layers}, inter_layers={inter_layers}")

        # 自动检测 compressor 类型
        ckpt_keys = set(ckpt['compressor'].keys())
        if any('gate' in k for k in ckpt_keys):
            from compressor import GatedVoCoCompressor
            compressor = GatedVoCoCompressor(K=K_seg, dim=3584, n_layers=n_layers)
        elif 'score_proj.weight' in ckpt_keys:
            from compressor import PoolingCompressor
            compressor = PoolingCompressor(K=K_seg, dim=3584, n_layers=n_layers)
        else:
            compressor = VoCoCompressor(K=K_seg, dim=3584, n_layers=n_layers)
        compressor.load_state_dict(ckpt["compressor"])
        compressor = compressor.to(device, dtype=torch.bfloat16)
        compressor.eval()

        if inter_layers > 0:
            inter_segment = InterSegmentAttention(dim=3584, n_layers=inter_layers)
            inter_segment.load_state_dict(ckpt["inter_segment"])
            inter_segment = inter_segment.to(device, dtype=torch.bfloat16)
            inter_segment.eval()

        for p in model.parameters():
            p.requires_grad = False

        inner = model
        try:
            from peft import PeftModel
            if isinstance(inner, PeftModel):
                inner = inner.base_model.model
        except ImportError:
            pass
        embed_layer = inner.get_input_embeddings()

    df = pd.read_parquet(args.parquet_path)
    short = df[df["duration"] == "short"].reset_index(drop=True)
    print(f"Video-MME Short: {len(short)} 条, {short['videoID'].nunique()} 视频")

    if args.num_shards > 1:
        short = short.iloc[args.shard_id::args.num_shards].reset_index(drop=True)

    # 建视频索引
    import glob
    video_index = {}
    for f in glob.glob(os.path.join(args.video_dir, "**", "*.*"), recursive=True):
        vid = os.path.splitext(os.path.basename(f))[0]
        video_index[vid] = f

    domain_stats = {}
    total_correct = 0
    total_count = 0

    for _, row in tqdm(short.iterrows(), total=len(short), desc="Video-MME Short", disable=(args.shard_id != 0)):
        vid = row["videoID"]
        if vid not in video_index:
            continue

        frames = extract_frames(video_index[vid], args.num_frames)
        if frames is None:
            continue

        try:
            if compressor is not None:
                # Compressor 模式
                from qwen_vl_utils import process_vision_info
                from model import get_video_embeds, split_into_segments
                tmp_paths_c = []
                for i, img in enumerate(frames):
                    p = os.path.join(tempfile.gettempdir(), f"_vmc_{os.getpid()}_{i}.jpg")
                    img.save(p)
                    tmp_paths_c.append(p)
                msgs = [{"role": "user", "content": [
                    {"type": "video", "video": tmp_paths_c, "fps": 1.0},
                    {"type": "text", "text": "x"},
                ]}]
                txt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                img_in, vid_in = process_vision_info(msgs)
                proc = processor(text=[txt], images=img_in, videos=vid_in, return_tensors="pt")
                for p in tmp_paths_c:
                    try: os.remove(p)
                    except OSError: pass
                pv = proc.get("pixel_values_videos")
                vg = proc.get("video_grid_thw")
                if pv is None:
                    continue
                with torch.no_grad():
                    ve, tpf, _ = get_video_embeds(model, pv, vg, device)
                    ve = ve.to(torch.bfloat16)
                    segs = split_into_segments(ve, tpf, 2)
                    comp_segs = [compressor(s) for s in segs]
                    if inter_segment is not None:
                        sl = [t.shape[0] for t in comp_segs]
                        comp_segs = inter_segment(torch.cat(comp_segs, dim=0), sl)
                    all_comp = torch.cat(comp_segs, dim=0)
                    comp_len = all_comp.shape[0]
                    comp_out = inner.model(
                        inputs_embeds=all_comp.unsqueeze(0),
                        attention_mask=torch.ones(1, comp_len, dtype=torch.long, device=device),
                        position_ids=torch.arange(comp_len, device=device).view(1,1,-1).expand(3,1,-1),
                        use_cache=True,
                    )
                    past_kv = comp_out.past_key_values
                    options_text = "\n".join(row["options"])
                    q_text = (f"<|im_start|>user\nSelect the best answer to the following multiple-choice question "
                              f"based on the video. Respond with only the letter (A, B, C, or D) of the correct option.\n"
                              f"{row['question']}\n{options_text}\nThe best answer is:"
                              f"<|im_end|>\n<|im_start|>assistant\n")
                    q_ids = tokenizer.encode(q_text, add_special_tokens=False, return_tensors="pt").to(device)
                    q_emb = embed_layer(q_ids).to(torch.bfloat16)
                    q_len = q_emb.shape[1]
                    q_out = inner.model(
                        inputs_embeds=q_emb,
                        attention_mask=torch.ones(1, comp_len+q_len, dtype=torch.long, device=device),
                        position_ids=torch.arange(comp_len, comp_len+q_len, device=device).view(1,1,-1).expand(3,1,-1),
                        past_key_values=past_kv, use_cache=False,
                    )
                    logits = inner.lm_head(q_out[0][0, -1, :])
                    option_ids = {c: tokenizer.encode(c, add_special_tokens=False)[0] for c in "ABCD"}
                    option_logits = {k: logits[v].item() for k, v in option_ids.items()}
                    pred = max(option_logits, key=option_logits.get)
            else:
                pred = eval_logit(model, processor, tokenizer, frames, row["question"], row["options"], device)
            gt = row["answer"].strip().upper()
            domain = row["domain"]

            if domain not in domain_stats:
                domain_stats[domain] = {"correct": 0, "total": 0}

            if pred == gt:
                total_correct += 1
                domain_stats[domain]["correct"] += 1
            total_count += 1
            domain_stats[domain]["total"] += 1

        except Exception:
            continue

    total_acc = total_correct / max(total_count, 1) * 100
    print(f"\nVideo-MME Short 总准确率: {total_acc:.2f}% ({total_correct}/{total_count})")
    for domain in sorted(domain_stats):
        s = domain_stats[domain]
        acc = s["correct"] / max(s["total"], 1) * 100
        print(f"  {domain}: {acc:.2f}% ({s['correct']}/{s['total']})")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"total_acc": total_acc, "total_correct": total_correct,
                        "total_count": total_count, "domain_stats": domain_stats,
                        "shard_id": args.shard_id, "num_shards": args.num_shards}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video-MME Short 评测")
    parser.add_argument("--video_dir", required=True)
    parser.add_argument("--parquet_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--checkpoint", default=None, help="compressor checkpoint，不指定则 zero-shot")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    main(args)

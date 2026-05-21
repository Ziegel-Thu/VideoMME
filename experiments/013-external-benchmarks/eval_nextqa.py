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

            if compressor is not None:
                from qwen_vl_utils import process_vision_info
                from model import get_video_embeds, split_into_segments
                tmp_paths_c = []
                for i, img in enumerate(frames):
                    p = os.path.join(tempfile.gettempdir(), f"_nqc_{os.getpid()}_{i}.jpg")
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
                    options = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(candidates)])
                    q_text = (f"<|im_start|>user\nSelect the best answer to the following multiple-choice question "
                              f"based on the video. Respond with only the letter (A, B, C, D, or E) of the correct option.\n"
                              f"{item['question']}\n{options}\nThe best answer is:"
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
                    option_ids = {chr(65+i): tokenizer.encode(chr(65+i), add_special_tokens=False)[0] for i in range(5)}
                    option_logits = {k: logits[v].item() for k, v in option_ids.items()}
                    pred = max(option_logits, key=option_logits.get)
            else:
                pred = eval_logit(model, processor, tokenizer, frames, item["question"], candidates, device)

            if pred == gt_letter:
                total_correct += 1
                if qtype in stats:
                    stats[qtype]["correct"] += 1
            total_count += 1
            if qtype in stats:
                stats[qtype]["total"] += 1

        except Exception:
            import traceback
            traceback.print_exc()
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
    parser.add_argument("--checkpoint", default=None, help="compressor checkpoint，不指定则 zero-shot")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    main(args)

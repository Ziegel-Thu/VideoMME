"""
004 VoCo 分段压缩 MCQ 评测

用法:
  python eval_voco.py \
    --checkpoint outputs/best_model.pt \
    --data_path /path/to/test.jsonl \
    --video_dirs /path/to/videos \
    --K_seg 4 --frames_per_segment 2 --max_samples 500
"""

import os
import json
import argparse

import torch
from tqdm import tqdm

from model import setup_voco_model, get_video_embeds, split_into_segments, \
    build_voco_sequence_embeds, build_voco_attention_mask
from data import VoCoVideoDataset, extract_frames, encode_text


def eval_voco(args):
    device = torch.device("cuda")

    model, processor, tokenizer = setup_voco_model(
        K_seg=args.K_seg, lora_r=16, device=device,
    )

    # 加载 checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.voco_embeds.data = ckpt["voco_embeds"].to(device, dtype=torch.bfloat16)

    from peft import PeftModel
    m = model.base
    if isinstance(m, PeftModel):
        missing = [k for k in ckpt["lora_state_dict"] if "lora_" in k]
        m.load_state_dict(ckpt["lora_state_dict"], strict=False)
        loaded = len([k for k in ckpt["lora_state_dict"] if "lora_" in k])
        print(f"  ✓ {loaded} LoRA keys 已加载")
    print(f"  Checkpoint: epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss')}")

    model.eval()

    # 数据
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

    import decord
    import uuid
    import tempfile
    from qwen_vl_utils import process_vision_info

    inner = model.base
    from peft import PeftModel
    if isinstance(inner, PeftModel):
        embed_layer = inner.base_model.model.get_input_embeddings()
    else:
        embed_layer = inner.get_input_embeddings()

    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")

    correct = 0
    total = 0
    total_loss = 0.0
    errors = 0

    for item in tqdm(samples, desc="VoCo 评测"):
        try:
            # 1fps 采帧
            try:
                vr = decord.VideoReader(item["_resolved_video"])
                duration = len(vr) / vr.get_avg_fps()
            except Exception:
                errors += 1
                continue

            n_frames = max(args.frames_per_segment,
                          min(int(duration * 1.0), args.max_frames))
            n_frames = (n_frames // args.frames_per_segment) * args.frames_per_segment
            if n_frames == 0:
                n_frames = args.frames_per_segment

            frames, _, _ = extract_frames(item["_resolved_video"], n_frames)

            # Vision encoder
            uid = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
            tmp_paths = []
            for i, img in enumerate(frames):
                p = os.path.join(tempfile.gettempdir(), f"_vf_{uid}_{i}.jpg")
                img.save(p)
                tmp_paths.append(p)

            messages = [{"role": "user", "content": [
                {"type": "video", "video": tmp_paths, "fps": 1.0},
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

            pv = proc_inputs.get("pixel_values_videos")
            vg = proc_inputs.get("video_grid_thw")
            if pv is None:
                errors += 1
                continue

            video_embeds, tokens_per_frame, _ = get_video_embeds(
                model.base, pv, vg, device,
            )

            # 切段
            segments = split_into_segments(
                video_embeds, tokens_per_frame, args.frames_per_segment,
            )

            # Q/A embeds
            q_text = f"<|im_start|>user\n{item['question']}<|im_end|>\n<|im_start|>assistant\n"
            a_text = f"{item['answer']}<|im_end|>"
            dtype = model.voco_embeds.dtype
            q_embeds, q_ids = encode_text(tokenizer, embed_layer, q_text, device, dtype)
            a_embeds, a_ids = encode_text(tokenizer, embed_layer, a_text, device, dtype)

            # 拼接序列
            voco_per_seg = model.voco_embeds.to(dtype)
            inputs_embeds, layout = build_voco_sequence_embeds(
                segments, voco_per_seg, q_embeds, a_embeds,
            )

            # Mask
            voco_4d_mask = build_voco_attention_mask(layout, device, dtype)
            padding_mask = torch.ones(1, layout["total_len"],
                                      dtype=torch.long, device=device)

            # Labels
            L = layout["total_len"]
            labels = torch.full((1, L), -100, dtype=torch.long, device=device)
            if layout["a"] is not None:
                a_s, a_e = layout["a"]
                labels[0, a_s:a_e] = a_ids.to(device)

            # Forward
            with torch.no_grad():
                logits, loss, _ = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=padding_mask,
                    voco_4d_mask=voco_4d_mask,
                    labels=labels,
                )

            if loss is not None:
                total_loss += loss.item()

            # 判断准确率：answer 首 token
            if layout["a"] is not None:
                a_s, _ = layout["a"]
                pred_token = logits[0, a_s - 1, :].argmax().item()
                gt_token = a_ids[0].item()
                if pred_token == gt_token:
                    correct += 1
                total += 1

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  [错误] {e}")
            continue

    acc = correct / max(total, 1)
    avg_loss = total_loss / max(total, 1)

    print(f"\n{'=' * 50}")
    print(f"VoCo MCQ 评测结果")
    print(f"{'=' * 50}")
    print(f"  准确率: {acc:.2%} ({correct}/{total})")
    print(f"  平均 loss: {avg_loss:.4f}")
    if errors > 0:
        print(f"  跳过 {errors} 条")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--video_dirs", required=True)
    parser.add_argument("--K_seg", type=int, default=4)
    parser.add_argument("--frames_per_segment", type=int, default=2)
    parser.add_argument("--max_frames", type=int, default=30)
    parser.add_argument("--max_samples", type=int, default=500)
    args = parser.parse_args()
    eval_voco(args)

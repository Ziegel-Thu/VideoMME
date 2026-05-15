"""
VoCo 蒸馏评测：加载蒸馏训好的 voco_embeds，用 VoCo 推理管线做 MCQ 评测

流程：
  1. 加载冻结 LLM + 蒸馏 voco_embeds
  2. 每个视频：vision encoder → 切段 → 每段 [vision, voco] forward 提取 KV cache → 拼接
  3. Forward [Q] with past_kv=voco_caches → 比较 ABCD logit

用法:
  python eval_distill.py \
      --data_path /path/to/test.jsonl \
      --video_dirs /path/to/videos \
      --model_path /path/to/qwen \
      --checkpoint /path/to/voco_embeds_epoch3.pt \
      --max_samples 200
"""

import os
import sys
import json
import glob
import argparse
import uuid
import tempfile

import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers.cache_utils import DynamicCache

sys.path.insert(0, os.path.dirname(__file__))
from data import extract_frames
from model import setup_voco_model, get_video_embeds, split_into_segments


def get_inner(base):
    m = base.module if hasattr(base, "module") else base
    try:
        from peft import PeftModel
        if isinstance(m, PeftModel):
            return m.base_model.model
    except ImportError:
        pass
    return m


def voco_inference(model, video_embeds, tokens_per_frame, frames_per_segment,
                   q_ids, tokenizer, device):
    """VoCo 推理：分段 → 提取 voco KV cache → 拼接 → forward Q → 取 ABCD logit。"""
    raw = model.module if hasattr(model, "module") else model
    inner = get_inner(raw.base)
    K = raw.K_seg
    voco_per_seg = raw.voco_embeds

    # 切段
    segments = split_into_segments(video_embeds, tokens_per_frame, frames_per_segment)

    # 每段提取 voco KV cache
    seg_caches = []
    position_offset = 0
    for seg in segments:
        inputs_embeds = torch.cat([seg.detach(), voco_per_seg], dim=0).unsqueeze(0)
        seg_len = inputs_embeds.shape[1]
        attn_mask = torch.ones(1, seg_len, dtype=torch.long, device=device)
        pos = torch.arange(position_offset, position_offset + seg_len, device=device)
        position_ids = pos.view(1, 1, -1).expand(3, 1, -1)

        cache = DynamicCache()
        out = inner.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )

        # 只保留 voco 部分 KV（最后 K 个）
        voco_cache = []
        for layer_idx in range(len(out.past_key_values)):
            k, v = out.past_key_values[layer_idx]
            voco_cache.append((k[:, :, -K:, :], v[:, :, -K:, :]))
        seg_caches.append(voco_cache)
        position_offset += seg_len

    # 拼接所有段的 voco KV cache
    full_cache = DynamicCache()
    n_layers = len(seg_caches[0])
    for layer_idx in range(n_layers):
        k_list = [seg_caches[s][layer_idx][0] for s in range(len(seg_caches))]
        v_list = [seg_caches[s][layer_idx][1] for s in range(len(seg_caches))]
        full_cache.update(torch.cat(k_list, dim=2), torch.cat(v_list, dim=2), layer_idx)

    total_voco = K * len(seg_caches)

    # Forward Q with past_kv
    embed_layer = inner.get_input_embeddings()
    q_embeds = embed_layer(q_ids.unsqueeze(0)).to(raw.voco_embeds.dtype)
    q_len = q_embeds.shape[1]

    full_attn_mask = torch.ones(1, total_voco + q_len, dtype=torch.long, device=device)
    text_pos = torch.arange(position_offset, position_offset + q_len, device=device)
    text_position_ids = text_pos.view(1, 1, -1).expand(3, 1, -1)

    out = inner.model(
        inputs_embeds=q_embeds,
        attention_mask=full_attn_mask,
        position_ids=text_position_ids,
        past_key_values=full_cache,
        use_cache=False,
    )
    logits = inner.lm_head(out[0])  # (1, q_len, vocab)

    # 取最后一个 token 的 logits（预测 answer 的第一个 token）
    last_logits = logits[0, -1, :].float()

    # 比较 ABCD
    option_ids = {}
    for letter in "ABCD":
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if ids:
            option_ids[letter] = ids[0]

    option_logits = {letter: last_logits[tid].item() for letter, tid in option_ids.items()}
    pred = max(option_logits, key=option_logits.get)
    return pred, option_logits


def main(args):
    device = torch.device("cuda")

    print("加载模型...")
    model, processor, tokenizer = setup_voco_model(
        model_name=args.model_path,
        K_seg=args.K_seg,
        device=device,
        use_lora=False,
        attn_implementation="eager",
    )

    # 加载蒸馏 checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.voco_embeds.data.copy_(ckpt["voco_embeds"].to(device))
    print(f"  已加载 voco_embeds from {args.checkpoint}")
    print(f"  K_seg={ckpt.get('K_seg', '?')}, train_loss={ckpt.get('train_loss', '?')}")

    model.eval()

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
    print(f"评测数据: {len(samples)} 条")

    correct = 0
    total = 0
    errors = 0

    inner = get_inner(model.base)

    for item in tqdm(samples, desc="VoCo 评测"):
        try:
            # 采帧 + vision encoder
            import decord
            from qwen_vl_utils import process_vision_info

            vr = decord.VideoReader(item["_resolved_video"])
            duration = len(vr) / vr.get_avg_fps()
            n_frames = max(args.frames_per_segment,
                          min(int(duration * args.fps), args.max_frames))
            n_frames = (n_frames // args.frames_per_segment) * args.frames_per_segment
            if n_frames == 0:
                n_frames = args.frames_per_segment

            frames, _, _ = extract_frames(item["_resolved_video"], n_frames)

            # Vision encoder
            uid = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
            tmp_dir = tempfile.gettempdir()
            tmp_paths = []
            for i, img in enumerate(frames):
                p = os.path.join(tmp_dir, f"_eval_{uid}_{i}.jpg")
                img.save(p)
                tmp_paths.append(p)

            messages = [{"role": "user", "content": [
                {"type": "video", "video": tmp_paths, "fps": args.fps},
                {"type": "text", "text": "x"},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=False)
            image_inputs, video_inputs = process_vision_info(messages)
            proc_inputs = processor(text=[text], images=image_inputs,
                                    videos=video_inputs, return_tensors="pt")

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

            with torch.no_grad():
                video_embeds, tokens_per_frame, _ = get_video_embeds(
                    model.base, pv, vg, device)
                video_embeds = video_embeds.to(model.voco_embeds.dtype)

            # 构造 Q
            q_text = f"<|im_start|>user\n{item['question']}<|im_end|>\n<|im_start|>assistant\n"
            q_ids = tokenizer.encode(q_text, add_special_tokens=False,
                                     return_tensors="pt").squeeze(0).to(device)

            with torch.no_grad():
                pred, logits = voco_inference(
                    model, video_embeds, tokens_per_frame,
                    args.frames_per_segment, q_ids, tokenizer, device,
                )

            gt = item["answer"].strip()
            if pred == gt:
                correct += 1
            total += 1

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  [错误] {e}")

    acc = correct / max(total, 1)
    print(f"\n{'=' * 50}")
    print(f"VoCo 蒸馏评测 ({args.checkpoint})")
    print(f"{'=' * 50}")
    print(f"  准确率: {acc:.2%} ({correct}/{total})")
    print(f"  错误: {errors}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"accuracy": acc, "correct": correct, "total": total,
                       "errors": errors, "checkpoint": args.checkpoint}, f, indent=2)
        print(f"  结果 → {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VoCo 蒸馏评测")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--video_dirs", required=True)
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--checkpoint", required=True, help="蒸馏 voco_embeds 文件")
    parser.add_argument("--K_seg", type=int, default=8)
    parser.add_argument("--frames_per_segment", type=int, default=2)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max_frames", type=int, default=30)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    main(args)

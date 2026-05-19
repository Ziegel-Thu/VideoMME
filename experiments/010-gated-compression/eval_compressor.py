"""
Cross-Attention 压缩模块评测

加载训练好的 compressor，对每段 dense vision tokens 做压缩，
用压缩后的 tokens 的 KV cache 回答 MCQ。

用法:
  python eval_compressor.py \
    --model_path /path/to/Qwen2.5-VL-7B-Instruct \
    --data_path /path/to/test.jsonl \
    --video_dirs /path/to/videos \
    --checkpoint /path/to/compressor_epoch3.pt \
    --max_samples 200
"""

import os
import argparse
import json
import glob
import uuid
import tempfile

import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
import decord

from compressor import InterSegmentAttention, VoCoCompressor
from model import get_video_embeds, split_into_segments


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


def get_inner(base_model):
    m = base_model.module if hasattr(base_model, "module") else base_model
    try:
        from peft import PeftModel
        if isinstance(m, PeftModel):
            return m.base_model.model
    except ImportError:
        pass
    return m


def get_checkpoint_config(ckpt):
    K_seg = ckpt.get("K_seg", 8)
    n_layers = ckpt.get("n_layers", 1)
    inter_layers = ckpt.get("inter_layers", 0) or 0
    return K_seg, n_layers, inter_layers


def eval_compressor(args):
    device = torch.device("cuda")

    # 加载 LLM
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    base.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer
    inner = get_inner(base)
    embed_layer = inner.get_input_embeddings()
    dtype = torch.bfloat16

    # 加载 compressor
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    K_seg, n_layers, inter_layers = get_checkpoint_config(ckpt)
    print(
        f"加载 compressor: K={K_seg}, n_layers={n_layers}, "
        f"inter_layers={inter_layers}"
    )
    print(f"  train_loss={ckpt.get('train_loss', '?')}")

    compressor = VoCoCompressor(K=K_seg, dim=3584, n_layers=n_layers)
    compressor.load_state_dict(ckpt["compressor"])
    compressor = compressor.to(device, dtype=dtype)
    compressor.eval()
    inter_segment = None
    if inter_layers > 0:
        inter_segment = InterSegmentAttention(
            dim=3584, n_layers=inter_layers,
        )
        inter_segment.load_state_dict(ckpt["inter_segment"])
        inter_segment = inter_segment.to(device, dtype=dtype)
        inter_segment.eval()

    n_params = compressor.num_params()
    if inter_segment is not None:
        n_params += inter_segment.num_params()
    print(f"  Compressor 参数: {n_params:,}")

    # 数据
    video_dirs = args.video_dirs.split(",")
    video_index = {}
    for vdir in video_dirs:
        if not os.path.isdir(vdir):
            continue
        for f in glob.glob(os.path.join(vdir, "**", "*.mp4"), recursive=True):
            video_index[os.path.basename(f)] = f

    with open(args.data_path) as f:
        samples = [json.loads(l.strip()) for l in f]

    resolved = []
    for item in samples:
        vname = os.path.basename(item["video_path"])
        if vname in video_index:
            item["_resolved_video"] = video_index[vname]
            resolved.append(item)
    samples = resolved[:args.max_samples] if args.max_samples > 0 else resolved
    # 分片：多卡并行评测时每卡只跑自己的分片
    if args.num_shards > 1:
        samples = samples[args.shard_id::args.num_shards]
    print(f"评测数据: {len(samples)} 条 (shard {args.shard_id}/{args.num_shards})")

    # ABCD token ids
    abcd_ids = {
        c: tokenizer.encode(c, add_special_tokens=False)[0]
        for c in ["A", "B", "C", "D"]
    }

    correct = 0
    total = 0

    for item in tqdm(samples, desc="Compressor 评测"):
        try:
            # 1. 视频解码 + vision encoder
            vr = decord.VideoReader(item["_resolved_video"])
            duration = len(vr) / vr.get_avg_fps()
            n_frames = max(2, min(int(duration * 1.0), 30))
            n_frames = (n_frames // 2) * 2
            if n_frames == 0:
                n_frames = 2

            frames, _, _ = extract_frames(item["_resolved_video"], n_frames)

            from qwen_vl_utils import process_vision_info
            uid = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
            tmp_dir = tempfile.gettempdir()
            tmp_paths = []
            for i, img in enumerate(frames):
                p = os.path.join(tmp_dir, f"_vf_{uid}_{i}.jpg")
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
                continue

            with torch.no_grad():
                video_embeds, tpf, _ = get_video_embeds(base, pv, vg, device)
                video_embeds = video_embeds.to(dtype)
                segments = split_into_segments(video_embeds, tpf, 2)

                # 2. 压缩每段 → KV cache
                all_compressed = []
                for seg in segments:
                    compressed = compressor(seg)  # (K, D)
                    all_compressed.append(compressed)
                if inter_segment is not None:
                    segment_lengths = [tokens.shape[0] for tokens in all_compressed]
                    all_tokens = torch.cat(all_compressed, dim=0)
                    all_compressed = inter_segment(all_tokens, segment_lengths)

                # 拼接所有段的压缩 tokens
                all_comp = torch.cat(all_compressed, dim=0)  # (n_seg*K, D)

                # 3. 过 LLM 得到 compressed tokens 的 KV cache
                comp_len = all_comp.shape[0]
                comp_input = all_comp.unsqueeze(0)
                comp_pos = torch.arange(comp_len, device=device)
                comp_pos_ids = comp_pos.view(1, 1, -1).expand(3, 1, -1)
                comp_attn = torch.ones(1, comp_len, dtype=torch.long, device=device)

                comp_out = inner.model(
                    inputs_embeds=comp_input,
                    attention_mask=comp_attn,
                    position_ids=comp_pos_ids,
                    use_cache=True,
                )
                past_kv = comp_out.past_key_values

                # 4. 用 past_kv 续写 question → 比较 ABCD logits
                q_text = (f"<|im_start|>user\n{item['question']}"
                          f"<|im_end|>\n<|im_start|>assistant\n")
                q_ids = tokenizer.encode(
                    q_text, add_special_tokens=False, return_tensors="pt",
                ).to(device)
                q_emb = embed_layer(q_ids).to(dtype)

                q_len = q_emb.shape[1]
                q_pos = torch.arange(comp_len, comp_len + q_len, device=device)
                q_pos_ids = q_pos.view(1, 1, -1).expand(3, 1, -1)
                q_attn = torch.ones(1, comp_len + q_len, dtype=torch.long, device=device)

                q_out = inner.model(
                    inputs_embeds=q_emb,
                    attention_mask=q_attn,
                    position_ids=q_pos_ids,
                    past_key_values=past_kv,
                    use_cache=False,
                )
                last_hidden = q_out[0][0, -1, :]  # (D,)
                logits = inner.lm_head(last_hidden)

                # 5. 比较 ABCD logits
                abcd_logits = {c: logits[tid].item() for c, tid in abcd_ids.items()}
                pred = max(abcd_logits, key=abcd_logits.get)
                gt = item["answer"].strip().upper()

                if pred == gt:
                    correct += 1
                total += 1

        except Exception as e:
            continue

    acc = correct / max(total, 1) * 100
    print(f"\n  准确率: {acc:.2f}% ({correct}/{total})")
    print(f"  K={K_seg}, n_layers={n_layers}, inter_layers={inter_layers}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--video_dirs", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--max_samples", type=int, default=200,
        help="评测样本数；设为 0 表示使用完整 test split。",
    )
    parser.add_argument("--shard_id", type=int, default=0, help="当前分片 ID（从 0 开始）")
    parser.add_argument("--num_shards", type=int, default=1, help="总分片数；1 表示不分片")
    args = parser.parse_args()
    eval_compressor(args)

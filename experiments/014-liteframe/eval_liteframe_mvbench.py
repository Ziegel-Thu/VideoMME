"""
MVBench 评测 — LiteFrame student encoder 模式

Student encoder 直接替换 teacher ViT，输出 256 tokens @ 3584D 送入 LLM。
不需要 LoRA 或任何额外训练（纯 CTD 特征质量评估）。

用法:
  # 4-shard 并行
  torchrun --nproc_per_node=4 eval_liteframe_mvbench.py \
    --video_base /mnt/default/bottleneck/benchmarks/mvbench/video \
    --json_base /mnt/default/bottleneck/benchmarks/mvbench/json \
    --model_path /mnt/default/bottleneck/models/Qwen2.5-VL-7B-Instruct \
    --student_ckpt checkpoint_final.pt \
    --output $AMLT_OUTPUT_DIR/shard{rank}.json
"""

import os
import sys
import json
import glob
import argparse
import tempfile

import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import decord

from liteframe_encoder import create_liteframe_base

# CLIP 归一化参数
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

TASK_VIDEO_PREFIX = {
    "action_antonym": "ssv2_video",
    "action_count": "clevrer",
    "action_localization": "sta",
    "action_prediction": "sta",
    "action_sequence": "star",
    "character_order": "clevrer",
    "counterfactual_inference": "clevrer",
    "egocentric_navigation": "vlnqa",
    "episodic_reasoning": "tvqa",
    "fine_grained_action": "Moments_in_Time_Raw",
    "fine_grained_pose": "nturgbd",
    "moving_attribute": "clevrer",
    "moving_count": "clevrer",
    "moving_direction": "clevrer",
    "object_existence": "clevrer",
    "object_interaction": "sta",
    "object_shuffle": "clevrer",
    "scene_transition": "scene_qa",
    "state_change": "perception",
    "unexpected_action": "FunQA_test",
}


def extract_frames(video_path, num_frames):
    """从视频均匀采样 num_frames 帧，返回 PIL Images。"""
    try:
        vr = decord.VideoReader(video_path)
        total = len(vr)
        indices = [min(int(i * total / num_frames), total - 1)
                   for i in range(num_frames)]
        frames = [Image.fromarray(vr[idx].asnumpy()) for idx in indices]
        return frames
    except Exception:
        return None


def frames_to_tensor(pil_frames, image_size=448):
    """PIL Images → (1, C, T, H, W) 归一化 tensor。"""
    tensors = []
    for img in pil_frames:
        img = img.convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        t = torch.from_numpy(
            __import__("numpy").array(img)
        ).permute(2, 0, 1).float() / 255.0  # (C, H, W)
        tensors.append(t)
    x = torch.stack(tensors, dim=0)  # (T, C, H, W)
    x = (x - CLIP_MEAN) / CLIP_STD
    return x.unsqueeze(0).permute(0, 2, 1, 3, 4)  # (1, C, T, H, W)


def resolve_video_path(video_field, task, video_base, video_index):
    basename = os.path.basename(video_field)
    if basename in video_index:
        return video_index[basename]
    for ext in [".mp4", ".avi", ".webm", ".mkv"]:
        if basename + ext in video_index:
            return video_index[basename + ext]
    return None


def eval_liteframe(student, base_model, tokenizer, frames_pil, question,
                   candidates, device, num_frames=8):
    """LiteFrame student encoder 评测。

    流程: frames → student encoder → 256 tokens → LLM (KV cache) → question → logits
    """
    from peft import PeftModel

    inner = base_model
    if hasattr(inner, "module"):
        inner = inner.module
    if isinstance(inner, PeftModel):
        inner = inner.base_model.model

    embed_layer = inner.get_input_embeddings()
    dtype = torch.bfloat16

    # Student forward: PIL → tensor → student → (1, 256, 3584)
    frames_tensor = frames_to_tensor(frames_pil, 448).to(device, dtype=dtype)
    with torch.no_grad():
        student_out, grid = student(frames_tensor)  # (1, N, 3584)

    vis_len = student_out.shape[1]
    vis_pos = torch.arange(vis_len, device=device)
    vis_pos_ids = vis_pos.view(1, 1, -1).expand(3, 1, -1)
    vis_attn = torch.ones(1, vis_len, dtype=torch.long, device=device)

    # First pass: visual tokens through LLM → KV cache
    with torch.no_grad():
        vis_out = inner.model(
            inputs_embeds=student_out.to(dtype),
            attention_mask=vis_attn,
            position_ids=vis_pos_ids,
            use_cache=True,
        )
        past_kv = vis_out.past_key_values

    # Second pass: question tokens
    options = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(candidates)])
    q_text = (
        f"<|im_start|>user\n"
        f"Select the best answer to the following multiple-choice question "
        f"based on the video. Respond with only the letter "
        f"({', '.join(chr(65+i) for i in range(len(candidates)))}) "
        f"of the correct option.\n{question}\n{options}\n"
        f"The best answer is:"
        f"<|im_end|>\n<|im_start|>assistant\n"
    )
    q_ids = tokenizer.encode(q_text, add_special_tokens=False, return_tensors="pt").to(device)
    q_emb = embed_layer(q_ids).to(dtype)

    q_len = q_emb.shape[1]
    q_pos = torch.arange(vis_len, vis_len + q_len, device=device)
    q_pos_ids = q_pos.view(1, 1, -1).expand(3, 1, -1)
    q_attn = torch.ones(1, vis_len + q_len, dtype=torch.long, device=device)

    with torch.no_grad():
        q_out = inner.model(
            inputs_embeds=q_emb,
            attention_mask=q_attn,
            position_ids=q_pos_ids,
            past_key_values=past_kv,
            use_cache=False,
        )
        last_hidden = q_out[0][0, -1, :]
        logits = inner.lm_head(last_hidden)

    option_ids = {chr(65+i): tokenizer.encode(chr(65+i), add_special_tokens=False)[0]
                  for i in range(len(candidates))}
    option_logits = {k: logits[v].item() for k, v in option_ids.items()}
    pred = max(option_logits, key=option_logits.get)
    return pred


def main(args):
    device = torch.device("cuda")

    # 1. 加载 Qwen2.5-VL (LLM only, no LoRA)
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer

    # 2. 加载 student encoder
    student = create_liteframe_base()
    ckpt = torch.load(args.student_ckpt, map_location="cpu", weights_only=True)
    student.load_state_dict(ckpt["student"])
    student = student.to(device, dtype=torch.bfloat16)
    student.eval()
    print(f"Student loaded: {sum(p.numel() for p in student.parameters())/1e6:.1f}M params, step {ckpt['step']}")

    # 3. 建视频文件名索引
    print(f"扫描视频目录 {args.video_base} ...")
    video_index = {}
    for root, dirs, files in os.walk(args.video_base):
        for f in files:
            video_index[f] = os.path.join(root, f)
    print(f"  索引 {len(video_index)} 个视频文件")

    # 4. 加载所有 task
    json_files = sorted(glob.glob(os.path.join(args.json_base, "*.json")))

    task_results = {}
    total_correct = 0
    total_count = 0

    for jf in json_files:
        task = os.path.basename(jf).replace(".json", "")
        with open(jf) as f:
            data = json.load(f)

        if args.num_shards > 1:
            data = data[args.shard_id::args.num_shards]

        correct = 0
        count = 0

        for item in tqdm(data, desc=task, disable=(args.shard_id != 0)):
            video_path = resolve_video_path(
                item["video"], task, args.video_base, video_index,
            )
            if video_path is None:
                continue

            frames = extract_frames(video_path, args.num_frames)
            if frames is None:
                continue

            try:
                gt = item["answer"]
                candidates = item["candidates"]
                gt_idx = candidates.index(gt)
                gt_letter = chr(65 + gt_idx)

                pred = eval_liteframe(
                    student, model, tokenizer,
                    frames, item["question"], candidates, device,
                    num_frames=args.num_frames,
                )

                if pred == gt_letter:
                    correct += 1
                count += 1

            except Exception as e:
                import traceback
                traceback.print_exc()
                continue

        acc = correct / max(count, 1) * 100
        task_results[task] = {"correct": correct, "total": count, "acc": acc}
        print(f"  {task}: {acc:.2f}% ({correct}/{count})")

    total_correct = sum(r["correct"] for r in task_results.values())
    total_count = sum(r["total"] for r in task_results.values())
    total_acc = total_correct / max(total_count, 1) * 100

    print(f"\nMVBench 总准确率: {total_acc:.2f}% ({total_correct}/{total_count})")
    print(f"  shard {args.shard_id}/{args.num_shards}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump({
                "total_acc": total_acc,
                "total_correct": total_correct,
                "total_count": total_count,
                "task_results": task_results,
                "shard_id": args.shard_id,
                "num_shards": args.num_shards,
            }, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MVBench LiteFrame eval")
    parser.add_argument("--video_base", required=True)
    parser.add_argument("--json_base", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--student_ckpt", required=True, help="CTD student checkpoint")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    main(args)

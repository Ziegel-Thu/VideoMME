"""
MVBench 评测脚本

支持 zero-shot 和 compressor 两种模式：
  - zero-shot: 直接用 Qwen2.5-VL 回答
  - compressor: 用压缩后的 tokens 回答

用法:
  # zero-shot
  python eval_mvbench.py \
    --video_base /mnt/default/bottleneck/benchmarks/mvbench/video \
    --json_base /mnt/default/bottleneck/benchmarks/mvbench/json \
    --model_path /mnt/default/bottleneck/models/Qwen2.5-VL-7B-Instruct

  # compressor
  python eval_mvbench.py \
    --video_base /mnt/default/bottleneck/benchmarks/mvbench/video \
    --json_base /mnt/default/bottleneck/benchmarks/mvbench/json \
    --model_path /mnt/default/bottleneck/models/Qwen2.5-VL-7B-Instruct \
    --checkpoint /mnt/default/bottleneck/checkpoints/006_compressor/110k_B2L_ep2.pt
"""

import os
import sys
import json
import glob
import argparse
import zipfile
import tempfile

import torch
from tqdm import tqdm
from PIL import Image
import decord

# MVBench task → video zip/dir 映射
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
    """从视频均匀采样 num_frames 帧。"""
    try:
        vr = decord.VideoReader(video_path)
        total = len(vr)
        indices = [min(int(i * total / num_frames), total - 1)
                   for i in range(num_frames)]
        frames = [Image.fromarray(vr[idx].asnumpy()) for idx in indices]
        return frames
    except Exception as e:
        return None


def resolve_video_path(video_field, task, video_base, video_index):
    """根据 video 字段在索引中查找实际路径。"""
    # 直接用文件名查
    basename = os.path.basename(video_field)
    if basename in video_index:
        return video_index[basename]
    # 加扩展名
    for ext in [".mp4", ".avi", ".webm", ".mkv"]:
        if basename + ext in video_index:
            return video_index[basename + ext]
    return None


def eval_zero_shot(model, processor, tokenizer, frames, question, candidates, device):
    """Zero-shot logit 评测：比较每个候选答案的 logit。"""
    from qwen_vl_utils import process_vision_info
    import uuid
    
    # 构造选项文本
    options = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(candidates)])
    prompt = (
        f"Select the best answer to the following multiple-choice question "
        f"based on the video. Respond with only the letter ({', '.join(chr(65+i) for i in range(len(candidates)))}) "
        f"of the correct option.\n{question}\n{options}\nThe best answer is:"
    )
    
    uid = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    tmp_dir = tempfile.gettempdir()
    tmp_paths = []
    for i, img in enumerate(frames):
        p = os.path.join(tmp_dir, f"_mv_{uid}_{i}.jpg")
        img.save(p)
        tmp_paths.append(p)
    
    messages = [{"role": "user", "content": [
        {"type": "video", "video": tmp_paths, "fps": 1.0},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        return_tensors="pt",
    ).to(device)
    
    for p in tmp_paths:
        try:
            os.remove(p)
        except OSError:
            pass
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]
    
    # 比较候选 token logit
    option_ids = {chr(65+i): tokenizer.encode(chr(65+i), add_special_tokens=False)[0]
                  for i in range(len(candidates))}
    option_logits = {k: logits[v].item() for k, v in option_ids.items()}
    pred = max(option_logits, key=option_logits.get)
    return pred


def eval_compressor_mode(base, processor, tokenizer, compressor, inter_segment,
                         frames, question, candidates, device):
    """Compressor 模式评测。"""
    from model import get_video_embeds, split_into_segments
    from qwen_vl_utils import process_vision_info
    import uuid
    
    inner = base.module if hasattr(base, "module") else base
    from peft import PeftModel
    if isinstance(inner, PeftModel):
        inner = inner.base_model.model
    embed_layer = inner.get_input_embeddings()
    dtype = torch.bfloat16
    
    uid = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    tmp_dir = tempfile.gettempdir()
    tmp_paths = []
    for i, img in enumerate(frames):
        p = os.path.join(tmp_dir, f"_mv_{uid}_{i}.jpg")
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
        return None
    
    with torch.no_grad():
        video_embeds, tpf, _ = get_video_embeds(base, pv, vg, device)
        video_embeds = video_embeds.to(dtype)
        segments = split_into_segments(video_embeds, tpf, 2)
        
        all_compressed = [compressor(seg) for seg in segments]
        if inter_segment is not None:
            seg_lens = [t.shape[0] for t in all_compressed]
            all_compressed = inter_segment(torch.cat(all_compressed, dim=0), seg_lens)
        
        all_comp = torch.cat(all_compressed, dim=0)
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
        last_hidden = q_out[0][0, -1, :]
        logits = inner.lm_head(last_hidden)
    
    option_ids = {chr(65+i): tokenizer.encode(chr(65+i), add_special_tokens=False)[0]
                  for i in range(len(candidates))}
    option_logits = {k: logits[v].item() for k, v in option_ids.items()}
    pred = max(option_logits, key=option_logits.get)
    return pred


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
    
    # 加载 compressor（如果指定了 checkpoint）
    compressor = None
    inter_segment = None
    if args.checkpoint:
        from compressor import VoCoCompressor, InterSegmentAttention
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
    
    # 解压视频 zip 到临时目录
    extracted_dirs = []
    if args.extract_dir:
        os.makedirs(args.extract_dir, exist_ok=True)
        extracted_dirs.append(args.extract_dir)
    
    # 建视频文件名索引（启动时扫描一次，避免每次 os.walk）
    print(f"扫描视频目录 {args.video_base} ...")
    video_index = {}
    for root, dirs, files in os.walk(args.video_base):
        for f in files:
            video_index[f] = os.path.join(root, f)
    print(f"  索引 {len(video_index)} 个视频文件")
    
    # 加载所有 task
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
                
                if compressor is not None:
                    pred = eval_compressor_mode(
                        model, processor, tokenizer, compressor, inter_segment,
                        frames, item["question"], candidates, device,
                    )
                else:
                    pred = eval_zero_shot(
                        model, processor, tokenizer, frames,
                        item["question"], candidates, device,
                    )
                
                if pred == gt_letter:
                    correct += 1
                count += 1
                
            except Exception as e:
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
    parser = argparse.ArgumentParser(description="MVBench 评测")
    parser.add_argument("--video_base", required=True, help="视频目录（含解压后的子目录）")
    parser.add_argument("--json_base", required=True, help="json 标注目录")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--checkpoint", default=None, help="compressor checkpoint，不指定则 zero-shot")
    parser.add_argument("--extract_dir", default=None, help="zip 解压目标目录")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--output", default=None, help="保存结果 JSON")
    args = parser.parse_args()
    main(args)

"""
013 Temporal Head 训练

基于已训好的 compressor checkpoint，冻结 compressor，
训练 temporal head 预测证据时间段。

流程:
  1. 加载视频 → vision encoder → dense tokens → compressor → compressed
  2. 加载 question → embeddings
  3. temporal_head(compressed, q_embeds) → bin scores
  4. BCE loss with ground truth bins

用法:
  python train_temporal_v2.py \
    --compressor_checkpoint /path/to/compressor_epoch1.pt \
    --data_path /path/to/temporal_evidence.jsonl \
    --video_dirs /path/to/videos \
    --model_path /path/to/Qwen2.5-VL-7B-Instruct \
    --head_type B \
    --epochs 10 --lr 1e-4
"""

import os
import json
import glob
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import decord

from compressor import VoCoCompressor, InterSegmentAttention
from model import get_video_embeds, split_into_segments
from temporal_head import SegmentTemporalHead, BinQueryTemporalHead


def extract_frames(video_path, num_frames):
    try:
        vr = decord.VideoReader(video_path)
        total = len(vr)
        fps = vr.get_avg_fps()
        indices = [min(int(i * total / num_frames), total - 1)
                   for i in range(num_frames)]
        frames = [Image.fromarray(vr[idx].asnumpy()) for idx in indices]
        duration = total / fps
        return frames, duration
    except Exception as e:
        return None, 0.0


def compute_bin_labels(duration, evidence_segments, num_bins=16, expand=1.0):
    """将 evidence_segments 转为 num_bins 个 bin 的 0/1 标签。"""
    if not evidence_segments or duration <= 0:
        return [0.0] * num_bins

    bin_size = duration / num_bins
    labels = [0.0] * num_bins

    for seg in evidence_segments:
        if seg is None or len(seg) < 2:
            continue
        t_s, t_e = seg[0], seg[1]
        if t_s == t_e:
            t_s = max(0, t_s - expand)
            t_e = min(duration, t_e + expand)
        for b in range(num_bins):
            bin_start = b * bin_size
            bin_end = (b + 1) * bin_size
            overlap = max(0, min(bin_end, t_e) - max(bin_start, t_s))
            if overlap > 0:
                labels[b] = 1.0
    return labels


class TemporalDataset(Dataset):
    def __init__(self, data_path, video_dirs, max_samples=None):
        video_index = {}
        for vdir in video_dirs:
            if not os.path.isdir(vdir):
                continue
            for f in glob.glob(os.path.join(vdir, "**", "*.*"), recursive=True):
                video_index[os.path.basename(f)] = f

        self.samples = []
        with open(data_path) as f:
            for line in f:
                item = json.loads(line.strip())
                vname = os.path.basename(item.get("video_path", ""))
                if vname in video_index:
                    item["_resolved_video"] = video_index[vname]
                    self.samples.append(item)
                if max_samples and len(self.samples) >= max_samples:
                    break

        print(f"TemporalDataset: {len(self.samples)} 条 (from {len(video_index)} 视频)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train(args):
    device = torch.device("cuda")

    # 加载 LLM + vision encoder
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    base.eval()
    for p in base.parameters():
        p.requires_grad = False
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = processor.tokenizer
    inner = base.module if hasattr(base, "module") else base
    try:
        from peft import PeftModel
        if isinstance(inner, PeftModel):
            inner = inner.base_model.model
    except ImportError:
        pass
    embed_layer = inner.get_input_embeddings()
    dtype = torch.bfloat16

    # 加载 compressor（冻结）
    ckpt = torch.load(args.compressor_checkpoint, map_location="cpu", weights_only=False)
    K_seg = ckpt.get("K_seg", 8)
    n_layers = ckpt.get("n_layers", 1)
    inter_layers = ckpt.get("inter_layers", 0) or 0
    print(f"Compressor: K={K_seg}, n_layers={n_layers}, inter_layers={inter_layers}")

    compressor = VoCoCompressor(K=K_seg, dim=3584, n_layers=n_layers)
    compressor.load_state_dict(ckpt["compressor"])
    compressor = compressor.to(device, dtype=dtype)
    compressor.eval()
    for p in compressor.parameters():
        p.requires_grad = False

    inter_segment = None
    if inter_layers > 0:
        inter_segment = InterSegmentAttention(dim=3584, n_layers=inter_layers)
        inter_segment.load_state_dict(ckpt["inter_segment"])
        inter_segment = inter_segment.to(device, dtype=dtype)
        inter_segment.eval()
        for p in inter_segment.parameters():
            p.requires_grad = False

    # 创建 temporal head
    if args.head_type == "B":
        temporal_head = SegmentTemporalHead(
            dim=3584, num_bins=args.num_bins,
        ).to(device, dtype=dtype)
    elif args.head_type == "C":
        temporal_head = BinQueryTemporalHead(
            dim=3584, num_bins=args.num_bins,
        ).to(device, dtype=dtype)
    else:
        raise ValueError(f"未知 head_type: {args.head_type}")

    print(f"Temporal Head ({args.head_type}): {temporal_head.num_params():,} 参数")

    # 数据
    video_dirs = args.video_dirs.split(",")
    dataset = TemporalDataset(args.data_path, video_dirs, args.max_samples)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda b: b[0])

    # 优化器
    optimizer = torch.optim.AdamW(temporal_head.parameters(), lr=args.lr, weight_decay=0.01)
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        temporal_head.train()
        epoch_loss = 0.0
        epoch_n = 0

        for item in tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            try:
                # 视频 → dense tokens → compressor
                frames, duration = extract_frames(item["_resolved_video"], args.num_frames)
                if frames is None or duration <= 0:
                    continue

                from qwen_vl_utils import process_vision_info
                import uuid, tempfile
                uid = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
                tmp_dir = tempfile.gettempdir()
                tmp_paths = []
                for i, img in enumerate(frames):
                    p = os.path.join(tmp_dir, f"_tp_{uid}_{i}.jpg")
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
                    segments = split_into_segments(video_embeds, tpf, args.frames_per_segment)

                    # Compressor forward
                    compressed_segments = [compressor(seg) for seg in segments]
                    if inter_segment is not None:
                        seg_lens = [t.shape[0] for t in compressed_segments]
                        compressed_segments = inter_segment(
                            torch.cat(compressed_segments, dim=0), seg_lens,
                        )

                # Question embeddings
                q_text = f"<|im_start|>user\n{item['question']}<|im_end|>\n<|im_start|>assistant\n"
                q_ids = tokenizer.encode(q_text, add_special_tokens=False, return_tensors="pt").to(device)
                with torch.no_grad():
                    q_embeds = embed_layer(q_ids).squeeze(0).to(dtype)

                # Temporal head forward
                optimizer.zero_grad()
                if args.head_type == "B":
                    bin_scores = temporal_head(compressed_segments, q_embeds, len(segments))
                else:  # C
                    compressed_flat = torch.cat(compressed_segments, dim=0)
                    bin_scores = temporal_head(compressed_flat, q_embeds)

                # Ground truth bins
                gt_bins = compute_bin_labels(
                    duration, item.get("evidence_segments"),
                    num_bins=args.num_bins,
                )
                gt_tensor = torch.tensor(gt_bins, device=device, dtype=torch.float32)

                # BCE loss with pos_weight
                pos_count = gt_tensor.sum()
                neg_count = args.num_bins - pos_count
                pos_weight = (neg_count / max(pos_count, 1)).clamp(max=10)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    bin_scores.float(), gt_tensor,
                    pos_weight=pos_weight.expand_as(gt_tensor),
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(temporal_head.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_n += 1

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                continue

        avg_loss = epoch_loss / max(epoch_n, 1)
        print(f"  Epoch {epoch+1} 完成: avg_loss={avg_loss:.6f} (n={epoch_n})")

        # Save checkpoint
        ckpt_path = os.path.join(args.output_dir, f"temporal_head_epoch{epoch+1}.pt")
        torch.save({
            "temporal_head": temporal_head.state_dict(),
            "head_type": args.head_type,
            "num_bins": args.num_bins,
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "compressor_checkpoint": args.compressor_checkpoint,
        }, ckpt_path)
        print(f"  💾 {ckpt_path}")

    print("\n训练完成.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal Head 训练 (方案 B/C)")
    parser.add_argument("--compressor_checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--video_dirs", required=True)
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--head_type", choices=["B", "C"], default="B")
    parser.add_argument("--num_bins", type=int, default=16)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frames_per_segment", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()
    train(args)

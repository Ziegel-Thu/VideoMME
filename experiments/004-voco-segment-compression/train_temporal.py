"""
004 Stage 3: VoCo Temporal Head 训练

从 voco tokens 的 hidden states 预测 evidence 时间 bins。
VoCo 模型中 Q 在所有 voco 之后（question-agnostic），
voco hidden states 天然携带了视觉信息的压缩表示。

用法:
  python train_temporal.py \
    --voco_checkpoint checkpoints/voco-single-10k/big-chimp/best_model.pt \
    --data_path ../../data/parsed/temporal_evidence.jsonl \
    --video_dirs ../../data/llava-video/0_30_s_academic_v0_1,../../data/llava-video/30_60_s_academic_v0_1 \
    --output_dir outputs_temporal \
    --K_seg 4 --num_bins 16 --epochs 10
"""

import os
import json
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import glob

from model import (
    setup_voco_model,
    get_video_embeds,
    split_into_segments,
    build_voco_sequence_embeds,
    build_voco_attention_mask,
)
from data import extract_frames, encode_text


# ============================================================
# Temporal Head
# ============================================================

class TemporalHead(nn.Module):
    """从 voco hidden states 预测 evidence time bins。

    输入: (B, n_voco_total, D) 所有段的 voco hidden states
    输出: (B, num_bins) 每个 bin 的 evidence logits
    """

    def __init__(self, hidden_dim=3584, num_bins=16):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_bins),
        )

    def forward(self, voco_hidden):
        # voco_hidden: (B, n_voco, D)
        pooled = voco_hidden.mean(dim=1)  # (B, D)
        return self.proj(pooled)  # (B, num_bins)


# ============================================================
# Dataset
# ============================================================

class TemporalDataset(Dataset):
    """temporal evidence 数据集。"""

    def __init__(self, data_path, video_dirs, max_samples=None,
                 evidence_expand=1.0):
        self.video_index = {}
        for vdir in video_dirs:
            if not os.path.isdir(vdir):
                continue
            for f in glob.glob(os.path.join(vdir, "**", "*.mp4"),
                               recursive=True):
                self.video_index[os.path.basename(f)] = f

        self.samples = []
        skipped_no_video = 0
        skipped_no_evidence = 0

        with open(data_path) as f:
            for line in f:
                item = json.loads(line.strip())
                segments = item.get("evidence_segments")
                if not segments:
                    skipped_no_evidence += 1
                    continue
                vname = os.path.basename(item.get("video_path", ""))
                if vname not in self.video_index:
                    skipped_no_video += 1
                    continue
                item["_resolved_video"] = self.video_index[vname]
                self.samples.append(item)

        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"Temporal 数据集: {len(self.samples)} 条可用, "
              f"{skipped_no_video} 无视频, "
              f"{skipped_no_evidence} 无 evidence")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def compute_bin_labels(duration, evidence_segments, num_bins=16,
                       expand=1.0):
    """计算 fixed-bin temporal labels。"""
    if duration <= 0:
        return [0.0] * num_bins

    bin_size = duration / num_bins
    labels = [0.0] * num_bins

    for s, e in evidence_segments:
        if e <= s:
            s = max(0, s - expand)
            e = min(duration, s + 2 * expand)
        for b in range(num_bins):
            bin_start = b * bin_size
            bin_end = (b + 1) * bin_size
            overlap = max(0, min(bin_end, e) - max(bin_start, s))
            labels[b] = max(labels[b], overlap / bin_size)

    return labels


# ============================================================
# Hidden state 捕获
# ============================================================

class VoCoHiddenCapture:
    """在最后一层 decoder 捕获 voco 位置的 hidden states。"""

    def __init__(self, last_layer):
        self.last_layer = last_layer
        self._hidden = None
        self._hook = None

    def install(self):
        def hook_fn(module, input, output):
            self._hidden = output[0]  # (B, L, D)
        self._hook = self.last_layer.register_forward_hook(hook_fn)

    def get_voco_hidden(self, layout):
        """从 layout 信息提取 voco 位置的 hidden states。"""
        if self._hidden is None:
            return None

        voco_positions = []
        for (_, _, voco_s, voco_e) in layout["segments"]:
            voco_positions.extend(range(voco_s, voco_e))

        if not voco_positions:
            return None

        # (1, n_voco, D)
        return self._hidden[:, voco_positions, :]

    def remove(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        self._hidden = None


# ============================================================
# Collate
# ============================================================

def temporal_collate(batch, voco_model, processor, tokenizer,
                     fps=1.0, frames_per_segment=2, max_frames=30,
                     num_bins=16, evidence_expand=1.0):
    """构造 VoCo forward 的输入 + temporal bin labels。"""
    assert len(batch) == 1
    item = batch[0]

    base = voco_model.base
    device = voco_model.voco_embeds.device
    dtype = voco_model.voco_embeds.dtype

    # 1. 读视频
    import decord
    try:
        vr = decord.VideoReader(item["_resolved_video"])
        duration = len(vr) / vr.get_avg_fps()
    except Exception:
        return None

    n_frames = max(frames_per_segment,
                   min(int(duration * fps), max_frames))
    n_frames = (n_frames // frames_per_segment) * frames_per_segment
    if n_frames == 0:
        n_frames = frames_per_segment

    frames, _, _ = extract_frames(item["_resolved_video"], n_frames)

    # 2. vision encoder
    from qwen_vl_utils import process_vision_info
    import uuid, tempfile

    uid = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    tmp_dir = tempfile.gettempdir()
    tmp_paths = []
    for i, img in enumerate(frames):
        p = os.path.join(tmp_dir, f"_tf_{uid}_{i}.jpg")
        img.save(p)
        tmp_paths.append(p)

    messages = [{"role": "user", "content": [
        {"type": "video", "video": tmp_paths, "fps": fps},
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

    video_embeds, tokens_per_frame, actual_frames = get_video_embeds(
        base, pv, vg, device,
    )

    # 3. 切段 + 构造序列
    segments = split_into_segments(
        video_embeds, tokens_per_frame, frames_per_segment,
    )

    m = base.module if hasattr(base, "module") else base
    from peft import PeftModel
    if isinstance(m, PeftModel):
        embed_layer = m.base_model.model.get_input_embeddings()
    else:
        embed_layer = m.get_input_embeddings()

    # Q text（temporal 任务也需要 Q，虽然 Q 在 voco 之后）
    q_text = (f"<|im_start|>user\n{item['question']}"
              f"<|im_end|>\n<|im_start|>assistant\n")
    a_text = f"{item.get('answer', 'unknown')}<|im_end|>"
    q_embeds, _ = encode_text(tokenizer, embed_layer, q_text, device, dtype)
    a_embeds, _ = encode_text(tokenizer, embed_layer, a_text, device, dtype)

    voco_per_seg = voco_model.voco_embeds.to(dtype)
    inputs_embeds, layout = build_voco_sequence_embeds(
        segments, voco_per_seg, q_embeds, a_embeds,
    )

    voco_4d_mask = build_voco_attention_mask(layout, device, dtype)
    padding_mask = torch.ones(
        1, layout["total_len"], dtype=torch.long, device=device,
    )

    # temporal bin labels
    bin_labels = compute_bin_labels(
        duration, item["evidence_segments"],
        num_bins=num_bins, expand=evidence_expand,
    )

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": padding_mask,
        "voco_4d_mask": voco_4d_mask,
        "layout": layout,
        "bin_labels": torch.tensor([bin_labels], dtype=torch.float,
                                   device=device),
        "duration": duration,
        "n_segments": len(segments),
    }


# ============================================================
# Checkpoint 加载
# ============================================================

def load_voco_checkpoint(model, ckpt_path):
    """加载 VoCo checkpoint（voco_embeds + LoRA weights）。"""
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # 恢复 voco_embeds
    model.voco_embeds.data.copy_(ckpt["voco_embeds"])

    # 恢复 LoRA weights
    m = model.base.module if hasattr(model.base, "module") else model.base
    m.load_state_dict(ckpt["lora_state_dict"], strict=False)

    print(f"✓ VoCo checkpoint 加载: epoch={ckpt['epoch']}, "
          f"val_loss={ckpt['val_loss']:.4f}, K_seg={ckpt['K_seg']}")
    return ckpt["epoch"], ckpt["val_loss"]


# ============================================================
# 评测工具
# ============================================================

def bins_to_segments(probs, duration, threshold=0.5):
    """把 bin 概率转成时间段列表。"""
    num_bins = len(probs)
    bin_size = duration / num_bins
    segments = []
    start = None

    for b in range(num_bins):
        if probs[b] >= threshold:
            if start is None:
                start = b * bin_size
        else:
            if start is not None:
                segments.append([start, b * bin_size])
                start = None
    if start is not None:
        segments.append([start, duration])
    return segments


def compute_tiou(pred_segments, gt_segments):
    """计算 temporal IoU。"""
    if not pred_segments or not gt_segments:
        return 0.0

    def to_set(segs):
        s = set()
        for start, end in segs:
            t = start
            while t < end:
                s.add(round(t, 0.1) if end - start > 1 else round(t, 0.1))
                t += 0.1
        return s

    pred_set = to_set(pred_segments)
    gt_set = to_set(gt_segments)

    if not pred_set or not gt_set:
        return 0.0

    intersection = len(pred_set & gt_set)
    union = len(pred_set | gt_set)
    return intersection / union


# ============================================================
# 训练
# ============================================================

def train(args):
    device = torch.device("cuda")

    print("=" * 60)
    print("004 VoCo Temporal Head 训练")
    print(f"  K_seg={args.K_seg}, num_bins={args.num_bins}")
    print(f"  lr={args.lr}, epochs={args.epochs}")
    print("=" * 60)

    # 加载 VoCo 模型
    model, processor, tokenizer = setup_voco_model(
        K_seg=args.K_seg, device=device,
        gradient_checkpointing=False,  # 不训 LLM，不需要
    )

    # 加载 checkpoint
    load_voco_checkpoint(model, args.voco_checkpoint)

    # 冻结所有 VoCo 参数
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # Temporal Head
    temporal_head = TemporalHead(3584, num_bins=args.num_bins).to(device)
    n_params = sum(p.numel() for p in temporal_head.parameters())
    print(f"Temporal Head: {n_params:,} params")

    # 数据
    video_dirs = args.video_dirs.split(",")
    dataset = TemporalDataset(
        args.data_path, video_dirs,
        max_samples=args.max_samples,
        evidence_expand=args.evidence_expand,
    )

    if len(dataset) == 0:
        print("没有可用数据！")
        return

    # 按视频分组划分
    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"train={len(train_set)}, val={len(val_set)}")

    def collate(batch):
        return temporal_collate(
            batch, model, processor, tokenizer,
            fps=args.fps, frames_per_segment=args.frames_per_segment,
            max_frames=args.max_frames, num_bins=args.num_bins,
            evidence_expand=args.evidence_expand,
        )

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                              collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            collate_fn=collate)

    # pos_weight
    pos, neg = 0, 0
    for sample in dataset.samples[:min(500, len(dataset.samples))]:
        labels = compute_bin_labels(
            1.0, sample["evidence_segments"],
            args.num_bins, args.evidence_expand,
        )
        pos += sum(1 for l in labels if l > 0.5)
        neg += sum(1 for l in labels if l <= 0.5)
    pw = min(neg / max(pos, 1), 5.0)
    print(f"pos_weight: {pw:.2f} (pos={pos}, neg={neg})")
    pos_weight = torch.tensor([pw], device=device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        temporal_head.parameters(), lr=args.lr, weight_decay=0.01,
    )

    # 获取最后一层 decoder
    m = model.base.module if hasattr(model.base, "module") else model.base
    from peft import PeftModel
    if isinstance(m, PeftModel):
        inner = m.base_model.model
    else:
        inner = m
    layers = inner.model.language_model.layers
    last_layer = layers[-1]

    os.makedirs(args.output_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(args.epochs):
        temporal_head.train()
        total_loss = 0.0
        n_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in pbar:
            if batch is None:
                continue

            try:
                # VoCo forward（冻结），捕获 hidden states
                capture = VoCoHiddenCapture(last_layer)
                capture.install()

                # 注入 VoCo 4D mask via hooks
                voco_4d_mask = batch["voco_4d_mask"]
                hooks = []
                def make_hook(mask_4d):
                    def fn(module, args, kwargs):
                        kwargs["attention_mask"] = mask_4d
                        return args, kwargs
                    return fn
                for layer in layers:
                    hooks.append(layer.register_forward_pre_hook(
                        make_hook(voco_4d_mask), with_kwargs=True,
                    ))

                try:
                    with torch.no_grad():
                        inner.model(
                            inputs_embeds=batch["inputs_embeds"],
                            attention_mask=batch["attention_mask"],
                        )
                finally:
                    for h in hooks:
                        h.remove()

                # 提取 voco hidden states
                voco_hidden = capture.get_voco_hidden(batch["layout"])
                capture.remove()

                if voco_hidden is None:
                    continue

                # Temporal head forward
                logits = temporal_head(voco_hidden.float())
                loss = F.binary_cross_entropy_with_logits(
                    logits, batch["bin_labels"], pos_weight=pos_weight,
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    temporal_head.parameters(), 1.0,
                )
                optimizer.step()

                total_loss += loss.item()
                n_steps += 1
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    segs=batch["n_segments"],
                )

            except Exception as e:
                if n_steps < 3:
                    print(f"  [错误] {e}")
                capture.remove()
                continue

        avg_loss = total_loss / max(n_steps, 1)
        print(f"  Epoch {epoch + 1}: train_loss={avg_loss:.4f} "
              f"({n_steps} steps)")

        # Validation
        temporal_head.eval()
        val_loss = 0.0
        val_steps = 0
        val_tious = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="  Val", leave=False):
                if batch is None:
                    continue

                try:
                    capture = VoCoHiddenCapture(last_layer)
                    capture.install()
                    hooks = []
                    for layer in layers:
                        hooks.append(layer.register_forward_pre_hook(
                            make_hook(batch["voco_4d_mask"]),
                            with_kwargs=True,
                        ))
                    try:
                        inner.model(
                            inputs_embeds=batch["inputs_embeds"],
                            attention_mask=batch["attention_mask"],
                        )
                    finally:
                        for h in hooks:
                            h.remove()

                    voco_hidden = capture.get_voco_hidden(batch["layout"])
                    capture.remove()

                    if voco_hidden is None:
                        continue

                    logits = temporal_head(voco_hidden.float())
                    loss = F.binary_cross_entropy_with_logits(
                        logits, batch["bin_labels"],
                        pos_weight=pos_weight,
                    )
                    val_loss += loss.item()
                    val_steps += 1

                    # tIoU
                    probs = torch.sigmoid(logits[0]).cpu().tolist()
                    pred_segs = bins_to_segments(
                        probs, batch["duration"],
                    )
                    gt_segs = []
                    bin_labels = batch["bin_labels"][0].cpu().tolist()
                    gt_segs = bins_to_segments(
                        bin_labels, batch["duration"], threshold=0.5,
                    )
                    tiou = compute_tiou(pred_segs, gt_segs)
                    val_tious.append(tiou)

                except Exception:
                    capture.remove()
                    continue

        avg_val = val_loss / max(val_steps, 1)
        avg_tiou = sum(val_tious) / max(len(val_tious), 1)
        print(f"  Epoch {epoch + 1}: val_loss={avg_val:.4f}, "
              f"val_tIoU={avg_tiou:.4f} ({val_steps} steps)")

        # Save
        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                "temporal_head": temporal_head.state_dict(),
                "epoch": epoch + 1,
                "val_loss": avg_val,
                "val_tiou": avg_tiou,
                "num_bins": args.num_bins,
            }, os.path.join(args.output_dir, "temporal_best.pt"))
            print(f"  ★ best model saved (val_loss={avg_val:.4f})")

        torch.save({
            "temporal_head": temporal_head.state_dict(),
            "epoch": epoch + 1,
            "val_loss": avg_val,
            "val_tiou": avg_tiou,
            "num_bins": args.num_bins,
        }, os.path.join(args.output_dir, f"temporal_epoch{epoch + 1}.pt"))

    print(f"\n训练完成. best_val_loss={best_val:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--voco_checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--video_dirs", required=True)
    parser.add_argument("--output_dir", default="outputs_temporal")
    parser.add_argument("--K_seg", type=int, default=4)
    parser.add_argument("--num_bins", type=int, default=16)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--frames_per_segment", type=int, default=2)
    parser.add_argument("--max_frames", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--evidence_expand", type=float, default=1.0)
    parser.add_argument("--overfit", action="store_true")
    train(parser.parse_args())

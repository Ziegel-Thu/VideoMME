"""
Stage 2+3: Multi-frame video + Bottleneck + Temporal Head

输入: 视频 N 帧 + 问题 → [vision][question][latent][answer]
输出: L_ans (NLL on answer) + L_temp (BCE on segment labels)

用法:
  # Overfit test
  python train_video.py --overfit --epochs 30 --num_frames 4

  # 全量训练（只有 L_ans）
  python train_video.py --epochs 3 --num_frames 8

  # 加 Temporal Head
  python train_video.py --epochs 3 --num_frames 8 --temporal_head --temporal_weight 0.5
"""

import os
import sys
import json
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from mvp import (
    setup_model_and_tokenizer,
    build_bottleneck_mask,
    build_training_input_video,
)


# ============================================================
# Dataset
# ============================================================

class VideoQADataset(Dataset):
    """加载视频 QA 数据，支持 MCQ + temporal 标注"""

    def __init__(self, data_path, processor, tokenizer, latent_tokens,
                 num_frames=8, max_samples=None):
        self.processor = processor
        self.tokenizer = tokenizer
        self.latent_tokens = latent_tokens
        self.num_frames = num_frames

        self.samples = []
        with open(data_path) as f:
            for line in f:
                d = json.loads(line)
                if os.path.exists(d.get("video_path", "")):
                    self.samples.append(d)

        if max_samples:
            self.samples = self.samples[:max_samples]
        print(f"Dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "video_path": sample["video_path"],
            "question": sample["question"],
            "answer": sample["answer"],
            "evidence_segments": sample.get("evidence_segments", None),
        }


def collate_fn_video(batch, processor, tokenizer, latent_tokens, num_frames,
                      latent_token_ids):
    """构造 video batch"""
    from qwen_vl_utils import process_vision_info
    import decord

    texts = []
    all_video_inputs = []
    all_image_inputs = []
    temporal_labels = []
    latent_str = "".join(latent_tokens)

    for item in batch:
        # 采样帧
        try:
            vr = decord.VideoReader(item["video_path"])
            total = len(vr)
            fps = vr.get_avg_fps()
            duration = total / fps
            indices = [int(i * total / num_frames) for i in range(num_frames)]
            frames = [vr[idx].asnumpy() for idx in indices]
            frame_times = [idx / fps for idx in indices]
        except Exception:
            # fallback: 黑帧
            from PIL import Image
            frames = [Image.new("RGB", (224, 224), "black")] * num_frames
            duration = 30.0
            frame_times = [i * 4.0 for i in range(num_frames)]

        from PIL import Image
        import tempfile
        tmp_paths = []
        for i, f in enumerate(frames):
            if not isinstance(f, Image.Image):
                f = Image.fromarray(f)
            p = os.path.join(tempfile.gettempdir(), f"_vf_{os.getpid()}_{i}.jpg")
            f.save(p)
            tmp_paths.append(p)

        # Token 顺序: [video] [question] [latent] [answer]
        messages = [
            {"role": "user", "content": [
                {"type": "video", "video": tmp_paths, "fps": 1.0},
                {"type": "text", "text": f"{item['question']}\n{latent_str}"},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": item["answer"]},
            ]},
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)

        img_in, vid_in = process_vision_info(messages)
        all_image_inputs.extend(img_in if img_in else [])
        all_video_inputs.extend(vid_in if vid_in else [])

        # 清理
        for p in tmp_paths:
            os.remove(p)

        # Temporal labels: 哪些帧的时间段包含证据
        if item["evidence_segments"]:
            t_labels = []
            for ft in frame_times:
                is_evidence = False
                for ts, te in item["evidence_segments"]:
                    if ts <= ft <= te:
                        is_evidence = True
                        break
                t_labels.append(1.0 if is_evidence else 0.0)
            temporal_labels.append(t_labels)
        else:
            temporal_labels.append(None)

    inputs = processor(
        text=texts,
        images=all_image_inputs if all_image_inputs else None,
        videos=all_video_inputs if all_video_inputs else None,
        return_tensors="pt",
        padding=True,
    )

    # Labels: 只在 answer 部分计算 loss
    input_ids = inputs["input_ids"]
    labels = input_ids.clone()
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")

    for b in range(input_ids.shape[0]):
        ids = input_ids[b].tolist()
        last_start = 0
        for i, tid in enumerate(ids):
            if tid == im_start_id:
                last_start = i
        labels[b, :last_start + 3] = -100
        pad_id = tokenizer.pad_token_id or 0
        labels[b, input_ids[b] == pad_id] = -100

    inputs["labels"] = labels
    inputs["_temporal_labels"] = temporal_labels

    return inputs


# ============================================================
# Temporal Head
# ============================================================

class TemporalHead(nn.Module):
    """从 latent token hidden states 预测 per-frame 证据概率

    输入: latent_hidden (B, K, D) + num_bins
    输出: (B, num_bins) logits
    """

    def __init__(self, d_model, num_bins=16):
        super().__init__()
        self.num_bins = num_bins
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_bins),
        )

    def forward(self, latent_hidden):
        # latent_hidden: (B, K, D) → pool → (B, D)
        pooled = latent_hidden.mean(dim=1)
        return self.proj(pooled)  # (B, num_bins)


# ============================================================
# Training
# ============================================================

def train(args):
    device = torch.device("cuda")

    model, processor, tokenizer, latent_token_ids = setup_model_and_tokenizer(
        K=args.K, lora_r=args.lora_r
    )
    latent_tokens = [f"<latent_{i}>" for i in range(args.K)]

    dataset = VideoQADataset(
        args.data_path, processor, tokenizer, latent_tokens,
        num_frames=args.num_frames, max_samples=args.max_samples,
    )

    def collate(batch):
        return collate_fn_video(batch, processor, tokenizer, latent_tokens,
                                 args.num_frames, latent_token_ids)

    if args.overfit:
        subset = Subset(dataset, range(min(args.batch_size, len(dataset))))
        loader = DataLoader(subset, batch_size=args.batch_size, collate_fn=collate)
        val_loader = loader
        print(f"Overfit: {len(subset)} samples")
    else:
        n_val = min(500, len(dataset) // 10)
        n_train = len(dataset) - n_val
        train_set, val_set = random_split(dataset, [n_train, n_val],
                                           generator=torch.Generator().manual_seed(42))
        loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, collate_fn=collate)
        print(f"Train: {n_train}, Val: {n_val}")

    # Temporal Head
    temporal_head = None
    if args.temporal_head:
        temporal_head = TemporalHead(3584, num_bins=args.num_frames).to(device)
        print(f"Temporal Head: {sum(p.numel() for p in temporal_head.parameters())} params")

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    if temporal_head:
        params += list(temporal_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    lang = model.base_model.model.model.language_model
    os.makedirs(args.output_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        if temporal_head:
            temporal_head.train()
        total_loss = 0
        n = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            t_labels = batch.pop("_temporal_labels")
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Bottleneck hooks
            hooks = []
            if args.bottleneck:
                bn_mask = build_bottleneck_mask(
                    batch["input_ids"], latent_token_ids, enable_bottleneck=True
                ).to(device, dtype=torch.bfloat16)

                def lh(m):
                    def fn(module, args, kwargs):
                        kwargs["attention_mask"] = m
                        return args, kwargs
                    return fn
                for layer in lang.layers:
                    hooks.append(layer.register_forward_pre_hook(lh(bn_mask), with_kwargs=True))

            # Forward
            outputs = model(**batch, output_hidden_states=args.temporal_head)
            loss_ans = outputs.loss

            # Temporal loss
            loss_temp = torch.tensor(0.0, device=device)
            if temporal_head and outputs.hidden_states is not None:
                hidden = outputs.hidden_states[-1]  # (B, L, D)
                # 找 latent token 位置
                latent_set = set(latent_token_ids)
                for b in range(batch["input_ids"].shape[0]):
                    if t_labels[b] is None:
                        continue
                    ids = batch["input_ids"][b].tolist()
                    lat_pos = [i for i, t in enumerate(ids) if t in latent_set]
                    if not lat_pos:
                        continue
                    lat_hidden = hidden[b, lat_pos, :].unsqueeze(0)  # (1, K, D)
                    logits = temporal_head(lat_hidden)  # (1, num_bins)
                    target = torch.tensor([t_labels[b]], dtype=torch.float, device=device)
                    loss_temp = loss_temp + F.binary_cross_entropy_with_logits(logits, target)

                if batch["input_ids"].shape[0] > 0:
                    loss_temp = loss_temp / batch["input_ids"].shape[0]

            loss = loss_ans + args.temporal_weight * loss_temp

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            for h in hooks:
                h.remove()

            total_loss += loss.item()
            n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg = total_loss / max(n, 1)
        print(f"  Epoch {epoch+1}: train_loss={avg:.4f}")

        # Validation
        model.eval()
        if temporal_head:
            temporal_head.eval()
        val_loss = 0
        vn = 0
        with torch.no_grad():
            for batch in val_loader:
                batch.pop("_temporal_labels")
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                hooks = []
                if args.bottleneck:
                    bn_mask = build_bottleneck_mask(
                        batch["input_ids"], latent_token_ids, enable_bottleneck=True
                    ).to(device, dtype=torch.bfloat16)
                    def lh(m):
                        def fn(module, args, kwargs):
                            kwargs["attention_mask"] = m; return args, kwargs
                        return fn
                    for layer in lang.layers:
                        hooks.append(layer.register_forward_pre_hook(lh(bn_mask), with_kwargs=True))
                out = model(**batch)
                val_loss += out.loss.item()
                vn += 1
                for h in hooks:
                    h.remove()

        val_loss /= max(vn, 1)
        print(f"  Epoch {epoch+1}: val_loss={val_loss:.4f}")

        # Save
        from train_mvp import save_checkpoint
        save_checkpoint(model, tokenizer, latent_token_ids,
                        os.path.join(args.output_dir, f"checkpoint_epoch{epoch+1}.pt"),
                        epoch=epoch+1, val_loss=val_loss)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, tokenizer, latent_token_ids,
                            os.path.join(args.output_dir, "best_model.pt"),
                            epoch=epoch+1, val_loss=val_loss)
            if temporal_head:
                torch.save(temporal_head.state_dict(),
                           os.path.join(args.output_dir, "temporal_head.pt"))
            print(f"  ★ New best model (val_loss={val_loss:.4f})")

    print(f"\nDone. best_val_loss={best_val:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home/v-shuzheng/video/data/parsed/visual_qa_v2.jsonl")
    parser.add_argument("--output_dir", default=os.path.join(os.path.dirname(__file__), "outputs_video"))
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--bottleneck", action="store_true")
    parser.add_argument("--mask_mode", default="livr", choices=["livr", "answer_only"])
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--temporal_head", action="store_true")
    parser.add_argument("--temporal_weight", type=float, default=0.5)
    args = parser.parse_args()
    train(args)

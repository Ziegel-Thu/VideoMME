"""
MVP 训练: Latent Visual Bottleneck on Qwen2.5-VL-7B

用法:
  # Overfit test (1 batch)
  python train_mvp.py --overfit --epochs 20

  # Bottleneck vs No-Bottleneck 对比
  python train_mvp.py --bottleneck --epochs 5
  python train_mvp.py --epochs 5
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import argparse
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from mvp import setup_model_and_tokenizer, build_bottleneck_mask


# ============================================================
# Dataset
# ============================================================

class ImageQADataset(Dataset):
    """加载视觉 QA 数据（MCQ + temporal-spatial），用视频关键帧当图片"""

    def __init__(self, data_path, processor, tokenizer, latent_tokens, max_samples=None):
        self.processor = processor
        self.tokenizer = tokenizer
        self.latent_tokens = latent_tokens
        self.latent_str = "".join(latent_tokens)

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

    def _extract_keyframe(self, video_path, time_sec=None):
        """从视频中提取一帧作为图片"""
        import decord
        try:
            vr = decord.VideoReader(video_path)
            if time_sec is not None:
                fps = vr.get_avg_fps()
                frame_idx = min(int(time_sec * fps), len(vr) - 1)
            else:
                frame_idx = len(vr) // 2  # 取中间帧
            frame = vr[frame_idx].numpy()
            return Image.fromarray(frame)
        except Exception:
            return Image.new("RGB", (224, 224), color="gray")

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = self._extract_keyframe(sample["video_path"])

        return {
            "image": img,
            "question": sample["question"],
            "answer": sample["answer"],
        }


def collate_fn_factory(processor, tokenizer, latent_tokens):
    """返回一个 collate function，动态处理 batch"""
    from qwen_vl_utils import process_vision_info

    latent_str = "".join(latent_tokens)

    def collate_fn(batch):
        texts = []
        all_images = []

        for item in batch:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["image"]},
                        {"type": "text", "text": f"{latent_str}\n{item['question']}"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": item["answer"]}],
                },
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)

            image_inputs, _ = process_vision_info(messages)
            all_images.extend(image_inputs)

        inputs = processor(
            text=texts,
            images=all_images if all_images else None,
            return_tensors="pt",
            padding=True,
        )

        # 构造 labels: 只在 answer 部分计算 loss
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()

        im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

        for b in range(input_ids.shape[0]):
            ids = input_ids[b].tolist()
            # 找最后一个 <|im_start|>（assistant 的开始）
            last_start = 0
            for i, tid in enumerate(ids):
                if tid == im_start_id:
                    last_start = i
            # mask 掉 assistant header 之前的所有 token
            # assistant header: <|im_start|>assistant\n → 大约 3 个 token
            labels[b, :last_start + 3] = -100
            # 也 mask 掉 padding
            pad_id = tokenizer.pad_token_id or 0
            labels[b, input_ids[b] == pad_id] = -100

        inputs["labels"] = labels
        return inputs

    return collate_fn


# ============================================================
# Hook helpers
# ============================================================

def install_bottleneck_hooks(model, input_ids, latent_token_ids, device):
    """Build bottleneck mask and install attention hooks. Returns hook handles."""
    custom_mask = build_bottleneck_mask(
        input_ids, latent_token_ids, enable_bottleneck=True
    ).to(device, dtype=torch.bfloat16)

    def make_hook(mask_4d):
        def hook_fn(module, args, kwargs):
            if "attention_mask" in kwargs:
                kwargs["attention_mask"] = mask_4d
            return args, kwargs
        return hook_fn

    hooks = []
    lang_model = model.base_model.model.model.language_model
    for layer in lang_model.layers:
        h = layer.self_attn.register_forward_pre_hook(
            make_hook(custom_mask), with_kwargs=True
        )
        hooks.append(h)
    return hooks


# ============================================================
# Checkpoint helpers
# ============================================================

def save_checkpoint(model, tokenizer, latent_token_ids, path, epoch=None, val_loss=None):
    """Save LoRA weights + latent embeddings."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    embed = model.get_input_embeddings()
    latent_embeds = {tid: embed.weight.data[tid].cpu().clone() for tid in latent_token_ids}

    ckpt = {
        "lora_state_dict": {
            k: v.cpu() for k, v in model.state_dict().items()
            if "lora_" in k or "latent" in k
        },
        "latent_embeddings": latent_embeds,
        "latent_token_ids": latent_token_ids,
    }
    if epoch is not None:
        ckpt["epoch"] = epoch
    if val_loss is not None:
        ckpt["val_loss"] = val_loss

    torch.save(ckpt, path)
    print(f"  Checkpoint saved → {path}")


# ============================================================
# Validation loop
# ============================================================

@torch.no_grad()
def validate(model, val_loader, latent_token_ids, device, use_bottleneck):
    """Run validation, return average loss."""
    model.eval()
    total_loss = 0.0
    n = 0

    for batch in val_loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        hooks = []
        if use_bottleneck:
            hooks = install_bottleneck_hooks(model, batch["input_ids"], latent_token_ids, device)

        outputs = model(**batch)

        for h in hooks:
            h.remove()

        total_loss += outputs.loss.item()
        n += 1

    model.train()
    return total_loss / max(n, 1)


# ============================================================
# Training
# ============================================================

def train(args):
    device = torch.device("cuda")

    # 模型
    model, processor, tokenizer, latent_token_ids = setup_model_and_tokenizer(
        K=args.K, lora_r=args.lora_r
    )
    latent_tokens = [f"<latent_{i}>" for i in range(args.K)]

    # 数据
    dataset = ImageQADataset(
        args.evidence_path,
        processor, tokenizer, latent_tokens,
        max_samples=args.max_samples,
    )

    collate = collate_fn_factory(processor, tokenizer, latent_tokens)

    if args.overfit:
        from torch.utils.data import Subset
        subset = Subset(dataset, range(min(args.batch_size, len(dataset))))
        loader = DataLoader(subset, batch_size=args.batch_size, collate_fn=collate)
        val_loader = loader  # overfit mode: val on same data
        print(f"Overfit mode: {len(subset)} samples")
    else:
        # 90/10 train/val split (deterministic)
        from torch.utils.data import Subset
        n_total = len(dataset)
        n_val = max(1, int(n_total * 0.1))
        n_train = n_total - n_val
        train_subset = Subset(dataset, range(n_train))
        val_subset = Subset(dataset, range(n_train, n_total))
        loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, collate_fn=collate)
        print(f"Train: {n_train} samples, Val: {n_val} samples")

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01
    )

    # Output dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float("inf")

    # Training loop
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        n = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            # Move to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # 构造 bottleneck mask
            hooks = []
            if args.bottleneck:
                hooks = install_bottleneck_hooks(model, batch["input_ids"], latent_token_ids, device)

            outputs = model(**batch)
            loss = outputs.loss

            # 移除 hooks
            for h in hooks:
                h.remove()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train = total_loss / max(n, 1)
        print(f"  Epoch {epoch+1}: train_loss={avg_train:.4f}")

        # --- Validation ---
        val_loss = validate(model, val_loader, latent_token_ids, device, args.bottleneck)
        print(f"  Epoch {epoch+1}: val_loss={val_loss:.4f}")

        # Save epoch checkpoint
        save_checkpoint(
            model, tokenizer, latent_token_ids,
            os.path.join(output_dir, f"checkpoint_epoch{epoch+1}.pt"),
            epoch=epoch + 1, val_loss=val_loss,
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, tokenizer, latent_token_ids,
                os.path.join(output_dir, "best_model.pt"),
                epoch=epoch + 1, val_loss=val_loss,
            )
            print(f"  ★ New best model (val_loss={val_loss:.4f})")

    print(f"\nDone. bottleneck={args.bottleneck}, best_val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence_path",
                        default="/home/v-shuzheng/video/data/parsed/visual_qa.jsonl")
    parser.add_argument("--video_dir",
                        default="/home/v-shuzheng/video/data/open-o3-video/videos/stgr")
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--bottleneck", action="store_true")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--output_dir", default=os.path.join(os.path.dirname(__file__), "outputs"))
    args = parser.parse_args()
    train(args)

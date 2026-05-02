"""
Sanity Check: Verify Bottleneck Attention Mask Actually Works

Runs 4 conditions on a small set of samples:
  1. Normal         — real image + bottleneck ON  → baseline
  2. Blank vision   — black image + bottleneck ON → should be HIGH loss
  3. Shuffled vision — wrong image + bottleneck ON → should be HIGH loss
  4. No bottleneck  — real image + bottleneck OFF  → comparison baseline

Expected if bottleneck works:
  Normal ≈ No bottleneck   (latent tokens carry info)
  Blank >> Normal          (model needs the image)
  Shuffled >> Normal       (model uses the *specific* image)

Usage:
  python sanity_check.py
  python sanity_check.py --max_samples 10
  python sanity_check.py --checkpoint path/to/lora_checkpoint
"""

import os
import sys
import json
import random
import argparse
from copy import deepcopy

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from mvp import setup_model_and_tokenizer, build_bottleneck_mask
from train_mvp import ImageQADataset, collate_fn_factory


# ============================================================
# Condition-aware collate wrappers
# ============================================================

def _make_condition_collate(base_collate, condition, all_images=None):
    """Wrap the base collate_fn to modify images per condition.

    Args:
        base_collate: original collate_fn from collate_fn_factory
        condition: "normal" | "blank" | "shuffled" | "no_bottleneck"
        all_images: pool of PIL images for shuffled condition
    """

    def collate_fn(batch):
        if condition == "blank":
            for item in batch:
                w, h = item["image"].size
                item["image"] = Image.new("RGB", (w, h), color="black")
        elif condition == "shuffled":
            for item in batch:
                # Pick a random image that is NOT the original
                item["image"] = random.choice(all_images)

        return base_collate(batch)

    return collate_fn


# ============================================================
# Evaluation with bottleneck hook injection
# ============================================================

@torch.no_grad()
def evaluate_condition(
    model, loader, latent_token_ids, device,
    enable_bottleneck=True, desc="eval",
):
    """Run forward pass over loader, return avg NLL loss."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(loader, desc=desc, leave=False):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        hooks = []
        if enable_bottleneck:
            custom_mask = build_bottleneck_mask(
                batch["input_ids"], latent_token_ids, enable_bottleneck=True
            ).to(device, dtype=torch.bfloat16)

            def make_hook(mask_4d):
                def hook_fn(module, args, kwargs):
                    if "attention_mask" in kwargs:
                        kwargs["attention_mask"] = mask_4d
                    return args, kwargs
                return hook_fn

            lang_model = model.base_model.model.model.language_model
            for layer in lang_model.layers:
                h = layer.self_attn.register_forward_pre_hook(
                    make_hook(custom_mask), with_kwargs=True
                )
                hooks.append(h)

        outputs = model(**batch)

        for h in hooks:
            h.remove()

        # Count non-ignored tokens for proper averaging
        labels = batch["labels"]
        n_tokens = (labels != -100).sum().item()
        total_loss += outputs.loss.item() * max(n_tokens, 1)
        total_tokens += max(n_tokens, 1)

    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss


# ============================================================
# Main
# ============================================================

def main(args):
    device = torch.device("cuda")

    # --- Load model ---
    print("=" * 60)
    print("Loading model...")
    model, processor, tokenizer, latent_token_ids = setup_model_and_tokenizer(
        K=args.K, lora_r=args.lora_r,
    )
    latent_tokens = [f"<latent_{i}>" for i in range(args.K)]

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        # Load LoRA weights
        lora_state = ckpt.get("lora_state_dict", {})
        if lora_state:
            model.load_state_dict(lora_state, strict=False)
        # Load latent embeddings
        latent_embeds = ckpt.get("latent_embeddings", {})
        embed = model.get_input_embeddings()
        for tid, emb in latent_embeds.items():
            embed.weight.data[tid] = emb.to(embed.weight.device)
        print(f"  Loaded {len(lora_state)} LoRA params, {len(latent_embeds)} latent embeds")

    # --- Load dataset ---
    print("Loading dataset...")
    dataset = ImageQADataset(
        args.evidence_path,
        processor, tokenizer, latent_tokens,
        max_samples=args.max_samples,
    )

    if len(dataset) == 0:
        print("ERROR: No samples found. Check paths.")
        sys.exit(1)

    # Collect all images for shuffled condition (pre-extract once)
    print("Pre-extracting images for shuffled condition...")
    all_images = []
    for i in range(len(dataset)):
        item = dataset[i]
        all_images.append(item["image"])

    # Base collate
    base_collate = collate_fn_factory(processor, tokenizer, latent_tokens)

    # --- Define conditions ---
    conditions = [
        ("Normal (bn ON)",    "normal",      True),
        ("Blank vision",      "blank",       True),
        ("Shuffled vision",   "shuffled",    True),
        ("No bottleneck",     "no_bottleneck", False),
    ]

    results = {}

    for name, condition, bn_on in conditions:
        print(f"\n{'=' * 60}")
        print(f"Condition: {name}")
        print(f"  bottleneck={'ON' if bn_on else 'OFF'}, image={condition}")
        print("-" * 60)

        coll = _make_condition_collate(base_collate, condition, all_images)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=coll)

        avg_loss = evaluate_condition(
            model, loader, latent_token_ids, device,
            enable_bottleneck=bn_on, desc=name,
        )

        results[name] = avg_loss
        print(f"  → Avg NLL loss: {avg_loss:.4f}")

    # --- Report ---
    print("\n" + "=" * 60)
    print("SANITY CHECK RESULTS")
    print("=" * 60)
    print(f"{'Condition':<25} {'Avg NLL':>10}")
    print("-" * 37)
    for name, loss in results.items():
        print(f"{name:<25} {loss:>10.4f}")

    normal = results.get("Normal (bn ON)", 0)
    blank = results.get("Blank vision", 0)
    shuffled = results.get("Shuffled vision", 0)
    no_bn = results.get("No bottleneck", 0)

    print("\n" + "-" * 37)
    print("Deltas vs Normal:")
    if normal > 0:
        print(f"  Blank    - Normal = {blank - normal:+.4f}")
        print(f"  Shuffled - Normal = {shuffled - normal:+.4f}")
        print(f"  No BN    - Normal = {no_bn - normal:+.4f}")

    # Interpretation
    print("\nInterpretation:")
    if blank > normal * 1.05:
        print("  ✅ Blank >> Normal → model needs the image (good)")
    else:
        print("  ⚠️  Blank ≈ Normal → vision may not be used (bad)")

    if shuffled > normal * 1.05:
        print("  ✅ Shuffled >> Normal → model uses specific image (good)")
    else:
        print("  ⚠️  Shuffled ≈ Normal → model ignores image content (bad)")

    if abs(no_bn - normal) < normal * 0.2:
        print("  ✅ No BN ≈ Normal → latent tokens carry info well (good)")
    else:
        gap_pct = abs(no_bn - normal) / max(normal, 1e-6) * 100
        print(f"  ℹ️  No BN vs Normal differ by {gap_pct:.1f}% — "
              "latent tokens may need more training")

    print(f"\nNote: With an untrained model, Blank/Shuffled deltas may be "
          "small.\nRe-run after training for definitive results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bottleneck sanity check")
    parser.add_argument("--evidence_path",
                        default="/home/v-shuzheng/video/data/parsed/visual_qa.jsonl")
    parser.add_argument("--video_dir",
                        default="/home/v-shuzheng/video/data/open-o3-video/videos/stgr")
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=20)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained LoRA checkpoint (optional)")
    args = parser.parse_args()
    main(args)

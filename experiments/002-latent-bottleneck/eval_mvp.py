"""
MVP 评估: Latent Visual Bottleneck on Qwen2.5-VL-7B

用法:
  # Evaluate with bottleneck
  python eval_mvp.py --checkpoint outputs/best_model.pt --bottleneck

  # Evaluate without bottleneck
  python eval_mvp.py --checkpoint outputs/best_model.pt

  # Custom max generation length
  python eval_mvp.py --checkpoint outputs/best_model.pt --bottleneck --max_new_tokens 128
"""

import os
import sys
import json
import torch
import argparse
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from mvp import setup_model_and_tokenizer, build_bottleneck_mask
from train_mvp import ImageQADataset, collate_fn_factory


# ============================================================
# Checkpoint loading
# ============================================================

def load_checkpoint(model, checkpoint_path, latent_token_ids):
    """Load LoRA weights + latent embeddings from a checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Restore LoRA weights
    lora_sd = ckpt["lora_state_dict"]
    model_sd = model.state_dict()
    matched = 0
    for k, v in lora_sd.items():
        if k in model_sd:
            model_sd[k].copy_(v)
            matched += 1
    model.load_state_dict(model_sd, strict=False)
    print(f"  Loaded {matched} LoRA parameters from checkpoint")

    # Restore latent embeddings
    embed = model.get_input_embeddings()
    latent_embeds = ckpt.get("latent_embeddings", {})
    for tid, emb in latent_embeds.items():
        embed.weight.data[tid].copy_(emb)
    print(f"  Restored {len(latent_embeds)} latent embeddings")

    info = {}
    if "epoch" in ckpt:
        info["epoch"] = ckpt["epoch"]
    if "val_loss" in ckpt:
        info["val_loss"] = ckpt["val_loss"]
    return info


# ============================================================
# Hook helpers (reused from train_mvp)
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
# Generation collate (no labels, for generation input)
# ============================================================

def gen_collate_fn_factory(processor, tokenizer, latent_tokens):
    """Collate that builds generation-ready inputs (prompt only, no answer)."""
    from qwen_vl_utils import process_vision_info

    latent_str = "".join(latent_tokens)

    def collate_fn(batch):
        texts = []
        all_images = []
        answers = []

        for item in batch:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": item["image"]},
                        {"type": "text", "text": f"{latent_str}\n{item['question']}"},
                    ],
                },
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text)

            image_inputs, _ = process_vision_info(messages)
            all_images.extend(image_inputs)
            answers.append(item["answer"])

        inputs = processor(
            text=texts,
            images=all_images if all_images else None,
            return_tensors="pt",
            padding=True,
        )
        return inputs, answers

    return collate_fn


# ============================================================
# Metrics
# ============================================================

def compute_token_accuracy(pred_ids, ref_ids):
    """Token-level accuracy: fraction of positions where pred matches ref."""
    min_len = min(len(pred_ids), len(ref_ids))
    if min_len == 0:
        return 0.0
    matches = sum(1 for p, r in zip(pred_ids[:min_len], ref_ids[:min_len]) if p == r)
    return matches / max(len(ref_ids), 1)


def normalize_text(text):
    """Normalize text for exact match comparison."""
    return text.strip().lower()


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(args):
    device = torch.device("cuda")

    # --- Load model ---
    print("=== Loading model ===")
    model, processor, tokenizer, latent_token_ids = setup_model_and_tokenizer(
        K=args.K, lora_r=args.lora_r
    )
    latent_tokens = [f"<latent_{i}>" for i in range(args.K)]

    # --- Load checkpoint ---
    if args.checkpoint:
        print(f"\n=== Loading checkpoint: {args.checkpoint} ===")
        ckpt_info = load_checkpoint(model, args.checkpoint, latent_token_ids)
        print(f"  Checkpoint info: {ckpt_info}")

    model.eval()

    # --- Load data (val split = last 10%) ---
    print("\n=== Loading data ===")
    dataset = ImageQADataset(
        args.evidence_path,
        args.video_dir,
        processor, tokenizer, latent_tokens,
        max_samples=args.max_samples,
    )

    n_total = len(dataset)
    n_val = max(1, int(n_total * 0.1))
    n_train = n_total - n_val
    val_subset = Subset(dataset, range(n_train, n_total))
    print(f"Val set: {len(val_subset)} samples (last 10%)")

    # --- 1) NLL loss on val set ---
    print("\n=== Computing NLL loss ===")
    loss_collate = collate_fn_factory(processor, tokenizer, latent_tokens)
    loss_loader = DataLoader(val_subset, batch_size=1, collate_fn=loss_collate)

    total_loss = 0.0
    n_loss = 0
    for batch in tqdm(loss_loader, desc="NLL loss"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        hooks = []
        if args.bottleneck:
            hooks = install_bottleneck_hooks(model, batch["input_ids"], latent_token_ids, device)

        outputs = model(**batch)

        for h in hooks:
            h.remove()

        total_loss += outputs.loss.item()
        n_loss += 1

    avg_nll = total_loss / max(n_loss, 1)
    print(f"  Average NLL loss: {avg_nll:.4f}")

    # --- 2) Generation + metrics ---
    print("\n=== Generating answers ===")
    gen_collate = gen_collate_fn_factory(processor, tokenizer, latent_tokens)
    gen_loader = DataLoader(val_subset, batch_size=1, collate_fn=gen_collate)

    results = []
    total_token_acc = 0.0
    exact_matches = 0
    n_gen = 0

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    for inputs, answers in tqdm(gen_loader, desc="Generating"):
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}
        ref_answer = answers[0]

        hooks = []
        if args.bottleneck:
            hooks = install_bottleneck_hooks(model, inputs["input_ids"], latent_token_ids, device)

        gen_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=1.0,
            eos_token_id=[im_end_id, tokenizer.eos_token_id],
        )

        for h in hooks:
            h.remove()

        # Extract only the generated tokens (after the prompt)
        prompt_len = inputs["input_ids"].shape[1]
        new_ids = gen_ids[0, prompt_len:].tolist()

        # Remove trailing eos/im_end tokens
        clean_ids = []
        for tid in new_ids:
            if tid in (im_end_id, tokenizer.eos_token_id):
                break
            clean_ids.append(tid)

        pred_text = tokenizer.decode(clean_ids, skip_special_tokens=True)
        ref_ids = tokenizer.encode(ref_answer, add_special_tokens=False)

        # Token accuracy
        token_acc = compute_token_accuracy(clean_ids, ref_ids)
        total_token_acc += token_acc

        # Exact match
        is_exact = normalize_text(pred_text) == normalize_text(ref_answer)
        if is_exact:
            exact_matches += 1

        results.append({
            "question": val_subset[n_gen]["question"],
            "reference": ref_answer,
            "predicted": pred_text,
            "token_accuracy": round(token_acc, 4),
            "exact_match": is_exact,
        })

        n_gen += 1

    avg_token_acc = total_token_acc / max(n_gen, 1)
    exact_match_rate = exact_matches / max(n_gen, 1)

    # --- Summary ---
    print("\n" + "=" * 50)
    print("  EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Checkpoint:       {args.checkpoint or 'none (base model)'}")
    print(f"  Bottleneck:       {args.bottleneck}")
    print(f"  Val samples:      {n_gen}")
    print(f"  Avg NLL loss:     {avg_nll:.4f}")
    print(f"  Token accuracy:   {avg_token_acc:.4f}")
    print(f"  Exact match rate: {exact_match_rate:.4f} ({exact_matches}/{n_gen})")
    print("=" * 50)

    # --- Save results ---
    output = {
        "config": {
            "checkpoint": args.checkpoint,
            "bottleneck": args.bottleneck,
            "K": args.K,
            "max_new_tokens": args.max_new_tokens,
            "n_val": n_gen,
        },
        "metrics": {
            "avg_nll_loss": round(avg_nll, 4),
            "token_accuracy": round(avg_token_acc, 4),
            "exact_match_rate": round(exact_match_rate, 4),
            "exact_matches": exact_matches,
        },
        "samples": results,
    }

    output_path = args.output or os.path.join(
        os.path.dirname(__file__), "outputs",
        f"eval_results{'_bn' if args.bottleneck else '_nobn'}.json"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Latent Visual Bottleneck model")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (best_model.pt)")
    parser.add_argument("--evidence_path",
                        default="/home/v-shuzheng/video/data/parsed/temporal_evidence.jsonl")
    parser.add_argument("--video_dir",
                        default="/home/v-shuzheng/video/data/open-o3-video/videos/stgr")
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--bottleneck", action="store_true",
                        help="Enable bottleneck attention mask during eval")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: outputs/eval_results_[bn|nobn].json)")
    args = parser.parse_args()
    evaluate(args)

"""
Multi-frame video sanity check

用视频输入验证 bottleneck 是否让 latent token 承载视觉信息。
4 个条件: normal / zero_vision / bn_normal / bn_zero
"""

import torch
import sys
import os
import json
import glob
import argparse
import decord
from PIL import Image
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(__file__))
from mvp import (
    setup_model_and_tokenizer,
    build_bottleneck_mask,
    build_training_input_video,
)
from train_video import VideoQADataset, collate_fn_video


def run_sanity_video(checkpoint_path, data_path, num_frames=4, max_samples=20, K=8):
    model, processor, tokenizer, latent_token_ids = setup_model_and_tokenizer(K=K)
    latent_tokens = [f"<latent_{i}>" for i in range(8)]
    device = next(model.parameters()).device

    # 加载 checkpoint
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["lora_state_dict"], strict=False)
    embed = model.get_input_embeddings()
    for tid, emb in ckpt["latent_embeddings"].items():
        embed.weight.data[tid] = emb.to(embed.weight.device)
    model.eval()

    dataset = VideoQADataset(
        data_path, processor, tokenizer, latent_tokens,
        num_frames=num_frames, max_samples=max_samples,
    )

    def collate(batch):
        return collate_fn_video(batch, processor, tokenizer, latent_tokens,
                                 num_frames, latent_token_ids)

    loader = DataLoader(dataset, batch_size=1, collate_fn=collate)
    lang = model.base_model.model.model.language_model

    results = {"normal": [], "zero_no_bn": [], "bn_normal": [], "bn_zero": []}

    for batch in loader:
        batch.pop("_temporal_labels")
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        ids = batch["input_ids"][0]

        # Vision mask: vision_start(151652) 到 vision_end(151653) 之间
        vmask = torch.zeros(len(ids), dtype=torch.bool, device=device)
        in_v = False
        for i, t in enumerate(ids.tolist()):
            if t == 151652:
                in_v = True
            if in_v:
                vmask[i] = True
            if t == 151653:
                in_v = False

        bn = build_bottleneck_mask(
            batch["input_ids"], latent_token_ids, enable_bottleneck=True
        ).to(device, dtype=torch.bfloat16)

        def zero_hook(module, args, kwargs):
            if "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is not None:
                e = kwargs["inputs_embeds"].clone()
                e[0, vmask] = 0.0
                kwargs["inputs_embeds"] = e
            return args, kwargs

        def layer_hook(m):
            def fn(module, args, kwargs):
                kwargs["attention_mask"] = m
                return args, kwargs
            return fn

        # 1. Normal
        with torch.no_grad():
            results["normal"].append(model(**batch).loss.item())

        # 2. Zero vision, no BN
        h = lang.register_forward_pre_hook(zero_hook, with_kwargs=True)
        with torch.no_grad():
            results["zero_no_bn"].append(model(**batch).loss.item())
        h.remove()

        # 3. BN normal
        hooks = [layer.register_forward_pre_hook(layer_hook(bn), with_kwargs=True)
                 for layer in lang.layers]
        with torch.no_grad():
            results["bn_normal"].append(model(**batch).loss.item())
        for h in hooks:
            h.remove()

        # 4. BN + zero vision
        hz = lang.register_forward_pre_hook(zero_hook, with_kwargs=True)
        hooks = [layer.register_forward_pre_hook(layer_hook(bn), with_kwargs=True)
                 for layer in lang.layers]
        with torch.no_grad():
            results["bn_zero"].append(model(**batch).loss.item())
        hz.remove()
        for h in hooks:
            h.remove()

    # 结果
    print("=" * 50)
    print("SANITY CHECK (multi-frame video)")
    print("=" * 50)
    for c, l in results.items():
        print(f"  {c:15s}: {sum(l)/len(l):.4f}")

    zd = sum(results["zero_no_bn"])/len(results["zero_no_bn"]) - sum(results["normal"])/len(results["normal"])
    bd = sum(results["bn_zero"])/len(results["bn_zero"]) - sum(results["bn_normal"])/len(results["bn_normal"])
    print(f"\n  Zero Δ (no BN): {zd:+.4f}  {'✅ PASS' if zd > 0.1 else '❌ FAIL'}")
    print(f"  Zero Δ (BN):    {bd:+.4f}  {'✅ PASS' if bd > 0.1 else '❌ FAIL'}")

    return zd > 0.1 and bd > 0.1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", default="/home/v-shuzheng/video/data/parsed/visual_qa_v2.jsonl")
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=20)
    args = parser.parse_args()
    passed = run_sanity_video(args.checkpoint, args.data_path, args.num_frames, args.max_samples, args.K)
    sys.exit(0 if passed else 1)

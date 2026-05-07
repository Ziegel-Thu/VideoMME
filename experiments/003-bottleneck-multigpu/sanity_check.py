"""
Sanity Check: 验证 bottleneck 是否让 latent tokens 承载视觉信息

测试 4 个条件:
  1. normal — 正常 forward
  2. zero_no_bn — 清零 vision embeddings，无 bottleneck
  3. bn_normal — 正常 vision + bottleneck mask
  4. bn_zero — 清零 vision + bottleneck mask

判断标准:
  - BN Zero Δ = bn_zero_loss - bn_normal_loss > 0.1 → bottleneck 生效
  - 说明模型依赖通过 latent 获取视觉信息

用法:
  python sanity_check.py \
    --checkpoint outputs_local_20k/checkpoint_epoch1.pt \
    --data_path /path/to/visual_qa_v3_short_val.jsonl \
    --video_dirs /path/to/videos \
    --K 32 --num_frames 4 --max_samples 30
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader, Subset

from model import (
    setup_model_and_tokenizer,
    build_bottleneck_mask,
    install_bottleneck_hooks,
    load_checkpoint,
    get_language_model_layers,
    VISION_START_ID,
    VISION_END_ID,
)
from data import VideoQADataset, collate_fn


def run_sanity(args):
    device = torch.device("cuda")

    # 加载模型
    model, processor, tokenizer, latent_token_ids = setup_model_and_tokenizer(
        K=args.K, device=device,
    )
    latent_tokens = [f"<latent_{i}>" for i in range(args.K)]

    # 加载 checkpoint
    epoch, val_loss = load_checkpoint(model, args.checkpoint)
    print(f"Checkpoint: epoch={epoch}, val_loss={val_loss}")
    model.eval()

    # 数据
    video_dirs = args.video_dirs.split(",")
    dataset = VideoQADataset(args.data_path, video_dirs,
                             max_samples=args.max_samples)

    def make_collate(proc, tok, lt, nf, ltids):
        def wrapper(batch):
            return collate_fn(batch, proc, tok, lt, nf, ltids)
        return wrapper

    collate = make_collate(processor, tokenizer, latent_tokens,
                           args.num_frames, latent_token_ids)
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate)

    layers = get_language_model_layers(model)
    # 需要访问 language_model 的 forward_pre_hook 来清零 vision
    m = model.module if hasattr(model, "module") else model
    lang = m.base_model.model.model.language_model

    results = {"normal": [], "zero_no_bn": [], "bn_normal": [], "bn_zero": []}

    for i, batch in enumerate(loader):
        batch.pop("_temporal_labels", None)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        ids = batch["input_ids"][0]

        # 找 vision token 位置
        vmask = torch.zeros(len(ids), dtype=torch.bool, device=device)
        in_v = False
        for j, t in enumerate(ids.tolist()):
            if t == VISION_START_ID:
                in_v = True
            if in_v:
                vmask[j] = True
            if t == VISION_END_ID:
                in_v = False

        # 构造 bottleneck mask
        bn_mask = build_bottleneck_mask(
            batch["input_ids"], latent_token_ids, enable_bottleneck=True,
        ).to(device, dtype=torch.bfloat16)

        # 清零 vision embeddings 的 hook
        def zero_hook(module, args, kwargs):
            if "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is not None:
                e = kwargs["inputs_embeds"].clone()
                e[0, vmask] = 0.0
                kwargs["inputs_embeds"] = e
            return args, kwargs

        with torch.no_grad():
            # 1. Normal
            results["normal"].append(model(**batch).loss.item())

            # 2. Zero vision, no BN
            h = lang.register_forward_pre_hook(zero_hook, with_kwargs=True)
            results["zero_no_bn"].append(model(**batch).loss.item())
            h.remove()

            # 3. BN normal
            hooks = install_bottleneck_hooks(layers, bn_mask)
            results["bn_normal"].append(model(**batch).loss.item())
            for h in hooks:
                h.remove()

            # 4. BN + zero vision
            hz = lang.register_forward_pre_hook(zero_hook, with_kwargs=True)
            hooks = install_bottleneck_hooks(layers, bn_mask)
            results["bn_zero"].append(model(**batch).loss.item())
            hz.remove()
            for h in hooks:
                h.remove()

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(loader)} 条已处理")

    # 输出结果
    print("\n" + "=" * 50)
    print("SANITY CHECK")
    print("=" * 50)
    for cond, losses in results.items():
        avg = sum(losses) / len(losses)
        print(f"  {cond:15s}: {avg:.4f}")

    avg_normal = sum(results["normal"]) / len(results["normal"])
    avg_zero = sum(results["zero_no_bn"]) / len(results["zero_no_bn"])
    avg_bn = sum(results["bn_normal"]) / len(results["bn_normal"])
    avg_bn_zero = sum(results["bn_zero"]) / len(results["bn_zero"])

    zero_delta = avg_zero - avg_normal
    bn_delta = avg_bn_zero - avg_bn

    print(f"\n  Zero Δ (no BN): {zero_delta:+.4f}  "
          f"{'✅ PASS' if zero_delta > 0.1 else '❌ FAIL'}")
    print(f"  Zero Δ (BN):    {bn_delta:+.4f}  "
          f"{'✅ PASS' if bn_delta > 0.1 else '❌ FAIL'}")

    return zero_delta > 0.1 and bn_delta > 0.1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bottleneck Sanity Check")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--video_dirs", required=True)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=30)
    args = parser.parse_args()
    passed = run_sanity(args)
    sys.exit(0 if passed else 1)

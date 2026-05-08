"""
MCQ 准确率评测

用 logits 判断答案首 token 是否匹配（不用 generate，避免 KV cache 和 mask 冲突）。

用法:
  python eval_mcq.py \
    --checkpoint outputs_local_20k/checkpoint_epoch1.pt \
    --data_path ../../data/parsed/visual_qa_v3_short_test.jsonl \
    --video_dirs ../../data/llava-video/0_30_s_academic_v0_1 \
    --K 32 --num_frames 4 --max_samples 200
"""

import os
import sys
import json
import argparse
import torch
from tqdm import tqdm

from model import (
    setup_model_and_tokenizer,
    build_bottleneck_mask,
    install_bottleneck_hooks,
    load_checkpoint,
    get_language_model_layers,
)
from data import VideoQADataset, collate_fn


def eval_mcq(args):
    device = torch.device("cuda")

    model, processor, tokenizer, latent_token_ids = setup_model_and_tokenizer(
        K=args.K, device=device,
    )
    latent_tokens = [f"<latent_{i}>" for i in range(args.K)]

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

    layers = get_language_model_layers(model)

    # 找 answer 起始位置
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")

    correct = 0
    total = 0
    total_loss = 0.0
    errors = 0

    for i, sample in enumerate(tqdm(dataset, desc="评测中")):
        try:
            batch = collate([sample])
            batch.pop("_temporal_labels", None)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            hooks = []
            try:
                if args.bottleneck:
                    bn_mask = build_bottleneck_mask(
                        batch["input_ids"], latent_token_ids,
                        enable_bottleneck=True,
                    ).to(device, dtype=torch.bfloat16)
                    hooks = install_bottleneck_hooks(layers, bn_mask)

                with torch.no_grad():
                    out = model(**batch)
                    if out.loss is not None:
                        total_loss += out.loss.item()
            finally:
                for h in hooks:
                    h.remove()

            # 找 answer 首 token 位置
            ids = batch["input_ids"][0].tolist()
            im_starts = [j for j, t in enumerate(ids) if t == im_start_id]
            if not im_starts:
                errors += 1
                continue

            # 最后一个 <|im_start|> 是 assistant
            ast = im_starts[-1]
            assistant_prefix = tokenizer.encode("assistant\n",
                                                add_special_tokens=False)
            ans_pos = ast + 1 + len(assistant_prefix)

            if ans_pos >= len(ids):
                errors += 1
                continue

            # logits[ans_pos - 1] 预测 ans_pos 位置的 token
            pred_token = out.logits[0, ans_pos - 1, :].argmax().item()
            gt_token = ids[ans_pos]

            if pred_token == gt_token:
                correct += 1
            total += 1

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"  [错误] {e}")
            continue

    acc = correct / max(total, 1)
    avg_loss = total_loss / max(total, 1)

    print(f"\n{'=' * 50}")
    print(f"MCQ 评测结果")
    print(f"{'=' * 50}")
    print(f"  准确率: {acc:.2%} ({correct}/{total})")
    print(f"  平均 loss: {avg_loss:.4f}")
    if errors > 0:
        print(f"  跳过 {errors} 条出错样本")

    return acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCQ 准确率评测")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--video_dirs", required=True)
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--bottleneck", action="store_true")
    args = parser.parse_args()
    eval_mcq(args)

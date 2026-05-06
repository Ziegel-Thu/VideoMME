"""
Temporal Grounding 评测

评测 temporal head 的时间段预测准确率。
指标: tIoU (temporal IoU), Recall@0.3/0.5/0.7

用法:
  python eval_temporal.py \
    --checkpoint outputs_k32_temporal/best_model.pt \
    --temporal_head_path outputs_k32_temporal/temporal_head.pt
"""

import os
import sys
import json
import glob
import torch
import torch.nn.functional as F
import argparse
import decord
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from mvp import setup_model_and_tokenizer, build_bottleneck_mask, build_training_input_video
from train_video import TemporalHead


def compute_tiou(pred_segments, gt_segments):
    """计算 temporal IoU between predicted and ground truth segments"""
    if not pred_segments or not gt_segments:
        return 0.0

    # 取预测的最大连续区间
    pred_start = min(s[0] for s in pred_segments)
    pred_end = max(s[1] for s in pred_segments)

    gt_start = min(s[0] for s in gt_segments)
    gt_end = max(s[1] for s in gt_segments)

    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    intersection = max(0, inter_end - inter_start)

    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def predict_temporal(model, temporal_head, processor, tokenizer, latent_tokens,
                     latent_token_ids, video_path, question, num_frames, device):
    """用 temporal head 预测时间段"""
    inputs = build_training_input_video(
        processor, tokenizer, latent_tokens,
        video_path, question, "placeholder",
        num_frames=num_frames,
    )

    # 获取视频时长
    duration = inputs.get("_video_duration", 30.0)
    frame_timestamps = inputs.get("_frame_timestamps", [])

    inputs_dev = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

    # Bottleneck hooks
    lang = model.base_model.model.model.language_model

    bn_mask = build_bottleneck_mask(
        inputs["input_ids"], latent_token_ids, enable_bottleneck=True
    ).to(device, dtype=torch.bfloat16)

    def lh(m):
        def fn(module, args, kwargs):
            kwargs["attention_mask"] = m
            return args, kwargs
        return fn

    hooks = [layer.register_forward_pre_hook(lh(bn_mask), with_kwargs=True)
             for layer in lang.layers]

    # 捕获最后一层 hidden state
    captured = {}
    def capture_hook(module, input, output):
        captured["hidden"] = output[0]
    hooks.append(lang.layers[-1].register_forward_hook(capture_hook))

    with torch.no_grad():
        model(**inputs_dev)

    for h in hooks:
        h.remove()

    if "hidden" not in captured:
        return [], duration

    hidden = captured["hidden"]  # (1, L, D)

    # 找 latent token 位置
    ids = inputs["input_ids"][0].tolist()
    latent_set = set(latent_token_ids)
    lat_pos = [i for i, t in enumerate(ids) if t in latent_set]

    if not lat_pos:
        return [], duration

    lat_hidden = hidden[0, lat_pos, :].unsqueeze(0).float()  # (1, K, D)
    logits = temporal_head(lat_hidden)  # (1, num_bins)
    probs = torch.sigmoid(logits[0]).cpu()

    # 将帧级概率转为时间段
    threshold = 0.5
    pred_segments = []
    in_segment = False
    seg_start = 0

    for i, p in enumerate(probs):
        if i < len(frame_timestamps):
            t = frame_timestamps[i]
        else:
            t = i * (duration / len(probs))

        if p > threshold and not in_segment:
            seg_start = t
            in_segment = True
        elif p <= threshold and in_segment:
            seg_end = t
            pred_segments.append([seg_start, seg_end])
            in_segment = False

    if in_segment:
        pred_segments.append([seg_start, duration])

    return pred_segments, duration, probs.tolist()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    print("加载模型...")
    model, processor, tokenizer, latent_token_ids = setup_model_and_tokenizer(K=args.K)
    latent_tokens = [f"<latent_{i}>" for i in range(args.K)]

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["lora_state_dict"], strict=False)
    embed = model.get_input_embeddings()
    for tid, emb in ckpt["latent_embeddings"].items():
        embed.weight.data[tid] = emb.to(embed.weight.device)
    model.eval()

    # 加载 temporal head
    temporal_head = TemporalHead(3584, num_bins=args.num_frames).to(device)
    temporal_head.load_state_dict(torch.load(args.temporal_head_path, map_location=device, weights_only=True))
    temporal_head.eval()
    print(f"Temporal Head 加载完成 (num_bins={args.num_frames})")

    # 加载评测数据
    video_index = {}
    for mp4 in glob.glob("/home/v-shuzheng/video/data/open-o3-video/videos/stgr/**/*.mp4", recursive=True):
        video_index[os.path.basename(mp4)] = mp4
    for mp4 in glob.glob("/home/v-shuzheng/video/data/llava-video/**/*.mp4", recursive=True):
        video_index[os.path.basename(mp4)] = mp4

    samples = []
    with open(args.data_path) as f:
        for line in f:
            d = json.loads(line)
            vname = os.path.basename(d["video_path"])
            if vname in video_index and d.get("evidence_segments"):
                d["_video_local"] = video_index[vname]
                samples.append(d)

    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"评测样本: {len(samples)}")

    # 评测
    tious = []
    recalls = {0.3: 0, 0.5: 0, 0.7: 0}
    results = []

    for sample in tqdm(samples, desc="评测"):
        try:
            pred_segs, duration, probs = predict_temporal(
                model, temporal_head, processor, tokenizer, latent_tokens,
                latent_token_ids, sample["_video_local"], sample["question"],
                args.num_frames, device,
            )
        except Exception as e:
            continue

        gt_segs = sample["evidence_segments"]
        tiou = compute_tiou(pred_segs, gt_segs)
        tious.append(tiou)

        for thresh in recalls:
            if tiou >= thresh:
                recalls[thresh] += 1

        results.append({
            "question": sample["question"][:60],
            "gt": gt_segs,
            "pred": pred_segs,
            "tiou": tiou,
            "probs": probs[:8],
        })

    # 汇总
    n = len(tious)
    print("\n" + "=" * 50)
    print("TEMPORAL GROUNDING RESULTS")
    print("=" * 50)
    print(f"  样本数: {n}")
    print(f"  平均 tIoU: {sum(tious)/max(n,1):.4f}")
    for thresh in sorted(recalls):
        print(f"  Recall@{thresh}: {recalls[thresh]/max(n,1):.4f} ({recalls[thresh]}/{n})")

    # 打印几个样例
    print("\n--- 样例 ---")
    for r in results[:5]:
        print(f"  Q: {r['question']}")
        print(f"  GT: {r['gt']}, Pred: {r['pred']}, tIoU: {r['tiou']:.3f}")
        print()

    # 保存结果
    out_path = os.path.join(os.path.dirname(args.checkpoint), "temporal_eval.json")
    with open(out_path, "w") as f:
        json.dump({
            "mean_tiou": sum(tious) / max(n, 1),
            "recalls": {str(k): v / max(n, 1) for k, v in recalls.items()},
            "n_samples": n,
            "results": results[:50],
        }, f, indent=2, ensure_ascii=False)
    print(f"\n结果保存到: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--temporal_head_path", required=True)
    parser.add_argument("--data_path", default="/home/v-shuzheng/video/data/parsed/temporal_evidence.jsonl")
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=200)
    args = parser.parse_args()
    main(args)

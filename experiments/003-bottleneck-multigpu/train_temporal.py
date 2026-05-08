"""
Stage 3: Temporal Head 训练

冻结 Stage 2 的 LoRA + latent embeddings，只训练 Temporal Head。
从 latent tokens 的 hidden states 预测 evidence 时间 bins。

用法:
  # 单卡
  python train_temporal.py \
    --stage2_checkpoint outputs/best_model.pt \
    --data_path ../../data/parsed/temporal_evidence.jsonl \
    --video_dirs ../../data/llava-video/0_30_s_academic_v0_1,../../data/open-o3-video/videos/stgr \
    --output_dir outputs_temporal \
    --K 32 --num_frames 8 --num_bins 8 --epochs 10

  # 多卡
  torchrun --nproc_per_node=4 train_temporal.py \
    --stage2_checkpoint outputs/best_model.pt \
    --data_path ... --video_dirs ... --output_dir ... \
    --K 48 --num_frames 12 --num_bins 16 --epochs 10
"""

import os
import json
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from model import (
    setup_model_and_tokenizer,
    build_bottleneck_mask,
    install_bottleneck_hooks,
    load_checkpoint,
    get_language_model_layers,
)
from data import extract_frames, build_model_input


# ============================================================
# Temporal Head
# ============================================================

class TemporalHead(nn.Module):
    """从 latent hidden states 预测 evidence time bins。

    输入: (B, K, D) latent tokens 的 hidden states
    输出: (B, num_bins) 每个 bin 的 evidence logits
    """

    def __init__(self, hidden_dim=3584, num_bins=16):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_bins),
        )

    def forward(self, latent_hidden):
        pooled = latent_hidden.mean(dim=1)  # (B, D)
        return self.proj(pooled)  # (B, num_bins)


# ============================================================
# Dataset
# ============================================================

class TemporalDataset(Dataset):
    """加载 temporal evidence 数据。"""

    def __init__(self, data_path, video_dirs, num_bins=16,
                 max_samples=None, evidence_expand=1.0):
        """
        Args:
            evidence_expand: 零长度 evidence 扩展半径（秒）
        """
        import glob

        # 建立视频索引
        self.video_index = {}
        for vdir in video_dirs:
            if not os.path.isdir(vdir):
                continue
            for f in glob.glob(os.path.join(vdir, "**", "*.mp4"), recursive=True):
                self.video_index[os.path.basename(f)] = f

        self.samples = []
        self.num_bins = num_bins
        self.evidence_expand = evidence_expand
        skipped_no_video = 0
        skipped_no_evidence = 0

        with open(data_path) as f:
            for line in f:
                item = json.loads(line.strip())
                # 必须有 evidence_segments
                segments = item.get("evidence_segments")
                if not segments:
                    skipped_no_evidence += 1
                    continue
                # 匹配视频
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
    """计算 fixed-bin temporal labels。

    Args:
        duration: 视频总时长（秒）
        evidence_segments: [[start, end], ...]
        num_bins: 时间 bin 数量
        expand: 零长度 evidence 扩展半径（秒）

    Returns:
        list[float]: 每个 bin 的 label (0.0 或 overlap 比例)
    """
    if duration <= 0:
        return [0.0] * num_bins

    bin_size = duration / num_bins
    labels = [0.0] * num_bins

    for s, e in evidence_segments:
        # 处理零长度 evidence（点标注）
        if e <= s:
            s = max(0, s - expand)
            e = min(duration, s + 2 * expand)

        for b in range(num_bins):
            bin_start = b * bin_size
            bin_end = (b + 1) * bin_size
            # 计算重叠比例作为 soft label
            overlap = max(0, min(bin_end, e) - max(bin_start, s))
            labels[b] = max(labels[b], overlap / bin_size)

    return labels


def temporal_collate_fn(batch, processor, tokenizer, latent_tokens,
                        num_frames, latent_token_ids, num_bins,
                        evidence_expand=1.0):
    """Temporal 数据的 collate function。"""
    assert len(batch) == 1
    item = batch[0]

    frames, timestamps, duration = extract_frames(
        item["_resolved_video"], num_frames
    )

    inputs = build_model_input(
        processor, tokenizer, latent_tokens,
        frames, item["question"], item.get("answer", ""),
    )

    # 不需要 answer labels（不训 LLM）
    input_ids = inputs["input_ids"][0]
    inputs["labels"] = torch.full_like(input_ids, -100).unsqueeze(0)

    # Temporal bin labels
    bin_labels = compute_bin_labels(
        duration, item["evidence_segments"],
        num_bins=num_bins, expand=evidence_expand,
    )
    inputs["_bin_labels"] = torch.tensor([bin_labels], dtype=torch.float)
    inputs["_duration"] = duration

    return inputs


# ============================================================
# 按视频分组划分数据集
# ============================================================

def video_group_split(dataset, train_ratio=0.85, val_ratio=0.05, seed=42):
    """按视频分组划分，防止同一视频跨 split 泄漏。"""
    import random

    # 按视频分组
    video_to_indices = defaultdict(list)
    for i, sample in enumerate(dataset.samples):
        video_to_indices[sample["_resolved_video"]].append(i)

    videos = list(video_to_indices.keys())
    random.seed(seed)
    random.shuffle(videos)

    n_val = max(1, int(len(videos) * val_ratio))
    n_test = max(1, int(len(videos) * (1 - train_ratio - val_ratio)))
    n_train = len(videos) - n_val - n_test

    train_videos = videos[:n_train]
    val_videos = videos[n_train:n_train + n_val]
    test_videos = videos[n_train + n_val:]

    train_idx = [i for v in train_videos for i in video_to_indices[v]]
    val_idx = [i for v in val_videos for i in video_to_indices[v]]
    test_idx = [i for v in test_videos for i in video_to_indices[v]]

    print(f"Temporal 划分 (按视频): "
          f"train={len(train_idx)} ({len(train_videos)} 视频), "
          f"val={len(val_idx)} ({len(val_videos)} 视频), "
          f"test={len(test_idx)} ({len(test_videos)} 视频)")

    return (Subset(dataset, train_idx),
            Subset(dataset, val_idx),
            Subset(dataset, test_idx))


# ============================================================
# Hidden state 捕获
# ============================================================

class HiddenStateCapture:
    """在最后一层 decoder 捕获 latent positions 的 hidden states。"""

    def __init__(self, last_layer, latent_token_ids):
        self.latent_set = set(latent_token_ids)
        self.last_layer = last_layer
        self._hidden = None
        self._hook = None

    def install(self):
        def hook_fn(module, input, output):
            self._hidden = output[0]
        self._hook = self.last_layer.register_forward_hook(hook_fn)

    def get(self, input_ids):
        if self._hidden is None:
            return None
        results = []
        for b in range(input_ids.shape[0]):
            ids = input_ids[b].tolist()
            lat_pos = [i for i, t in enumerate(ids) if t in self.latent_set]
            if lat_pos:
                results.append(self._hidden[b, lat_pos, :])
        if not results:
            return None
        return torch.stack(results)

    def remove(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        self._hidden = None


# ============================================================
# 分布式工具（复用 train_ddp.py 的逻辑）
# ============================================================

def setup_distributed():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return local_rank, world_size
    return 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(msg):
    if is_main():
        print(msg)


# ============================================================
# 评测工具
# ============================================================

def bins_to_segments(probs, duration, threshold=0.5, min_gap=0):
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

    # 合并间隔小于 min_gap 的段
    if min_gap > 0 and len(segments) > 1:
        merged = [segments[0]]
        for seg in segments[1:]:
            if seg[0] - merged[-1][1] <= min_gap:
                merged[-1][1] = seg[1]
            else:
                merged.append(seg)
        segments = merged

    return segments


def compute_tiou(pred_segments, gt_segments):
    """计算 interval-union tIoU。"""
    if not pred_segments or not gt_segments:
        return 0.0

    # 构建覆盖集合（离散化到 0.1 秒精度）
    def to_set(segments):
        s = set()
        for start, end in segments:
            t = start
            while t < end:
                s.add(round(t, 1))
                t += 0.1
        return s

    pred_set = to_set(pred_segments)
    gt_set = to_set(gt_segments)

    if not pred_set and not gt_set:
        return 1.0
    if not pred_set or not gt_set:
        return 0.0

    intersection = len(pred_set & gt_set)
    union = len(pred_set | gt_set)
    return intersection / union


# ============================================================
# 训练
# ============================================================

def train(args):
    local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    log("=" * 60)
    log("Stage 3: Temporal Head 训练")
    log(f"  GPU 数量: {world_size}")
    log(f"  K={args.K}, num_frames={args.num_frames}, num_bins={args.num_bins}")
    log(f"  lr={args.lr}, epochs={args.epochs}")
    log("=" * 60)

    # --- 加载 Stage 2 模型 ---
    model, processor, tokenizer, latent_token_ids = setup_model_and_tokenizer(
        K=args.K, device=device,
    )
    latent_tokens = [f"<latent_{i}>" for i in range(args.K)]

    # 加载 Stage 2 checkpoint
    epoch_s2, val_loss_s2 = load_checkpoint(model, args.stage2_checkpoint)
    log(f"Stage 2 checkpoint: epoch={epoch_s2}, val_loss={val_loss_s2}")

    # 冻结所有 Stage 2 参数
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    # --- Temporal Head ---
    temporal_head = TemporalHead(3584, num_bins=args.num_bins).to(device)
    log(f"Temporal Head: {sum(p.numel() for p in temporal_head.parameters())} params")

    if world_size > 1:
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        temporal_head = DDP(temporal_head, device_ids=[local_rank])

    # --- 数据 ---
    video_dirs = args.video_dirs.split(",")
    dataset = TemporalDataset(
        args.data_path, video_dirs,
        num_bins=args.num_bins,
        max_samples=args.max_samples,
        evidence_expand=args.evidence_expand,
    )

    if args.overfit:
        subset = Subset(dataset, range(min(32, len(dataset))))
        train_set, val_set, test_set = subset, subset, subset
        log(f"Overfit 模式: {len(subset)} 条")
    else:
        train_set, val_set, test_set = video_group_split(dataset)

    # 保存 test indices
    if is_main():
        os.makedirs(args.output_dir, exist_ok=True)
        if hasattr(test_set, 'indices'):
            with open(os.path.join(args.output_dir, "temporal_test_indices.json"), "w") as f:
                json.dump(test_set.indices, f)

    train_sampler = DistributedSampler(train_set, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if world_size > 1 else None

    def make_collate(proc, tok, lt, nf, ltids, nb, expand):
        def wrapper(batch):
            return temporal_collate_fn(batch, proc, tok, lt, nf, ltids, nb, expand)
        return wrapper

    collate = make_collate(processor, tokenizer, latent_tokens,
                           args.num_frames, latent_token_ids,
                           args.num_bins, args.evidence_expand)

    train_loader = DataLoader(
        train_set, batch_size=1, sampler=train_sampler,
        shuffle=(train_sampler is None), collate_fn=collate,
    )
    val_loader = DataLoader(
        val_set, batch_size=1, sampler=val_sampler,
        shuffle=False, collate_fn=collate,
    )

    # --- 计算 pos_weight ---
    if args.pos_weight_auto:
        pos, neg = 0, 0
        for sample in dataset.samples[:min(2000, len(dataset.samples))]:
            labels = compute_bin_labels(
                1.0, sample["evidence_segments"],
                args.num_bins, args.evidence_expand,
            )
            pos += sum(1 for l in labels if l > 0.5)
            neg += sum(1 for l in labels if l <= 0.5)
        pw = min(neg / max(pos, 1), 5.0)
        log(f"  pos_weight: {pw:.2f} (pos={pos}, neg={neg})")
    else:
        pw = 1.0

    pos_weight = torch.tensor([pw], device=device)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(temporal_head.parameters(),
                                  lr=args.lr, weight_decay=0.01)

    layers = get_language_model_layers(model)
    best_val = float("inf")

    # --- 训练循环 ---
    for epoch in range(args.epochs):
        temporal_head.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)

        total_loss = 0.0
        n_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}",
                    disable=not is_main())

        for batch in pbar:
            bin_labels = batch.pop("_bin_labels").to(device)
            batch.pop("_duration", None)
            batch.pop("_temporal_labels", None)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            hooks = []
            hidden_cap = HiddenStateCapture(layers[-1], latent_token_ids)
            try:
                # Bottleneck hooks（必须在 bottleneck 下 forward）
                bn_mask = build_bottleneck_mask(
                    batch["input_ids"], latent_token_ids,
                    enable_bottleneck=True,
                ).to(device, dtype=torch.bfloat16)
                hooks = install_bottleneck_hooks(layers, bn_mask)
                hidden_cap.install()

                with torch.no_grad():
                    model(**batch)

            finally:
                for h in hooks:
                    h.remove()

            # 提取 latent hidden → temporal head
            latent_hidden = hidden_cap.get(batch["input_ids"])
            hidden_cap.remove()

            if latent_hidden is None:
                continue

            logits = temporal_head(latent_hidden.float())
            loss = F.binary_cross_entropy_with_logits(
                logits, bin_labels, pos_weight=pos_weight,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(temporal_head.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_steps += 1
            if is_main():
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / max(n_steps, 1)
        log(f"  Epoch {epoch + 1}: train_loss={avg_loss:.4f}")

        # --- Validation ---
        val_loss, val_tiou, val_recall = evaluate(
            model, temporal_head, val_loader, layers,
            latent_token_ids, device, world_size,
            args.num_bins, pos_weight,
        )
        log(f"  Epoch {epoch + 1}: val_loss={val_loss:.4f}, "
            f"tIoU={val_tiou:.4f}, R@0.3={val_recall[0.3]:.2%}, "
            f"R@0.5={val_recall[0.5]:.2%}, R@0.7={val_recall[0.7]:.2%}")

        # --- Save ---
        if is_main():
            th = temporal_head.module if hasattr(temporal_head, "module") else temporal_head
            torch.save({
                "state_dict": th.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_tiou": val_tiou,
                "num_bins": args.num_bins,
            }, os.path.join(args.output_dir, f"temporal_epoch{epoch + 1}.pt"))

            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    "state_dict": th.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "val_tiou": val_tiou,
                    "num_bins": args.num_bins,
                }, os.path.join(args.output_dir, "temporal_best.pt"))
                log(f"  ★ New best (val_loss={val_loss:.4f}, tIoU={val_tiou:.4f})")

    log(f"\n训练完成. best_val_loss={best_val:.4f}")
    cleanup_distributed()


@torch.no_grad()
def evaluate(model, temporal_head, val_loader, layers,
             latent_token_ids, device, world_size,
             num_bins, pos_weight):
    """评测：loss + tIoU + Recall@k。"""
    temporal_head.eval()
    loss_sum = 0.0
    tiou_sum = 0.0
    recall_counts = {0.3: 0, 0.5: 0, 0.7: 0}
    count = 0

    for batch in val_loader:
        bin_labels = batch.pop("_bin_labels").to(device)
        duration = batch.pop("_duration", 1.0)
        batch.pop("_temporal_labels", None)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        hooks = []
        hidden_cap = HiddenStateCapture(layers[-1], latent_token_ids)
        try:
            bn_mask = build_bottleneck_mask(
                batch["input_ids"], latent_token_ids,
                enable_bottleneck=True,
            ).to(device, dtype=torch.bfloat16)
            hooks = install_bottleneck_hooks(layers, bn_mask)
            hidden_cap.install()
            model(**batch)
        finally:
            for h in hooks:
                h.remove()

        latent_hidden = hidden_cap.get(batch["input_ids"])
        hidden_cap.remove()
        if latent_hidden is None:
            continue

        logits = temporal_head(latent_hidden.float())
        loss = F.binary_cross_entropy_with_logits(
            logits, bin_labels, pos_weight=pos_weight,
        )
        loss_sum += loss.item()

        # tIoU
        probs = torch.sigmoid(logits[0]).cpu().tolist()
        pred_segs = bins_to_segments(probs, duration)
        gt_labels = bin_labels[0].cpu().tolist()
        gt_segs = bins_to_segments(gt_labels, duration, threshold=0.5)
        tiou = compute_tiou(pred_segs, gt_segs)
        tiou_sum += tiou

        for thresh in recall_counts:
            if tiou >= thresh:
                recall_counts[thresh] += 1

        count += 1

    # all_reduce
    if world_size > 1:
        stats = torch.tensor(
            [loss_sum, tiou_sum, float(count)] +
            [float(recall_counts[t]) for t in [0.3, 0.5, 0.7]],
            device=device,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        loss_sum = stats[0].item()
        tiou_sum = stats[1].item()
        count = stats[2].item()
        recall_counts = {0.3: stats[3].item(), 0.5: stats[4].item(), 0.7: stats[5].item()}

    avg_loss = loss_sum / max(count, 1)
    avg_tiou = tiou_sum / max(count, 1)
    recall = {t: c / max(count, 1) for t, c in recall_counts.items()}

    temporal_head.train()
    return avg_loss, avg_tiou, recall


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Temporal Head 训练")

    # 数据
    parser.add_argument("--data_path", required=True,
                        help="temporal_evidence.jsonl 路径")
    parser.add_argument("--video_dirs", required=True,
                        help="视频目录，逗号分隔")
    parser.add_argument("--output_dir", required=True,
                        help="输出目录")
    parser.add_argument("--max_samples", type=int, default=None)

    # 模型
    parser.add_argument("--stage2_checkpoint", required=True,
                        help="Stage 2 best_model.pt 路径")
    parser.add_argument("--K", type=int, default=32)
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--num_bins", type=int, default=16,
                        help="时间 bin 数量")

    # 训练
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--evidence_expand", type=float, default=1.0,
                        help="零长度 evidence 扩展半径（秒）")
    parser.add_argument("--pos_weight_auto", action="store_true",
                        help="自动计算 pos_weight")

    args = parser.parse_args()
    train(args)

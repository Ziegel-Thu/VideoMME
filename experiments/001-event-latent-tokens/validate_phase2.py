"""
Phase 2 验证: 用真实提取的 feature 跑 EventLatentModel 前向

验证:
  1. DataLoader 能正确加载 feature + 标注
  2. 模型前向无报错
  3. 输出 shape 和数值合理
"""

import os
import json
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class TemporalEvidenceDataset(Dataset):
    """加载预提取的 segment feature + temporal evidence 标注"""

    def __init__(
        self,
        evidence_path: str,
        feature_dir: str,
        segment_stride: float = 4.0,
        max_segments: int = 128,
    ):
        self.feature_dir = Path(feature_dir)
        self.segment_stride = segment_stride
        self.max_segments = max_segments

        # 建 feature 索引
        self.feature_index = {}
        for pt_file in glob.glob(str(self.feature_dir / "*.pt")):
            # 文件名格式: video_name.mp4.pt
            basename = os.path.basename(pt_file)
            video_name = basename.replace(".pt", "")
            self.feature_index[video_name] = pt_file

        # 加载标注，只保留有 feature 的
        self.samples = []
        with open(evidence_path) as f:
            for line in f:
                d = json.loads(line)
                video_name = os.path.basename(d["video_path"])
                if video_name in self.feature_index:
                    d["_video_name"] = video_name
                    self.samples.append(d)

        print(f"Dataset: {len(self.samples)} samples (from {len(self.feature_index)} features)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_name = sample["_video_name"]

        # 加载 feature
        data = torch.load(
            self.feature_index[video_name],
            weights_only=False,
            map_location="cpu",
        )
        embeddings = data["embeddings"].float()  # (M, D)
        timestamps = data["timestamps"]  # [(start, end), ...]

        M, D = embeddings.shape

        # 构造 temporal ground truth: segment-level binary label
        gt_segments = sample["evidence_segments"]  # [[t_s, t_e], ...]
        segment_labels = torch.zeros(M)
        for ts, te in gt_segments:
            for i, (seg_start, seg_end) in enumerate(timestamps):
                # 如果 segment 和 evidence 有交集
                if seg_start < te and seg_end > ts:
                    segment_labels[i] = 1.0

        # Pad or truncate to max_segments
        if M > self.max_segments:
            embeddings = embeddings[:self.max_segments]
            segment_labels = segment_labels[:self.max_segments]
            M = self.max_segments

        # 用问题文本的 hash 作为伪 question embedding（真训练时用 LLM encode）
        question_hash = hash(sample["question"]) % (2**31)
        torch.manual_seed(question_hash)
        question_embed = torch.randn(D)

        return {
            "segment_features": embeddings,       # (M, D)
            "question_embed": question_embed,      # (D,)
            "segment_labels": segment_labels,      # (M,)
            "num_segments": M,
            "question": sample["question"],
            "video_name": video_name,
        }


def collate_fn(batch):
    """动态 pad 到 batch 内最大 M"""
    max_M = max(b["num_segments"] for b in batch)
    D = batch[0]["segment_features"].shape[-1]
    B = len(batch)

    segment_features = torch.zeros(B, max_M, D)
    question_embeds = torch.stack([b["question_embed"] for b in batch])
    segment_labels = torch.zeros(B, max_M)
    mask = torch.zeros(B, max_M, dtype=torch.bool)

    for i, b in enumerate(batch):
        M = b["num_segments"]
        segment_features[i, :M] = b["segment_features"]
        segment_labels[i, :M] = b["segment_labels"]
        mask[i, :M] = True

    return {
        "segment_features": segment_features,
        "question_embed": question_embeds,
        "segment_labels": segment_labels,
        "mask": mask,
    }


def main():
    import sys
    sys.path.insert(0, "/home/v-shuzheng/video/experiments/001-event-latent-tokens")
    from models.event_latent import EventLatentModel

    EVIDENCE_PATH = "/home/v-shuzheng/video/data/parsed/temporal_evidence.jsonl"
    FEATURE_DIR = "/home/v-shuzheng/video/data/features"

    # 1. 加载数据
    print("=== 加载数据 ===")
    dataset = TemporalEvidenceDataset(EVIDENCE_PATH, FEATURE_DIR)

    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(loader))

    print(f"\nBatch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} ({v.dtype})")

    # 2. 模型前向
    print("\n=== 模型前向 ===")
    D = batch["segment_features"].shape[-1]
    model = EventLatentModel(d_model=D, n_latent=2, top_m=8, K=8, n_heads=8)
    print(f"d_model={D}, params={sum(p.numel() for p in model.parameters()):,}")

    with torch.no_grad():
        out = model(batch["segment_features"], batch["question_embed"])

    print(f"\nOutput shapes:")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape}")

    # 3. 计算 loss (BCE for temporal grounding)
    print("\n=== Loss 计算 ===")
    logits = out["segment_logits"]  # (B, M)
    labels = batch["segment_labels"]  # (B, M)
    mask = batch["mask"]  # (B, M)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[mask], labels[mask]
    )
    print(f"  BCE loss: {loss.item():.4f}")

    # 4. 检查 temporal grounding 预测
    print("\n=== Temporal Grounding 预测 ===")
    probs = torch.sigmoid(logits)
    for i in range(min(2, len(batch["segment_features"]))):
        M = batch["mask"][i].sum().item()
        gt_idx = batch["segment_labels"][i, :M].nonzero(as_tuple=True)[0].tolist()
        pred_top3 = probs[i, :M].topk(min(3, M)).indices.tolist()
        print(f"  Sample {i}: M={M}, GT segments={gt_idx}, Pred top-3={pred_top3}")

    # 5. 检查 selector
    print("\n=== Selector 选段 ===")
    for i in range(min(2, len(batch["segment_features"]))):
        sel = sorted(out["selected_indices"][i].tolist())
        print(f"  Sample {i}: selected={sel}")

    print("\n=== Phase 2 验证通过 ✅ ===")


if __name__ == "__main__":
    main()

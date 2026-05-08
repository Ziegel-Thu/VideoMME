"""
数据加载：VideoQA Dataset + collate function + TemporalHead

支持:
- MCQ 数据 (visual_qa_v2.jsonl)
- 可选 temporal 标注 (evidence_segments)
- 可配置帧数（--num_frames）
"""

import os
import json
import glob
import uuid
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import decord


class VideoQADataset(Dataset):
    """视频 QA 数据集。

    __getitem__ 返回原始数据 dict，视频处理在 collate_fn 中完成。
    """

    def __init__(self, data_path, video_dirs, max_samples=None):
        """
        Args:
            data_path: jsonl 文件路径
            video_dirs: 视频搜索目录列表
            max_samples: 最大样本数（调试用）
        """
        # 建立视频文件索引: basename → full_path
        self.video_index = {}
        for vdir in video_dirs:
            if not os.path.isdir(vdir):
                continue
            for f in glob.glob(os.path.join(vdir, "**", "*.mp4"), recursive=True):
                self.video_index[os.path.basename(f)] = f

        # 加载数据，过滤无视频的样本
        self.samples = []
        skipped = 0
        with open(data_path) as f:
            for line in f:
                item = json.loads(line.strip())
                vname = os.path.basename(item["video_path"])
                if vname in self.video_index:
                    item["_resolved_video"] = self.video_index[vname]
                    self.samples.append(item)
                else:
                    skipped += 1
        if max_samples:
            self.samples = self.samples[:max_samples]

        print(f"数据集: {len(self.samples)} 条可用, "
              f"{skipped} 条跳过 (视频索引: {len(self.video_index)} 个)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def extract_frames(video_path, num_frames):
    """从视频均匀采样 num_frames 帧。

    Returns:
        frames: list[PIL.Image]
        timestamps: list[float] — 每帧的时间戳（秒）
        duration: float — 视频总时长（秒）
    """
    try:
        vr = decord.VideoReader(video_path)
        total = len(vr)
        fps = vr.get_avg_fps()
        indices = [min(int(i * total / num_frames), total - 1)
                   for i in range(num_frames)]
        frames = [Image.fromarray(vr[idx].asnumpy()) for idx in indices]
        timestamps = [idx / fps for idx in indices]
        duration = total / fps
        return frames, timestamps, duration
    except Exception as e:
        print(f"  ⚠️ 视频读取失败 {video_path}: {e}，用黑帧替代")
        frames = [Image.new("RGB", (224, 224), (0, 0, 0))] * num_frames
        return frames, [0.0] * num_frames, 1.0


def build_model_input(processor, tokenizer, latent_tokens, frames,
                      question, answer, max_pixels=128 * 28 * 28):
    """构造模型输入，token 顺序: [vision][question][latent][answer]。"""
    from qwen_vl_utils import process_vision_info

    # 每个进程/rank 用唯一 ID 避免临时文件冲突
    uid = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    tmp_dir = tempfile.gettempdir()
    tmp_paths = []
    for i, img in enumerate(frames):
        p = os.path.join(tmp_dir, f"_vframe_{uid}_{i}.jpg")
        img.save(p)
        tmp_paths.append(p)

    latent_str = "".join(latent_tokens)
    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": tmp_paths, "fps": 1.0},
            {"type": "text", "text": f"{question}\n{latent_str}"},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": answer},
        ]},
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        return_tensors="pt", padding=True,
    )

    # 清理临时文件
    for p in tmp_paths:
        try:
            os.remove(p)
        except OSError:
            pass

    return inputs


def collate_fn(batch, processor, tokenizer, latent_tokens,
               num_frames, latent_token_ids):
    """batch 组装：视频采帧 → processor → 构造 labels。

    当前 per-GPU batch_size=1（显存限制），所以只处理 batch[0]。
    """
    assert len(batch) == 1, "当前只支持 batch_size=1"
    item = batch[0]

    # 视频采帧
    frames, timestamps, duration = extract_frames(
        item["_resolved_video"], num_frames
    )

    # 构造模型输入
    inputs = build_model_input(
        processor, tokenizer, latent_tokens,
        frames, item["question"], item["answer"],
    )

    # 构造 labels: 只在 assistant 回复部分计算 loss
    input_ids = inputs["input_ids"][0]
    labels = torch.full_like(input_ids, -100)

    ids_list = input_ids.tolist()
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")

    # 找最后一个 <|im_start|>（assistant 的起始标记）
    im_start_positions = [i for i, t in enumerate(ids_list) if t == im_start_id]
    if im_start_positions:
        ast = im_start_positions[-1]
        # 跳过 "<|im_start|>assistant\n"（约 3 个 token）
        assistant_prefix = tokenizer.encode("assistant\n", add_special_tokens=False)
        content_start = ast + 1 + len(assistant_prefix)
        labels[content_start:] = input_ids[content_start:]

    inputs["labels"] = labels.unsqueeze(0)

    # Temporal labels（如果有 evidence_segments）
    temporal_label = None
    if item.get("evidence_segments"):
        temporal_label = _compute_temporal_labels(
            timestamps, duration, item["evidence_segments"]
        )

    inputs["_temporal_labels"] = [temporal_label]
    return inputs


def _compute_temporal_labels(timestamps, duration, evidence_segments):
    """计算每帧是否在 evidence 时间段内。"""
    labels = []
    for t in timestamps:
        in_evidence = any(s <= t <= e for s, e in evidence_segments)
        labels.append(1.0 if in_evidence else 0.0)
    return labels


class TemporalHead(nn.Module):
    """MLP: 从 latent hidden states 预测时间段。

    输入: latent tokens 的 hidden states (B, K, D)
    输出: 每帧/每 bin 的 evidence logits (B, num_bins)
    """

    def __init__(self, hidden_dim=3584, num_bins=8):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_bins),
        )

    def forward(self, latent_hidden):
        pooled = latent_hidden.mean(dim=1)  # (B, D)
        return self.proj(pooled)  # (B, num_bins)

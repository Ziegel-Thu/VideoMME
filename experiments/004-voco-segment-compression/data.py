"""
004-voco-segment: Dataset + Collate

每个样本：
  视频 → 1fps 采样 → vision_encoder → dense embeds
  按 frames_per_segment 切段
  Q text → tokenize → embed
  A text → tokenize → embed (训练时)

Collate 输出: 已构造好的 inputs_embeds + attention_mask + labels + layout
"""

import os
import sys
import json
import glob
import uuid
import tempfile
import importlib.util

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image
import decord

# 显式从 003 加载 extract_frames，避免和本目录的 data.py 名字冲突
_003_data_path = os.path.join(os.path.dirname(__file__),
                              "../003-bottleneck-multigpu/data.py")
_spec = importlib.util.spec_from_file_location("data_003", _003_data_path)
_data_003 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_data_003)
extract_frames = _data_003.extract_frames

from model import (
    get_video_embeds,
    split_into_segments,
    build_voco_sequence_embeds,
    build_voco_attention_mask,
)


class VoCoVideoDataset(Dataset):
    """视频 MCQ 数据集，__getitem__ 返回原始 dict（视频处理在 collate）。"""

    def __init__(self, data_path, video_dirs, max_samples=None):
        self.video_index = {}
        for vdir in video_dirs:
            if not os.path.isdir(vdir):
                continue
            for f in glob.glob(os.path.join(vdir, "**", "*.mp4"), recursive=True):
                self.video_index[os.path.basename(f)] = f

        self.samples = []
        with open(data_path) as f:
            for line in f:
                item = json.loads(line.strip())
                vname = os.path.basename(item["video_path"])
                if vname in self.video_index:
                    item["_resolved_video"] = self.video_index[vname]
                    self.samples.append(item)

        if max_samples:
            self.samples = self.samples[:max_samples]
        print(f"VoCo 数据集: {len(self.samples)} 条")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def encode_text(tokenizer, embed_layer, text, device, dtype):
    """文本 → token ids → embeds"""
    ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt").to(device)
    embeds = embed_layer(ids).squeeze(0).to(dtype)  # (L, D)
    return embeds, ids.squeeze(0)


def voco_collate(batch, voco_model, processor, tokenizer,
                 fps=1.0, frames_per_segment=2, max_frames=30):
    """构造单个样本的 inputs_embeds + mask + labels。

    batch_size=1。

    Returns dict with:
        inputs_embeds: (1, L, D)
        attention_mask: (1, 1, L, L)
        labels: (1, L)
        layout: dict
    """
    assert len(batch) == 1
    item = batch[0]

    base = voco_model.base
    device = voco_model.voco_embeds.device
    dtype = voco_model.voco_embeds.dtype

    # 1. 1fps 采帧
    try:
        vr = decord.VideoReader(item["_resolved_video"])
        duration = len(vr) / vr.get_avg_fps()
    except Exception:
        return None

    n_frames = max(frames_per_segment, min(int(duration * fps), max_frames))
    # 确保 n_frames 是 frames_per_segment 的倍数
    n_frames = (n_frames // frames_per_segment) * frames_per_segment
    if n_frames == 0:
        n_frames = frames_per_segment

    frames, _, _ = extract_frames(item["_resolved_video"], n_frames)

    # 2. 走 processor 拿 pixel_values_videos + video_grid_thw
    from qwen_vl_utils import process_vision_info

    uid = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
    tmp_dir = tempfile.gettempdir()
    tmp_paths = []
    for i, img in enumerate(frames):
        p = os.path.join(tmp_dir, f"_vf_{uid}_{i}.jpg")
        img.save(p)
        tmp_paths.append(p)

    messages = [{"role": "user", "content": [
        {"type": "video", "video": tmp_paths, "fps": fps},
        {"type": "text", "text": "x"},  # placeholder
    ]}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    proc_inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        return_tensors="pt",
    )

    for p in tmp_paths:
        try:
            os.remove(p)
        except OSError:
            pass

    # 3. vision encoder → dense embeds
    pv = proc_inputs.get("pixel_values_videos")
    vg = proc_inputs.get("video_grid_thw")
    if pv is None:
        return None

    video_embeds, tokens_per_frame, actual_frames = get_video_embeds(
        base, pv, vg, device,
    )
    # video_embeds: (actual_frames * tokens_per_frame, D)

    # 4. 切段
    segments = split_into_segments(
        video_embeds, tokens_per_frame, frames_per_segment,
    )
    n_segments = len(segments)

    # 5. 构造 Q/A embeds
    # Qwen2.5-VL 的 chat template
    m = base.module if hasattr(base, "module") else base
    from peft import PeftModel
    if isinstance(m, PeftModel):
        embed_layer = m.base_model.model.get_input_embeddings()
    else:
        embed_layer = m.get_input_embeddings()

    # 构造 Q 和 A 的文本（包含 chat 标签）
    q_text = f"<|im_start|>user\n{item['question']}<|im_end|>\n<|im_start|>assistant\n"
    a_text = f"{item['answer']}<|im_end|>"

    q_embeds, q_ids = encode_text(tokenizer, embed_layer, q_text, device, dtype)
    a_embeds, a_ids = encode_text(tokenizer, embed_layer, a_text, device, dtype)

    # 6. 拼接 inputs_embeds
    voco_per_seg = voco_model.voco_embeds.to(dtype)
    inputs_embeds, layout = build_voco_sequence_embeds(
        segments, voco_per_seg, q_embeds, a_embeds,
    )

    # 7. 构造 mask
    voco_4d_mask = build_voco_attention_mask(layout, device, dtype)

    # 标准 2D padding mask (全 1 因为 batch_size=1，没 padding)
    padding_mask = torch.ones(1, layout["total_len"], dtype=torch.long, device=device)

    # 8. 构造 labels (只在 a 部分算 loss)
    L = layout["total_len"]
    labels = torch.full((L,), -100, dtype=torch.long, device=device)
    if layout["a"] is not None:
        a_s, a_e = layout["a"]
        labels[a_s:a_e] = a_ids.to(device)

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": padding_mask,
        "voco_4d_mask": voco_4d_mask,
        "labels": labels.unsqueeze(0),
        "layout": layout,
        "n_segments": n_segments,
    }

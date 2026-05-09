"""
004 拼接版 (KV cache concat): 真正分段 forward

每段独立 forward → 提取该段 voco 位置的 KV cache → 拼接所有段 → 最后一次 forward 跑 [all_voco_cache, Q, A]

优势:
- 每段 forward 序列短（vision_t + voco_t ≈ 200+4 = 204 tokens）
- 最终 forward 短（n_seg × K_seg + Q + A ≈ 60+50 = 110 tokens）
- 总计算量 << 单 forward 版

劣势:
- 实现复杂（KV cache 拼接、position_ids 处理）
- 不同段的 KV cache 拼起来 position_ids 要保持一致

用法:
  python train_concat.py --data_path ... --video_dirs ... --output_dir ...
"""

import os
import sys
import json
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers.cache_utils import DynamicCache

from model import setup_voco_model, get_video_embeds, split_into_segments
from data import VoCoVideoDataset, encode_text


def get_inner_model(base):
    """获取最内层 Qwen2_5_VLForConditionalGeneration（解 PEFT/DDP）。"""
    m = base.module if hasattr(base, "module") else base
    from peft import PeftModel
    if isinstance(m, PeftModel):
        return m.base_model.model
    return m


def forward_segment(inner, vision_embeds, voco_embeds, position_offset=0):
    """跑一段的 forward: [vision_t, voco_t]，返回 voco 位置的 KV cache。

    Args:
        inner: Qwen2_5_VLForConditionalGeneration
        vision_embeds: (V_t, D) 该段 vision tokens
        voco_embeds: (K, D) 该段 voco tokens
        position_offset: 段在最终序列中的起始位置（用于 position_ids）

    Returns:
        voco_cache: list of (K_proj, V_proj) 每层的 voco 部分 KV cache
        每层 cache shape: (1, n_heads, K, head_dim)
    """
    seg_len = vision_embeds.shape[0] + voco_embeds.shape[0]
    K = voco_embeds.shape[0]

    inputs_embeds = torch.cat([vision_embeds, voco_embeds], dim=0).unsqueeze(0)
    attention_mask = torch.ones(1, seg_len, dtype=torch.long,
                                device=vision_embeds.device)
    position_ids = torch.arange(
        position_offset, position_offset + seg_len,
        device=vision_embeds.device,
    ).unsqueeze(0)

    cache = DynamicCache()
    out = inner.model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=cache,
        use_cache=True,
    )

    # 提取 voco 位置（最后 K 个）的 KV cache
    voco_cache = []
    pkv = out.past_key_values
    n_layers = len(pkv)
    for layer_idx in range(n_layers):
        k, v = pkv[layer_idx]  # tuple (key, value)
        voco_cache.append((k[:, :, -K:, :], v[:, :, -K:, :]))

    return voco_cache


def concat_voco_caches(seg_caches):
    """把多段的 voco cache 拼接成一个完整 cache。

    Args:
        seg_caches: list of seg_cache, 每个 seg_cache 是 list of (k, v) per layer

    Returns:
        cache: DynamicCache containing all voco K/V
        total_voco_len: int
    """
    cache = DynamicCache()
    n_layers = len(seg_caches[0])
    K_per_seg = seg_caches[0][0][0].shape[2]
    total_K = K_per_seg * len(seg_caches)

    for layer_idx in range(n_layers):
        k_list = [seg_caches[s][layer_idx][0] for s in range(len(seg_caches))]
        v_list = [seg_caches[s][layer_idx][1] for s in range(len(seg_caches))]
        cache.update(
            torch.cat(k_list, dim=2),
            torch.cat(v_list, dim=2),
            layer_idx,
        )

    return cache, total_K


def voco_concat_forward(model, vision_embeds, tokens_per_frame, frames_per_segment,
                        q_embeds, a_embeds=None):
    """完整的拼接 forward 流程。

    1. 切段
    2. 每段独立 forward 提取 voco cache
    3. 拼接所有 voco cache
    4. 最后一次 forward 跑 [Q, A]，past_key_values 是拼接的 voco cache

    Returns:
        logits, hidden_states (Q+A 部分)
    """
    inner = get_inner_model(model.base)
    K = model.K_seg
    voco_per_seg = model.voco_embeds  # (K, D)

    # 切段
    segments = split_into_segments(vision_embeds, tokens_per_frame, frames_per_segment)
    n_segments = len(segments)

    # 每段独立 forward 提取 voco cache
    seg_caches = []
    position_offset = 0
    for seg in segments:
        # 段内 vision 占 V_t 个位置，voco 占 K 个位置
        # 但段间 vision 不共享位置（每段独立 forward），voco 在最终序列中是连续的
        # 简化: 每段 forward 用 [0, V_t + K) 的 position_ids
        seg_cache = forward_segment(inner, seg, voco_per_seg, position_offset=0)
        seg_caches.append(seg_cache)

    # 拼接 voco caches
    full_cache, total_voco = concat_voco_caches(seg_caches)

    # 最后一次 forward: 输入是 [Q, A]，past_kv 是 voco caches
    if a_embeds is not None:
        text_embeds = torch.cat([q_embeds, a_embeds], dim=0).unsqueeze(0)
    else:
        text_embeds = q_embeds.unsqueeze(0)

    text_len = text_embeds.shape[1]
    # attention_mask: (1, total_voco + text_len) — 标准 padding mask 全 1
    full_attn_mask = torch.ones(1, total_voco + text_len,
                                dtype=torch.long, device=text_embeds.device)

    # position_ids: text 部分接在 total_voco 之后
    text_position_ids = torch.arange(
        total_voco, total_voco + text_len, device=text_embeds.device,
    ).unsqueeze(0)

    out = inner.model(
        inputs_embeds=text_embeds,
        attention_mask=full_attn_mask,
        position_ids=text_position_ids,
        past_key_values=full_cache,
        use_cache=True,
    )
    hidden = out[0]
    logits = inner.lm_head(hidden)

    return logits, hidden, total_voco


def voco_concat_collate(batch, voco_model, processor, tokenizer,
                        fps=1.0, frames_per_segment=2, max_frames=30):
    """拼接版 collate: 准备 vision_embeds + Q/A embeds，不提前拼成大序列。"""
    assert len(batch) == 1
    item = batch[0]

    base = voco_model.base
    device = voco_model.voco_embeds.device
    dtype = voco_model.voco_embeds.dtype

    import uuid as _uuid, tempfile as _tmpfile
    from qwen_vl_utils import process_vision_info
    from data import extract_frames

    try:
        vr = decord.VideoReader(item["_resolved_video"])
        duration = len(vr) / vr.get_avg_fps()
    except Exception:
        return None

    n_frames = max(frames_per_segment, min(int(duration * fps), max_frames))
    n_frames = (n_frames // frames_per_segment) * frames_per_segment
    if n_frames == 0:
        n_frames = frames_per_segment

    frames, _, _ = extract_frames(item["_resolved_video"], n_frames)

    uid = f"{os.getpid()}_{_uuid.uuid4().hex[:8]}"
    tmp_dir = _tmpfile.gettempdir()
    tmp_paths = []
    for i, img in enumerate(frames):
        p = os.path.join(tmp_dir, f"_vf_{uid}_{i}.jpg")
        img.save(p)
        tmp_paths.append(p)

    messages = [{"role": "user", "content": [
        {"type": "video", "video": tmp_paths, "fps": fps},
        {"type": "text", "text": "x"},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(messages)
    proc_inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                            return_tensors="pt")

    for p in tmp_paths:
        try: os.remove(p)
        except OSError: pass

    pv = proc_inputs.get("pixel_values_videos")
    vg = proc_inputs.get("video_grid_thw")
    if pv is None:
        return None

    video_embeds, tokens_per_frame, _ = get_video_embeds(base, pv, vg, device)

    # Q/A embeds
    m = base.module if hasattr(base, "module") else base
    from peft import PeftModel
    if isinstance(m, PeftModel):
        embed_layer = m.base_model.model.get_input_embeddings()
    else:
        embed_layer = m.get_input_embeddings()

    q_text = f"<|im_start|>user\n{item['question']}<|im_end|>\n<|im_start|>assistant\n"
    a_text = f"{item['answer']}<|im_end|>"

    q_embeds, _ = encode_text(tokenizer, embed_layer, q_text, device, dtype)
    a_embeds, a_ids = encode_text(tokenizer, embed_layer, a_text, device, dtype)

    return {
        "video_embeds": video_embeds.to(dtype),
        "tokens_per_frame": tokens_per_frame,
        "frames_per_segment": frames_per_segment,
        "q_embeds": q_embeds,
        "a_embeds": a_embeds,
        "a_ids": a_ids,
    }


def train(args):
    device = torch.device("cuda")

    model, processor, tokenizer = setup_voco_model(
        K_seg=args.K_seg,
        lora_r=args.lora_r,
        device=device,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    video_dirs = args.video_dirs.split(",")
    dataset = VoCoVideoDataset(
        args.data_path, video_dirs, max_samples=args.max_samples,
    )

    n_val = min(50, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    def collate(batch):
        return voco_concat_collate(
            batch, model, processor, tokenizer,
            fps=args.fps,
            frames_per_segment=args.frames_per_segment,
            max_frames=args.max_frames,
        )

    train_loader = DataLoader(
        train_set, batch_size=1, shuffle=True, collate_fn=collate,
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        n = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch in pbar:
            if batch is None:
                continue

            try:
                logits, _, total_voco = voco_concat_forward(
                    model,
                    vision_embeds=batch["video_embeds"],
                    tokens_per_frame=batch["tokens_per_frame"],
                    frames_per_segment=batch["frames_per_segment"],
                    q_embeds=batch["q_embeds"],
                    a_embeds=batch["a_embeds"],
                )

                # logits 是 [Q, A] 部分；构造 labels（只算 A 的 loss）
                q_len = batch["q_embeds"].shape[0]
                a_len = batch["a_embeds"].shape[0]
                a_ids = batch["a_ids"]

                # logits[i] 预测 i+1 位置；A 部分位于 [q_len, q_len+a_len)
                # 我们要让 A 的每个位置（除第一个外）和 a_ids 对齐
                # 实际: logits[q_len-1:q_len-1+a_len] 预测 A 的 token
                shift_logits = logits[0, q_len - 1: q_len - 1 + a_len, :]
                shift_labels = a_ids.to(device)

                if shift_logits.shape[0] != shift_labels.shape[0]:
                    continue

                loss = nn.functional.cross_entropy(
                    shift_logits.float(), shift_labels,
                )

                if torch.isnan(loss):
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

                total_loss += loss.item()
                n += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}", voco=total_voco)

            except Exception as e:
                print(f"  [错误] {e}")
                import traceback; traceback.print_exc()
                continue

        avg_loss = total_loss / max(n, 1)
        print(f"  Epoch {epoch + 1}: train_loss={avg_loss:.4f}")

        # Save
        ckpt_path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch + 1}.pt")
        torch.save({
            "voco_embeds": model.voco_embeds.detach().cpu(),
            "lora_state_dict": {
                k: v.cpu() for k, v in model.base.state_dict().items()
                if "lora_" in k
            },
            "epoch": epoch + 1,
            "K_seg": args.K_seg,
        }, ckpt_path)
        print(f"  saved → {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--video_dirs", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--K_seg", type=int, default=4)
    parser.add_argument("--frames_per_segment", type=int, default=2)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max_frames", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    args = parser.parse_args()
    train(args)

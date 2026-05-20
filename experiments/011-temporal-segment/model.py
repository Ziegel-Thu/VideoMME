"""
004-voco-segment-compression: 模型加载 + VoCo 风格分段压缩

设计:
- 用 inputs_embeds 直接拼接（不通过 input_ids），更干净
- voco embeddings 是 nn.Parameter，K_seg 个/段
- 序列布局: [V1, voco1, V2, voco2, ..., Vn, voco_n, Q, A]
- attention mask:
  - 同段 vision 互看 (causal)
  - voco_t 看自己之前的 voco + 同段 vision
  - 跨段 vision 互不可见
  - Q/A 看不到任何 vision，只能看 voco + Q + 之前 A
"""

import os
import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model


class VoCoSegmentModel(nn.Module):
    """Qwen2.5-VL + 可学习 voco embeddings + LoRA。

    forward 接收已拼接的 inputs_embeds（vision + voco + text 混合），
    内部不做 token id 识别，完全靠位置信息。
    """

    def __init__(self, base_model, K_seg=4, hidden_dim=3584):
        super().__init__()
        self.base = base_model
        self.K_seg = K_seg
        self.hidden_dim = hidden_dim

        # K_seg 个可学习 voco embeddings（所有段共享同一组 base voco embeds，
        # 后续可加段位置编码区分段）
        self.voco_embeds = nn.Parameter(
            torch.randn(K_seg, hidden_dim) * 0.02
        )

    def get_voco_embeds(self, n_segments):
        """获取 n_segments 段的 voco embeddings (n_seg * K_seg, D)。

        这里所有段共享同一组 base voco，后续可加段位置编码。
        """
        # (n_seg, K_seg, D) = broadcast
        return self.voco_embeds.unsqueeze(0).expand(n_segments, -1, -1).reshape(
            n_segments * self.K_seg, self.hidden_dim,
        )

    def forward(self, inputs_embeds, attention_mask, labels=None,
                position_ids=None, voco_4d_mask=None):
        """走 LLM forward，返回 logits + loss。

        attention_mask: (B, L) 标准 padding mask（用于 Qwen 内部 mRoPE 等）
        voco_4d_mask: (B, 1, L, L) 自定义 4D mask，通过 hook 注入到每层
        """
        # 进入 Qwen2.5-VL 的 LLM 部分（绕过 visual scatter）
        m = self.base.module if hasattr(self.base, "module") else self.base
        from peft import PeftModel
        if isinstance(m, PeftModel):
            inner = m.base_model.model
        else:
            inner = m

        # 通过 forward_pre_hook 在每层 LLM decoder layer 上注入 4D mask
        layers = inner.model.language_model.layers
        hooks = []

        def make_hook(mask_4d):
            def fn(module, args, kwargs):
                kwargs["attention_mask"] = mask_4d
                return args, kwargs
            return fn

        if voco_4d_mask is not None:
            for layer in layers:
                hooks.append(layer.register_forward_pre_hook(
                    make_hook(voco_4d_mask), with_kwargs=True,
                ))

        try:
            out = inner.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
            hidden = out[0]
            logits = inner.lm_head(hidden)

            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

            return logits, loss, hidden
        finally:
            for h in hooks:
                h.remove()


def setup_voco_model(
    model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    K_seg=4,
    lora_r=16,
    lora_alpha=32,
    lora_targets=None,
    device=None,
    gradient_checkpointing=False,
    attn_implementation="eager",
    use_lora=True,
):
    """加载 Qwen2.5-VL，包装 VoCoSegmentModel。

    Args:
        attn_implementation: "eager" (支持自定义 4D mask) 或 "sdpa" (更快，拼接版用)
        lora_targets: LoRA target modules 列表，默认全部 7 个
        use_lora: True=加 LoRA 微调 LLM；False=完全冻结 LLM，只训 voco_embeds
    """
    if device is None:
        device = torch.device("cuda")

    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    ).to(device)
    base.config.use_cache = False

    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer

    # 冻结 vision encoder
    for p in base.visual.parameters():
        p.requires_grad = False

    if use_lora:
        if lora_targets is None:
            lora_targets = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules=lora_targets,
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        base = get_peft_model(base, lora_config)

        if gradient_checkpointing:
            base.enable_input_require_grads()
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        base.print_trainable_parameters()
    else:
        # 完全冻结 LLM，只有 VoCoSegmentModel.voco_embeds 可训练
        for p in base.parameters():
            p.requires_grad = False
        n_total = sum(p.numel() for p in base.parameters())
        print(f"LLM 完全冻结 ({n_total:,} 参数)，只训练 voco_embeds "
              f"({K_seg} × 3584 = {K_seg * 3584:,} 参数)")

    # 包装 VoCoSegmentModel
    model = VoCoSegmentModel(base, K_seg=K_seg, hidden_dim=3584).to(
        device, dtype=torch.bfloat16,
    )

    return model, processor, tokenizer


def get_video_embeds(base_model, pixel_values_videos, video_grid_thw, device):
    """跑 vision encoder 拿 dense video embeddings。

    Returns:
        video_embeds: (total_tokens, D) — 所有帧的 dense vision tokens
        tokens_per_frame: int
        n_frames: int
    """
    m = base_model.module if hasattr(base_model, "module") else base_model
    from peft import PeftModel
    if isinstance(m, PeftModel):
        inner = m.base_model.model
    else:
        inner = m

    pv = pixel_values_videos.to(device, dtype=torch.bfloat16)
    vg = video_grid_thw.to(device)

    with torch.no_grad():  # vision encoder 冻结
        out = inner.model.get_video_features(pv, vg)
        embeds = torch.cat(out.pooler_output, dim=0)  # (total, D)

    t, h, w = vg[0].tolist()
    tokens_per_frame = embeds.shape[0] // t

    return embeds, tokens_per_frame, t


def split_into_segments(video_embeds, tokens_per_frame, frames_per_segment):
    """把 dense video embeds 按帧切分成段。

    Args:
        video_embeds: (total, D)
        tokens_per_frame: int
        frames_per_segment: 每段帧数（如 2s/段, 1fps → 2帧/段）

    Returns:
        list of (seg_tokens, D) tensors
    """
    n_frames = video_embeds.shape[0] // tokens_per_frame
    segments = []
    for s in range(0, n_frames, frames_per_segment):
        e = min(s + frames_per_segment, n_frames)
        seg = video_embeds[s * tokens_per_frame: e * tokens_per_frame]
        segments.append(seg)
    return segments


def build_voco_sequence_embeds(
    segments, voco_embeds_per_seg, q_embeds, a_embeds=None,
):
    """构造 VoCo 序列的 inputs_embeds。

    序列: [V1, voco1, V2, voco2, ..., Vn, voco_n, Q, A?]

    Args:
        segments: list of (V_t_tokens, D) — 每段 vision embeds
        voco_embeds_per_seg: (K_seg, D) — 每段 voco（可学习参数共享）
        q_embeds: (Q_len, D)
        a_embeds: (A_len, D) or None

    Returns:
        inputs_embeds: (1, total_L, D)
        layout_info: dict with segment ranges, voco positions, q/a positions
    """
    parts = []
    layout = {
        "segments": [],  # list of (vision_start, vision_end, voco_start, voco_end)
    }

    cursor = 0
    K_seg = voco_embeds_per_seg.shape[0]
    for seg in segments:
        seg_len = seg.shape[0]
        v_start = cursor
        v_end = cursor + seg_len
        parts.append(seg)
        cursor = v_end

        voco_start = cursor
        voco_end = cursor + K_seg
        parts.append(voco_embeds_per_seg)
        cursor = voco_end

        layout["segments"].append((v_start, v_end, voco_start, voco_end))

    # Q
    q_start = cursor
    q_end = cursor + q_embeds.shape[0]
    parts.append(q_embeds)
    cursor = q_end
    layout["q"] = (q_start, q_end)

    # A (optional)
    if a_embeds is not None:
        a_start = cursor
        a_end = cursor + a_embeds.shape[0]
        parts.append(a_embeds)
        cursor = a_end
        layout["a"] = (a_start, a_end)
    else:
        layout["a"] = None

    layout["total_len"] = cursor
    inputs_embeds = torch.cat(parts, dim=0).unsqueeze(0)  # (1, L, D)

    return inputs_embeds, layout


def build_voco_attention_mask(layout, device, dtype=torch.bfloat16):
    """构造 VoCo 风格 attention mask (1, 1, L, L)。

    向量化实现，避免 Python 循环。

    规则:
    - 同段 vision 互看 (causal)
    - voco_t: 看自己之前 + 同段 vision_t (但不看其他段的 vision)
    - 跨段 vision 互不可见
    - Q/A: 看不到所有 vision，只能看 voco + Q + 之前的 A
    """
    L = layout["total_len"]

    # 构造 segment_id (vision 在段 t 标记 t, voco 在段 t 标记 t, 其他 -1)
    is_vision = torch.zeros(L, dtype=torch.bool, device=device)
    is_voco = torch.zeros(L, dtype=torch.bool, device=device)
    seg_id = torch.full((L,), -1, dtype=torch.long, device=device)

    for seg_idx, (vs, ve, voco_s, voco_e) in enumerate(layout["segments"]):
        is_vision[vs:ve] = True
        seg_id[vs:ve] = seg_idx
        is_voco[voco_s:voco_e] = True
        seg_id[voco_s:voco_e] = seg_idx

    qs, qe = layout["q"]
    is_q = torch.zeros(L, dtype=torch.bool, device=device)
    is_q[qs:qe] = True

    is_a = torch.zeros(L, dtype=torch.bool, device=device)
    if layout["a"] is not None:
        a_s, a_e = layout["a"]
        is_a[a_s:a_e] = True

    is_qa = is_q | is_a

    # Causal mask (i can see j iff j <= i)
    idx = torch.arange(L, device=device)
    causal = idx.unsqueeze(1) >= idx.unsqueeze(0)  # (L, L)

    # 起步: causal 允许的 = 0，禁止的 = -inf
    mask = torch.where(causal, 0.0, float("-inf"))

    # 计算"j 是 vision 但 i 不能看"
    # block 条件:
    #   1. j 是 vision，i 是 Q/A → block
    #   2. j 是 vision，i 是 vision，且 seg_id[i] != seg_id[j] → block
    #   3. j 是 vision，i 是 voco，且 seg_id[i] != seg_id[j] → block
    j_is_vision = is_vision.unsqueeze(0)  # (1, L)
    i_is_qa = is_qa.unsqueeze(1)  # (L, 1)
    i_is_vision = is_vision.unsqueeze(1)
    i_is_voco = is_voco.unsqueeze(1)

    seg_id_i = seg_id.unsqueeze(1)
    seg_id_j = seg_id.unsqueeze(0)
    cross_segment = seg_id_i != seg_id_j  # (L, L)

    # block_qa: i 是 Q/A 且 j 是 vision
    block_qa = i_is_qa & j_is_vision
    # block_vision_cross: i 是 vision 且 j 是 vision 且不同段
    block_v_cross = i_is_vision & j_is_vision & cross_segment
    # block_voco_cross: i 是 voco 且 j 是 vision 且不同段
    block_voco_cross = i_is_voco & j_is_vision & cross_segment

    block = block_qa | block_v_cross | block_voco_cross
    mask = torch.where(block, float("-inf"), mask)

    return mask.unsqueeze(0).unsqueeze(0).to(dtype)  # (1, 1, L, L)

"""
模型加载与 Bottleneck Mask 构造

核心组件:
- setup_model_and_tokenizer: 加载 Qwen2.5-VL + LoRA + latent tokens
- build_bottleneck_mask: 构造 LIVR-style 4D attention mask
- save/load checkpoint

注意: 必须使用 attn_implementation="eager"，SDPA 会忽略自定义 4D mask
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model

# Qwen2.5-VL vision token IDs（固定值，不同 transformers 版本可能不同）
VISION_START_ID = 151652
VISION_END_ID = 151653


def setup_model_and_tokenizer(
    model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    K=32,
    lora_r=16,
    lora_alpha=32,
    device=None,
    gradient_checkpointing=False,
):
    """加载 Qwen2.5-VL，添加 latent tokens，配置 LoRA。

    顺序: 加载模型 → resize embedding → 初始化 latent → LoRA → 冻结 → grad hook
    DDP wrap 应在此函数返回后进行。

    Args:
        model_name: HuggingFace 模型名
        K: latent token 数量
        lora_r: LoRA rank
        lora_alpha: LoRA alpha
        device: 目标设备 (e.g., torch.device("cuda:0"))
        gradient_checkpointing: 是否启用梯度检查点（节省显存，允许更多帧）

    Returns:
        model, processor, tokenizer, latent_token_ids
    """
    if device is None:
        device = torch.device("cuda")

    # 必须用 eager attention — SDPA 会忽略自定义 4D mask
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)

    # 训练时关闭 KV cache（gradient checkpointing 要求）
    model.config.use_cache = False

    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer

    # 添加 K 个 latent special tokens
    latent_tokens = [f"<latent_{i}>" for i in range(K)]
    tokenizer.add_special_tokens({"additional_special_tokens": latent_tokens})
    model.resize_token_embeddings(len(tokenizer))
    latent_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in latent_tokens]

    # 初始化 latent token embeddings（DDP wrap 前，所有 rank 用相同初始化）
    embed_layer = model.get_input_embeddings()
    for tid in latent_token_ids:
        embed_layer.weight.data[tid].normal_(mean=0.0, std=0.02)

    # 冻结 vision encoder
    for p in model.visual.parameters():
        p.requires_grad = False

    # LoRA on LLM
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Gradient checkpointing
    if gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # 解冻 embedding 但用 grad hook 只更新 latent 行
    embed_layer = model.get_input_embeddings()
    embed_layer.weight.requires_grad_(True)

    latent_ids_set = set(latent_token_ids)

    def _embedding_grad_hook(grad):
        mask = torch.zeros_like(grad)
        for tid in latent_ids_set:
            mask[tid] = 1.0
        return grad * mask

    embed_layer.weight.register_hook(_embedding_grad_hook)

    model.print_trainable_parameters()
    return model, processor, tokenizer, latent_token_ids


def get_language_model_layers(model):
    """获取 LLM decoder layers，兼容 DDP 和非 DDP 模型。"""
    m = model.module if hasattr(model, "module") else model
    try:
        return m.base_model.model.model.language_model.layers
    except AttributeError:
        # 部分 transformers 版本结构不同
        return m.base_model.model.model.layers


def install_bottleneck_hooks(layers, mask_4d):
    """在所有 decoder layers 上注册 bottleneck mask hook。

    返回 hooks 列表，调用方必须在 backward 完成后 remove。
    """
    hooks = []

    def make_hook(mask):
        def hook_fn(module, args, kwargs):
            kwargs["attention_mask"] = mask
            return args, kwargs
        return hook_fn

    for layer in layers:
        h = layer.register_forward_pre_hook(make_hook(mask_4d), with_kwargs=True)
        hooks.append(h)

    return hooks


def build_bottleneck_mask(input_ids, latent_token_ids, enable_bottleneck=True):
    """构造 LIVR-style 4D bottleneck attention mask。

    LIVR 规则:
      - vision/latent tokens: 正常 causal
      - question/answer tokens: 不能看 vision，只能看 latent + text

    Args:
        input_ids: (B, L) token IDs
        latent_token_ids: latent token ID 列表
        enable_bottleneck: False 则返回普通 causal mask

    Returns:
        (B, 1, L, L) float mask，被 block 的位置为 -inf
    """
    B, L = input_ids.shape
    device = input_ids.device

    # 基础 causal mask
    causal = torch.triu(
        torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1
    )
    mask = torch.where(causal, float("-inf"), 0.0)
    mask = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, L, L).clone()

    if not enable_bottleneck:
        return mask

    latent_set = torch.tensor(latent_token_ids, device=device)

    for b in range(B):
        ids = input_ids[b]

        # 找 vision token 区间
        is_vision = torch.zeros(L, dtype=torch.bool, device=device)
        starts = (ids == VISION_START_ID).nonzero(as_tuple=True)[0]
        ends = (ids == VISION_END_ID).nonzero(as_tuple=True)[0]
        for s, e in zip(starts, ends):
            is_vision[s:e + 1] = True

        # 找 latent token 位置
        is_latent = (ids.unsqueeze(-1) == latent_set).any(-1)

        if not is_vision.any() or not is_latent.any():
            continue

        # LIVR: 只有 vision 和 latent 能看 vision
        can_see_vision = is_vision | is_latent
        blocked_rows = ~can_see_vision
        vision_cols = is_vision
        block = blocked_rows.unsqueeze(1) & vision_cols.unsqueeze(0)
        causal_valid = (
            torch.arange(L, device=device).unsqueeze(1)
            >= torch.arange(L, device=device).unsqueeze(0)
        )
        block = block & causal_valid
        mask[b, 0, block] = float("-inf")

    return mask


def save_checkpoint(model, latent_token_ids, path, epoch=None, val_loss=None):
    """保存 LoRA 权重 + latent embeddings（自动 unwrap DDP）。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    m = model.module if hasattr(model, "module") else model
    embed = m.get_input_embeddings()

    ckpt = {
        "lora_state_dict": {
            k: v.cpu() for k, v in m.state_dict().items()
            if "lora_" in k or "latent" in k
        },
        "latent_embeddings": {
            tid: embed.weight.data[tid].cpu().clone()
            for tid in latent_token_ids
        },
        "latent_token_ids": latent_token_ids,
    }
    if epoch is not None:
        ckpt["epoch"] = epoch
    if val_loss is not None:
        ckpt["val_loss"] = val_loss

    torch.save(ckpt, path)
    print(f"  Checkpoint → {path}")


def load_checkpoint(model, ckpt_path):
    """加载 checkpoint，恢复 LoRA 权重和 latent embeddings。

    返回 (epoch, val_loss)。
    """
    m = model.module if hasattr(model, "module") else model
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    missing, unexpected = m.load_state_dict(
        ckpt["lora_state_dict"], strict=False
    )
    # 验证 LoRA 权重确实加载了
    lora_loaded = [k for k in ckpt["lora_state_dict"] if "lora_" in k]
    lora_missing = [k for k in missing if "lora_" in k]
    if lora_missing:
        print(f"  ⚠️ {len(lora_missing)} LoRA keys 未加载: {lora_missing[:5]}...")
    else:
        print(f"  ✓ {len(lora_loaded)} LoRA keys 已加载")

    embed = m.get_input_embeddings()
    for tid, emb in ckpt["latent_embeddings"].items():
        embed.weight.data[int(tid)] = emb.to(embed.weight.device)
    print(f"  ✓ {len(ckpt['latent_embeddings'])} latent embeddings 已恢复")

    return ckpt.get("epoch", 0), ckpt.get("val_loss", float("inf"))

"""
MVP: Latent Visual Bottleneck on Qwen2.5-VL-7B

核心: 在 Qwen2.5-VL 的输入序列中插入 K 个 latent token,
修改 attention mask 使 answer tokens 只能看 latent + text, 不能看 vision tokens。

Step 1: 加载模型 + 加 latent token
Step 2: 构造 bottleneck attention mask
Step 3: LoRA + NLL 训练
"""

import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model


# ============================================================
# Step 1: 加载模型，扩展 tokenizer 加 latent tokens
# ============================================================

def setup_model_and_tokenizer(
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    K: int = 8,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    """加载 Qwen2.5-VL，加 latent tokens，配 LoRA"""

    # 加载模型（用 sdpa attention）
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda")

    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer

    # 添加 K 个 latent special tokens
    latent_tokens = [f"<latent_{i}>" for i in range(K)]
    tokenizer.add_special_tokens({"additional_special_tokens": latent_tokens})
    model.resize_token_embeddings(len(tokenizer))

    latent_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in latent_tokens]

    # 冻 vision encoder
    for param in model.visual.parameters():
        param.requires_grad = False

    # LoRA on LLM
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # 解冻 latent token 的 embedding
    embed_layer = model.get_input_embeddings()
    # 初始化 latent token embeddings
    for tid in latent_token_ids:
        embed_layer.weight.data[tid].normal_(mean=0.0, std=0.02)
    # 解冻整个 embedding 但用 hook 只更新 latent 行
    embed_layer.weight.requires_grad_(True)

    latent_ids_set = set(latent_token_ids)
    def embedding_grad_hook(grad):
        mask = torch.zeros_like(grad)
        for tid in latent_ids_set:
            mask[tid] = 1.0
        return grad * mask

    embed_layer.weight.register_hook(embedding_grad_hook)

    model.print_trainable_parameters()

    return model, processor, tokenizer, latent_token_ids


# ============================================================
# Step 2: 构造 bottleneck attention mask
# ============================================================

def build_bottleneck_mask(
    input_ids: torch.Tensor,
    latent_token_ids: list[int],
    vision_start_id: int = 151652,
    vision_end_id: int = 151653,
    enable_bottleneck: bool = True,
) -> torch.Tensor:
    """构造 4D bottleneck attention mask（向量化，无 Python 循环）。"""
    B, L = input_ids.shape
    device = input_ids.device

    # 基础 causal mask
    causal = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
    mask = torch.where(causal, float("-inf"), 0.0)
    mask = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, L, L).clone()

    if not enable_bottleneck:
        return mask

    latent_set = torch.tensor(latent_token_ids, device=device)

    for b in range(B):
        ids = input_ids[b]

        # 找 vision 区间：vision_start 到 vision_end 之间
        is_vision = torch.zeros(L, dtype=torch.bool, device=device)
        starts = (ids == vision_start_id).nonzero(as_tuple=True)[0]
        ends = (ids == vision_end_id).nonzero(as_tuple=True)[0]
        for s, e in zip(starts, ends):
            is_vision[s:e+1] = True

        # 找 latent token 位置
        is_latent = (ids.unsqueeze(-1) == latent_set).any(-1)
        latent_pos = is_latent.nonzero(as_tuple=True)[0]

        if not is_vision.any() or len(latent_pos) == 0:
            continue

        last_latent = latent_pos[-1].item()

        # 所有 last_latent 之后的 token → 不能看 vision positions
        # 构造 block mask: (L,) × (L,) → (L, L)
        after_latent = torch.arange(L, device=device) > last_latent  # (L,)
        vision_cols = is_vision  # (L,)

        # block[i,j] = True if i > last_latent AND j is vision AND j <= i (causal)
        block = after_latent.unsqueeze(1) & vision_cols.unsqueeze(0)  # (L, L)
        causal_valid = torch.arange(L, device=device).unsqueeze(1) >= torch.arange(L, device=device).unsqueeze(0)
        block = block & causal_valid

        mask[b, 0, block] = float("-inf")

    return mask


# ============================================================
# Step 3: 构造训练数据
# ============================================================

def build_training_input(
    processor,
    tokenizer,
    latent_tokens: list[str],
    image_path: str,
    question: str,
    answer: str,
):
    """构造单条训练样本的输入。

    输入序列: <|im_start|>system\n...<|im_end|>
              <|im_start|>user\n<image>\n<latent_0>...<latent_K>\nQuestion<|im_end|>
              <|im_start|>assistant\nAnswer<|im_end|>
    """
    from PIL import Image

    latent_str = "".join(latent_tokens)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": f"{latent_str}\n{question}"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": answer},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # 加载图片
    from qwen_vl_utils import process_vision_info
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    )

    return inputs


# ============================================================
# Step 4: Smoke test
# ============================================================

def smoke_test():
    """快速验证: 模型能加载，bottleneck mask 能构造，forward 不报错"""

    print("=== MVP Smoke Test ===\n")

    K = 8
    print(f"[1/4] 加载模型 + {K} latent tokens...")
    model, processor, tokenizer, latent_token_ids = setup_model_and_tokenizer(K=K)

    latent_tokens = [f"<latent_{i}>" for i in range(K)]
    print(f"  Latent token IDs: {latent_token_ids}")

    print("\n[2/4] 构造训练输入...")
    # 用一张简单的测试图片
    from PIL import Image
    import tempfile, os
    # 创建一张纯色测试图片
    img = Image.new("RGB", (224, 224), color="red")
    tmp_path = os.path.join(tempfile.gettempdir(), "test_red.png")
    img.save(tmp_path)

    inputs = build_training_input(
        processor, tokenizer, latent_tokens,
        image_path=tmp_path,
        question="What color is this image?",
        answer="The image is red.",
    )

    input_ids = inputs["input_ids"]
    print(f"  input_ids shape: {input_ids.shape}")
    print(f"  token 序列中的 latent tokens: ", end="")
    for i, tid in enumerate(input_ids[0].tolist()):
        if tid in latent_token_ids:
            print(f"pos={i} ", end="")
    print()

    # 找 vision tokens
    vision_count = 0
    for tid in input_ids[0].tolist():
        if tid >= 151652 and tid <= 151656:
            vision_count += 1
    print(f"  vision-related tokens: {vision_count}")

    print("\n[3/4] 构造 bottleneck mask...")
    mask_bn = build_bottleneck_mask(input_ids, latent_token_ids, enable_bottleneck=True)
    mask_no = build_bottleneck_mask(input_ids, latent_token_ids, enable_bottleneck=False)

    # 统计被 block 的位置数
    bn_blocked = (mask_bn == float("-inf")).sum().item()
    no_blocked = (mask_no == float("-inf")).sum().item()
    print(f"  Bottleneck ON  — blocked positions: {bn_blocked}")
    print(f"  Bottleneck OFF — blocked positions: {no_blocked}")
    print(f"  额外 block 了: {bn_blocked - no_blocked} 个位置")

    print("\n[4/4] Forward pass...")
    device = next(model.parameters()).device

    # 构造 labels（只算 answer 部分的 loss）
    labels = input_ids.clone()
    # 找 assistant 回答开始的位置，之前的都 mask 掉
    ids_list = input_ids[0].tolist()
    # 简单方式：找最后一个 <|im_start|> 之后的内容
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    last_im_start = 0
    for i, tid in enumerate(ids_list):
        if tid == im_start_id:
            last_im_start = i
    # mask 掉 assistant header (im_start + "assistant\n")
    labels[0, :last_im_start + 2] = -100

    # Move to device
    inputs_on_device = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    labels = labels.to(device)

    with torch.no_grad():
        outputs = model(
            **inputs_on_device,
            labels=labels,
        )

    print(f"  Loss (no custom mask): {outputs.loss.item():.4f}")
    print(f"\n=== Smoke Test PASSED ✅ ===")

    # 清理
    os.remove(tmp_path)

    return model, processor, tokenizer, latent_token_ids


if __name__ == "__main__":
    smoke_test()

"""
Cross-Attention 压缩模块蒸馏训练

基于预提取的 teacher 特征训练，不需要加载 vision encoder。
数据来自 extract_teacher.py 的输出。

流程 (per segment):
  1. compressed = compressor(dense_vision_seg)       # Cross-Attention 压缩
  2. student_input = [compressed, q_embeds]
  3. student_hidden = LLM(student_input)[-Q_len:]    # 冻结 LLM forward
  4. loss = MSE(student_hidden, teacher_q_hidden)     # 预提取的 teacher target
  5. backward → 更新 compressor 参数

用法:
  torchrun --nproc_per_node=4 train_compressor.py \
    --cache_dir /nvmessd/lifanhong/video/teacher_cache_10k \
    --output_dir /nvmessd/lifanhong/video/outputs_compressor_1L \
    --model_path /path/to/Qwen2.5-VL-7B-Instruct \
    --n_layers 1 --K_seg 8 --lr 1e-4 --epochs 3
"""

import os
import sys
import argparse
import glob

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from compressor import VoCoCompressor


# ============================================================
# Dataset: 从预提取的 .pt 文件加载
# ============================================================

class TeacherCacheDataset(Dataset):
    """加载 extract_teacher.py 预提取的 teacher 特征。"""

    def __init__(self, cache_dir, max_samples=None):
        self.cache_dir = validate_cache_dir(cache_dir)
        self.sample_index = []
        self._loaded_shard_path = None
        self._loaded_shard_samples = None

        shard_files = sorted(glob.glob(os.path.join(self.cache_dir, "teacher_shard_*.pt")))
        if not shard_files:
            raise ValueError(
                f"{self.cache_dir} 中没有 teacher_shard_*.pt。"
                "请先用 pack_teacher_cache.py 或 shard 版 extract_teacher.py 生成 shard cache。"
            )
        for shard_path in shard_files:
            shard_samples = torch.load(shard_path, map_location="cpu", weights_only=False)
            for item_idx in range(len(shard_samples)):
                self.sample_index.append((shard_path, item_idx))

        if max_samples:
            self.sample_index = self.sample_index[:max_samples]
        print(f"TeacherCacheDataset: {len(self.sample_index)} 个样本 from {self.cache_dir}")

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        try:
            file_path, item_idx = self.sample_index[idx]
            if file_path != self._loaded_shard_path:
                self._loaded_shard_samples = torch.load(
                    file_path, map_location="cpu", weights_only=False,
                )
                self._loaded_shard_path = file_path
            return self._loaded_shard_samples[item_idx]
        except Exception as e:
            print(f"  [加载错误] {self.sample_index[idx][0]}: {e}")
            return None


def collate_fn(batch):
    """batch_size=1，直接返回单个样本。"""
    return batch[0]


# ============================================================
# 工具函数
# ============================================================

def get_inner(base_model):
    """获取最内层 Qwen model。"""
    m = base_model.module if hasattr(base_model, "module") else base_model
    try:
        from peft import PeftModel
        if isinstance(m, PeftModel):
            return m.base_model.model
    except ImportError:
        pass
    return m


def validate_cache_dir(cache_dir):
    """只允许从本地 SSD 读取 teacher cache，拒绝 NFS。"""
    real_path = os.path.realpath(cache_dir)
    if real_path.startswith("/beegfs_hdd/"):
        raise ValueError(
            f"禁止直接从 NFS 读取 teacher cache: {real_path}。"
            "请先同步到本地 SSD (/nvmessd/...) 再训练。"
        )
    return real_path


def lm_forward_kv(inner, tokens, device):
    """单次 LLM forward → 返回 pooled KV cache (per-layer mean)。

    Args:
        tokens: (N, D) — vision/compressed tokens
    Returns:
        pooled_kv: (n_layers, D) — 每层 KV 的 mean pool
    """
    t_input = tokens.unsqueeze(0)
    t_len = tokens.shape[0]
    t_pos = torch.arange(t_len, device=device)
    t_pos_ids = t_pos.view(1, 1, -1).expand(3, 1, -1)
    t_attn = torch.ones(1, t_len, dtype=torch.long, device=device)

    out = inner.model(
        inputs_embeds=t_input,
        attention_mask=t_attn,
        position_ids=t_pos_ids,
        use_cache=True,
    )
    kv_cache = out.past_key_values

    pooled = []
    for layer_kv in kv_cache:
        k, v = layer_kv  # each (1, n_heads, N, head_dim)
        # mean pool over tokens, concat K and V
        k_pool = k.squeeze(0).mean(dim=1).mean(dim=0)  # (head_dim,)
        v_pool = v.squeeze(0).mean(dim=1).mean(dim=0)
        pooled.append(torch.cat([k_pool, v_pool]))      # (2*head_dim,)
    return torch.stack(pooled)  # (n_layers, 2*head_dim)


def student_forward_segment(inner, compressed, q_embeds, device):
    """Student forward: [compressed (K), Q (Q_len)] → Q 位置 hidden。

    Args:
        inner: Qwen 内层模型
        compressed: (K, D) — compressor 输出，有梯度
        q_embeds: (Q_len, D) — question embeddings，detached

    Returns:
        student_q_hidden: (Q_len, D) — Q 位置的 hidden states
    """
    K = compressed.shape[0]
    Q_len = q_embeds.shape[0]
    q_detached = q_embeds.detach()

    s_input = torch.cat([compressed, q_detached], dim=0).unsqueeze(0)
    s_len = K + Q_len
    s_pos = torch.arange(s_len, device=device)
    s_pos_ids = s_pos.view(1, 1, -1).expand(3, 1, -1)
    s_attn = torch.ones(1, s_len, dtype=torch.long, device=device)

    s_out = inner.model(
        inputs_embeds=s_input,
        attention_mask=s_attn,
        position_ids=s_pos_ids,
        use_cache=False,
    )
    s_hidden = s_out[0].squeeze(0)  # (K+Q_len, D)
    return s_hidden[-Q_len:]         # (Q_len, D)


# ============================================================
# 训练循环
# ============================================================

def train(args):
    # 分布式初始化
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        import datetime as _dt
        dist.init_process_group("nccl", timeout=_dt.timedelta(hours=2))
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0
        world_size = 1

    device = torch.device(f"cuda:{local_rank}")
    is_main = (local_rank == 0)

    def log(msg):
        if is_main:
            print(msg)

    log("=" * 60)
    log("Cross-Attention 压缩模块蒸馏训练")
    log(f"  GPU 数量: {world_size}")
    log(f"  K_seg: {args.K_seg}, n_layers: {args.n_layers}")
    log(f"  loss_type: {args.loss_type}")
    log(f"  lr: {args.lr}, epochs: {args.epochs}")
    log(f"  save_steps: {args.save_steps}")
    log(f"  cache_dir: {args.cache_dir}")
    log("=" * 60)

    # 加载 LLM（冻结，用于 student forward）
    from transformers import Qwen2_5_VLForConditionalGeneration
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device)
    base.eval()
    for p in base.parameters():
        p.requires_grad = False
    inner = get_inner(base)
    n_llm = sum(p.numel() for p in base.parameters())
    log(f"  LLM 冻结: {n_llm:,} 参数")

    # 创建压缩模块
    compressor = VoCoCompressor(
        K=args.K_seg, dim=3584,
        n_heads=args.n_heads, n_layers=args.n_layers,
    ).to(device, dtype=torch.bfloat16)
    n_comp = compressor.num_params()
    log(f"  Compressor: {n_comp:,} 参数 ({args.n_layers} 层)")

    # 多卡同步 compressor 初始权重
    if world_size > 1:
        for p in compressor.parameters():
            dist.broadcast(p.data, src=0)

    # 数据集
    dataset = TeacherCacheDataset(args.cache_dir, max_samples=args.max_samples)
    if len(dataset) == 0:
        log("⚠️ 数据集为空，退出")
        return

    # shard 文件很大，保持按 shard 顺序访问，避免在不同 shard 间频繁来回切换。
    train_sampler = DistributedSampler(dataset, shuffle=False) if world_size > 1 else None
    loader = DataLoader(
        dataset, batch_size=1,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=collate_fn,
    )

    # 优化器
    optimizer = torch.optim.AdamW(
        compressor.parameters(), lr=args.lr, weight_decay=0.01,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0
    start_epoch = 0

    # 从 checkpoint 恢复
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu",
                          weights_only=False)
        compressor.load_state_dict(ckpt["compressor"])
        start_epoch = ckpt.get("epoch", 0)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        log(f"  从 {args.resume_from} 恢复，start_epoch={start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        compressor.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)
        epoch_loss = 0.0
        epoch_n = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}",
                    disable=not is_main)
        for sample in pbar:
            if sample is None:
                continue

            segments = sample["segments"]       # list of (V_i, D) tensors
            q_embeds = sample["q_embeds"]       # (Q_len, D)
            teacher_targets = sample["teacher_q_hidden"]  # list of (Q_len, D)
            global_step += 1

            try:
                optimizer.zero_grad()
                sample_loss = 0.0
                n_segs = 0

                # 逐段: compress → loss
                for seg_vision, teacher_target in zip(segments, teacher_targets):
                    seg_vision = seg_vision.to(device, dtype=torch.bfloat16)
                    q_emb = q_embeds.to(device, dtype=torch.bfloat16)

                    # Compress: (V, D) → (K, D)
                    compressed = compressor(seg_vision)

                    loss = torch.tensor(0.0, device=device)

                    if args.loss_type in ("B", "BD"):
                        teacher_target = teacher_target.to(device, dtype=torch.bfloat16)
                        student_q_hidden = student_forward_segment(
                            inner, compressed, q_emb, device,
                        )
                        loss_b = nn.functional.mse_loss(
                            student_q_hidden, teacher_target.detach(),
                        )
                        loss = loss + loss_b

                    if args.loss_type in ("D", "BD"):
                        # Teacher KV: 现场算（no_grad）
                        with torch.no_grad():
                            teacher_kv = lm_forward_kv(
                                inner, seg_vision.detach(), device,
                            )
                        # Student KV
                        student_kv = lm_forward_kv(inner, compressed, device)
                        loss_d = nn.functional.mse_loss(student_kv, teacher_kv)
                        loss = loss + loss_d

                    loss.backward()
                    sample_loss += loss.item()
                    n_segs += 1

                # 梯度同步
                if world_size > 1:
                    for p in compressor.parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                            p.grad.div_(world_size)

                torch.nn.utils.clip_grad_norm_(compressor.parameters(), 1.0)
                optimizer.step()

                avg_seg_loss = sample_loss / max(n_segs, 1)
                epoch_loss += sample_loss
                epoch_n += n_segs

                if is_main:
                    pbar.set_postfix(
                        loss=f"{avg_seg_loss:.6f}", segs=n_segs, step=global_step,
                    )
                    if global_step % 10 == 0:
                        print(f"  [step {global_step}] loss={avg_seg_loss:.6f} "
                              f"(n_segs={n_segs})")

            except torch.cuda.OutOfMemoryError:
                if is_main:
                    print(f"  [OOM] step {global_step}，跳过")
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                if is_main:
                    print(f"  [错误] step {global_step}: {e}")
                continue

            # 定期保存
            if args.save_steps > 0 and global_step % args.save_steps == 0 and is_main:
                ckpt_path = os.path.join(
                    args.output_dir,
                    f"compressor_e{epoch+1}_s{global_step}.pt",
                )
                torch.save({
                    "compressor": compressor.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "step": global_step,
                    "K_seg": args.K_seg,
                    "n_layers": args.n_layers,
                }, ckpt_path)
                print(f"  💾 checkpoint → {ckpt_path}")

        # Epoch 结束
        avg_epoch_loss = epoch_loss / max(epoch_n, 1)
        log(f"  Epoch {epoch + 1} 完成: avg_loss={avg_epoch_loss:.6f} "
            f"(total_segs={epoch_n})")

        if is_main:
            ckpt_path = os.path.join(
                args.output_dir, f"compressor_epoch{epoch+1}.pt",
            )
            torch.save({
                "compressor": compressor.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "train_loss": avg_epoch_loss,
                "K_seg": args.K_seg,
                "n_layers": args.n_layers,
            }, ckpt_path)
            print(f"  💾 epoch checkpoint → {ckpt_path}")
        if dist.is_initialized():
            dist.barrier()

    log("\n训练完成.")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cross-Attention 压缩模块蒸馏训练",
    )
    parser.add_argument("--cache_dir", required=True,
                        help="预提取 teacher 特征目录")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--model_path", type=str,
                        default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="LLM 模型路径（用于 student forward）")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--K_seg", type=int, default=8,
                        help="每段压缩 token 数")
    parser.add_argument("--n_layers", type=int, default=1,
                        help="Cross-Attention 层数")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Attention heads 数")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--resume_from", type=str, default=None,
                        help="从 checkpoint 恢复训练")
    parser.add_argument("--loss_type", type=str, default="B",
                        choices=["B", "D", "BD"],
                        help="蒸馏 loss 类型: B=attn output, D=KV cache, BD=两者")
    args = parser.parse_args()
    train(args)

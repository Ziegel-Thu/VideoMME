"""
Stage 2: Bottleneck SFT 多卡 DDP 训练

启动方式:
  # 单卡
  python train_ddp.py --data_path ... --video_dirs ... --output_dir ... --bottleneck

  # 多卡 (torchrun)
  torchrun --nproc_per_node=4 train_ddp.py --data_path ... --video_dirs ... --output_dir ... --bottleneck

  # amlt 集群
  amlt run amlt.yaml
"""

import os
import json
import argparse
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from model import (
    setup_model_and_tokenizer,
    build_bottleneck_mask,
    install_bottleneck_hooks,
    save_checkpoint,
    load_checkpoint,
    get_language_model_layers,
)
from data import VideoQADataset, collate_fn, TemporalHead


# ============================================================
# 分布式工具
# ============================================================

def setup_distributed():
    """初始化分布式环境，返回 (local_rank, world_size)。"""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return local_rank, world_size
    else:
        return 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def log(msg):
    if is_main():
        print(msg)


def reduce_mean(value, world_size):
    """跨 GPU 平均一个标量值。"""
    if world_size <= 1:
        return value
    t = torch.tensor(value, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item() / world_size


# ============================================================
# 数据划分
# ============================================================

def split_dataset(dataset, n_val=500, n_test=500, seed=42):
    """将数据集划分为 train / val / test。

    Args:
        dataset: 完整数据集
        n_val: 验证集大小（上限）
        n_test: 测试集大小（上限）
        seed: 随机种子（所有 rank 必须相同）

    Returns:
        train_set, val_set, test_set
    """
    total = len(dataset)
    n_test = min(n_test, total // 10)
    n_val = min(n_val, total // 10)
    n_train = total - n_val - n_test

    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )
    log(f"数据划分: train={n_train}, val={n_val}, test={n_test}")
    return train_set, val_set, test_set


# ============================================================
# Hidden state 捕获 hook（替代 output_hidden_states=True，省显存）
# ============================================================

class HiddenStateCapture:
    """在最后一层 decoder 后捕获 hidden states，只保留 latent 位置。

    用法:
        cap = HiddenStateCapture(layers[-1], latent_token_ids)
        cap.install()
        outputs = model(**batch)
        latent_hidden = cap.get(batch["input_ids"])  # (B, K, D)
        cap.remove()
    """

    def __init__(self, last_layer, latent_token_ids):
        self.latent_set = set(latent_token_ids)
        self.last_layer = last_layer
        self._hidden = None
        self._hook = None

    def install(self):
        def hook_fn(module, input, output):
            # output[0] 是 hidden_states (B, L, D)
            self._hidden = output[0]
        self._hook = self.last_layer.register_forward_hook(hook_fn)

    def get(self, input_ids):
        """提取 latent token 位置的 hidden states，返回 (B, K, D)。"""
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
        return torch.stack(results)  # (B, K, D)

    def remove(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        self._hidden = None


# ============================================================
# 训练
# ============================================================

def train(args):
    local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    log("=" * 60)
    log("Bottleneck DDP 训练")
    log(f"  GPU 数量: {world_size}")
    log(f"  帧数: {args.num_frames}")
    log(f"  K={args.K}, lr={args.lr}, epochs={args.epochs}")
    log(f"  梯度累积: {args.grad_accum}")
    log(f"  等效 batch size: {world_size * args.grad_accum}")
    log(f"  Gradient checkpointing: {args.gradient_checkpointing}")
    log(f"  Bottleneck: {args.bottleneck}")
    log("=" * 60)

    # --- 模型 ---
    model, processor, tokenizer, latent_token_ids = setup_model_and_tokenizer(
        K=args.K, lora_r=args.lora_r, device=device,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    latent_tokens = [f"<latent_{i}>" for i in range(args.K)]

    # 加载 checkpoint
    start_epoch = 0
    if args.resume_from:
        start_epoch, _ = load_checkpoint(model, args.resume_from)
        log(f"从 epoch {start_epoch} 继续训练")

    # DDP 广播 rank 0 的参数（确保 latent embedding 一致）
    if world_size > 1:
        for p in model.parameters():
            dist.broadcast(p.data, src=0)

    # DDP wrap
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=False)

    # --- 数据 ---
    video_dirs = args.video_dirs.split(",")
    dataset = VideoQADataset(
        args.data_path, video_dirs, max_samples=args.max_samples,
    )

    if args.overfit:
        subset = Subset(dataset, range(min(4, len(dataset))))
        train_set, val_set = subset, subset
        test_set = subset
        log(f"Overfit 模式: {len(subset)} 条")
    else:
        train_set, val_set, test_set = split_dataset(
            dataset, n_val=args.n_val, n_test=args.n_test,
        )

    # 保存 test set 的 indices 供后续评测使用
    if is_main():
        os.makedirs(args.output_dir, exist_ok=True)
        if hasattr(test_set, 'indices'):
            with open(os.path.join(args.output_dir, "test_indices.json"), "w") as f:
                json.dump(test_set.indices, f)
            log(f"Test set indices 已保存 → {args.output_dir}/test_indices.json")

    # Sampler
    train_sampler = DistributedSampler(train_set, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if world_size > 1 else None

    def make_collate(proc, tok, lt, nf, ltids):
        def wrapper(batch):
            return collate_fn(batch, proc, tok, lt, nf, ltids)
        return wrapper

    collate = make_collate(processor, tokenizer, latent_tokens,
                           args.num_frames, latent_token_ids)

    train_loader = DataLoader(
        train_set, batch_size=1, sampler=train_sampler,
        shuffle=(train_sampler is None), collate_fn=collate,
    )
    val_loader = DataLoader(
        val_set, batch_size=1, sampler=val_sampler,
        shuffle=False, collate_fn=collate,
    )

    # --- Temporal Head（可选）---
    temporal_head = None
    if args.temporal_head:
        temporal_head = TemporalHead(3584, num_bins=args.num_frames).to(device)
        if world_size > 1:
            temporal_head = DDP(temporal_head, device_ids=[local_rank],
                                find_unused_parameters=False)
        th_raw = temporal_head.module if hasattr(temporal_head, "module") else temporal_head
        log(f"Temporal Head: {sum(p.numel() for p in th_raw.parameters())} params")

    # --- Optimizer ---
    params = [p for p in model.parameters() if p.requires_grad]
    if temporal_head:
        params += [p for p in temporal_head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)

    # LLM layers（用于挂 hook）
    layers = get_language_model_layers(model)
    os.makedirs(args.output_dir, exist_ok=True)
    best_val = float("inf")

    # --- 训练循环 ---
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        if temporal_head:
            temporal_head.train()
        if train_sampler:
            train_sampler.set_epoch(epoch)

        total_loss = 0.0
        n_steps = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}",
                    disable=not is_main())

        for step, batch in enumerate(pbar):
            t_labels = batch.pop("_temporal_labels")
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # 是否在这一步同步梯度
            is_sync_step = (step + 1) % args.grad_accum == 0
            sync_ctx = (
                nullcontext() if (world_size <= 1 or is_sync_step)
                else model.no_sync()
            )

            # Bottleneck hooks + forward + backward（try/finally 防泄漏）
            hooks = []
            hidden_cap = None
            try:
                if args.bottleneck:
                    bn_mask = build_bottleneck_mask(
                        batch["input_ids"], latent_token_ids,
                        enable_bottleneck=True,
                    ).to(device, dtype=torch.bfloat16)
                    hooks = install_bottleneck_hooks(layers, bn_mask)

                # 用 hook 捕获最后层 hidden state（temporal head 用）
                if temporal_head:
                    hidden_cap = HiddenStateCapture(layers[-1], latent_token_ids)
                    hidden_cap.install()

                with sync_ctx:
                    outputs = model(**batch)
                    loss = outputs.loss / args.grad_accum

                    # Temporal loss
                    if temporal_head and hidden_cap:
                        loss_temp = _compute_temporal_loss(
                            hidden_cap, batch["input_ids"],
                            temporal_head, t_labels, device,
                        )
                        loss = loss + args.temporal_weight * loss_temp / args.grad_accum

                    loss.backward()

            finally:
                for h in hooks:
                    h.remove()
                if hidden_cap:
                    hidden_cap.remove()

            # 梯度累积完成 → 更新参数
            if is_sync_step:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * args.grad_accum
            n_steps += 1

            if is_main():
                pbar.set_postfix(loss=f"{loss.item() * args.grad_accum:.4f}")

        # 处理末尾不完整的累积步
        if n_steps % args.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = reduce_mean(total_loss / max(n_steps, 1), world_size)
        log(f"  Epoch {epoch + 1}: train_loss={avg_loss:.4f}")

        # --- Validation ---
        val_loss = validate(model, val_loader, layers, latent_token_ids,
                            args, device, world_size)
        log(f"  Epoch {epoch + 1}: val_loss={val_loss:.4f}")

        # --- Checkpoint（只 rank 0 保存）---
        if is_main():
            save_checkpoint(
                model, latent_token_ids,
                os.path.join(args.output_dir, f"checkpoint_epoch{epoch + 1}.pt"),
                epoch=epoch + 1, val_loss=val_loss,
            )
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    model, latent_token_ids,
                    os.path.join(args.output_dir, "best_model.pt"),
                    epoch=epoch + 1, val_loss=val_loss,
                )
                if temporal_head:
                    th = (temporal_head.module
                          if hasattr(temporal_head, "module")
                          else temporal_head)
                    torch.save(
                        th.state_dict(),
                        os.path.join(args.output_dir, "temporal_head.pt"),
                    )
                log(f"  ★ New best (val_loss={val_loss:.4f})")

        # 同步 best_val
        if world_size > 1:
            best_t = torch.tensor(best_val, device=device)
            dist.broadcast(best_t, src=0)
            best_val = best_t.item()

    log(f"\n训练完成. best_val_loss={best_val:.4f}")
    cleanup_distributed()


@torch.no_grad()
def validate(model, val_loader, layers, latent_token_ids, args, device,
             world_size):
    """验证循环，返回跨 GPU 平均的 val_loss。"""
    model.eval()
    loss_sum = 0.0
    count = 0

    for batch in val_loader:
        batch.pop("_temporal_labels")
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        hooks = []
        try:
            if args.bottleneck:
                bn_mask = build_bottleneck_mask(
                    batch["input_ids"], latent_token_ids,
                    enable_bottleneck=True,
                ).to(device, dtype=torch.bfloat16)
                hooks = install_bottleneck_hooks(layers, bn_mask)

            out = model(**batch)
            loss_sum += out.loss.item()
            count += 1
        finally:
            for h in hooks:
                h.remove()

    # all_reduce loss_sum 和 count，避免各 rank 样本数不同导致偏差
    if world_size > 1:
        stats = torch.tensor([loss_sum, float(count)], device=device)
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        loss_sum, count = stats[0].item(), stats[1].item()

    return loss_sum / max(count, 1)


def _compute_temporal_loss(hidden_cap, input_ids, temporal_head,
                           t_labels, device):
    """计算 temporal head 的 BCE loss。"""
    latent_hidden = hidden_cap.get(input_ids)  # (B, K, D) or None

    if latent_hidden is None or t_labels[0] is None:
        # 没有 temporal label 时，构造 dummy loss 使参数留在计算图
        th = temporal_head.module if hasattr(temporal_head, "module") else temporal_head
        return 0.0 * sum(p.sum() for p in th.parameters())

    logits = temporal_head(latent_hidden)  # (B, num_bins)
    target = torch.tensor([t_labels[0]], dtype=torch.float, device=device)
    return F.binary_cross_entropy_with_logits(logits, target)


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bottleneck SFT DDP 训练")

    # 数据
    parser.add_argument("--data_path", required=True,
                        help="训练数据 jsonl 路径")
    parser.add_argument("--video_dirs", required=True,
                        help="视频目录，逗号分隔")
    parser.add_argument("--output_dir", required=True,
                        help="输出目录（checkpoints + logs）")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大样本数（调试用）")
    parser.add_argument("--n_val", type=int, default=500,
                        help="验证集大小上限")
    parser.add_argument("--n_test", type=int, default=1000,
                        help="测试集大小上限")

    # 模型
    parser.add_argument("--K", type=int, default=32,
                        help="latent token 数量")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--num_frames", type=int, default=8,
                        help="每视频采样帧数")

    # 训练
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="学习率")
    parser.add_argument("--epochs", type=int, default=10,
                        help="训练轮数")
    parser.add_argument("--grad_accum", type=int, default=2,
                        help="梯度累积步数")
    parser.add_argument("--bottleneck", action="store_true",
                        help="启用 bottleneck mask")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="启用梯度检查点（节省显存，允许更多帧）")
    parser.add_argument("--overfit", action="store_true",
                        help="Overfit 模式（调试用）")

    # Temporal Head
    parser.add_argument("--temporal_head", action="store_true",
                        help="启用 temporal head")
    parser.add_argument("--temporal_weight", type=float, default=0.5,
                        help="temporal loss 权重")

    # 恢复训练
    parser.add_argument("--resume_from", default=None,
                        help="从 checkpoint 继续训练")

    args = parser.parse_args()
    train(args)

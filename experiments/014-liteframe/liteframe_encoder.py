"""
LiteFrame Student Encoder — 轻量视频编码器

基于 arXiv:2605.17260 实现。ViT-Base + DW temporal conv + progressive strided downsampling。
蒸馏 Qwen2.5-VL ViT (1280D, 32层) → 轻量 ViT-Base (768D, 12层)。

架构 (4帧 448×448 输入):
  PatchEmbed (1,14,14) → T=4, H=32, W=32 = 4096 tokens @ 768D
  Blocks 0-3:  spatial attn + DW temp conv (T=4)
  StridedConv [2,2,2] → T=2, H=16, W=16 = 512 tokens
  Blocks 4-7:  spatial attn + DW temp conv (T=2)
  StridedConv [2,1,1] → T=1, H=16, W=16 = 256 tokens
  Blocks 8-11: spatial attn (T=1, 无需 temp conv)
  Output: 256 tokens → proj → 3584D

参数量: ~91M
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed3D(nn.Module):
    """3D Patch Embedding (temporal_patch_size=1 保留完整时间分辨率)。"""

    def __init__(self, patch_size=14, in_channels=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels, embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
            bias=True,
        )

    def forward(self, x):
        """(B, C, T, H, W) → (B, T', H', W', D)"""
        x = self.proj(x)  # (B, D, T, H', W')
        return x.permute(0, 2, 3, 4, 1)


class SpatialAttention(nn.Module):
    """Per-frame spatial self-attention (F.scaled_dot_product_attention 自动用 Flash)。"""

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """(B_T, HW, D) → (B_T, HW, D)"""
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, hd)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, D)
        return self.proj(x)


class DWTemporalConv(nn.Module):
    """Depth-wise 1D temporal convolution (参数极少: dim × kernel_size)。"""

    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=dim,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, T, H, W):
        """(B, T*H*W, D) → (B, T*H*W, D)"""
        B, _, D = x.shape
        # (B, T, H*W, D) → (B*H*W, D, T) for temporal conv
        x = x.view(B, T, H * W, D).permute(0, 2, 3, 1).reshape(B * H * W, D, T)
        x = self.conv(x)
        x = x.reshape(B, H * W, D, T).permute(0, 3, 1, 2).reshape(B, T * H * W, D)
        return self.norm(x)


class StridedDWConv3D(nn.Module):
    """Depth-wise strided 3D convolution for progressive downsampling。"""

    def __init__(self, dim, stride=(2, 2, 2)):
        super().__init__()
        self.conv = nn.Conv3d(
            dim, dim, kernel_size=3, stride=stride,
            padding=1, groups=dim,
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, T, H, W):
        """(B, T*H*W, D) → (B, T'*H'*W', D), T', H', W'"""
        B, _, D = x.shape
        x = x.view(B, T, H, W, D).permute(0, 4, 1, 2, 3)  # (B, D, T, H, W)
        x = self.conv(x)
        T2, H2, W2 = x.shape[2], x.shape[3], x.shape[4]
        x = x.permute(0, 2, 3, 4, 1).reshape(B, T2 * H2 * W2, D)
        return self.norm(x), T2, H2, W2


class LiteFrameBlock(nn.Module):
    """Transformer block: pre-norm spatial attn + FFN + DW temporal conv。"""

    def __init__(self, dim, num_heads, ffn_mult=4, has_temporal_conv=True):
        super().__init__()
        self.has_temporal_conv = has_temporal_conv
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SpatialAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        ffn_dim = int(dim * ffn_mult)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )
        self.temporal_conv = DWTemporalConv(dim) if has_temporal_conv else None

    def forward(self, x, T, H, W):
        B = x.shape[0]
        # Spatial self-attention (per-frame)
        res = x
        x_normed = self.norm1(x)
        x_frames = x_normed.view(B, T, H * W, -1).reshape(B * T, H * W, -1)
        x_frames = self.attn(x_frames)
        x = res + x_frames.reshape(B, T * H * W, -1)
        # FFN
        x = x + self.ffn(self.norm2(x))
        # DW temporal conv (always applied, even when T=1)
        if self.temporal_conv is not None:
            x = x + self.temporal_conv(x, T, H, W)
        return x


class LiteFrameEncoder(nn.Module):
    """LiteFrame 轻量视频编码器。

    Args:
        output_dim: 输出维度 (3584 = Qwen2.5-VL LLM hidden)
        downsample_config: {layer_idx: (stride_t, stride_h, stride_w)}
    """

    def __init__(
        self,
        patch_size=14,
        in_channels=3,
        hidden_dim=768,
        num_heads=12,
        num_layers=12,
        ffn_mult=4,
        output_dim=3584,
        downsample_config=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        if downsample_config is None:
            downsample_config = {4: (2, 2, 2), 8: (2, 1, 1)}

        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, in_channels=in_channels, embed_dim=hidden_dim,
        )

        # Factorized 3D 位置编码 (默认 T=4, H=32, W=32)
        self.pos_embed_t = nn.Parameter(torch.randn(1, 4, 1, 1, hidden_dim) * 0.02)
        self.pos_embed_h = nn.Parameter(torch.randn(1, 1, 32, 1, hidden_dim) * 0.02)
        self.pos_embed_w = nn.Parameter(torch.randn(1, 1, 1, 32, hidden_dim) * 0.02)

        # Transformer blocks + downsamplers
        self.blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleDict()
        for i in range(num_layers):
            self.blocks.append(LiteFrameBlock(
                dim=hidden_dim, num_heads=num_heads, ffn_mult=ffn_mult,
                has_temporal_conv=True,
            ))
            if i in downsample_config:
                self.downsamplers[str(i)] = StridedDWConv3D(
                    hidden_dim, stride=downsample_config[i],
                )

        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv1d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _interpolate_pos(self, pe, dim_idx, target_size):
        """对单个维度的位置编码做插值。"""
        current_size = pe.shape[dim_idx + 1]  # +1 because dim 0 is batch
        if current_size == target_size:
            return pe
        # 用 1D interpolate
        # pe shape: (1, ..., size_at_dim, ..., D)
        # 先把目标维度移到倒数第二位
        ndim = pe.dim()
        perm = list(range(ndim))
        perm.remove(dim_idx + 1)
        perm.insert(-1, dim_idx + 1)
        pe = pe.permute(*perm)  # (..., size, D)
        shape_prefix = pe.shape[:-2]
        pe = pe.reshape(-1, current_size, self.hidden_dim)  # (N, size, D)
        pe = pe.permute(0, 2, 1)  # (N, D, size)
        pe = F.interpolate(pe, size=target_size, mode="linear", align_corners=False)
        pe = pe.permute(0, 2, 1)  # (N, target_size, D)
        pe = pe.reshape(*shape_prefix, target_size, self.hidden_dim)
        # 逆排列
        inv_perm = [0] * ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i
        pe = pe.permute(*inv_perm)
        return pe

    def get_pos_embed(self, T, H, W):
        # dim_idx: 0=T(dim1), 1=H(dim2), 2=W(dim3) — offset +1 inside _interpolate_pos
        pe_t = self._interpolate_pos(self.pos_embed_t, 0, T) if T != 4 else self.pos_embed_t
        pe_h = self._interpolate_pos(self.pos_embed_h, 1, H) if H != 32 else self.pos_embed_h
        pe_w = self._interpolate_pos(self.pos_embed_w, 2, W) if W != 32 else self.pos_embed_w
        pos = pe_t + pe_h + pe_w  # broadcast → (1, T, H, W, D)
        return pos.reshape(1, T * H * W, self.hidden_dim)

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: (B, C, T, H, W) 归一化视频帧
        Returns:
            tokens: (B, N_out, output_dim)
            grid: (T_out, H_out, W_out)
        """
        x = self.patch_embed(pixel_values)  # (B, T, H, W, D)
        B, T, H, W, D = x.shape
        x = x.reshape(B, T * H * W, D)
        x = x + self.get_pos_embed(T, H, W)

        for i, block in enumerate(self.blocks):
            x = block(x, T, H, W)
            if str(i) in self.downsamplers:
                x, T, H, W = self.downsamplers[str(i)](x, T, H, W)

        x = self.out_norm(x)
        x = self.out_proj(x)
        return x, (T, H, W)

    def num_params(self, trainable_only=True):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def create_liteframe_base(output_dim=3584, **kwargs):
    """创建 LiteFrame ViT-Base。"""
    return LiteFrameEncoder(
        patch_size=14, in_channels=3, hidden_dim=768,
        num_heads=12, num_layers=12, ffn_mult=4,
        output_dim=output_dim, **kwargs,
    )

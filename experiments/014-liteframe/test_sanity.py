"""LiteFrame sanity check — 验证 forward pass 和 shapes。"""

import torch
from liteframe_encoder import create_liteframe_base, LiteFrameEncoder


def test_forward_shapes():
    """测试 4帧 448×448 输入的 forward pass。"""
    print("=== LiteFrame Sanity Check ===\n")

    model = create_liteframe_base(output_dim=3584)
    print(f"参数量: {model.num_params():,} ({model.num_params() / 1e6:.1f}M)")

    # 4帧 448×448 输入
    x = torch.randn(1, 3, 4, 448, 448)
    print(f"输入: {x.shape}  (B, C, T, H, W)")

    with torch.no_grad():
        out, (T, H, W) = model(x)

    print(f"输出: {out.shape}  (B, N, D)")
    print(f"Grid: T={T}, H={H}, W={W}")
    print(f"压缩比: {4*32*32} → {T*H*W} = {4*32*32 // (T*H*W)}×")

    assert out.shape == (1, 256, 3584), f"Expected (1, 256, 3584), got {out.shape}"
    assert (T, H, W) == (1, 16, 16), f"Expected (1, 16, 16), got ({T}, {H}, {W})"
    print("\n✓ Forward pass 正确!")


def test_different_frames():
    """测试不同帧数输入。"""
    print("\n=== 不同帧数测试 ===\n")
    model = create_liteframe_base(output_dim=3584)

    for T_in in [4, 8, 16]:
        x = torch.randn(1, 3, T_in, 448, 448)
        with torch.no_grad():
            out, (T, H, W) = model(x)
        tokens = T * H * W
        print(f"  T={T_in:2d} 帧: {T_in*32*32:5d} tokens → {tokens:4d} tokens "
              f"({T_in*32*32 // tokens}×), grid=({T},{H},{W})")


def test_gradient():
    """测试梯度能正常反传。"""
    print("\n=== 梯度测试 ===\n")
    model = create_liteframe_base(output_dim=3584)
    x = torch.randn(1, 3, 4, 448, 448)
    out, _ = model(x)
    loss = out.mean()
    loss.backward()

    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total = sum(1 for p in model.parameters())
    print(f"  {has_grad}/{total} 参数有梯度")
    assert has_grad > 0, "没有参数有梯度!"
    print("  ✓ 梯度反传正常!")


def test_bf16():
    """测试 bf16 forward。"""
    print("\n=== BF16 测试 ===\n")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_liteframe_base(output_dim=3584).to(device, dtype=torch.bfloat16)
    x = torch.randn(1, 3, 4, 448, 448, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        out, grid = model(x)
    print(f"  输出 dtype: {out.dtype}, shape: {out.shape}")
    assert out.dtype == torch.bfloat16
    print("  ✓ BF16 正常!")


if __name__ == "__main__":
    test_forward_shapes()
    test_different_frames()
    test_gradient()
    if torch.cuda.is_available():
        test_bf16()
    print("\n✅ 全部测试通过!")

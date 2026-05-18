"""
将按样本存储的小 .pt teacher cache 打包成 shard，降低 NFS metadata 压力。

用法:
  python pack_teacher_cache.py \
    --input_dir /nvmessd/lifanhong/video/teacher_cache_10k \
    --output_dir /nvmessd/lifanhong/video/teacher_cache_10k_sharded \
    --shard_size 1000
"""

import argparse
import glob
import os

import torch


def pack_teacher_cache(input_dir, output_dir, shard_size=1000):
    sample_files = sorted(glob.glob(os.path.join(input_dir, "*.pt")))
    os.makedirs(output_dir, exist_ok=True)

    shard_paths = []
    shard_samples = []
    shard_idx = 0

    for sample_file in sample_files:
        shard_samples.append(torch.load(sample_file, map_location="cpu", weights_only=False))
        if len(shard_samples) == shard_size:
            shard_path = os.path.join(output_dir, f"teacher_shard_{shard_idx:03d}.pt")
            torch.save(shard_samples, shard_path)
            shard_paths.append(shard_path)
            shard_samples = []
            shard_idx += 1

    if shard_samples:
        shard_path = os.path.join(output_dir, f"teacher_shard_{shard_idx:03d}.pt")
        torch.save(shard_samples, shard_path)
        shard_paths.append(shard_path)

    return shard_paths


def main():
    parser = argparse.ArgumentParser(description="打包 teacher cache 为 shard")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--shard_size", type=int, default=1000)
    args = parser.parse_args()

    shard_paths = pack_teacher_cache(args.input_dir, args.output_dir, args.shard_size)
    print(f"打包完成: {len(shard_paths)} 个 shard")
    for path in shard_paths:
        print(path)


if __name__ == "__main__":
    main()

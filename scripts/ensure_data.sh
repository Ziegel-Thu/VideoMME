#!/bin/bash
# 数据完整性检查 + 自动从 NFS 同步
# 用法: bash ensure_data.sh [machine_name]
#
# 在任何机器上训练/提取前运行此脚本，确保本地 SSD 有所需数据。
# 缺失的数据会自动从 NFS 拷贝。

set -e

NFS_BASE="/beegfs_hdd/data/nfs_share/users/lifanhong/nishome/video"
SSD_BASE="/nvmessd/lifanhong/video"
MODEL_CACHE="/nvmessd/lifanhong/.cache/modelscope/Qwen/Qwen2___5-VL-7B-Instruct"

echo "=== 数据完整性检查 ==="
echo "  NFS: $NFS_BASE"
echo "  SSD: $SSD_BASE"
echo ""

MISSING=0

# ---- 1. 训练数据 (jsonl) ----
echo "[1] 训练/测试数据 (jsonl)..."
for f in visual_qa_v3_0_60_train.jsonl visual_qa_v3_0_60_test.jsonl; do
    SSD_PATH="$SSD_BASE/parsed/$f"
    NFS_PATH="$NFS_BASE/data/parsed/$f"
    # jsonl 也可能在 NFS 的 checkpoints 目录
    NFS_ALT="$NFS_BASE/checkpoints/$f"

    if [ -f "$SSD_PATH" ]; then
        echo "  ✅ $f ($(wc -l < "$SSD_PATH") 行)"
    else
        echo "  ❌ $f 不在 SSD"
        mkdir -p "$SSD_BASE/parsed"
        if [ -f "$NFS_PATH" ]; then
            echo "     → 从 NFS 拷贝: $NFS_PATH"
            cp "$NFS_PATH" "$SSD_PATH"
            echo "     ✅ 已拷贝 ($(wc -l < "$SSD_PATH") 行)"
        elif [ -f "$NFS_ALT" ]; then
            echo "     → 从 NFS 拷贝: $NFS_ALT"
            cp "$NFS_ALT" "$SSD_PATH"
            echo "     ✅ 已拷贝 ($(wc -l < "$SSD_PATH") 行)"
        else
            echo "     ⚠️  NFS 上也没有此文件，需手动准备"
            MISSING=$((MISSING + 1))
        fi
    fi
done

# ---- 2. 视频文件 ----
echo ""
echo "[2] 视频文件..."
VIDEO_DIR="$SSD_BASE/llava-video/0_30_s_academic_v0_1/academic_source"
if [ -d "$VIDEO_DIR" ]; then
    N_VIDEOS=$(find "$VIDEO_DIR" -name "*.mp4" 2>/dev/null | wc -l)
    echo "  ✅ 视频目录存在 ($N_VIDEOS 个 mp4)"
    if [ "$N_VIDEOS" -lt 100 ]; then
        echo "  ⚠️  视频数量偏少，可能需要解压/下载更多"
    fi
else
    echo "  ❌ 视频目录不存在: $VIDEO_DIR"
    echo "     需要手动解压视频数据到 SSD"
    MISSING=$((MISSING + 1))
fi

# ---- 3. 模型权重 ----
echo ""
echo "[3] Qwen2.5-VL-7B 模型..."
if [ -f "$MODEL_CACHE/config.json" ]; then
    echo "  ✅ 模型存在: $MODEL_CACHE"
else
    echo "  ❌ 模型不存在"
    echo "     运行: python -c \"from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct', cache_dir='/nvmessd/lifanhong/.cache/modelscope')\""
    MISSING=$((MISSING + 1))
fi

# ---- 4. Teacher cache (预提取特征) ----
echo ""
echo "[4] Teacher cache (预提取特征)..."
TEACHER_SSD="$SSD_BASE/teacher_cache_10k"
TEACHER_NFS="$NFS_BASE/teacher_cache_10k"

if [ -d "$TEACHER_SSD" ]; then
    N_FILES=$(ls "$TEACHER_SSD"/*.pt 2>/dev/null | wc -l)
    echo "  ✅ Teacher cache 存在 ($N_FILES 个文件)"
else
    echo "  ❌ Teacher cache 不在 SSD"
    if [ -d "$TEACHER_NFS" ]; then
        N_NFS=$(ls "$TEACHER_NFS"/*.pt 2>/dev/null | wc -l)
        echo "     NFS 有 $N_NFS 个文件，开始 rsync..."
        mkdir -p "$TEACHER_SSD"
        rsync -a --info=progress2 "$TEACHER_NFS/" "$TEACHER_SSD/"
        echo "     ✅ 已同步"
    else
        echo "     NFS 上也没有，需要先运行 extract_teacher.py"
        MISSING=$((MISSING + 1))
    fi
fi

# ---- 5. Checkpoints (训练产物) ----
echo ""
echo "[5] 训练 checkpoints..."
for method in distill_attn_10k distill_kv_10k; do
    CKPT_SSD="$SSD_BASE/outputs_${method}"
    CKPT_NFS="$NFS_BASE/checkpoints/${method}"
    if [ -d "$CKPT_SSD" ]; then
        N_CKPTS=$(ls "$CKPT_SSD"/voco_embeds_*.pt 2>/dev/null | wc -l)
        echo "  ✅ $method: $N_CKPTS 个 checkpoint"
    elif [ -d "$CKPT_NFS" ]; then
        echo "  ⚠️  $method: SSD 无，NFS 有，按需 rsync"
    else
        echo "  ℹ️  $method: 无 checkpoint（尚未训练）"
    fi
done

# ---- 结果 ----
echo ""
if [ "$MISSING" -eq 0 ]; then
    echo "🎉 所有数据检查通过！"
else
    echo "⚠️  有 $MISSING 项数据缺失，请处理后再开始训练"
fi

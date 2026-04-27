#!/bin/bash
# 下载 arXiv 论文的 LaTeX 源码
# 用法: bash scripts/download_paper.sh <arXiv_ID> [自定义文件夹名]
# 示例:
#   bash scripts/download_paper.sh 2405.21075
#   bash scripts/download_paper.sh 2405.21075 video-mme
#   bash scripts/download_paper.sh https://arxiv.org/abs/2405.21075
#   bash scripts/download_paper.sh https://arxiv.org/pdf/2405.21075

set -e

PAPER_DIR="$(cd "$(dirname "$0")/.." && pwd)/paper"

if [ -z "$1" ]; then
    echo "用法: bash scripts/download_paper.sh <arXiv_ID_or_URL> [自定义文件夹名]"
    echo "示例: bash scripts/download_paper.sh 2405.21075 video-mme"
    exit 1
fi

# 从 URL 或纯 ID 中提取 arXiv ID
INPUT="$1"
ARXIV_ID=$(echo "$INPUT" | grep -oP '(\d{4}\.\d{4,5}(v\d+)?)')

if [ -z "$ARXIV_ID" ]; then
    echo "❌ 无法从 '$INPUT' 中提取 arXiv ID"
    echo "   支持格式: 2405.21075, https://arxiv.org/abs/2405.21075, https://arxiv.org/pdf/2405.21075"
    exit 1
fi

# 确定目标文件夹名
if [ -n "$2" ]; then
    FOLDER_NAME="$2"
else
    FOLDER_NAME="$ARXIV_ID"
fi

TARGET_DIR="$PAPER_DIR/$FOLDER_NAME"

if [ -d "$TARGET_DIR" ]; then
    echo "⚠️  目录已存在: $TARGET_DIR"
    echo "   如需重新下载，请先删除该目录"
    exit 1
fi

mkdir -p "$TARGET_DIR"

echo "📥 正在下载 arXiv:$ARXIV_ID 的 LaTeX 源码..."
DOWNLOAD_URL="https://arxiv.org/e-print/$ARXIV_ID"
TMP_FILE="$TARGET_DIR/_source.tar.gz"

# 下载源码包
if ! curl -L -o "$TMP_FILE" -f --retry 3 --retry-delay 5 \
    -H "User-Agent: Mozilla/5.0" "$DOWNLOAD_URL" 2>/dev/null; then
    echo "❌ 下载失败，请检查 arXiv ID 是否正确: $ARXIV_ID"
    rm -rf "$TARGET_DIR"
    exit 1
fi

# 检测文件类型并解压
FILE_TYPE=$(file -b "$TMP_FILE")
echo "📦 文件类型: $FILE_TYPE"

cd "$TARGET_DIR"

if echo "$FILE_TYPE" | grep -qi "gzip\|tar"; then
    tar xzf "_source.tar.gz" 2>/dev/null || gzip -d "_source.tar.gz" 2>/dev/null || true
elif echo "$FILE_TYPE" | grep -qi "pdf"; then
    echo "⚠️  该论文只提供 PDF，无 LaTeX 源码"
    mv "_source.tar.gz" "paper.pdf"
    echo "✅ PDF 已保存到: $TARGET_DIR/paper.pdf"
    exit 0
else
    # 尝试直接作为 tar 解压
    tar xf "_source.tar.gz" 2>/dev/null || true
fi

# 清理临时文件
rm -f "_source.tar.gz"

# 统计结果
TEX_COUNT=$(find . -name "*.tex" | wc -l)
BIB_COUNT=$(find . -name "*.bib" | wc -l)
FIG_COUNT=$(find . \( -name "*.pdf" -o -name "*.png" -o -name "*.jpg" -o -name "*.eps" \) | wc -l)

echo ""
echo "✅ 下载完成: $TARGET_DIR"
echo "   📄 .tex 文件: $TEX_COUNT"
echo "   📚 .bib 文件: $BIB_COUNT"
echo "   🖼️  图片文件: $FIG_COUNT"

# 列出 tex 文件
if [ "$TEX_COUNT" -gt 0 ]; then
    echo ""
    echo "   TeX 文件列表:"
    find . -name "*.tex" | sort | sed 's|^\./|   - |'
fi

echo ""
echo "💡 主文件通常是 main.tex 或包含 \\documentclass 的文件"
if [ "$TEX_COUNT" -gt 1 ]; then
    MAIN=$(grep -rl '\\documentclass' . --include='*.tex' 2>/dev/null | head -1)
    if [ -n "$MAIN" ]; then
        echo "   🎯 检测到主文件: $MAIN"
    fi
fi

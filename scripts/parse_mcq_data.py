#!/usr/bin/env python3
"""
从 LLaVA-Video-178K 和 Open-o3-Video (STGR) 生成干净的 MCQ 训练数据。

用法:
    python scripts/parse_mcq_data.py

输出:
    data/parsed/visual_qa_v3_train.jsonl
    data/parsed/visual_qa_v3_val.jsonl
    data/parsed/visual_qa_v3_test.jsonl
"""

import json
import os
import re
import random
from collections import Counter, defaultdict
from pathlib import Path

# ── 常量 ──────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LLAVA_VIDEO_DIR = DATA_DIR / "llava-video"
LLAVA_ANNOT_DIR = DATA_DIR / "llava-video-annotations"
STGR_JSON = DATA_DIR / "open-o3-video" / "json_data" / "STGR-SFT.json"
STGR_VIDEO_DIR = DATA_DIR / "open-o3-video" / "videos" / "stgr"
OUTPUT_DIR = DATA_DIR / "parsed"

SEED = 42
TRAIN_RATIO = 0.85
VAL_RATIO = 0.05
TEST_RATIO = 0.10

# LLaVA-Video-178K MCQ 标注文件列表
LLAVA_ANNOT_FILES = [
    "0_30_s_academic_v0_1/0_30_s_academic_mc_v0_1_qa_processed.json",
    "0_30_s_nextqa/0_30_s_nextqa_mc_qa_processed.json",
    "0_30_s_perceptiontest/0_30_s_perceptiontest_mc_qa_processed.json",
    "0_30_s_youtube_v0_1/0_30_s_youtube_mc_v0_1_qa_processed.json",
    "30_60_s_academic_v0_1/30_60_s_academic_mc_v0_1_qa_processed.json",
    "30_60_s_nextqa/30_60_s_nextqa_mc_qa_processed.json",
    "30_60_s_perceptiontest/30_60_s_perceptiontest_mc_qa_processed.json",
    "30_60_s_youtube_v0_1/30_60_s_youtube_mc_v0_1_qa_processed.json",
    "1_2_m_academic_v0_1/1_2_m_academic_mc_v0_1_qa_processed.json",
    "1_2_m_nextqa/1_2_m_nextqa_mc_qa_processed.json",
    "1_2_m_youtube_v0_1/1_2_m_youtube_mc_v0_1_qa_processed.json",
    "2_3_m_academic_v0_1/2_3_m_academic_mc_v0_1_qa_processed.json",
    "2_3_m_nextqa/2_3_m_nextqa_mc_qa_processed.json",
    "2_3_m_youtube_v0_1/2_3_m_youtube_mc_v0_1_qa_processed.json",
]


def build_video_index():
    """构建视频 basename → 绝对路径的索引。"""
    print("正在构建视频文件索引...")
    index = {}
    search_dirs = [LLAVA_VIDEO_DIR, STGR_VIDEO_DIR]

    for search_dir in search_dirs:
        if not search_dir.exists():
            print(f"  警告: 目录不存在 {search_dir}")
            continue
        for root, _, files in os.walk(search_dir):
            for f in files:
                if f.endswith(".mp4"):
                    full_path = os.path.join(root, f)
                    # basename 冲突时保留第一个
                    if f not in index:
                        index[f] = full_path

    print(f"  索引完成，共 {len(index)} 个视频文件")
    return index


def extract_answer_letter(raw_answer):
    """从 GPT 回答中提取选项字母 (A-E)。

    支持格式: "A", "A.", "A. Some text.", "The answer is A." 等
    """
    raw_answer = raw_answer.strip()

    # 最常见：以字母开头
    m = re.match(r"^([A-E])\b", raw_answer)
    if m:
        return m.group(1)

    # "The answer is X" 格式
    m = re.search(r"answer\s+is\s+([A-E])\b", raw_answer, re.IGNORECASE)
    if m:
        return m.group(1)

    return None


def extract_question_text(human_value):
    """从 human value 中提取 question 文本（去掉 <image>/<video> 标签和尾部指令）。

    返回格式：问题文本 + Options 部分
    """
    text = human_value.strip()
    # 去掉开头的 <image> / <video> 标签
    text = re.sub(r"^<(?:image|video)>\s*", "", text)

    # 去掉尾部的 "Please provide/respond..." 指令
    text = re.sub(
        r"\s*Please\s+(?:provide|respond|select|choose|answer).*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # 去掉尾部的 "Answer with" 指令
    text = re.sub(
        r"\s*Answer\s+with\s+.*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # 标准化选项格式：确保 Options 之前有换行
    # 有些标注把选项紧跟在问题后面
    text = re.sub(r"\s*\n?\s*((?:[A-E]\.\s))", r"\n\1", text)

    # 如果选项没有 "Options:" 前缀，加上
    if re.search(r"\n[A-E]\.\s", text) and "Options" not in text:
        # 在第一个选项前加 "Options:\n"
        text = re.sub(r"\n([A-E]\.\s)", r"\nOptions:\n\1", text, count=1)

    return text.strip()


def parse_llava_video_annotations(video_index):
    """解析 LLaVA-Video-178K 的 MCQ 标注。"""
    print("\n" + "=" * 60)
    print("解析 LLaVA-Video-178K MCQ 标注")
    print("=" * 60)

    samples = []
    skipped_no_answer = 0
    skipped_no_video = 0
    file_stats = {}

    for annot_file in LLAVA_ANNOT_FILES:
        filepath = LLAVA_ANNOT_DIR / annot_file
        if not filepath.exists():
            print(f"  警告: 文件不存在 {filepath}")
            continue

        with open(filepath, "r") as f:
            data = json.load(f)

        count = 0
        for item in data:
            video_rel = item.get("video", "")
            video_basename = os.path.basename(video_rel)
            # 从视频路径推断 source
            source = infer_source_from_path(video_rel)

            # 每个 item 有多轮 QA
            convs = item.get("conversations", [])
            for i in range(0, len(convs) - 1, 2):
                human_msg = convs[i]
                gpt_msg = convs[i + 1]

                if human_msg.get("from") != "human" or gpt_msg.get("from") != "gpt":
                    continue

                question = extract_question_text(human_msg["value"])
                answer = extract_answer_letter(gpt_msg["value"])

                if answer is None:
                    skipped_no_answer += 1
                    continue

                # 查找视频路径
                video_path = video_index.get(video_basename)
                if video_path is None:
                    skipped_no_video += 1
                    continue

                samples.append({
                    "question": question,
                    "answer": answer,
                    "video_path": video_path,
                    "task": "MCQ",
                    "source": source,
                })
                count += 1

        file_stats[annot_file] = count
        print(f"  {annot_file}: {len(data)} 条目 → {count} 个 QA")

    print(f"\nLLaVA-Video 汇总:")
    print(f"  有效样本: {len(samples)}")
    print(f"  跳过（无法解析答案）: {skipped_no_answer}")
    print(f"  跳过（视频不存在）: {skipped_no_video}")

    return samples


def parse_stgr_annotations(video_index):
    """解析 Open-o3-Video STGR MCQ 标注。"""
    print("\n" + "=" * 60)
    print("解析 Open-o3-Video STGR MCQ 标注")
    print("=" * 60)

    if not STGR_JSON.exists():
        print(f"  警告: 文件不存在 {STGR_JSON}")
        return []

    with open(STGR_JSON, "r") as f:
        data = json.load(f)

    print(f"  总条目: {len(data)}")

    # 筛选 MCQ 条目
    mcq_items = [
        d for d in data
        if d.get("task") == "General video QA MCQ"
        and "Options" in d.get("question", "")
    ]
    print(f"  MCQ 含 Options: {len(mcq_items)}")

    samples = []
    skipped_temporal = 0
    skipped_no_answer = 0
    skipped_no_video = 0

    for item in mcq_items:
        answer_raw = item.get("answer", "").strip()

        # 检查是否为时间段格式（无法提取选项字母，跳过）
        if "<t>" in answer_raw:
            skipped_temporal += 1
            continue

        answer = extract_answer_letter(answer_raw)
        if answer is None:
            skipped_no_answer += 1
            continue

        # 解析视频路径
        video_rel = item.get("video_path", "")
        video_basename = os.path.basename(video_rel)
        video_path = video_index.get(video_basename)

        if video_path is None:
            skipped_no_video += 1
            continue

        # 提取 source（从 video_path 推断）
        source = infer_source_from_path(video_rel)

        # question 已经包含 Options，直接使用
        question = item["question"].strip()

        samples.append({
            "question": question,
            "answer": answer,
            "video_path": video_path,
            "task": "MCQ",
            "source": source,
        })

    print(f"\nSTGR 汇总:")
    print(f"  有效样本: {len(samples)}")
    print(f"  跳过（时间段答案）: {skipped_temporal}")
    print(f"  跳过（无法解析答案）: {skipped_no_answer}")
    print(f"  跳过（视频不存在）: {skipped_no_video}")

    return samples


def infer_source_from_path(video_path):
    """从视频路径推断数据来源。"""
    path_lower = video_path.lower()
    if "activitynet" in path_lower:
        return "activitynet"
    if "charades" in path_lower:
        return "charades"
    if "nextqa" in path_lower or "next-qa" in path_lower or "NeXT-QA" in video_path:
        return "nextqa"
    if "perceptiontest" in path_lower:
        return "perceptiontest"
    if "youcook" in path_lower:
        return "youcook2"
    if "clevrer" in path_lower:
        return "clevrer"
    if "/STAR/" in video_path:
        return "star"
    if "youtube" in path_lower or "ytb_" in path_lower:
        return "youtube"
    if "ego" in path_lower:
        return "ego"
    if "coin" in path_lower or "/COIN/" in video_path:
        return "coin"
    if "textvr" in path_lower:
        return "textvr"
    if "sharegpt" in path_lower:
        return "sharegpt4video"
    if "videochatgpt" in path_lower:
        return "videochatgpt"
    return "other"


def deduplicate(samples):
    """按 (video_path, question) 去重，保留第一条。"""
    print("\n" + "=" * 60)
    print("去重")
    print("=" * 60)

    seen = set()
    unique = []
    dup_count = 0

    for s in samples:
        key = (s["video_path"], s["question"])
        if key in seen:
            dup_count += 1
            continue
        seen.add(key)
        unique.append(s)

    print(f"  去重前: {len(samples)}")
    print(f"  重复数: {dup_count}")
    print(f"  去重后: {len(unique)}")

    return unique


def filter_video_exists(samples):
    """过滤掉视频文件不存在的样本。"""
    print("\n" + "=" * 60)
    print("视频可用性过滤")
    print("=" * 60)

    # 缓存检查结果避免重复 stat
    exists_cache = {}
    valid = []
    missing = 0

    for s in samples:
        vp = s["video_path"]
        if vp not in exists_cache:
            exists_cache[vp] = os.path.isfile(vp)
        if exists_cache[vp]:
            valid.append(s)
        else:
            missing += 1

    print(f"  过滤前: {len(samples)}")
    print(f"  视频不存在: {missing}")
    print(f"  过滤后: {len(valid)}")

    return valid


def split_by_video(samples, seed=SEED):
    """按视频级别 split，同一视频的所有问题在同一个 split。"""
    print("\n" + "=" * 60)
    print("按视频级别划分 train/val/test")
    print("=" * 60)

    # 按视频分组
    video_to_indices = defaultdict(list)
    for i, s in enumerate(samples):
        video_to_indices[s["video_path"]].append(i)

    videos = sorted(video_to_indices.keys())
    random.seed(seed)
    random.shuffle(videos)

    n_videos = len(videos)
    n_train = int(n_videos * TRAIN_RATIO)
    n_val = int(n_videos * VAL_RATIO)

    train_videos = set(videos[:n_train])
    val_videos = set(videos[n_train:n_train + n_val])
    test_videos = set(videos[n_train + n_val:])

    train_samples = [samples[i] for v in videos if v in train_videos for i in video_to_indices[v]]
    val_samples = [samples[i] for v in videos if v in val_videos for i in video_to_indices[v]]
    test_samples = [samples[i] for v in videos if v in test_videos for i in video_to_indices[v]]

    print(f"  总视频数: {n_videos}")
    print(f"  train: {len(train_samples)} 条 / {len(train_videos)} 视频")
    print(f"  val:   {len(val_samples)} 条 / {len(val_videos)} 视频")
    print(f"  test:  {len(test_samples)} 条 / {len(test_videos)} 视频")

    return train_samples, val_samples, test_samples


def save_jsonl(samples, filepath):
    """保存为 JSONL 格式。"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"  已保存: {filepath} ({len(samples)} 条)")


def print_statistics(train, val, test):
    """打印详细统计信息。"""
    all_samples = train + val + test

    print("\n" + "=" * 60)
    print("统计报告")
    print("=" * 60)

    # 每个 source 的样本数
    source_counter = Counter(s["source"] for s in all_samples)
    print("\n各 source 样本数:")
    for src, cnt in source_counter.most_common():
        print(f"  {src}: {cnt}")
    print(f"  合计: {len(all_samples)}")

    # 答案分布
    answer_counter = Counter(s["answer"] for s in all_samples)
    print("\n答案分布:")
    for letter in sorted(answer_counter.keys()):
        cnt = answer_counter[letter]
        pct = cnt / len(all_samples) * 100
        print(f"  {letter}: {cnt} ({pct:.1f}%)")

    # 检查 question 跨 split 重复
    train_questions = set(s["question"] for s in train)
    val_questions = set(s["question"] for s in val)
    test_questions = set(s["question"] for s in test)

    tv_overlap = train_questions & val_questions
    tt_overlap = train_questions & test_questions
    vt_overlap = val_questions & test_questions

    print(f"\nQuestion 跨 split 重复检查:")
    print(f"  train ∩ val:  {len(tv_overlap)} 个相同 question")
    print(f"  train ∩ test: {len(tt_overlap)} 个相同 question")
    print(f"  val ∩ test:   {len(vt_overlap)} 个相同 question")
    if tv_overlap or tt_overlap or vt_overlap:
        print("  注意: question 文本有重复，但对应不同视频（视频级别无泄漏）")


def main():
    print("=" * 60)
    print("MCQ 训练数据生成")
    print(f"基础目录: {BASE_DIR}")
    print("=" * 60)

    # 1. 构建视频索引
    video_index = build_video_index()

    # 2. 解析标注
    llava_samples = parse_llava_video_annotations(video_index)
    stgr_samples = parse_stgr_annotations(video_index)

    # 3. 合并
    all_samples = llava_samples + stgr_samples
    print(f"\n合并后总样本数: {len(all_samples)}")

    # 4. 去重
    all_samples = deduplicate(all_samples)

    # 5. 视频可用性过滤
    all_samples = filter_video_exists(all_samples)

    # 6. 按视频级别 split
    train, val, test = split_by_video(all_samples)

    # 7. 保存
    print("\n" + "=" * 60)
    print("保存结果")
    print("=" * 60)
    save_jsonl(train, OUTPUT_DIR / "visual_qa_v3_train.jsonl")
    save_jsonl(val, OUTPUT_DIR / "visual_qa_v3_val.jsonl")
    save_jsonl(test, OUTPUT_DIR / "visual_qa_v3_test.jsonl")

    # 8. 统计
    print_statistics(train, val, test)

    print("\n✅ 完成！")


if __name__ == "__main__":
    main()

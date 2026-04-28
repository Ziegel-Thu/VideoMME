"""
Phase 0.2: 从 Open-o3-Video 和 Video-o3 Seeker-173K 提取训练标注

输出:
  data/parsed/temporal_evidence.jsonl  — 时间证据监督 (Q, A, video_path, segments)
  data/parsed/selector_trajectories.jsonl — 选段器轨迹 (video_id, question, selected_segments)
"""

import json
import re
import os
import glob
from pathlib import Path

DATA_ROOT = Path("/home/v-shuzheng/video/data")
OUTPUT_DIR = DATA_ROOT / "parsed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_open_o3_video():
    """从 Open-o3-Video STGR 数据中提取时间证据标注"""
    results = []

    for json_file in ["STGR-SFT.json", "STGR-RL.json"]:
        path = DATA_ROOT / "open-o3-video" / "json_data" / json_file
        if not path.exists():
            print(f"  跳过 {path}（不存在）")
            continue

        with open(path) as f:
            data = json.load(f)

        split = "sft" if "SFT" in json_file else "rl"
        count = 0

        for item in data:
            answer = item.get("answer", "")
            segments = []

            # 格式 1 (SFT): "From <t>2</t>s to <t>6</t>s"
            time_matches = re.findall(r"<t>([\d.]+)</t>", answer)
            if len(time_matches) >= 2:
                for i in range(0, len(time_matches) - 1, 2):
                    segments.append([float(time_matches[i]), float(time_matches[i + 1])])

            # 格式 2 (RL): answer = "[1.0, 4.0]"
            if not segments:
                try:
                    parsed = json.loads(answer)
                    if isinstance(parsed, list) and len(parsed) == 2:
                        segments.append([float(parsed[0]), float(parsed[1])])
                except (json.JSONDecodeError, ValueError):
                    pass

            # 从 reasoning_process 中提取内嵌时间
            if not segments:
                reasoning = item.get("reasoning_process", "")
                inline_times = re.findall(r"at<t>([\d.]+)</t>s", reasoning)
                for t in inline_times:
                    segments.append([float(t), float(t)])

            if not segments:
                continue

            results.append({
                "id": item.get("id", ""),
                "source": f"open-o3-video/{split}",
                "video_path": item.get("video_path", ""),
                "question": item.get("question", ""),
                "answer": answer,
                "task": item.get("task", ""),
                "evidence_segments": segments,
            })
            count += 1

        print(f"  {json_file}: {len(data)} 条 → {count} 条有时间证据")

    out_path = OUTPUT_DIR / "temporal_evidence.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  写入 {out_path} ({len(results)} 条)")
    return results


def parse_seeker_173k():
    """从 Seeker-173K 中提取选段轨迹"""
    results = []
    json_files = glob.glob(str(DATA_ROOT / "seeker-173k" / "**" / "*w_tool*.json"), recursive=True)

    for json_file in sorted(json_files):
        with open(json_file) as f:
            data = json.load(f)

        split = "sft" if "/SFT/" in json_file else "rl"
        fname = os.path.basename(json_file)
        count = 0

        for item in data:
            selected_segments = []
            question = ""
            video_url = ""

            # === 格式 1 (SFT): messages + grounding tags ===
            if "messages" in item:
                messages = item.get("messages", [])
                videos = item.get("videos", [])
                video_url = videos[0]["url"] if videos else ""

                for msg in messages:
                    content = msg.get("content", "")
                    role = msg.get("role", "")

                    if role == "user" and not question:
                        q_match = re.search(r'["""](.+?)["""]', content)
                        if q_match:
                            question = q_match.group(1)
                        else:
                            question = content[:200]

                    grounding_matches = re.findall(
                        r'<grounding>\s*(\{.*?\})\s*</grounding>',
                        content, re.DOTALL
                    )
                    for g in grounding_matches:
                        try:
                            g_data = json.loads(g)
                            seg = g_data.get("temporal_segment", [])
                            strategy = g_data.get("sampling_strategy", "")
                            if len(seg) == 2:
                                selected_segments.append({
                                    "start": seg[0],
                                    "end": seg[1],
                                    "strategy": strategy,
                                })
                        except json.JSONDecodeError:
                            continue

            # === 格式 2 (RL): solution.clue[].timestamp ===
            elif "solution" in item:
                question = item.get("question", "")
                video_url = item.get("video", "")
                solution = item.get("solution", {})
                clues = solution.get("clue", [])
                for clue in clues:
                    ts = clue.get("timestamp", [])
                    if len(ts) == 2:
                        selected_segments.append({
                            "start": ts[0],
                            "end": ts[1],
                            "strategy": "clue",
                        })

            if not selected_segments:
                continue

            video_id = item.get("id", item.get("doc_id", ""))
            results.append({
                "id": video_id,
                "source": f"seeker-173k/{split}/{fname}",
                "video_url": video_url,
                "question": question,
                "selected_segments": selected_segments,
                "num_turns": len(selected_segments),
            })
            count += 1

        print(f"  {fname}: {len(data)} 条 → {count} 条有选段轨迹")

    out_path = OUTPUT_DIR / "selector_trajectories.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  写入 {out_path} ({len(results)} 条)")
    return results


if __name__ == "__main__":
    print("=== Phase 0.2: 解析训练标注 ===\n")

    print("[1/2] Open-o3-Video: 提取时间证据...")
    temporal = parse_open_o3_video()

    print(f"\n[2/2] Seeker-173K: 提取选段轨迹...")
    selector = parse_seeker_173k()

    print(f"\n=== 完成 ===")
    print(f"  时间证据: {len(temporal)} 条")
    print(f"  选段轨迹: {len(selector)} 条")

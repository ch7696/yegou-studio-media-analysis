#!/usr/bin/env python3
"""Pure visual analysis entrypoint for Media Analysis Vision Studio."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ANALYZER = ROOT / "director_detail_analyzer.py"
MODEL_NAME = os.environ.get("MINICPM_MODEL", "MiniCPM-V-4_5-GPTQ")


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, text=True)


def write_documents(output: Path, source: Path) -> list[Path]:
    raw_path = output / "视觉分析文本汇总.md"
    if not raw_path.exists():
        raise FileNotFoundError(f"视觉分析文本汇总不存在：{raw_path}")

    info_path = output / "测试视频信息.md"
    info = info_path.read_text(encoding="utf-8").strip() if info_path.exists() else ""
    content = [
        "# 纯视觉分析",
        "",
        f"来源视频：{source.name}",
        f"视觉模型：{MODEL_NAME}",
        "分析口径：每秒 1 张高清帧；每 20 秒一个上下文联系图；每 5 秒一组细节帧。",
        "本文件只保留画面、镜头、构图、主体、动作、方位、色彩、可见文字和连续性观察，不包含 ASR。",
        "",
    ]
    if info:
        content.extend(["## 处理信息", "", info, ""])
    content.extend([raw_path.read_text(encoding="utf-8").strip(), ""])
    report_path = output / "01_纯视觉分析.md"
    report_path.write_text("\n".join(content), encoding="utf-8")

    frames = sorted((output / "01_每秒高清帧").glob("frame_*.jpg"))
    context = sorted((output / "02_20秒联系图").glob("联系图_*.jpg"))
    detail = sorted((output / "02_20秒联系图").glob("细节联系图_*.jpg"))
    manifest = {
        "schema_version": "media-analysis-vision-studio/v1",
        "source": str(source),
        "output": str(output),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "models": {"visual": MODEL_NAME},
        "granularity": {
            "frame_interval_sec": 1,
            "context_interval_sec": 20,
            "detail_interval_sec": 5,
        },
        "evidence_counts": {
            "frames_1fps": len(frames),
            "context_sheets_20s": len(context),
            "detail_sheets_5s": len(detail),
        },
        "documents": [
            "01_纯视觉分析.md",
            "视觉分析文本汇总.md",
            "视觉片段索引.jsonl",
            "测试视频信息.md",
        ],
    }
    manifest_path = output / "分析清单.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return [report_path, manifest_path]


def main() -> None:
    parser = argparse.ArgumentParser(description="纯视觉视频导演拉片分析")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-duration", type=float, default=0.0, help="只处理前 N 秒；默认全片")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise SystemExit(f"视频不存在：{source}")
    output.mkdir(parents=True, exist_ok=True)

    print("阶段 1/2：逐秒高清帧 + 20 秒视觉上下文", flush=True)
    command = [sys.executable, str(ANALYZER), "--source", str(source), "--output", str(output)]
    if args.max_duration > 0:
        command.extend(["--max-duration", str(args.max_duration)])
    run(command)

    print("阶段 2/2：整理视觉证据与纯视觉报告", flush=True)
    paths = write_documents(output, source)
    for path in paths:
        print(f"已生成：{path}", flush=True)


if __name__ == "__main__":
    main()

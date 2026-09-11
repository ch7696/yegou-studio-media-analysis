#!/usr/bin/env python3
"""Fixed four-document pipeline for long-video director analysis.

The first three documents are deterministic machine artifacts:
1) visual evidence and visual-model prose;
2) ASR plus ForcedAligner timestamps;
3) code-only interval merge of visual blocks and audio segments.

The fourth document is an optional human/LLM interpretation.  A WebUI can
generate it after the user reviews or downloads the first three artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_ROOT", str(ROOT / "models")))
ASR_IMAGE = os.environ.get("ASR_IMAGE", "ragflow-qwen-asr:0.0.6")
MINICPM_CONTAINER = os.environ.get("MINICPM_CONTAINER", "vision-minicpm")
ASR_CONTAINER = os.environ.get("ASR_CONTAINER", "vision-asr")
ASR_MAX_NEW_TOKENS = os.environ.get("ASR_MAX_NEW_TOKENS", "2048")
ASR_LANGUAGE = os.environ.get("ASR_LANGUAGE", "").strip()


def run(command: list[str], *, check: bool = True, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, check=check, timeout=timeout)


def stop_visual_service() -> None:
    subprocess.run(
        [
            "docker",
            "stop",
            MINICPM_CONTAINER,
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def ensure_visual_service() -> None:
    # run_model.sh waits for /health; the extra timeout covers cold-start logs
    # on slower cloud disks before the first multimodal request is sent.
    run([str(ROOT / "scripts" / "run_model.sh")], timeout=900)


def run_asr(source: Path, output: Path) -> None:
    # ASR and VLM are run sequentially so the same GPU can be used safely.
    # Only the new project's visual container is stopped here.
    stop_visual_service()
    if subprocess.run(
        ["docker", "image", "inspect", ASR_IMAGE],
        text=True,
        capture_output=True,
        check=False,
    ).returncode:
        raise RuntimeError(f"找不到 ASR 镜像：{ASR_IMAGE}")

    mounted_source = f"/data/source/{source.name}"
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        ASR_CONTAINER,
        "--gpus",
        "all",
        "--ipc=host",
        "--shm-size=4g",
        "-e",
        "TRANSFORMERS_OFFLINE=1",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-v",
        f"{ROOT}:/app:ro",
        "-v",
        f"{MODEL_DIR / 'Qwen3-ASR-0.6B'}:/models/Qwen3-ASR-0.6B:ro",
        "-v",
        f"{MODEL_DIR / 'Qwen3-ForcedAligner-0.6B'}:/models/Qwen3-ForcedAligner-0.6B:ro",
        "-v",
        f"{source}:{mounted_source}:ro",
        "-v",
        f"{output}:/data/output",
        "--entrypoint",
        "python3",
        ASR_IMAGE,
        "/app/audio_timeline.py",
        "--source-root",
        "/data/source",
        "--source-file",
        mounted_source,
        "--output-root",
        "/data/output",
        "--asr-model",
        "/models/Qwen3-ASR-0.6B",
        "--aligner-model",
        "/models/Qwen3-ForcedAligner-0.6B",
        "--max-new-tokens",
        ASR_MAX_NEW_TOKENS,
        "--force",
    ]
    if ASR_LANGUAGE:
        command.extend(["--language", ASR_LANGUAGE])
    run(command)


def run_visual(source: Path, output: Path, max_duration: float) -> None:
    ensure_visual_service()
    command = [
        sys.executable,
        str(ROOT / "app" / "director_detail_analyzer.py"),
        "--source",
        str(source),
        "--output",
        str(output),
    ]
    if max_duration > 0:
        command.extend(["--max-duration", str(max_duration)])
    run(command)


def fmt_time(seconds: float) -> str:
    value = max(0.0, float(seconds))
    hours = int(value // 3600)
    minutes = int((value - hours * 3600) // 60)
    remainder = value - hours * 3600 - minutes * 60
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remainder:05.2f}"
    return f"{minutes:02d}:{remainder:05.2f}"


def number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def readable_units(units: list[dict[str, Any]], language: str) -> str:
    values = [str(item.get("text", "")).strip() for item in units if str(item.get("text", "")).strip()]
    if language.lower().startswith(("en", "eng")):
        return " ".join(values)
    return "".join(values)


def load_audio(output: Path) -> dict[str, Any]:
    files = sorted((output / "02_视频" / "音轨与时间轴").glob("*.json"))
    if not files:
        # Qwen3-ASR records videos without an audio stream in audio_records.jsonl
        # and intentionally does not create an empty transcript JSON.  Keep the
        # four-document contract intact for silent videos instead of failing at
        # the packaging stage.
        records_path = output / "audio_records.jsonl"
        if records_path.exists():
            for line in records_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("status") == "no_audio":
                    return {
                        "schema_version": "media-audio-timeline/v1",
                        "asr": {
                            "model": "Qwen3-ASR-0.6B",
                            "forced_aligner": "Qwen3-ForcedAligner-0.6B",
                            "language": "none",
                            "text": "",
                            "segments": [],
                            "units": [],
                        },
                    }
        raise FileNotFoundError("没有找到 ASR JSON：请先完成音频分析")
    return json.loads(files[0].read_text(encoding="utf-8"))


def load_visual_blocks(output: Path) -> list[dict[str, Any]]:
    index_path = output / "视觉片段索引.jsonl"
    blocks: list[dict[str, Any]] = []
    if index_path.exists():
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if item.get("kind") == "context":
                    blocks.append(item)
        if blocks:
            return sorted(blocks, key=lambda item: number(item.get("start_sec")))

    def decode_stamp(value: str) -> float:
        pieces, fraction = value.split(".")
        values = [int(item) for item in pieces.split("-")]
        if len(values) == 2:
            return values[0] * 60 + values[1] + int(fraction) / 100
        return values[0] * 3600 + values[1] * 60 + values[2] + int(fraction) / 100

    pattern = re.compile(
        r"(?P<start>\d{2}(?:-\d{2})?-\d{2}\.\d{2})-(?P<end>\d{2}(?:-\d{2})?-\d{2}\.\d{2})"
    )
    text_dir = output / "03_模型文字分析"
    context_dir = output / "02_20秒联系图"
    for text_path in sorted(text_dir.glob("上下文_*.md")):
        match = pattern.search(text_path.name)
        if not match:
            continue
        image_matches = sorted(context_dir.glob(f"联系图_*{match.group('start')}-{match.group('end')}*.jpg"))
        blocks.append(
            {
                "kind": "context",
                "start_sec": decode_stamp(match.group("start")),
                "end_sec": decode_stamp(match.group("end")),
                "image": str(image_matches[0].relative_to(output)) if image_matches else "",
                "text": str(text_path.relative_to(output)),
                "detail_files": [],
            }
        )
    return sorted(blocks, key=lambda item: number(item.get("start_sec")))


def audio_segments(audio: dict[str, Any]) -> list[dict[str, Any]]:
    asr = audio.get("asr") or {}
    language = str(asr.get("language") or "auto")
    result = []
    for segment in asr.get("segments", []):
        segment_start = number(segment.get("start_sec"))
        segment_end = number(segment.get("end_sec"))
        units = [
            unit
            for unit in asr.get("units", [])
            if number(unit.get("start_sec")) >= segment_start - 1e-6
            and number(unit.get("start_sec")) < segment_end - 1e-6
        ]
        item = dict(segment)
        reconstructed = readable_units(units, language)
        item["display_text"] = reconstructed or str(segment.get("text") or "")
        item["_units"] = units
        result.append(item)
    return result


def write_pure_visual(output: Path, source: Path) -> Path:
    raw_path = output / "视觉分析文本汇总.md"
    if not raw_path.exists():
        raise FileNotFoundError("没有找到视觉模型文字汇总：请先完成视觉分析")
    content = [
        "# 01_纯视觉分析",
        "",
        f"来源视频：{source.name}",
        "分析口径：每秒 1 张高清帧；每 20 秒一个上下文联系图；5 秒细节组用于动作和方位复核。",
        "本文件只保留视觉模型看到的画面、镜头、构图、主体、动作、方位、色彩和可见文字，不混入 ASR。",
        "",
        raw_path.read_text(encoding="utf-8").strip(),
        "",
    ]
    path = output / "01_纯视觉分析.md"
    path.write_text("\n".join(content), encoding="utf-8")
    return path


def write_pure_asr(output: Path, source: Path, audio: dict[str, Any]) -> Path:
    asr = audio.get("asr") or {}
    language = str(asr.get("language") or "auto")
    segments = audio_segments(audio)
    lines = [
        "# 02_纯ASR与时间戳",
        "",
        f"来源视频：{source.name}",
        f"语言：{language}",
        "ASR：Qwen3-ASR-0.6B",
        "时间对齐：Qwen3-ForcedAligner-0.6B",
        "",
        "## 完整转写",
        "",
        str(asr.get("text") or "（未识别到语音）"),
        "",
        "## 对齐后的时间线",
        "",
    ]
    for item in segments:
        lines.append(
            f"{fmt_time(number(item.get('start_sec')))}–{fmt_time(number(item.get('end_sec')))}  "
            f"{item.get('display_text') or item.get('text') or '（空）'}"
        )
    lines.extend(
        [
            "",
            "## 机器文件",
            "",
            "完整逐词边界保存在同目录的 JSON；字幕文件同时提供 SRT 和 VTT。",
            "",
        ]
    )
    path = output / "02_纯ASR与时间戳.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_code_merge(output: Path, source: Path, audio: dict[str, Any]) -> Path:
    blocks = load_visual_blocks(output)
    segments = audio_segments(audio)
    lines = [
        "# 03_代码综合时间线",
        "",
        f"来源视频：{source.name}",
        "合并方式：代码按视觉上下文区间与 ASR 时间区间做区间相交；本文件不让语言模型改写视觉或语音原文。",
        "视觉时间轴：每 20 秒一个上下文块；音频时间轴：ForcedAligner 分段；两者只在时间重叠时归入同一块。",
        "",
    ]
    for index, block in enumerate(blocks, 1):
        start = number(block.get("start_sec"))
        end = number(block.get("end_sec"))
        text_path = output / str(block.get("text") or "")
        context_text = text_path.read_text(encoding="utf-8").strip() if text_path.exists() else "（视觉文本缺失）"
        overlaps = [
            item
            for item in segments
            if number(item.get("end_sec")) > start and number(item.get("start_sec")) < end
        ]
        lines.extend(
            [
                f"## {index:03d}｜{fmt_time(start)}–{fmt_time(end)}",
                "",
                "### 视觉原文",
                "",
                context_text,
                "",
                "### 同期 ASR",
                "",
            ]
        )
        if overlaps:
            for item in overlaps:
                clipped_start = max(start, number(item.get("start_sec")))
                clipped_end = min(end, number(item.get("end_sec")))
                units = [
                    unit
                    for unit in item.get("_units", [])
                    if number(unit.get("end_sec")) > start
                    and number(unit.get("start_sec")) < end
                ]
                display_text = readable_units(
                    units,
                    str(audio.get("asr", {}).get("language") or "auto"),
                ) or str(item.get("display_text") or item.get("text") or "（空）")
                lines.append(
                    f"- {fmt_time(clipped_start)}–{fmt_time(clipped_end)}：{display_text}"
                )
        else:
            lines.append("- 该视觉区间内没有 ASR 语音。")
        detail_files = block.get("detail_files") or []
        if isinstance(detail_files, list) and detail_files:
            lines.extend(["", "### 5 秒高清细节与同期语音", ""])
            for detail in detail_files:
                if not isinstance(detail, dict):
                    continue
                detail_start = number(detail.get("start_sec"))
                detail_end = number(detail.get("end_sec"))
                detail_text_path = output / str(detail.get("text") or "")
                detail_text = (
                    detail_text_path.read_text(encoding="utf-8").strip()
                    if detail_text_path.exists()
                    else "（高清细节文字缺失）"
                )
                lines.extend(
                    [
                        f"#### {fmt_time(detail_start)}–{fmt_time(detail_end)}",
                        "",
                        "视觉细节原文：",
                        "",
                        detail_text,
                        "",
                        "同期 ASR：",
                        "",
                    ]
                )
                detail_overlaps = [
                    item
                    for item in segments
                    if number(item.get("end_sec")) > detail_start
                    and number(item.get("start_sec")) < detail_end
                ]
                if detail_overlaps:
                    for item in detail_overlaps:
                        clipped_start = max(detail_start, number(item.get("start_sec")))
                        clipped_end = min(detail_end, number(item.get("end_sec")))
                        units = [
                            unit
                            for unit in item.get("_units", [])
                            if number(unit.get("end_sec")) > detail_start
                            and number(unit.get("start_sec")) < detail_end
                        ]
                        display_text = readable_units(
                            units,
                            str(audio.get("asr", {}).get("language") or "auto"),
                        ) or str(item.get("display_text") or item.get("text") or "（空）")
                        lines.append(
                            f"- {fmt_time(clipped_start)}–{fmt_time(clipped_end)}：{display_text}"
                        )
                else:
                    lines.append("- 该 5 秒细节区间内没有 ASR 语音。")
                if detail.get("image"):
                    lines.append(f"- 高清细节证据：{detail['image']}")
                lines.append("")
        lines.extend(["", "### 证据", ""])
        if block.get("image"):
            lines.append(f"- 20 秒联系图：{block['image']}")
        if block.get("text"):
            lines.append(f"- 视觉文字：{block['text']}")
        lines.append("")
    path = output / "03_代码综合时间线.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_final_document(output: Path, source: Path, final_source: Path | None) -> Path:
    path = output / "04_最终导演分析.md"
    if final_source:
        body = final_source.read_text(encoding="utf-8").strip()
        content = [
            "# 04_最终导演分析",
            "",
            f"来源视频：{source.name}",
            "处理方式：在前三份机器产物基础上，由 Codex/人工进行导演拉片、叙事和视觉语法分析。",
            "",
            body,
            "",
        ]
    else:
        content = [
            "# 04_最终导演分析",
            "",
            f"来源视频：{source.name}",
            "状态：待人工或大模型处理。",
            "",
            "本文件是最终解释层，不参与前三份机器产物的生成。用户可以下载 01_纯视觉分析.md、02_纯ASR与时间戳.md 和 03_代码综合时间线.md，交给 Codex、其他大模型或人工继续处理。",
            "",
            "建议最终分析重点：镜头切换、构图方位、主体动作、动作变化、旁白与画面的关系、剪辑节奏、叙事推进、视觉母题和不确定项。",
            "",
        ]
    path.write_text("\n".join(content), encoding="utf-8")
    return path


def package_documents(output: Path, source: Path, final_source: Path | None) -> list[Path]:
    audio = load_audio(output)
    paths = [
        write_pure_visual(output, source),
        write_pure_asr(output, source, audio),
        write_code_merge(output, source, audio),
        write_final_document(output, source, final_source),
    ]
    manifest = {
        "schema_version": "media-analysis-pipeline/v1",
        "source": str(source),
        "output": str(output),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "granularity": {
            "visual_frame_interval_sec": 1,
            "visual_context_interval_sec": 20,
            "visual_detail_interval_sec": 5,
        },
        "documents": [path.name for path in paths],
        "models": {
            "visual": "MiniCPM-V-4_5-GPTQ",
            "asr": "Qwen3-ASR-0.6B",
            "forced_aligner": "Qwen3-ForcedAligner-0.6B",
        },
    }
    (output / "分析清单.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="固定的视频视觉+ASR+时间轴分析流程")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-duration", type=float, default=0.0, help="只处理前 N 秒；默认 0 表示全片")
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-visual", action="store_true")
    parser.add_argument("--package-only", action="store_true", help="只根据已有视觉/ASR产物生成四份文档")
    parser.add_argument("--final-source", type=Path, help="已有的 Codex/人工最终分析稿")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise SystemExit(f"视频不存在：{source}")
    output.mkdir(parents=True, exist_ok=True)

    if not args.package_only:
        if not args.skip_asr:
            print("阶段 1/3：ASR + ForcedAligner", flush=True)
            run_asr(source, output)
            print("阶段 1/3 完成：ASR + ForcedAligner", flush=True)
        if not args.skip_visual:
            print("阶段 2/3：逐秒高清帧 + 20秒视觉上下文", flush=True)
            run_visual(source, output, args.max_duration)
            print("阶段 2/3 完成：视觉拉片", flush=True)

    print("阶段 3/3：生成四份文档", flush=True)
    paths = package_documents(output, source, args.final_source.resolve() if args.final_source else None)
    for path in paths:
        print(f"已生成：{path}", flush=True)


if __name__ == "__main__":
    main()

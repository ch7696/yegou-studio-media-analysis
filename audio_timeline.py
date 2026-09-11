#!/usr/bin/env python3
"""Transcribe media audio and produce a searchable timestamped timeline.

The script deliberately keeps audio analysis separate from visual analysis:
MiniCPM owns images/video frames, while Qwen3-ASR and ForcedAligner own the
audio track. A later merge step can join both records by relative_path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from asset_naming import artifact_stem, derived_relative_path, media_category


SOURCE_ROOT = Path("/mnt/c/Users/Administrator/Documents/ComfyUI/output")
OUTPUT_ROOT = Path("/mnt/c/Users/Administrator/Documents/AI_Asset_Output/media_analysis")
RECORDS_FILE_NAME = "audio_records.jsonl"
INDEX_FILE_NAME = "audio_index.md"
TRANSCRIPT_DIR_NAME = "transcripts"
NORMALIZED_AUDIO_DIR_NAME = "normalized_audio"
ANALYZER_VERSION = "qwen3-asr-0.6b-forced-aligner-0.6b-v2"

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".wmv", ".flv", ".gif"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".wma"}
SENTENCE_END = set("。！？!?；;：:\n")


def run(command: list[str], *, check: bool = True, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=check,
        timeout=timeout,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ffprobe(path: Path) -> dict[str, Any]:
    result = run(
        [
            "ffprobe",
            "-hide_banner",
            "-loglevel",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        timeout=180,
    )
    return json.loads(result.stdout)


def number(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def media_metadata(path: Path, kind: str) -> dict[str, Any]:
    info = ffprobe(path)
    fmt = info.get("format", {})
    streams = info.get("streams", [])
    audio = next((item for item in streams if item.get("codec_type") == "audio"), None)
    video = next((item for item in streams if item.get("codec_type") == "video"), None)
    metadata: dict[str, Any] = {
        "duration_sec": number(fmt.get("duration")),
        "format": fmt.get("format_name"),
        "audio_stream_present": audio is not None,
    }
    if audio:
        metadata["audio"] = {
            "index": audio.get("index"),
            "codec": audio.get("codec_name"),
            "codec_long_name": audio.get("codec_long_name"),
            "sample_rate": audio.get("sample_rate"),
            "channels": audio.get("channels"),
            "channel_layout": audio.get("channel_layout"),
            "language": (audio.get("tags") or {}).get("language"),
        }
    if video:
        metadata["video"] = {
            "index": video.get("index"),
            "codec": video.get("codec_name"),
            "width": video.get("width"),
            "height": video.get("height"),
            "fps": video.get("r_frame_rate"),
        }
    return metadata


def extract_wav(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(target),
        ],
        timeout=900,
    )


def extract_flac(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-sn",
            "-dn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "flac",
            str(target),
        ],
        timeout=900,
    )


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            relative = item.get("relative_path")
            if relative:
                records[relative] = item
    return records


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        stream.flush()


def object_value(item: Any, *names: str) -> Any:
    if isinstance(item, dict):
        for name in names:
            if name in item:
                return item[name]
        return None
    for name in names:
        if hasattr(item, name):
            return getattr(item, name)
    return None


def flatten_timestamps(value: Any) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, (dict, list, tuple)) and hasattr(value, "items"):
        value = getattr(value, "items")
    if isinstance(value, (list, tuple)):
        if len(value) == 1 and isinstance(value[0], (list, tuple)):
            return list(value[0])
        return list(value)
    return []


def parse_units(result: Any) -> list[dict[str, Any]]:
    raw = object_value(result, "time_stamps", "timestamps", "timestamp")
    units: list[dict[str, Any]] = []
    for item in flatten_timestamps(raw):
        text = object_value(item, "text", "word", "token")
        start = number(object_value(item, "start_time", "start", "start_sec"))
        end = number(object_value(item, "end_time", "end", "end_sec"))
        if text is None or start is None or end is None:
            continue
        text = str(text)
        if not text:
            continue
        units.append(
            {
                "text": text,
                "start_sec": round(max(0.0, start), 3),
                "end_sec": round(max(start, end), 3),
            }
        )
    return units


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def build_segments(units: list[dict[str, Any]], transcript: str, duration: float | None) -> list[dict[str, Any]]:
    if not units:
        if not transcript:
            return []
        return [
            {
                "id": 0,
                "start_sec": 0.0,
                "end_sec": round(duration or 0.0, 3),
                "text": transcript,
            }
        ]

    segments: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []

    def close_segment() -> None:
        if not current:
            return
        text = clean_text("".join(item["text"] for item in current))
        if not text:
            current.clear()
            return
        segments.append(
            {
                "id": len(segments),
                "start_sec": current[0]["start_sec"],
                "end_sec": current[-1]["end_sec"],
                "text": text,
            }
        )
        current.clear()

    for unit in units:
        current.append(unit)
        text = "".join(item["text"] for item in current)
        elapsed = unit["end_sec"] - current[0]["start_sec"]
        last_char = unit["text"][-1:]
        if last_char in SENTENCE_END or elapsed >= 15.0 or len(text) >= 80:
            close_segment()
    close_segment()
    if len(segments) == 1 and transcript:
        segments[0]["text"] = transcript
    return segments


def load_asr(asr_path: str, aligner_path: str, max_new_tokens: int):
    import torch
    from qwen_asr import Qwen3ASRModel

    dtype = torch.bfloat16
    print(f"加载 ASR: {asr_path}", flush=True)
    print(f"加载 ForcedAligner: {aligner_path}", flush=True)
    return Qwen3ASRModel.from_pretrained(
        asr_path,
        dtype=dtype,
        device_map="cuda:0",
        max_inference_batch_size=1,
        max_new_tokens=max_new_tokens,
        forced_aligner=aligner_path,
        forced_aligner_kwargs={
            "dtype": dtype,
            "device_map": "cuda:0",
        },
    )


def transcribe(model: Any, wav_path: Path, language: str | None) -> dict[str, Any]:
    results = model.transcribe(
        audio=str(wav_path),
        language=language,
        return_time_stamps=True,
    )
    result = results[0] if isinstance(results, (list, tuple)) else results
    text = clean_text(object_value(result, "text"))
    detected_language = object_value(result, "language")
    units = parse_units(result)
    return {
        "language": str(detected_language or language or "auto"),
        "text": text,
        "units": units,
    }


def srt_time(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def vtt_time(seconds: float) -> str:
    return srt_time(seconds).replace(",", ".")


def write_outputs(
    output_root: Path,
    record: dict[str, Any],
    transcript: dict[str, Any],
    keep_audio: bool,
) -> dict[str, Any]:
    digest = record["sha256"]
    stem = artifact_stem(
        {**record, "audio_analysis": {"transcript": transcript["text"]}},
        prefix="音轨",
    )
    segments = build_segments(
        transcript["units"],
        transcript["text"],
        record.get("metadata", {}).get("duration_sec"),
    )
    payload = {
        "schema_version": "media-audio-timeline/v1",
        "relative_path": record["relative_path"],
        "source_path": record["source_path"],
        "source_sha256": digest,
        "media_kind": record["kind"],
        "media_metadata": record.get("metadata", {}),
        "asr": {
            "model": "Qwen3-ASR-0.6B",
            "forced_aligner": "Qwen3-ForcedAligner-0.6B",
            "language": transcript["language"],
            "text": transcript["text"],
            "segments": segments,
            "units": transcript["units"],
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    transcript_dir = output_root / media_category(record) / "音轨与时间轴"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    json_path = transcript_dir / f"{stem}.json"
    srt_path = transcript_dir / f"{stem}.srt"
    vtt_path = transcript_dir / f"{stem}.vtt"
    md_path = transcript_dir / f"{stem}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    srt_lines: list[str] = []
    vtt_lines = ["WEBVTT", ""]
    md_lines = [
        f"# {record['relative_path']}",
        "",
        f"- 原始文件：{record['relative_path']}",
        f"- 音频语言：{transcript['language']}",
        "- ASR：Qwen3-ASR-0.6B",
        "- 强制对齐：Qwen3-ForcedAligner-0.6B",
        "",
        "## 完整转写",
        "",
        transcript["text"] or "（未识别到语音）",
        "",
        "## 时间轴",
        "",
    ]
    for index, segment in enumerate(segments, 1):
        start = float(segment["start_sec"])
        end = max(start, float(segment["end_sec"]))
        text = segment["text"]
        srt_lines.extend([str(index), f"{srt_time(start)} --> {srt_time(end)}", text, ""])
        vtt_lines.extend([f"{vtt_time(start)} --> {vtt_time(end)}", text, ""])
        md_lines.append(f"- [{vtt_time(start)} - {vtt_time(end)}] {text}")
    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
    vtt_path.write_text("\n".join(vtt_lines) + "\n", encoding="utf-8")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    result = {
        "transcript_json": str(json_path.relative_to(output_root)),
        "transcript_srt": str(srt_path.relative_to(output_root)),
        "transcript_vtt": str(vtt_path.relative_to(output_root)),
        "transcript_markdown": str(md_path.relative_to(output_root)),
        "language": transcript["language"],
        "transcript": transcript["text"],
        "segments": segments,
        "units": transcript["units"],
    }
    if keep_audio:
        normalized_path = output_root / media_category(record) / "规范化音频" / f"{stem}.flac"
        extract_flac(Path(record["source_path"]), normalized_path)
        result["normalized_audio"] = str(normalized_path.relative_to(output_root))
    return result


def format_duration(value: Any) -> str:
    seconds = number(value)
    if seconds is None:
        return ""
    total = int(round(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def write_index(output_root: Path, records: dict[str, dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for item in records.values():
        status = item.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    lines = [
        "# 音轨与时间戳分析",
        "",
        f"更新时间：{datetime.now().isoformat(timespec='seconds')}",
        "模型：Qwen3-ASR-0.6B + Qwen3-ForcedAligner-0.6B",
        "",
        f"记录 {len(records)} 个；完成 {counts.get('done', 0)}，无音轨 {counts.get('no_audio', 0)}，重复 {counts.get('duplicate', 0)}，失败 {counts.get('failed', 0)}。",
        "",
        "每个已完成文件同时生成 JSON、SRT、VTT 和 Markdown。JSON 保留逐字/逐词单元，Markdown 用于后续 RAGFlow 检索。",
        "",
        "## 文件",
        "",
    ]
    for item in sorted(records.values(), key=lambda x: x.get("relative_path", "")):
        relative = item.get("relative_path", "")
        status = item.get("status")
        if status == "done":
            asr = item.get("asr", {})
            lines.append(
                f"- {relative} · {asr.get('language', 'auto')} · {format_duration(item.get('metadata', {}).get('duration_sec'))} · "
                f"Markdown: {asr.get('transcript_markdown', '')}"
            )
        elif status == "no_audio":
            lines.append(f"- {relative} · 无音轨")
        elif status == "duplicate":
            lines.append(f"- {relative} · 重复，复用 {item.get('duplicate_of', '')}")
        else:
            lines.append(f"- {relative} · 失败：{item.get('error', '')}")
    (output_root / INDEX_FILE_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def classify(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTS:
        return "video"
    if suffix in AUDIO_EXTS:
        return "audio"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract audio and create timestamped Qwen3-ASR transcripts.")
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--source-file", type=Path, help="只处理指定的单个媒体文件；用于 WebUI 单任务")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--asr-model", default="/models/Qwen3-ASR-0.6B")
    parser.add_argument("--aligner-model", default="/models/Qwen3-ForcedAligner-0.6B")
    parser.add_argument("--language", default=None, help="例如 Chinese；留空则自动检测")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-audio", action="store_true", help="额外保存 16kHz 单声道 FLAC")
    args = parser.parse_args()

    if not args.source_root.exists():
        raise SystemExit(f"来源目录不存在: {args.source_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    records_path = args.output_root / RECORDS_FILE_NAME
    records = load_jsonl(records_path)
    hash_index = {
        item.get("sha256"): item
        for item in records.values()
        if item.get("sha256") and item.get("status") in {"done", "duplicate", "no_audio"}
    }
    if args.source_file:
        if not args.source_file.is_file():
            raise SystemExit(f"来源文件不存在: {args.source_file}")
        if classify(args.source_file) is None:
            raise SystemExit(f"不支持的媒体类型: {args.source_file}")
        paths = [args.source_file]
        selected = paths
        start_index = 1
    else:
        paths = sorted(path for path in args.source_root.rglob("*") if path.is_file() and classify(path))
        selected = paths[args.offset :]
        start_index = args.offset + 1
    print(f"音视频总数={len(paths)}，已有音轨记录={len(records)}，输出={args.output_root}", flush=True)

    model = None
    processed = 0
    for index, path in enumerate(selected, start_index):
        try:
            relative = path.relative_to(args.source_root).as_posix()
        except ValueError:
            relative = path.name
        kind = classify(path)
        digest = sha256(path)
        previous = records.get(relative)
        if (
            not args.force
            and previous
            and previous.get("sha256") == digest
            and previous.get("analyzer_version") == ANALYZER_VERSION
            and previous.get("status") in {"done", "duplicate", "no_audio"}
        ):
            print(f"[{index}/{len(paths)}] 跳过 {relative}", flush=True)
            continue

        base: dict[str, Any] = {
            "relative_path": relative,
            "source_path": str(path),
            "kind": kind,
            "sha256": digest,
            "analyzer_version": ANALYZER_VERSION,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        print(f"[{index}/{len(paths)}] 检查 {relative}", flush=True)
        try:
            metadata = media_metadata(path, kind)
            base["metadata"] = metadata
            if not metadata.get("audio_stream_present"):
                record = {**base, "status": "no_audio"}
                print("    -> 无音轨", flush=True)
            else:
                duplicate = hash_index.get(digest)
                if duplicate and duplicate.get("status") in {"done", "duplicate"}:
                    record = {
                        **base,
                        "status": "duplicate",
                        "duplicate_of": duplicate.get("relative_path"),
                        "asr": duplicate.get("asr", {}),
                    }
                    print(f"    -> 重复，复用 {record.get('duplicate_of')}", flush=True)
                else:
                    if model is None:
                        model = load_asr(args.asr_model, args.aligner_model, args.max_new_tokens)
                    with tempfile.TemporaryDirectory(prefix="qwen-asr-") as temp_dir:
                        wav_path = Path(temp_dir) / "audio.wav"
                        extract_wav(path, wav_path)
                        transcript = transcribe(model, wav_path, args.language)
                    asr = write_outputs(args.output_root, base, transcript, args.keep_audio)
                    record = {**base, "status": "done", "asr": asr}
                    print(
                        f"    -> 完成: {transcript['language']}，{len(asr['segments'])} 段，"
                        f"{transcript['text'][:160]}",
                        flush=True,
                    )
            records[relative] = record
            append_jsonl(records_path, record)
            if record.get("status") in {"done", "duplicate", "no_audio"}:
                hash_index[digest] = record
        except Exception as exc:
            record = {**base, "status": "failed", "error": str(exc)}
            records[relative] = record
            append_jsonl(records_path, record)
            print(f"    -> 失败: {exc}", flush=True)
        processed += 1
        if processed % 5 == 0 or record.get("status") == "failed":
            write_index(args.output_root, records)
        if args.limit and processed >= args.limit:
            break
    write_index(args.output_root, records)


if __name__ == "__main__":
    main()

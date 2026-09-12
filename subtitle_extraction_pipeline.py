#!/usr/bin/env python3
"""Visual subtitle extraction variant for Media Analysis Vision Studio.

This is deliberately separate from the director-analysis pipeline.  It uses
the same MiniCPM-V service and one-second evidence frames, but asks for one
structured observation per frame and emits a subtitle track that can be
consumed by a dubbing workflow.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
APP_ROOT = ROOT / "app"
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

# Reuse the existing visual client and container helpers.  The original
# director-detail pipeline itself is not changed by this import.
from director_detail_analyzer import call_visual, docker_cp, run  # noqa: E402
from vision_core import (  # noqa: E402
    MODEL_CONTAINER,
    MODEL_NAME,
    decodable_video_duration,
    ffprobe,
    fmt_time,
    metadata,
    number,
    sha256,
)


FRAME_INTERVAL_SEC = 1.0
FRAMES_PER_REQUEST = 5
RELEASE_GPU_AFTER_JOB = os.environ.get("RELEASE_GPU_AFTER_JOB", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}


def stop_visual_service() -> None:
    subprocess.run(
        ["docker", "stop", MODEL_CONTAINER],
        text=True,
        capture_output=True,
        check=False,
    )


def ensure_visual_service() -> None:
    run([str(ROOT / "scripts" / "run_model.sh")], timeout=900)


def parse_time(value: Any, default: float) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        pass
    pieces = raw.replace(",", ".").split(":")
    try:
        if len(pieces) == 2:
            return float(pieces[0]) * 60.0 + float(pieces[1])
        if len(pieces) == 3:
            return float(pieces[0]) * 3600.0 + float(pieces[1]) * 60.0 + float(pieces[2])
    except ValueError:
        return default
    return default


def confidence(value: Any) -> float:
    if isinstance(value, str):
        mapping = {"high": 0.9, "medium": 0.65, "low": 0.35}
        if value.strip().lower() in mapping:
            return mapping[value.strip().lower()]
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def truthy(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"0", "false", "no", "none", "null", "否"}


def clean_caption(value: Any) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if text in {"无", "无字幕", "没有字幕", "none", "null", "n/a", "N/A"}:
        return ""
    return text


def subtitle_prompt(times: list[float]) -> str:
    mapping = "、".join(f"第{i + 1}张={time:.3f}秒" for i, time in enumerate(times))
    return f"""你是视频字幕取证助手。请只检查发送给你的连续视频帧，不要做导演拉片，不要总结剧情，也不要根据声音猜台词。

这些图片按发送顺序对应视频时间：{mapping}。

任务：找出画面中真正作为对白、旁白或翻译字幕出现的可读文字，用于后续重新配音。不要把台标、水印、按钮、聊天框、海报、书本、路牌、包装、衣服上的字、片名或普通场景文字当成对白字幕。字幕可能有两行，请合并为一条文字并保留原标点；看不清、被遮挡或无法确认时不要猜测。

你必须为每一张图片返回一条 frame 记录，即使这一帧没有字幕也要返回 visible=false、text=""。只能输出一个 JSON 对象，禁止 Markdown 代码块、解释文字和额外字段。格式如下：
{{"frames":[{{"frame_time_sec":{times[0] if times else 0:.3f},"visible":true,"text":"画面中实际读到的字幕","text_type":"dialogue","dubbing_candidate":true,"confidence":0.0}}]}}

字段规则：
- frame_time_sec 必须从上面的时间映射中原样选择，不能自行改成别的时间。
- text_type 只能是 dialogue、narration、title_card、sign、watermark 或 other。
- 只有对白字幕或旁白字幕的 dubbing_candidate 才能为 true；其他可见文字为 false。
- confidence 是 0 到 1 的数字，按文字清晰度和字幕判断把握填写。
- 同一句字幕在连续帧中保持完全相同的文字；字幕切换后立即返回新文字。
- 没有字幕时返回 visible=false、text=""、dubbing_candidate=false、confidence=0。
"""


def decode_json(text: str) -> Any:
    """Decode JSON even when a vision model adds a fence or short preamble."""

    raw = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    candidates = [fenced.group(1).strip()] if fenced else []
    candidates.append(raw)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        for index, char in enumerate(candidate):
            if char not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
                return value
            except json.JSONDecodeError:
                continue
    raise ValueError("视觉模型没有返回可解析的 JSON")


def payload_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        values: Any = None
        for key in ("frames", "observations", "items", "results", "segments"):
            if isinstance(payload.get(key), list):
                values = payload[key]
                break
        if values is None:
            values = [payload]
    else:
        values = []
    return [item for item in values if isinstance(item, dict)]


def normalize_observation(item: dict[str, Any], fallback_time: float, frame_index: int) -> dict[str, Any] | None:
    nested = item.get("subtitle")
    if isinstance(nested, dict):
        merged = dict(item)
        merged.update(nested)
        item = merged
    text = clean_caption(item.get("text", item.get("caption", item.get("content", ""))))
    visible = truthy(item.get("visible"), bool(text))
    text_type = str(item.get("text_type", item.get("type", "dialogue")) or "other").strip().lower()
    candidate_value = item.get("dubbing_candidate", item.get("is_subtitle"))
    candidate = truthy(candidate_value, bool(text))
    if text_type in {"title_card", "sign", "watermark", "other"}:
        candidate = False
    if not visible or not text or not candidate:
        return None
    if text.lower() in {"none", "null", "n/a", "无", "无字幕"}:
        return None
    timestamp = item.get(
        "frame_time_sec",
        item.get("time_sec", item.get("timestamp", item.get("start_sec", fallback_time))),
    )
    start = parse_time(timestamp, fallback_time)
    return {
        "frame_index": frame_index,
        "frame_time_sec": round(start, 3),
        "text": text,
        "text_type": text_type,
        "confidence": round(confidence(item.get("confidence")), 3),
    }


def caption_key(value: str) -> str:
    return re.sub(r"[\s，。！？、；：,.!?;:\"'“”‘’（）()【】\[\]…—-]", "", value).lower()


def same_caption(left: str, right: str) -> bool:
    left_key = caption_key(left)
    right_key = caption_key(right)
    if not left_key or not right_key:
        return False
    if left_key == right_key:
        return True
    if min(len(left_key), len(right_key)) >= 4 and (
        left_key in right_key or right_key in left_key
    ):
        return True
    if min(len(left_key), len(right_key)) >= 6:
        return SequenceMatcher(None, left_key, right_key).ratio() >= 0.94
    return False


def merge_observations(
    observations: list[dict[str, Any]],
    duration: float,
    frame_interval: float = FRAME_INTERVAL_SEC,
) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for observation in sorted(observations, key=lambda item: float(item["frame_time_sec"])):
        start = max(0.0, float(observation["frame_time_sec"]))
        end = min(duration, max(start + frame_interval, start))
        if current and start <= float(current["end_sec"]) + 0.02 and same_caption(
            str(current["text"]), str(observation["text"])
        ):
            current["end_sec"] = round(max(float(current["end_sec"]), end), 3)
            current["observed_frames"].append(int(observation["frame_index"]))
            current["confidence_values"].append(float(observation["confidence"]))
            if len(str(observation["text"])) > len(str(current["text"])):
                current["text"] = observation["text"]
            continue
        if current:
            current["confidence"] = round(
                sum(current.pop("confidence_values")) / len(current["observed_frames"]), 3
            )
            current["id"] = len(segments)
            segments.append(current)
        current = {
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "text": observation["text"],
            "text_type": observation["text_type"],
            "confidence_values": [float(observation["confidence"])],
            "observed_frames": [int(observation["frame_index"])],
            "source": "visual_model",
            "dubbing_candidate": True,
        }
    if current:
        current["confidence"] = round(
            sum(current.pop("confidence_values")) / len(current["observed_frames"]), 3
        )
        current["id"] = len(segments)
        segments.append(current)
    return segments


def srt_time(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_outputs(
    output: Path,
    source: Path,
    source_hash: str,
    duration: float,
    time_offset: float,
    metadata_value: dict[str, Any],
    observations: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    batch_records: list[dict[str, Any]],
) -> list[Path]:
    payload = {
        "schema_version": "media-subtitle-track/v1",
        "mode": "visual_subtitle_extraction",
        "source": str(source),
        "source_sha256": source_hash,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL_NAME,
        "media_metadata": metadata_value,
        "timing": {
            "analysis_start_sec": round(time_offset, 3),
            "frame_interval_sec": FRAME_INTERVAL_SEC,
            "timestamp_precision_sec": FRAME_INTERVAL_SEC,
            "boundary_note": "字幕边界依据每秒画面采样估计，适合交给后续配音流程复核。",
        },
        "segments": segments,
        "observations": observations,
        "batch_count": len(batch_records),
    }
    json_path = output / "字幕提取时间轴.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    srt_lines: list[str] = []
    vtt_lines = ["WEBVTT", ""]
    for index, segment in enumerate(segments, 1):
        start = float(segment["start_sec"])
        end = max(start, float(segment["end_sec"]))
        text = str(segment["text"])
        srt_lines.extend([str(index), f"{srt_time(start)} --> {srt_time(end)}", text, ""])
        vtt_lines.extend([f"{srt_time(start).replace(',', '.')} --> {srt_time(end).replace(',', '.')}", text, ""])
    srt_path = output / "字幕提取时间轴.srt"
    vtt_path = output / "字幕提取时间轴.vtt"
    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")
    vtt_path.write_text("\n".join(vtt_lines) + "\n", encoding="utf-8")

    raw_path = output / "字幕提取原始结果.jsonl"
    raw_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in batch_records) + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "# 视觉字幕提取报告",
        "",
        f"- 来源视频：`{source.name}`",
        f"- 视觉模型：`{MODEL_NAME}`",
        f"- 处理范围：`{fmt_time(time_offset)}` 起，分析片段 {fmt_time(duration)}",
        f"- 字幕条数：`{len(segments)}`",
        "- 处理方式：每秒抽取一张画面；模型逐帧判断可见对白/旁白字幕；代码合并连续相同字幕。",
        "- 时间精度：当前以 1 秒画面采样估计起止边界，重配音前建议结合音频时间轴微调。",
        "",
        "## 字幕时间轴",
        "",
    ]
    if segments:
        for index, segment in enumerate(segments, 1):
            report_lines.append(
                f"{index}. `{fmt_time(float(segment['start_sec']))}–{fmt_time(float(segment['end_sec']))}` "
                f"（置信度 {float(segment['confidence']):.2f}）：{segment['text']}"
            )
    else:
        report_lines.append("未提取到可用于重配音的画面字幕。")
    report_lines.extend(
        [
            "",
            "## 机器文件",
            "",
            "- `字幕提取时间轴.json`：包含字幕段、逐帧观察和时间口径。",
            "- `字幕提取时间轴.srt` / `字幕提取时间轴.vtt`：可直接导入剪辑或配音工具。",
            "- `字幕提取原始结果.jsonl`：每批模型原始返回和解析结果，用于复核模型判断。",
        ]
    )
    report_path = output / "字幕提取报告.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": "media-subtitle-extraction/v1",
        "source": str(source),
        "output": str(output),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "visual_subtitle_extraction",
        "models": {"visual": MODEL_NAME},
        "granularity": {"frame_interval_sec": FRAME_INTERVAL_SEC, "frames_per_request": FRAMES_PER_REQUEST},
        "evidence_counts": {"frames": len(list((output / '01_每秒高清帧').glob('frame_*.jpg'))), "observations": len(observations), "segments": len(segments)},
        "documents": [path.name for path in (json_path, srt_path, vtt_path, raw_path, report_path)],
    }
    manifest_path = output / "分析清单.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return [json_path, srt_path, vtt_path, raw_path, report_path, manifest_path]


def analyze(
    source: Path,
    output: Path,
    max_duration: float | None = None,
    time_offset: float = 0.0,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    frames_dir = output / "01_每秒高清帧"
    frames_dir.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(source)
    container_video = f"/tmp/subtitle-extraction-{source_hash[:12]}.mp4"
    remote_frames = f"/tmp/subtitle-extraction-frames-{source_hash[:12]}"
    ensure_visual_service()
    docker_cp(str(source), f"{MODEL_CONTAINER}:{container_video}")
    batch_records: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    try:
        info = ffprobe(container_video)
        metadata_value = metadata(info)
        declared = max(0.5, number(metadata_value.get("duration_sec")))
        actual = decodable_video_duration(container_video)
        duration = max(0.5, min(declared, actual or declared))
        if max_duration and max_duration > 0:
            duration = min(duration, max_duration)
        (output / "字幕提取任务信息.md").write_text(
            "\n".join(
                [
                    "# 字幕提取任务信息",
                    "",
                    f"- 视频：`{source.name}`",
                    f"- SHA256：`{source_hash}`",
                    f"- 视频时间起点：`{fmt_time(time_offset)}`",
                    f"- 分析片段时长：`{fmt_time(duration)}`",
                    f"- 画面：`{metadata_value.get('video', {}).get('width', '未知')} × {metadata_value.get('video', {}).get('height', '未知')}`",
                    "- 视觉模式：每秒一张画面，提取画面中的对白/旁白字幕。",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        run(["docker", "exec", MODEL_CONTAINER, "mkdir", "-p", remote_frames], timeout=120)
        run(
            [
                "docker",
                "exec",
                MODEL_CONTAINER,
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                container_video,
                "-map",
                "0:v:0",
                "-vf",
                "fps=1",
                "-q:v",
                "2",
                f"{remote_frames}/frame_%06d.jpg",
            ],
            timeout=1200,
        )
        docker_cp(f"{MODEL_CONTAINER}:{remote_frames}/.", str(frames_dir))
        frames = sorted(frames_dir.glob("frame_*.jpg"))
        with (frames_dir / "帧索引.md").open("w", encoding="utf-8") as stream:
            for index, frame in enumerate(frames):
                stream.write(
                    f"- 片段 {fmt_time(index)}；视频 {fmt_time(time_offset + index)}：`{frame.name}`\n"
                )

        print(f"阶段 1/2：已抽取 {len(frames)} 张字幕证据帧", flush=True)
        total_batches = max(1, math.ceil(len(frames) / FRAMES_PER_REQUEST))
        for batch_index in range(total_batches):
            start_index = batch_index * FRAMES_PER_REQUEST
            selected = frames[start_index : start_index + FRAMES_PER_REQUEST]
            if not selected:
                break
            times = [time_offset + (start_index + index) * FRAME_INTERVAL_SEC for index in range(len(selected))]
            raw_text = ""
            parsed_items: list[dict[str, Any]] = []
            parse_error = ""
            try:
                raw_text = call_visual(selected, subtitle_prompt(times), max_tokens=2048)
                payload = decode_json(raw_text)
                for item_index, item in enumerate(payload_items(payload)):
                    fallback_time = times[min(item_index, len(times) - 1)]
                    observation = normalize_observation(
                        item,
                        fallback_time,
                        start_index + min(item_index, len(times) - 1),
                    )
                    if observation:
                        # The prompt only allows the supplied frame times.  Snap
                        # a model-produced timestamp back to the nearest real
                        # frame so a hallucinated time cannot enter the track.
                        nearest = min(
                            range(len(times)),
                            key=lambda index: abs(float(observation["frame_time_sec"]) - times[index]),
                        )
                        observation["frame_time_sec"] = round(times[nearest], 3)
                        observation["frame_index"] = start_index + nearest
                        parsed_items.append(observation)
            except Exception as exc:
                parse_error = f"{type(exc).__name__}: {exc}"
            observations.extend(parsed_items)
            batch_records.append(
                {
                    "batch": batch_index + 1,
                    "frame_indices": list(range(start_index, start_index + len(selected))),
                    "frame_times_sec": [round(value, 3) for value in times],
                    "observations": parsed_items,
                    "parse_error": parse_error,
                    "raw_response": raw_text,
                }
            )
            progress = round((batch_index + 1) / total_batches * 100)
            print(
                f"字幕批次完成：{batch_index + 1}/{total_batches}，"
                f"视频 {fmt_time(times[0])}–{fmt_time(times[-1] + FRAME_INTERVAL_SEC)}，进度={progress}%",
                flush=True,
            )
        segments = merge_observations(observations, time_offset + duration)
        paths = write_outputs(
            output,
            source,
            source_hash,
            duration,
            time_offset,
            metadata_value,
            observations,
            segments,
            batch_records,
        )
        print(f"阶段 2/2：视觉字幕时间轴已生成，共 {len(segments)} 条", flush=True)
        for path in paths:
            print(f"已生成：{path}", flush=True)
    finally:
        for remote_path in (container_video, remote_frames):
            subprocess.run(
                ["docker", "exec", MODEL_CONTAINER, "rm", "-rf", remote_path],
                check=False,
                timeout=180,
                capture_output=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="视觉字幕提取变体")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-duration", type=float, default=0.0, help="只处理前 N 秒；默认全片")
    parser.add_argument("--time-offset", type=float, default=0.0, help="输出时间相对原视频的起始偏移秒数")
    parser.add_argument("--release-gpu-after", dest="release_gpu_after", action="store_true")
    parser.add_argument("--keep-gpu-after", dest="release_gpu_after", action="store_false")
    parser.set_defaults(release_gpu_after=None)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise SystemExit(f"视频不存在：{source}")
    if args.time_offset < 0 or not math.isfinite(args.time_offset):
        raise SystemExit("--time-offset 必须是大于等于 0 的数字")
    release_gpu_after = RELEASE_GPU_AFTER_JOB if args.release_gpu_after is None else args.release_gpu_after
    try:
        analyze(source, output, args.max_duration or None, args.time_offset)
    finally:
        if release_gpu_after:
            print("字幕提取结束：正在停止视觉模型容器并释放 GPU 显存", flush=True)
            stop_visual_service()
            print("视觉模型容器已停止，GPU 显存已释放", flush=True)


if __name__ == "__main__":
    main()

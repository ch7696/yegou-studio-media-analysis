"""Shared runtime helpers for the pure-visual video pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any


MODEL_CONTAINER = os.environ.get("MINICPM_CONTAINER", "vision-minicpm")
MODEL_URL = os.environ.get("MINICPM_URL", "http://127.0.0.1:8002/v1/chat/completions")
MODEL_NAME = os.environ.get("MINICPM_MODEL", "MiniCPM-V-4_5-GPTQ")


def docker_file(args: list[str], timeout: int = 600) -> bytes:
    result = subprocess.run(
        ["docker", "exec", MODEL_CONTAINER, *args],
        capture_output=True,
        timeout=timeout,
    )
    if result.returncode:
        detail = result.stderr.decode("utf-8", "replace")[-1600:]
        raise RuntimeError(f"容器文件工具失败: {' '.join(args[:2])}: {detail}")
    return result.stdout


def ffprobe(container_path: str) -> dict[str, Any]:
    raw = docker_file(
        [
            "ffprobe",
            "-hide_banner",
            "-loglevel",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            "-i",
            container_path,
        ],
        timeout=180,
    )
    return json.loads(raw.decode("utf-8"))


def number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def metadata(info: dict[str, Any]) -> dict[str, Any]:
    fmt = info.get("format") or {}
    streams = info.get("streams") or []
    video = next((item for item in streams if item.get("codec_type") == "video"), {})
    audio = next((item for item in streams if item.get("codec_type") == "audio"), {})
    return {
        "duration_sec": number(fmt.get("duration")),
        "format": fmt.get("format_name"),
        "size_bytes": number(fmt.get("size"), 0),
        "video": {
            "codec": video.get("codec_name"),
            "width": video.get("width"),
            "height": video.get("height"),
            "fps": video.get("r_frame_rate"),
            "frames": video.get("nb_frames"),
        },
        "audio": {
            "present": bool(audio),
            "codec": audio.get("codec_name"),
            "sample_rate": audio.get("sample_rate"),
            "channels": audio.get("channels"),
            "channel_layout": audio.get("channel_layout"),
            "language": (audio.get("tags") or {}).get("language"),
        },
    }


def decodable_video_duration(container_path: str) -> float:
    """Find the last decodable video timestamp for truncated MP4 files."""

    result = subprocess.run(
        [
            "docker",
            "exec",
            MODEL_CONTAINER,
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "info",
            "-i",
            container_path,
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            "showinfo",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        timeout=600,
    )
    log = result.stderr.decode("utf-8", "replace")
    timestamps = [float(value) for value in re.findall(r"pts_time:([0-9]+(?:\.[0-9]+)?)", log)]
    return (max(timestamps) + 0.05) if timestamps else 0.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fmt_time(seconds: Any) -> str:
    value = max(0.0, number(seconds))
    minutes, sec = divmod(value, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:05.2f}"
    return f"{minutes:02d}:{sec:05.2f}"

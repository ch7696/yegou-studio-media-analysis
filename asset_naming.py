#!/usr/bin/env python3
"""Generate readable, Windows-safe names for derived media artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


_INVALID_WINDOWS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_WHITESPACE = re.compile(r"\s+")


def media_category(item: dict[str, Any]) -> str:
    """Return the human-facing category for a media record."""

    suffix = str(item.get("suffix") or Path(str(item.get("relative_path") or "")).suffix).lower()
    kind = str(item.get("kind") or "").lower()
    if suffix == ".gif":
        return "04_动图"
    if kind == "image":
        return "01_纯图片"
    if kind == "video":
        return "02_视频"
    if kind == "audio":
        return "03_纯音频"
    return "99_其他"


def derived_relative_path(item: dict[str, Any], area: str, filename: str) -> str:
    return (Path(media_category(item)) / area / filename).as_posix()


def clean_component(value: Any, limit: int = 72) -> str:
    """Keep a filename component readable and valid on Windows."""

    text = str(value or "").replace("`", " ").replace("\r", " ").replace("\n", " ")
    text = _INVALID_WINDOWS.sub("_", text)
    text = _WHITESPACE.sub(" ", text).strip(" .")
    if not text:
        return "未命名资产"
    return text[:limit].rstrip(" ._") or "未命名资产"


def truncate_utf8(value: str, max_bytes: int) -> str:
    """Trim a filename component by UTF-8 bytes, not Python characters."""

    text = str(value or "")
    while text and len(text.encode("utf-8")) > max_bytes:
        text = text[:-1]
    return text.rstrip(" ._")


def source_stem(item: dict[str, Any]) -> str:
    relative = str(item.get("relative_path") or "")
    return clean_component(Path(relative).stem, limit=48)


def natural_title(item: dict[str, Any], transcript: str | None = None) -> str:
    """Prefer the model's short description, then transcript, then source name."""

    analysis = item.get("analysis") or {}
    audio = item.get("audio_analysis") or item.get("asr") or {}
    candidates = [analysis.get("summary"), transcript, audio.get("transcript")]
    generic = {
        "暂无视觉分析",
        "媒体信息已登记",
        "（未识别到语音）",
    }
    for candidate in candidates:
        title = clean_component(candidate, limit=64)
        if title != "未命名资产" and title not in generic:
            return title.rstrip("。.!！？") or title
    return source_stem(item)


def artifact_stem(
    item: dict[str, Any],
    prefix: str,
    transcript: str | None = None,
) -> str:
    """Return a readable stem with a short stable suffix for collision safety."""

    title = truncate_utf8(natural_title(item, transcript=transcript), 96)
    source = truncate_utf8(source_stem(item), 60)
    digest = str(item.get("sha256") or "")[:8]
    parts = [clean_component(prefix, 16), title or "未命名资产"]
    if source and source not in title:
        parts.append(source)
    if digest:
        parts.append(digest)
    result = "__".join(parts)
    suffix = f"__{digest}" if digest else ""
    head = "__".join(parts[:-1]) if digest else result
    head = truncate_utf8(head, max(120, 220 - len(suffix.encode("utf-8"))))
    return (head + suffix).rstrip(" ._") or "未命名资产"


def artifact_filename(
    item: dict[str, Any],
    prefix: str,
    extension: str,
    transcript: str | None = None,
) -> str:
    return artifact_stem(item, prefix=prefix, transcript=transcript) + extension


def asset_card_filename(item: dict[str, Any], sequence: int) -> str:
    return f"{sequence:04d}_{artifact_stem(item, prefix='资产卡片')}.md"

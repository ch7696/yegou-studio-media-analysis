#!/usr/bin/env python3
"""Small dependency-free WebUI for the fixed media-analysis pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import mimetypes
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
PIPELINE = REPO_ROOT / "media_analysis_pipeline.py"
ASSET_ROOT = REPO_ROOT / "assets"
WEBUI_STATE = Path(os.environ.get("WEBUI_STATE", str(REPO_ROOT / "state")))
UPLOAD_DIR = WEBUI_STATE / "uploads"
JOB_STATE_DIR = WEBUI_STATE / "jobs"
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("MEDIA_OUTPUT_ROOT", "/mnt/c/Users/Administrator/Desktop/media_analysis"))
ASR_IMAGE = os.environ.get("ASR_IMAGE", "ragflow-qwen-asr:0.0.6")
MEDIA_TOOL_IMAGE = os.environ.get(
    "MEDIA_TOOL_IMAGE",
    os.environ.get("MINICPM_IMAGE", "swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai:v0.26.0"),
)
MAX_UPLOAD_BYTES = 20 * 1024 * 1024 * 1024
ALLOWED_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".wmv", ".flv"}

for directory in (UPLOAD_DIR, JOB_STATE_DIR, DEFAULT_OUTPUT_ROOT):
    directory.mkdir(parents=True, exist_ok=True)

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.RLock()
# A single process owns the GPU transition ASR -> visual -> packaging.  This
# prevents two WebUI uploads from stopping/restarting the same VLM container
# underneath each other.
ANALYSIS_LOCK = threading.Lock()
GPU_STATS_LOCK = threading.Lock()
GPU_STATS_CACHE_AT = 0.0
GPU_STATS_CACHE: dict[str, object] = {
    "available": False,
    "message": "等待 GPU 状态",
}


def gpu_status() -> dict[str, object]:
    """Return cached nvidia-smi telemetry without blocking the analysis worker."""

    global GPU_STATS_CACHE_AT, GPU_STATS_CACHE
    now = time.monotonic()
    with GPU_STATS_LOCK:
        if now - GPU_STATS_CACHE_AT < 1.0:
            return dict(GPU_STATS_CACHE)

    value: dict[str, object]
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode:
            value = {"available": False, "message": "nvidia-smi 不可用"}
        else:
            rows: list[dict[str, object]] = []
            for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
                if len(row) < 7:
                    continue
                try:
                    def metric(raw: str) -> float | None:
                        clean = raw.strip()
                        if clean.upper() in {"N/A", "NOT SUPPORTED", "[NOT SUPPORTED]"}:
                            return None
                        return float(clean)

                    rows.append(
                        {
                            "index": row[0].strip(),
                            "name": row[1].strip(),
                            "utilization": metric(row[2]),
                            "memory_used_mb": metric(row[3]),
                            "memory_total_mb": metric(row[4]),
                            "temperature_c": metric(row[5]),
                            "power_w": metric(row[6]),
                        }
                    )
                except (TypeError, ValueError):
                    continue
            if not rows:
                value = {"available": False, "message": "未读取到 GPU"}
            else:
                utilization = [float(item["utilization"]) for item in rows if item["utilization"] is not None]
                memory_used = [float(item["memory_used_mb"]) for item in rows if item["memory_used_mb"] is not None]
                memory_total = [float(item["memory_total_mb"]) for item in rows if item["memory_total_mb"] is not None]
                temperatures = [float(item["temperature_c"]) for item in rows if item["temperature_c"] is not None]
                power = [float(item["power_w"]) for item in rows if item["power_w"] is not None]
                value = {
                    "available": True,
                    "name": str(rows[0]["name"]) if len(rows) == 1 else f"{len(rows)} 张 GPU",
                    "count": len(rows),
                    "utilization_pct": round(max(utilization), 1) if utilization else None,
                    "memory_used_mb": round(sum(memory_used), 1) if memory_used else None,
                    "memory_total_mb": round(sum(memory_total), 1) if memory_total else None,
                    "temperature_c": round(max(temperatures), 1) if temperatures else None,
                    "power_w": round(sum(power), 1) if power else None,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
    except (OSError, subprocess.SubprocessError) as exc:
        value = {"available": False, "message": f"GPU 状态不可用：{type(exc).__name__}"}

    with GPU_STATS_LOCK:
        GPU_STATS_CACHE_AT = time.monotonic()
        GPU_STATS_CACHE = value
        return dict(GPU_STATS_CACHE)


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>野构 Studio · 视频导演拉片工作台</title>
  <style>
    :root { color-scheme: dark; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; --ink: #edf3ff; --muted: #91a1bd; --line: #273755; --blue: #6f96ff; --blue-dark: #4d72e8; --cyan: #62e3ff; --soft: #111d34; --panel: rgba(15, 25, 46, .86); }
    * { box-sizing: border-box; }
    body { position: relative; margin: 0; min-height: 100vh; overflow-x: hidden; background: radial-gradient(circle at 10% -8%, #263c78 0, transparent 34%), radial-gradient(circle at 96% 7%, #30235f 0, transparent 29%), #080e1b; color: var(--ink); }
    body::before { position: fixed; inset: 0; z-index: -2; background-image: linear-gradient(#a6c2ff0d 1px, transparent 1px), linear-gradient(90deg, #a6c2ff0d 1px, transparent 1px); background-size: 52px 52px; mask-image: linear-gradient(to bottom, #000 0, transparent 78%); content: ""; pointer-events: none; }
    body::after { position: fixed; top: -180px; right: -100px; z-index: -1; width: 460px; height: 460px; border: 1px solid #6f96ff24; border-radius: 50%; box-shadow: 0 0 0 28px #6f96ff08, 0 0 0 56px #6f96ff05; content: ""; animation: ambient-pulse 8s ease-in-out infinite; pointer-events: none; }
    main { width: min(1240px, calc(100% - 40px)); margin: 0 auto; padding: 34px 0 76px; }
    .hero { position: relative; margin-bottom: 25px; }
    .eyebrow { display: inline-flex; align-items: center; gap: 8px; color: var(--cyan); font-size: 11px; font-weight: 800; letter-spacing: .16em; }
    .eyebrow::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: #47c58a; box-shadow: 0 0 0 4px #47c58a22; }
    .hero-row { display: flex; align-items: center; justify-content: space-between; gap: 30px; margin-top: 12px; }
    .brand-lockup { display: flex; align-items: center; gap: 20px; min-width: 0; }
    .brand-logo { display: block; width: 132px; height: 132px; flex: 0 0 auto; object-fit: contain; border: 1px solid #d5e6ef; border-radius: 25px; background: #fff; box-shadow: 0 14px 30px #4d83a51c, 0 0 0 7px #ffffffb8; mix-blend-mode: multiply; }
    .brand-copy { min-width: 0; }
    .brand-name { margin-bottom: 8px; color: #3b86b6; font: 800 12px ui-monospace, SFMono-Regular, Consolas, monospace; letter-spacing: .16em; text-transform: uppercase; }
    .hero-tools { display: flex; align-items: center; gap: 12px; }
    .hero-stage { position: relative; width: 176px; height: 112px; flex: 0 0 auto; perspective: 700px; }
    .stage-halo { position: absolute; top: 50%; left: 50%; width: 84px; height: 84px; border-radius: 50%; background: #5f8bff22; filter: blur(18px); transform: translate(-50%, -50%); animation: halo-breathe 5s ease-in-out infinite; }
    .stage-core { position: absolute; top: 50%; left: 50%; display: grid; place-content: center; width: 69px; height: 69px; border: 1px solid #9eb7ff8c; border-radius: 18px; background: linear-gradient(135deg, #527cf0cc, #6e4fdbcc); box-shadow: 0 0 28px #5f8bff55, inset 0 1px #ffffff66; color: #eef4ff; text-align: center; transform: translate(-50%, -50%) rotateX(16deg) rotateY(-18deg) rotateZ(8deg); transform-style: preserve-3d; animation: core-float 5s ease-in-out infinite; }
    .stage-core::before, .stage-core::after { position: absolute; width: 9px; height: 9px; border-radius: 50%; background: var(--cyan); box-shadow: 0 0 12px var(--cyan); content: ""; }
    .stage-core::before { top: 9px; right: 10px; }
    .stage-core::after { bottom: 9px; left: 10px; background: #b79cff; box-shadow: 0 0 12px #b79cff; }
    .stage-core span { font: 800 8px ui-monospace, SFMono-Regular, Consolas, monospace; letter-spacing: .16em; opacity: .75; }
    .stage-core b { margin-top: 3px; font: 700 23px ui-monospace, SFMono-Regular, Consolas, monospace; letter-spacing: -.1em; }
    .stage-orbit { position: absolute; top: 50%; left: 50%; width: 142px; height: 54px; border: 1px solid #6f96ff90; border-radius: 50%; transform-style: preserve-3d; }
    .stage-orbit::after { position: absolute; top: -4px; left: 50%; width: 7px; height: 7px; border-radius: 50%; background: var(--cyan); box-shadow: 0 0 12px var(--cyan); content: ""; }
    .stage-orbit-a { transform: translate(-50%, -50%) rotateX(67deg) rotateZ(18deg); animation: orbit-a 12s linear infinite; }
    .stage-orbit-b { width: 115px; height: 42px; border-color: #b79cff70; transform: translate(-50%, -50%) rotateX(67deg) rotateY(62deg) rotateZ(-24deg); animation: orbit-b 16s linear infinite reverse; }
    .stage-spark { position: absolute; width: 4px; height: 4px; border-radius: 50%; background: #fff; box-shadow: 0 0 9px #fff; animation: spark-drift 4s ease-in-out infinite; }
    .stage-spark.one { top: 15px; left: 25px; }
    .stage-spark.two { right: 17px; bottom: 16px; animation-delay: 1.4s; }
    @keyframes ambient-pulse { 0%, 100% { opacity: .45; transform: scale(1); } 50% { opacity: .8; transform: scale(1.04); } }
    @keyframes halo-breathe { 0%, 100% { opacity: .6; transform: translate(-50%, -50%) scale(.9); } 50% { opacity: 1; transform: translate(-50%, -50%) scale(1.16); } }
    @keyframes core-float { 0%, 100% { margin-top: 0; } 50% { margin-top: -7px; } }
    @keyframes orbit-a { from { transform: translate(-50%, -50%) rotateX(67deg) rotateZ(18deg); } to { transform: translate(-50%, -50%) rotateX(67deg) rotateZ(378deg); } }
    @keyframes orbit-b { from { transform: translate(-50%, -50%) rotateX(67deg) rotateY(62deg) rotateZ(-24deg); } to { transform: translate(-50%, -50%) rotateX(67deg) rotateY(62deg) rotateZ(-384deg); } }
    @keyframes spark-drift { 0%, 100% { opacity: .35; transform: translate3d(0, 0, 0); } 50% { opacity: 1; transform: translate3d(7px, -10px, 18px); } }
    h1 { margin: 0 0 10px; font-size: clamp(28px, 5vw, 42px); letter-spacing: -.04em; line-height: 1.08; }
    .title-mark { display: inline-block; margin-left: 5px; color: var(--cyan); font: 800 .34em ui-monospace, SFMono-Regular, Consolas, monospace; letter-spacing: .14em; vertical-align: .25em; }
    h2 { margin: 0; font-size: 18px; letter-spacing: -.02em; }
    .hint { max-width: 700px; margin: 0; color: var(--muted); line-height: 1.65; }
    .local-pill, .tag { border: 1px solid #304466; background: #111d34cc; color: #a8b8d4; border-radius: 999px; padding: 9px 13px; font-size: 12px; white-space: nowrap; }
    .local-pill { display: flex; align-items: center; gap: 8px; }
    .local-pill span { width: 7px; height: 7px; border-radius: 50%; background: #47c58a; }
    .studio-grid { display: grid; grid-template-columns: minmax(0, 1.62fr) minmax(270px, .72fr); gap: 16px; align-items: start; }
    .studio-sidebar { display: grid; gap: 16px; position: sticky; top: 18px; }
    .card { background: var(--panel); border: 1px solid var(--line); border-radius: 19px; padding: 24px; margin-bottom: 16px; box-shadow: 0 20px 60px #0000002b, inset 0 1px #ffffff0a; backdrop-filter: blur(16px); }
    .studio-sidebar .card { margin-bottom: 0; }
    .card-head { display: flex; align-items: center; justify-content: space-between; gap: 18px; margin-bottom: 18px; }
    .section-label { margin-bottom: 5px; color: #7185aa; font-size: 10px; font-weight: 800; letter-spacing: .17em; }
    label { display: block; font-weight: 650; margin-bottom: 10px; }
    .upload-zone { position: relative; display: flex; align-items: center; gap: 15px; min-height: 94px; padding: 18px; border: 1px dashed #3b527a; border-radius: 14px; background: linear-gradient(135deg, #142341, #101b32); cursor: pointer; transition: border .2s, background .2s, transform .2s, box-shadow .2s; }
    .upload-zone:hover, .upload-zone.dragging { border-color: var(--cyan); background: linear-gradient(135deg, #172b51, #12203b); box-shadow: 0 10px 28px #00000026; transform: translateY(-1px); }
    .upload-zone input[type=file] { position: absolute; inset: 0; width: 100%; height: 100%; opacity: 0; cursor: pointer; }
    .upload-icon { display: grid; place-items: center; width: 45px; height: 45px; border: 1px solid #6f96ff55; border-radius: 13px; background: #3455ad55; color: var(--cyan); font-size: 23px; flex: 0 0 auto; }
    .upload-copy strong { display: block; margin-bottom: 4px; }
    .upload-copy span { color: var(--muted); font-size: 13px; }
    .file-name { margin-top: 9px; color: var(--cyan); font-size: 13px; overflow-wrap: anywhere; }
    .preview-panel { margin-top: 18px; padding: 15px; border: 1px solid #2b3d61; border-radius: 15px; background: linear-gradient(145deg, #111f3a, #0e192f); }
    .preview-panel[hidden] { display: none; }
    .preview-top { display: flex; align-items: center; justify-content: space-between; gap: 16px; }
    .preview-top h3 { margin: 0; font-size: 16px; letter-spacing: -.02em; }
    .preview-meta { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 7px; color: #62718a; font-size: 12px; }
    .preview-meta span { padding: 6px 9px; border: 1px solid #32476d; border-radius: 999px; background: #0d172acc; }
    .video-frame { position: relative; display: grid; place-items: center; min-height: 170px; margin-top: 12px; overflow: hidden; border-radius: 13px; background: #10182a; box-shadow: 0 10px 24px #1b2e5b1c; }
    .video-preview { display: block; width: 100%; max-height: 340px; aspect-ratio: 16 / 9; object-fit: contain; background: #10182a; }
    .video-loading { position: absolute; color: #cbd5e1; font-size: 13px; pointer-events: none; }
    .video-loading[hidden] { display: none; }
    .timeline-card { margin-top: 12px; padding: 13px 14px 11px; border: 1px solid #2c3f64; border-radius: 12px; background: #0b1528; }
    .timeline-head, .timeline-actions, .timeline-scale { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    .timeline-head { color: #8192b0; font-size: 12px; }
    .timeline-head strong { color: var(--ink); font-size: 13px; }
    .timeline-readout { display: flex; align-items: baseline; gap: 8px; margin-top: 7px; }
    .time-chip { color: var(--blue); font: 750 14px ui-monospace, SFMono-Regular, Consolas, monospace; }
    .time-chip.end { color: #5a55e8; }
    .time-divider { color: #a7b2c5; }
    .selection-duration { margin-left: auto; color: #7d8aa0; font-size: 12px; }
    .timeline-shell { position: relative; height: 39px; margin: 5px 7px 0; }
    .timeline-track { position: absolute; top: 15px; right: 0; left: 0; height: 8px; overflow: hidden; border-radius: 999px; background: #1e2c47; }
    .timeline-fill { position: absolute; top: 0; bottom: 0; left: 0; right: 0; border-radius: 999px; background: linear-gradient(90deg, #4776f7, #6758e8); box-shadow: 0 2px 7px #4776f744; }
    .timeline-playhead { position: absolute; top: 6px; bottom: 8px; left: 0; width: 2px; border-radius: 2px; background: #17233a; opacity: .7; pointer-events: none; transform: translateX(-1px); }
    .timeline-range { position: absolute; top: 0; left: 0; z-index: 3; width: 100%; height: 39px; margin: 0; appearance: none; background: transparent; pointer-events: none; }
    .timeline-range::-webkit-slider-runnable-track { height: 8px; background: transparent; }
    .timeline-range::-moz-range-track { height: 8px; background: transparent; }
    .timeline-range::-webkit-slider-thumb { width: 18px; height: 18px; margin-top: -5px; appearance: none; border: 3px solid #fff; border-radius: 50%; background: var(--blue); box-shadow: 0 2px 7px #22376a55; cursor: ew-resize; pointer-events: auto; }
    .timeline-range::-moz-range-thumb { width: 12px; height: 12px; border: 3px solid #fff; border-radius: 50%; background: var(--blue); box-shadow: 0 2px 7px #22376a55; cursor: ew-resize; pointer-events: auto; }
    #timeline-end-range { z-index: 4; }
    #timeline-start-range { z-index: 5; }
    .timeline-scale { margin-top: -1px; color: #7183a4; font: 10px ui-monospace, SFMono-Regular, Consolas, monospace; }
    .timeline-actions { margin-top: 10px; align-items: center; }
    .range-summary { color: #8ea0be; font-size: 12px; }
    .quick-actions { display: flex; gap: 7px; }
    .mini-button { width: auto; margin: 0; padding: 7px 9px; border: 1px solid #33486e; background: #111f39; color: #9eafd0; font-size: 11px; box-shadow: none; }
    .mini-button:hover { border-color: #6f96ff; background: #182c50; color: var(--cyan); box-shadow: none; }
    .range-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 20px; }
    .range-grid label { margin: 0; color: #9aabc7; font-size: 13px; }
    .range-grid input { width: 100%; padding: 11px 12px; margin-top: 8px; border: 1px solid #2d4268; border-radius: 10px; outline: none; background: #0b1528; color: var(--ink); font-size: 14px; transition: border .2s, box-shadow .2s; }
    .range-grid input:focus { border-color: #7393f6; box-shadow: 0 0 0 4px #3568f233; }
    .release-option { display: flex; align-items: flex-start; gap: 10px; margin-top: 16px; padding: 12px 13px; border: 1px solid #263b60; border-radius: 12px; background: #0e1a30; cursor: pointer; }
    .release-option input { width: 16px; height: 16px; flex: 0 0 auto; margin: 2px 0 0; accent-color: var(--blue); }
    .release-option strong, .release-option small { display: block; }
    .release-option strong { font-size: 12px; }
    .release-option small { margin-top: 4px; color: #8293b0; font-size: 11px; line-height: 1.5; }
    button { width: 100%; margin-top: 18px; border: 0; border-radius: 11px; padding: 13px 20px; background: linear-gradient(135deg, var(--blue), #5a55e8); color: white; font-size: 15px; font-weight: 700; cursor: pointer; box-shadow: 0 8px 18px #3568f233; transition: transform .2s, box-shadow .2s, opacity .2s; }
    button:hover { transform: translateY(-1px); box-shadow: 0 11px 22px #3568f33d; }
    button:disabled { background: #34415c; color: #93a1b9; box-shadow: none; cursor: wait; transform: none; }
    .output-note { display: flex; align-items: center; gap: 12px; margin-top: 19px; padding: 13px 14px; border: 1px solid #243758; border-radius: 12px; background: #101d35; }
    .note-icon { display: grid; place-items: center; width: 29px; height: 29px; border: 1px solid #5b79dc55; border-radius: 9px; background: #253d8255; color: var(--cyan); font-weight: 800; }
    .output-note strong { display: block; margin-bottom: 5px; font-size: 12px; }
    .output-note code { padding: 0; background: transparent; color: #9aafd4; font: 12px ui-monospace, SFMono-Regular, Consolas, monospace; }
    .flow { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 4px; }
    .flow-item { min-height: 118px; padding: 15px; border: 1px solid #243758; border-radius: 13px; background: #0e1a30; transition: border .25s, transform .25s, background .25s; }
    .flow-item:hover { border-color: #5275d7; background: #122341; transform: translateY(-3px); }
    .flow-num { display: block; margin-bottom: 15px; color: #8ca5ee; font: 800 11px ui-monospace, monospace; }
    .flow-item strong { display: block; margin-bottom: 6px; font-size: 13px; }
    .flow-item span:last-child { display: block; color: var(--muted); font-size: 12px; line-height: 1.55; }
    progress { width: 100%; height: 9px; margin-top: 4px; border: 0; border-radius: 999px; overflow: hidden; }
    progress::-webkit-progress-bar { background: #1c2a45; border-radius: 999px; }
    progress::-webkit-progress-value { background: linear-gradient(90deg, var(--blue), #6b59e9); border-radius: 999px; }
    progress::-moz-progress-bar { background: linear-gradient(90deg, var(--blue), #6b59e9); border-radius: 999px; }
    .progress-head { display: flex; align-items: flex-end; justify-content: space-between; gap: 18px; margin-top: 18px; }
    .progress-head strong { display: block; font-size: 16px; }
    .progress-percent { color: var(--blue); font: 800 30px/1 ui-monospace, SFMono-Regular, Consolas, monospace; letter-spacing: -.06em; }
    .run-plan { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin-top: 15px; }
    .plan-step { min-width: 0; padding: 10px 11px; border: 1px solid #263b60; border-radius: 11px; background: #0e1a30; opacity: .62; transition: border .2s, background .2s, opacity .2s, transform .2s; }
    .plan-step.active { border-color: #6f96ff; background: #172a50; opacity: 1; box-shadow: 0 0 0 1px #6f96ff2b, 0 8px 20px #3154a51c; transform: translateY(-1px); }
    .plan-step.done { border-color: #4a9985; background: #102d2d; opacity: 1; }
    .plan-index { display: block; margin-bottom: 6px; color: #8198ca; font: 800 10px ui-monospace, monospace; }
    .plan-step strong, .plan-step span:last-child { display: block; }
    .plan-step strong { overflow: hidden; font-size: 12px; text-overflow: ellipsis; white-space: nowrap; }
    .plan-step span:last-child { margin-top: 4px; overflow: hidden; color: #8293b0; font-size: 10px; line-height: 1.4; text-overflow: ellipsis; white-space: nowrap; }
    .telemetry-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 9px; margin-top: 14px; }
    .telemetry-item { min-width: 0; padding: 12px; border: 1px solid #263b60; border-radius: 12px; background: linear-gradient(145deg, #12213d, #0d182c); }
    .telemetry-label { display: block; margin-bottom: 7px; color: #8195b7; font-size: 10px; letter-spacing: .07em; }
    .telemetry-value { display: block; overflow: hidden; color: #e1eaff; font: 750 17px/1.1 ui-monospace, SFMono-Regular, Consolas, monospace; text-overflow: ellipsis; white-space: nowrap; }
    .telemetry-sub { display: block; margin-top: 6px; overflow: hidden; color: #8293b0; font-size: 10px; text-overflow: ellipsis; white-space: nowrap; }
    .telemetry-foot { margin-top: 10px; color: #7489aa; font: 10px ui-monospace, SFMono-Regular, Consolas, monospace; overflow-wrap: anywhere; }
    .row { display: flex; justify-content: space-between; gap: 18px; align-items: center; }
    .status { padding: 6px 10px; border: 1px solid #3e5d9b; border-radius: 999px; background: #1a2e5b; color: #a8c0ff; font-size: 12px; font-weight: 700; }
    .error { color: #ff9b9b; white-space: pre-wrap; }
    pre { margin: 14px 0 0; padding: 15px; min-height: 180px; max-height: 420px; overflow: auto; background: #111a2d; color: #dbeafe; border-radius: 12px; font: 13px/1.6 ui-monospace, SFMono-Regular, Consolas, monospace; white-space: pre-wrap; }
    code { display: block; padding: 10px 12px; background: #0d182c; border: 1px solid #243758; border-radius: 8px; overflow-wrap: anywhere; color: #b9c8e2; }
    ul { padding-left: 20px; line-height: 1.9; }
    a { color: var(--blue); text-decoration: none; }
    a:hover { text-decoration: underline; }
    .small { color: var(--muted); font-size: 13px; line-height: 1.6; }
    .studio-main { min-width: 0; }
    .pipeline-card { margin-top: 0; }
    .side-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 18px; }
    .side-head h2 { font-size: 17px; }
    .live-badge { display: inline-flex; align-items: center; gap: 6px; padding: 6px 8px; border: 1px solid #2e7b73; border-radius: 999px; background: #123832; color: #7df0d1; font: 700 9px ui-monospace, SFMono-Regular, Consolas, monospace; letter-spacing: .08em; }
    .live-badge::before { width: 5px; height: 5px; border-radius: 50%; background: #66f1c4; box-shadow: 0 0 9px #66f1c4; content: ""; }
    .model-stack { display: grid; gap: 10px; }
    .model-row { display: flex; align-items: center; gap: 11px; min-width: 0; padding: 11px; border: 1px solid #243758; border-radius: 13px; background: linear-gradient(135deg, #12213d, #0c172b); transition: border .25s, transform .25s, box-shadow .25s; }
    .model-row:hover { border-color: #5678d5; box-shadow: 0 10px 25px #0000002b; transform: translate3d(0, -2px, 8px); }
    .model-mark { display: grid; place-items: center; width: 31px; height: 31px; flex: 0 0 auto; border: 1px solid #6f96ff66; border-radius: 10px; background: #3656a955; color: #b9c9ff; font: 800 12px ui-monospace, SFMono-Regular, Consolas, monospace; }
    .model-mark.audio { border-color: #62e3ff66; background: #1a648055; color: #8ff0ff; }
    .model-mark.align { border-color: #b79cff66; background: #684e9b55; color: #d6c7ff; }
    .model-info { min-width: 0; }
    .model-info small, .model-info strong, .model-info em { display: block; }
    .model-info small { margin-bottom: 3px; color: #7185a7; font-size: 10px; letter-spacing: .03em; }
    .model-info strong { overflow: hidden; color: #e4ecff; font: 650 12px ui-monospace, SFMono-Regular, Consolas, monospace; text-overflow: ellipsis; white-space: nowrap; }
    .model-info em { margin-top: 4px; color: #8293b0; font-size: 10px; font-style: normal; }
    .gpu-note { display: flex; align-items: flex-start; gap: 8px; margin-top: 15px; padding-top: 13px; border-top: 1px solid #223452; color: #8293b0; font-size: 11px; line-height: 1.55; }
    .gpu-note::before { width: 7px; height: 7px; flex: 0 0 auto; margin-top: 4px; border: 1px solid #6f96ff; border-radius: 50%; box-shadow: 0 0 10px #6f96ff; content: ""; }
    .delivery-stat { display: flex; align-items: center; gap: 12px; padding: 13px; border: 1px solid #293d61; border-radius: 13px; background: #0d192f; }
    .delivery-stat strong { color: var(--cyan); font: 700 30px/1 ui-monospace, SFMono-Regular, Consolas, monospace; }
    .delivery-stat span { color: #9aabca; font-size: 12px; line-height: 1.5; }
    .delivery-list { display: grid; gap: 8px; margin: 14px 0 0; }
    .delivery-item { display: flex; align-items: center; gap: 8px; color: #a5b4cf; font-size: 11px; }
    .delivery-item::before { display: grid; place-items: center; width: 16px; height: 16px; border: 1px solid #3b5481; border-radius: 5px; color: #7fe8da; content: "✓"; font-size: 10px; }
    .path-label { margin: 16px 0 7px; color: #7185a7; font-size: 10px; letter-spacing: .08em; }
    .side-path { padding: 10px; color: #9db0d1; font: 11px/1.5 ui-monospace, SFMono-Regular, Consolas, monospace; overflow-wrap: anywhere; }
    .studio-footer { display: flex; align-items: center; justify-content: space-between; gap: 16px; margin-top: 4px; color: #5f7091; font-size: 11px; }
    .studio-footer span:last-child { color: #7387aa; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
    @media (prefers-reduced-motion: reduce) { *, *::before, *::after { scroll-behavior: auto !important; animation-duration: .01ms !important; animation-iteration-count: 1 !important; transition-duration: .01ms !important; } }
    @media (max-width: 920px) { .studio-grid { grid-template-columns: 1fr; } .studio-sidebar { position: static; grid-template-columns: 1fr 1fr; } }
    @media (max-width: 720px) { main { width: min(100% - 24px, 1240px); padding-top: 26px; } .hero-row { align-items: flex-start; flex-direction: column; } .brand-lockup { align-items: flex-start; gap: 14px; } .brand-logo { width: 96px; height: 96px; border-radius: 19px; } .hero-stage { align-self: center; margin-top: -8px; } .flow { grid-template-columns: 1fr 1fr; } .preview-top, .timeline-actions { align-items: flex-start; flex-direction: column; } .preview-meta { justify-content: flex-start; } .quick-actions { flex-wrap: wrap; } .studio-sidebar { grid-template-columns: 1fr; } .run-plan { grid-template-columns: repeat(2, minmax(0, 1fr)); } .telemetry-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
    @media (max-width: 480px) { main { width: min(100% - 20px, 980px); } .card { padding: 17px; border-radius: 15px; } .range-grid, .flow, .run-plan, .telemetry-grid { grid-template-columns: 1fr; } .local-pill { align-self: flex-start; } }

    /* Minimal geometric light theme */
    :root { color-scheme: light; --ink: #17354f; --muted: #718aa0; --line: #dceaf3; --blue: #3988c2; --blue-dark: #266c9f; --cyan: #55b9d8; --soft: #f1f8fc; --panel: rgba(255, 255, 255, .94); }
    body { background: linear-gradient(135deg, #f9fcff 0%, #edf7fc 52%, #f7fbfe 100%); color: var(--ink); }
    body::before { z-index: 0; background-image: linear-gradient(#6fa8c915 1px, transparent 1px), linear-gradient(90deg, #6fa8c915 1px, transparent 1px); background-size: 56px 56px; mask-image: linear-gradient(to bottom, #000 0, transparent 82%); }
    body::after { display: none; }
    .ripple-field { position: fixed; inset: 0; z-index: 0; overflow: hidden; perspective: 1000px; pointer-events: none; }
    .ripple-field::before { position: absolute; top: -130px; right: -80px; width: 430px; height: 280px; border-radius: 50%; background: radial-gradient(ellipse, #a6d9ed38 0, transparent 68%); filter: blur(8px); content: ""; }
    .ripple-field::after { position: absolute; right: 12%; bottom: 7%; width: 210px; height: 92px; border-radius: 50%; background: #a8dbed2b; filter: blur(32px); content: ""; }
    .ripple-ring { position: absolute; display: block; border: 1px solid #63accd4f; border-radius: 50%; transform-style: preserve-3d; animation: light-ripple 18s ease-in-out infinite; }
    .ripple-one { top: -35px; right: -120px; width: 680px; height: 300px; transform: rotateX(67deg) rotateZ(-16deg) scale(.97); }
    .ripple-two { top: 13px; right: -98px; width: 590px; height: 260px; border-color: #77b9d64a; animation-delay: -4s; transform: rotateX(67deg) rotateZ(-16deg) scale(.97); }
    .ripple-three { top: 62px; right: -76px; width: 500px; height: 220px; border-color: #4c9fc852; animation-delay: -8s; transform: rotateX(67deg) rotateZ(-16deg) scale(.97); }
    .ripple-four { top: 111px; right: -54px; width: 410px; height: 180px; border-color: #a0cfe04c; animation-delay: -12s; transform: rotateX(67deg) rotateZ(-16deg) scale(.97); }
    @keyframes light-ripple { 0%, 100% { opacity: .42; transform: rotateX(67deg) rotateZ(-16deg) scale(.97) translateZ(0); } 50% { opacity: .78; transform: rotateX(67deg) rotateZ(-16deg) scale(1.03) translateZ(12px); } }
    main { position: relative; z-index: 1; }
    .hero { margin-bottom: 25px; }
    .hero-tools { margin-left: auto; }
    .hero-stage { display: none; }
    .eyebrow { color: #3b86b6; }
    .eyebrow::before { background: #55c7a1; box-shadow: 0 0 0 4px #55c7a122; }
    .brand-logo { border-color: #d5e6ef; box-shadow: 0 14px 30px #4d83a51c, 0 0 0 7px #ffffffb8; }
    .brand-name { color: #3b86b6; }
    .title-mark { color: #3b9bc1; }
    .hint { color: #718aa0; }
    .local-pill, .tag { border-color: #d3e5ef; background: #ffffffc9; color: #608098; box-shadow: 0 4px 15px #5d96b312; }
    .local-pill span { background: #45b58d; }
    .card { background: var(--panel); border-color: #dbeaf3; box-shadow: 0 18px 45px #4d83a512, inset 0 1px #ffffff; backdrop-filter: blur(13px); }
    .section-label { color: #86a1b5; }
    .upload-zone { border-color: #a7ccdf; background: linear-gradient(135deg, #f8fcff, #edf7fc); }
    .upload-zone:hover, .upload-zone.dragging { border-color: #5ca8d0; background: linear-gradient(135deg, #f0faff, #e7f5fb); box-shadow: 0 9px 25px #4d93b51a; }
    .upload-icon { border-color: #a7d3e4; background: #e2f3fa; color: #368bb5; }
    .upload-copy span, .small { color: #718aa0; }
    .file-name { color: #378dbb; }
    .preview-panel { border-color: #cfe4ee; background: linear-gradient(145deg, #f6fbfe, #edf7fc); }
    .preview-meta { color: #6b879e; }
    .preview-meta span { border-color: #d4e6ef; background: #ffffffc9; }
    .video-frame { background: #e8f3f8; box-shadow: 0 10px 24px #4d83a516; }
    .video-preview { background: #e8f3f8; }
    .video-loading { color: #6e8ba0; }
    .timeline-card { border-color: #cfe3ed; background: #ffffff; box-shadow: 0 7px 18px #5b91aa0c; }
    .timeline-head { color: #7290a5; }
    .timeline-head strong { color: #224661; }
    .time-chip { color: #2e86b8; }
    .time-chip.end { color: #6f78ba; }
    .time-divider { color: #a8bdca; }
    .selection-duration { color: #7892a4; }
    .timeline-track { background: #deedf4; }
    .timeline-fill { background: linear-gradient(90deg, #58abd0, #829fdd); box-shadow: 0 2px 7px #5b9ec744; }
    .timeline-playhead { background: #2c5872; }
    .timeline-range::-webkit-slider-thumb { border-color: #fff; background: #3988c2; box-shadow: 0 2px 7px #397da455; }
    .timeline-range::-moz-range-thumb { border-color: #fff; background: #3988c2; box-shadow: 0 2px 7px #397da455; }
    .timeline-scale { color: #8aa4b5; }
    .range-summary { color: #6e889d; }
    .mini-button { border-color: #cde1eb; background: #f5fbfe; color: #54758b; }
    .mini-button:hover { border-color: #83bdd8; background: #eaf7fc; color: #2b7eae; }
    .range-grid label { color: #58778e; }
    .range-grid input { border-color: #cfe2ec; background: #fbfdff; color: #234761; }
    .range-grid input:focus { border-color: #70afd0; box-shadow: 0 0 0 4px #3988c21c; }
    .release-option { border-color: #d6e7ef; background: #f8fcfe; }
    .release-option small { color: #7891a3; }
    button { background: linear-gradient(135deg, #3988c2, #5c9ed0); color: #fff; box-shadow: 0 8px 18px #3988c233; }
    button:hover { box-shadow: 0 11px 22px #3988c23d; }
    button:disabled { background: #b5cbd7; color: #edf7fb; }
    .flow-item { border-color: #dceaf2; background: #f8fcfe; }
    .flow-item:hover { border-color: #9dc9dd; background: #f0f9fd; }
    .flow-num { color: #75a9c4; }
    .flow-item span:last-child { color: #718aa0; }
    progress::-webkit-progress-bar { background: #e0edf4; }
    progress::-webkit-progress-value { background: linear-gradient(90deg, #55a8ce, #809fdd); }
    progress::-moz-progress-bar { background: linear-gradient(90deg, #55a8ce, #809fdd); }
    .progress-percent { color: #3988c2; }
    .plan-step { border-color: #d6e7ef; background: #f8fcfe; }
    .plan-step.active { border-color: #65a8ca; background: #eaf6fb; box-shadow: 0 0 0 1px #65a8ca24, 0 8px 20px #4d83a514; }
    .plan-step.done { border-color: #8cc9b0; background: #f0faf5; }
    .plan-index { color: #70a0b9; }
    .plan-step span:last-child { color: #7891a3; }
    .telemetry-item { border-color: #d6e7ef; background: linear-gradient(145deg, #f8fcfe, #eef8fc); box-shadow: 0 3px 10px #4d83a50b; }
    .telemetry-label { color: #7b96a8; }
    .telemetry-value { color: #26516b; }
    .telemetry-sub, .telemetry-foot { color: #7891a3; }
    .status { border-color: #b8d8e8; background: #edf8fc; color: #327baa; }
    .error { color: #c45555; }
    pre { background: #eef7fb; border: 1px solid #d6e8f0; color: #355a72; }
    code { background: #f3f9fc; border-color: #d6e8f0; color: #4c6f84; }
    a { color: #2e82b3; }
    .live-badge { border-color: #adddca; background: #effaf5; color: #388466; }
    .live-badge::before { background: #52c495; box-shadow: 0 0 9px #52c49588; }
    .model-row { border-color: #d6e7ef; background: linear-gradient(135deg, #f9fdff, #eef8fc); box-shadow: 0 3px 9px #4d83a50a; }
    .model-row:hover { border-color: #99c6db; box-shadow: 0 8px 20px #4d83a51a; transform: translateY(-1px); }
    .model-mark { border-color: #acd3e4; background: #e6f5fa; color: #3988b5; }
    .model-mark.audio { border-color: #a8dce5; background: #e8f8f9; color: #3e98a7; }
    .model-mark.align { border-color: #c8c9eb; background: #f0f0fc; color: #7478b4; }
    .model-info small { color: #7994a7; }
    .model-info strong { color: #244b66; }
    .model-info em { color: #7891a3; }
    .gpu-note { border-top-color: #d9eaf1; color: #7891a3; }
    .gpu-note::before { border-color: #5ca8d0; box-shadow: 0 0 10px #5ca8d066; }
    .delivery-stat { border-color: #d5e7ef; background: #f3fafe; }
    .delivery-stat strong { color: #398bb9; }
    .delivery-stat span, .delivery-item { color: #648196; }
    .delivery-item::before { border-color: #b9d8e5; color: #48a984; }
    .path-label { color: #829caf; }
    .side-path { color: #5a788c; }
    .studio-footer { color: #89a1b0; }
    .studio-footer span:last-child { color: #7795a7; }
    @media (max-width: 720px) { .hero-tools { width: 100%; justify-content: flex-start; } }
  </style>
</head>
<body>
  <div class="ripple-field" aria-hidden="true">
    <i class="ripple-ring ripple-one"></i>
    <i class="ripple-ring ripple-two"></i>
    <i class="ripple-ring ripple-three"></i>
    <i class="ripple-ring ripple-four"></i>
  </div>
  <main>
    <header class="hero">
      <div class="eyebrow">野构 STUDIO · LOCAL MEDIA WORKBENCH</div>
      <div class="hero-row">
        <div class="brand-lockup">
          <img class="brand-logo" src="/assets/brand/yegou-studio-logo.png" alt="野构 Studio 创意标志">
          <div class="brand-copy">
            <div class="brand-name">野构 Studio</div>
            <h1>视频导演拉片工作台 <span class="title-mark">MEDIA LAB</span></h1>
            <p class="hint">把视频交给野构 Studio，得到可回看、可下载、可继续加工的视觉与声音时间线。</p>
          </div>
        </div>
        <div class="hero-tools">
          <div class="local-pill"><span></span>本地运行 · 不上传云端</div>
        </div>
      </div>
    </header>

    <div class="studio-grid">
      <div class="studio-main">
    <section class="card input-card">
      <div class="card-head">
        <div><div class="section-label">INPUT</div><h2>选择一个视频</h2></div>
        <span class="tag">默认分析整片</span>
      </div>
      <form id="upload-form">
        <label class="upload-zone" id="drop-zone" for="video">
          <input id="video" name="video" type="file" accept="video/*" required>
          <span class="upload-icon">↑</span>
          <span class="upload-copy"><strong>点击选择，或把视频拖到这里</strong><span>支持 MP4、MOV、MKV、AVI、WebM 等常见格式</span></span>
        </label>
        <div id="file-name" class="file-name">尚未选择视频</div>
        <div id="preview-panel" class="preview-panel" hidden>
          <div class="preview-top">
            <div><div class="section-label">VISUAL RANGE</div><h3>可视时间轴</h3></div>
            <div class="preview-meta"><span id="preview-duration">总时长 --:--</span><span id="preview-time">当前 --:--.-</span></div>
          </div>
          <div class="video-frame">
            <video id="video-preview" class="video-preview" controls preload="metadata"></video>
            <span id="video-loading" class="video-loading">正在读取视频时长…</span>
          </div>
          <div class="timeline-card">
            <div class="timeline-head"><strong>拖动两端选择分析区间</strong><span>蓝色区域 = 将要送入模型的片段</span></div>
            <div class="timeline-readout">
              <span id="timeline-start-label" class="time-chip">00:00.0</span>
              <span class="time-divider">—</span>
              <span id="timeline-end-label" class="time-chip end">--:--.-</span>
              <span id="selection-duration" class="selection-duration">选中 --:--.-</span>
            </div>
            <div id="timeline-shell" class="timeline-shell">
              <div class="timeline-track"><div id="timeline-fill" class="timeline-fill"></div></div>
              <div id="timeline-playhead" class="timeline-playhead"></div>
              <input id="timeline-start-range" class="timeline-range" type="range" min="0" max="0" step="0.1" value="0" aria-label="分析开始位置">
              <input id="timeline-end-range" class="timeline-range" type="range" min="0" max="0" step="0.1" value="0" aria-label="分析结束位置">
            </div>
            <div id="timeline-scale" class="timeline-scale"><span>00:00</span><span>25%</span><span>50%</span><span>75%</span><span>结尾</span></div>
            <div class="timeline-actions">
              <span id="range-summary" class="range-summary">整部视频</span>
              <div class="quick-actions">
                <button id="set-start" class="mini-button" type="button">以当前播放位置设为开始</button>
                <button id="set-end" class="mini-button" type="button">以当前播放位置设为结束</button>
              </div>
            </div>
          </div>
          <p class="small">时间轴只在浏览器本地预览，不会上传额外数据。拖动后下方秒数会同步更新；不设结束点仍表示分析到视频结尾。</p>
        </div>
        <div class="range-grid">
          <label for="start-sec">开始秒
            <input id="start-sec" type="number" min="0" step="0.1" placeholder="0">
          </label>
          <label for="end-sec">结束秒
            <input id="end-sec" type="number" min="0" step="0.1" placeholder="留空 = 视频结尾">
          </label>
        </div>
        <p class="small">例如填写 20 和 40，只分析原视频的 00:20–00:40。截取片段内的报告时间码从 00:00 重新计时，任务信息中会保留原视频区间。</p>
        <label class="release-option" for="release-gpu">
          <input id="release-gpu" type="checkbox" checked>
          <span><strong>分析完成后释放显存</strong><small>默认停止 MiniCPM-V 容器，释放 GPU；取消勾选可保留热模型以加快下一次任务。</small></span>
        </label>
        <button id="start" type="submit">开始分析</button>
      </form>
      <p class="small">开始秒留空按 0 处理，结束秒留空按视频结尾处理。长视频会分批执行，页面显示的是阶段级大概进度。</p>
    </section>
      </div>
      <aside class="studio-sidebar">
        <section class="card model-card">
          <div class="side-head">
            <div><div class="section-label">LOCAL MODELS</div><h2>分析引擎</h2></div>
            <span class="live-badge">GPU READY</span>
          </div>
          <div class="model-stack">
            <div class="model-row">
              <span class="model-mark">V</span>
              <div class="model-info"><small>视觉理解 · 导演拉片</small><strong>MiniCPM-V-4.5-GPTQ</strong><em>逐秒高清帧 + 20 秒上下文</em></div>
            </div>
            <div class="model-row">
              <span class="model-mark audio">A</span>
              <div class="model-info"><small>语音识别 · ASR</small><strong>Qwen3-ASR-0.6B</strong><em>原始语音与分段转写</em></div>
            </div>
            <div class="model-row">
              <span class="model-mark align">T</span>
              <div class="model-info"><small>时间对齐 · ForcedAligner</small><strong>Qwen3-ForcedAligner-0.6B</strong><em>词级时间戳与字幕轨</em></div>
            </div>
          </div>
          <div class="gpu-note">模型按 ASR → 视觉 → 整理顺序切换，尽量避免多个大模型同时占用显存。</div>
        </section>
        <section class="card delivery-card">
          <div class="side-head">
            <div><div class="section-label">DELIVERY</div><h2>输出工作包</h2></div>
            <span class="tag">V1</span>
          </div>
          <div class="delivery-stat"><strong>04</strong><span>份主文档<br>可下载、可继续加工</span></div>
          <div class="delivery-list">
            <div class="delivery-item">纯视觉分析</div>
            <div class="delivery-item">纯 ASR 与时间戳</div>
            <div class="delivery-item">代码综合时间线</div>
            <div class="delivery-item">最终导演分析模板</div>
          </div>
          <div class="path-label">DEFAULT OUTPUT</div>
          <div class="side-path">C:\\Users\\Administrator\\Desktop\\media_analysis</div>
        </section>
      </aside>
    </div>

    <section class="card pipeline-card">
      <div class="card-head">
        <div><div class="section-label">PIPELINE</div><h2>这次任务会做什么</h2></div>
      </div>
      <div class="flow">
        <div class="flow-item"><span class="flow-num">01</span><strong>截取范围</strong><span>按开始秒和结束秒生成分析片段。</span></div>
        <div class="flow-item"><span class="flow-num">02</span><strong>声音时间线</strong><span>ASR + ForcedAligner 生成旁白和时间戳。</span></div>
        <div class="flow-item"><span class="flow-num">03</span><strong>视觉拉片</strong><span>1 秒高清帧、20 秒联系图、5 秒细节组。</span></div>
        <div class="flow-item"><span class="flow-num">04</span><strong>整理下载</strong><span>输出四份主文档及 SRT/VTT 字幕。</span></div>
      </div>
      <p class="small">前三份是可复核的机器产物；第四份是解释层。没有额外提交最终分析时，第四份会显示待处理模板。</p>
    </section>

    <section id="progress-card" class="card" hidden>
      <div class="row">
        <strong id="filename">等待任务</strong>
        <span id="status" class="status">等待中</span>
      </div>
      <div class="progress-head">
        <div><div class="section-label">LIVE PROCESS</div><strong>处理进度</strong></div>
        <span id="progress-percent" class="progress-percent">0%</span>
      </div>
      <progress id="progress" value="0" max="100"></progress>
      <div id="phase" class="small">尚未开始</div>
      <div class="run-plan" aria-label="任务计划">
        <div class="plan-step" data-plan-step="0"><span class="plan-index">01</span><strong>准备与截取</strong><span>生成分析输入</span></div>
        <div class="plan-step" data-plan-step="1"><span class="plan-index">02</span><strong>ASR + 对齐</strong><span>语音与词级时间戳</span></div>
        <div class="plan-step" data-plan-step="2"><span class="plan-index">03</span><strong>视觉拉片</strong><span>1fps / 20秒 / 5秒</span></div>
        <div class="plan-step" data-plan-step="3"><span class="plan-index">04</span><strong>综合打包</strong><span>四份文档与字幕</span></div>
      </div>
      <div class="telemetry-grid">
        <div class="telemetry-item"><span class="telemetry-label">GPU 利用率</span><strong id="gpu-utilization" class="telemetry-value">--</strong><span id="gpu-utilization-sub" class="telemetry-sub">等待采样</span></div>
        <div class="telemetry-item"><span class="telemetry-label">显存占用</span><strong id="gpu-memory" class="telemetry-value">--</strong><span id="gpu-memory-sub" class="telemetry-sub">已用 / 总量</span></div>
        <div class="telemetry-item"><span class="telemetry-label">温度 / 功耗</span><strong id="gpu-thermal" class="telemetry-value">--</strong><span id="gpu-power" class="telemetry-sub">功耗 --</span></div>
        <div class="telemetry-item"><span class="telemetry-label">视觉设备</span><strong id="gpu-name" class="telemetry-value">--</strong><span id="gpu-refresh" class="telemetry-sub">状态等待</span></div>
      </div>
      <pre id="logs"></pre>
    </section>

    <section id="result-card" class="card" hidden>
      <h2>输出结果</h2>
      <p>输出目录：</p>
      <code id="output-dir"></code>
      <ul id="files"></ul>
      <p id="error" class="error"></p>
    </section>
    <div class="studio-footer"><span>野构 STUDIO · MEDIA ANALYSIS LAB</span><span>MiniCPM-V / Qwen3-ASR / ForcedAligner</span></div>
  </main>
  <script>
    const form = document.getElementById("upload-form");
    const fileInput = document.getElementById("video");
    const startButton = document.getElementById("start");
    const progressCard = document.getElementById("progress-card");
    const resultCard = document.getElementById("result-card");
    const filename = document.getElementById("filename");
    const status = document.getElementById("status");
    const progress = document.getElementById("progress");
    const progressPercent = document.getElementById("progress-percent");
    const phase = document.getElementById("phase");
    const planSteps = Array.from(document.querySelectorAll("[data-plan-step]"));
    const logs = document.getElementById("logs");
    const gpuUtilization = document.getElementById("gpu-utilization");
    const gpuUtilizationSub = document.getElementById("gpu-utilization-sub");
    const gpuMemory = document.getElementById("gpu-memory");
    const gpuMemorySub = document.getElementById("gpu-memory-sub");
    const gpuThermal = document.getElementById("gpu-thermal");
    const gpuPower = document.getElementById("gpu-power");
    const gpuName = document.getElementById("gpu-name");
    const gpuRefresh = document.getElementById("gpu-refresh");
    const outputDir = document.getElementById("output-dir");
    const files = document.getElementById("files");
    const errorBox = document.getElementById("error");
    const startSec = document.getElementById("start-sec");
    const endSec = document.getElementById("end-sec");
    const releaseGpu = document.getElementById("release-gpu");
    const dropZone = document.getElementById("drop-zone");
    const fileName = document.getElementById("file-name");
    const previewPanel = document.getElementById("preview-panel");
    const videoPreview = document.getElementById("video-preview");
    const videoLoading = document.getElementById("video-loading");
    const previewDuration = document.getElementById("preview-duration");
    const previewTime = document.getElementById("preview-time");
    const timelineStartRange = document.getElementById("timeline-start-range");
    const timelineEndRange = document.getElementById("timeline-end-range");
    const timelineFill = document.getElementById("timeline-fill");
    const timelinePlayhead = document.getElementById("timeline-playhead");
    const timelineStartLabel = document.getElementById("timeline-start-label");
    const timelineEndLabel = document.getElementById("timeline-end-label");
    const selectionDuration = document.getElementById("selection-duration");
    const timelineScale = document.getElementById("timeline-scale");
    const rangeSummary = document.getElementById("range-summary");
    const setStartButton = document.getElementById("set-start");
    const setEndButton = document.getElementById("set-end");
    let timer = null;
    let previewUrl = null;
    let videoDuration = 0;

    function formatSeconds(value, withTenths = true) {
      if (!Number.isFinite(value)) return withTenths ? "--:--.-" : "--:--";
      const safe = Math.max(0, value);
      const minutes = Math.floor(safe / 60);
      const seconds = safe - minutes * 60;
      const rendered = withTenths ? seconds.toFixed(1).padStart(4, "0") : String(Math.floor(seconds)).padStart(2, "0");
      return String(minutes).padStart(2, "0") + ":" + rendered;
    }

    function formatFileSize(bytes) {
      if (!Number.isFinite(bytes) || bytes < 1024) return String(bytes || 0) + " B";
      const units = ["KB", "MB", "GB", "TB"];
      let value = bytes;
      let unit = "B";
      for (let index = 0; value >= 1024 && index < units.length; index += 1) {
        value /= 1024;
        unit = units[index];
      }
      return value.toFixed(value >= 100 ? 0 : value >= 10 ? 1 : 2) + " " + unit;
    }

    function minimumSpan() {
      return Math.min(0.1, Math.max(0.01, videoDuration / 100));
    }

    function normalizeRange(start, end, changed) {
      if (!videoDuration) return { start: 0, end: 0 };
      const span = minimumSpan();
      start = Math.max(0, Math.min(videoDuration, Number.isFinite(start) ? start : 0));
      end = Math.max(0, Math.min(videoDuration, Number.isFinite(end) ? end : videoDuration));
      if (changed === "start") start = Math.min(start, Math.max(0, end - span));
      if (changed === "end") end = Math.max(end, Math.min(videoDuration, start + span));
      if (end <= start) {
        if (changed === "start") end = Math.min(videoDuration, start + span);
        else start = Math.max(0, end - span);
      }
      return { start: Math.max(0, start), end: Math.min(videoDuration, end) };
    }

    function syncFieldsFromTimeline(start, end) {
      startSec.value = start < 0.05 ? "0" : start.toFixed(1);
      endSec.value = Math.abs(end - videoDuration) < 0.05 ? "" : end.toFixed(1);
    }

    function updatePlayhead() {
      if (!videoDuration) return;
      const current = Number.isFinite(videoPreview.currentTime) ? videoPreview.currentTime : 0;
      const percent = Math.max(0, Math.min(100, current / videoDuration * 100));
      timelinePlayhead.style.left = percent + "%";
      previewTime.textContent = "当前 " + formatSeconds(current);
    }

    function updateTimeline(start, end) {
      if (!videoDuration) return;
      const startPercent = Math.max(0, Math.min(100, start / videoDuration * 100));
      const endPercent = Math.max(0, Math.min(100, end / videoDuration * 100));
      timelineStartRange.value = String(start);
      timelineEndRange.value = String(end);
      timelineFill.style.left = startPercent + "%";
      timelineFill.style.right = (100 - endPercent) + "%";
      timelineStartLabel.textContent = formatSeconds(start);
      timelineEndLabel.textContent = formatSeconds(end);
      selectionDuration.textContent = "选中 " + formatSeconds(end - start);
      rangeSummary.textContent = (start < 0.05 && Math.abs(end - videoDuration) < 0.05 ? "整部视频" : "分析区间") + " · " + formatSeconds(start) + "–" + formatSeconds(end);
      updatePlayhead();
    }

    function syncTimelineFromFields(changed, commit) {
      if (!videoDuration) return;
      const rawStart = Number.parseFloat(startSec.value);
      const rawEnd = Number.parseFloat(endSec.value);
      const range = normalizeRange(
        Number.isFinite(rawStart) ? rawStart : 0,
        Number.isFinite(rawEnd) ? rawEnd : videoDuration,
        changed,
      );
      updateTimeline(range.start, range.end);
      if (commit) syncFieldsFromTimeline(range.start, range.end);
    }

    function timelineChanged(changed) {
      const range = normalizeRange(Number(timelineStartRange.value), Number(timelineEndRange.value), changed);
      updateTimeline(range.start, range.end);
      syncFieldsFromTimeline(range.start, range.end);
    }

    function updateTimelineScale() {
      if (!videoDuration) return;
      timelineScale.innerHTML = [0, 0.25, 0.5, 0.75, 1]
        .map(function(ratio) { return "<span>" + formatSeconds(videoDuration * ratio, false) + "</span>"; })
        .join("");
    }

    function resetPreview() {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      previewUrl = null;
      videoDuration = 0;
      videoPreview.removeAttribute("src");
      videoPreview.load();
      previewPanel.hidden = true;
      videoLoading.hidden = false;
      previewDuration.textContent = "总时长 --:--";
      previewTime.textContent = "当前 --:--.-";
      rangeSummary.textContent = "尚未读取视频";
    }

    function loadPreview(file) {
      resetPreview();
      previewPanel.hidden = false;
      videoLoading.hidden = false;
      videoLoading.textContent = "正在读取视频时长…";
      previewUrl = URL.createObjectURL(file);
      videoPreview.src = previewUrl;
      videoPreview.load();
    }

    function showSelectedFile(file) {
      fileName.textContent = file ? "已选择： " + file.name + " · " + formatFileSize(file.size) : "尚未选择视频";
      startSec.value = "";
      endSec.value = "";
      if (file) loadPreview(file);
      else resetPreview();
    }

    fileInput.addEventListener("change", function() {
      showSelectedFile(fileInput.files[0]);
    });
    videoPreview.addEventListener("loadedmetadata", function() {
      videoDuration = Number.isFinite(videoPreview.duration) ? videoPreview.duration : 0;
      if (!videoDuration) {
        videoLoading.textContent = "无法读取视频时长，请直接填写秒数后重试";
        return;
      }
      timelineStartRange.max = String(videoDuration);
      timelineEndRange.max = String(videoDuration);
      timelineStartRange.step = "0.1";
      timelineEndRange.step = "0.1";
      previewDuration.textContent = "总时长 " + formatSeconds(videoDuration);
      videoLoading.hidden = true;
      updateTimelineScale();
      syncTimelineFromFields(null, true);
    });
    videoPreview.addEventListener("timeupdate", updatePlayhead);
    videoPreview.addEventListener("loadeddata", updatePlayhead);
    timelineStartRange.addEventListener("input", function() { timelineChanged("start"); });
    timelineEndRange.addEventListener("input", function() { timelineChanged("end"); });
    startSec.addEventListener("input", function() { syncTimelineFromFields("start", false); });
    endSec.addEventListener("input", function() { syncTimelineFromFields("end", false); });
    startSec.addEventListener("change", function() { syncTimelineFromFields("start", true); });
    endSec.addEventListener("change", function() { syncTimelineFromFields("end", true); });
    setStartButton.addEventListener("click", function() {
      if (!videoDuration) return;
      const range = normalizeRange(videoPreview.currentTime, Number(timelineEndRange.value), "start");
      updateTimeline(range.start, range.end);
      syncFieldsFromTimeline(range.start, range.end);
    });
    setEndButton.addEventListener("click", function() {
      if (!videoDuration) return;
      const range = normalizeRange(Number(timelineStartRange.value), videoPreview.currentTime, "end");
      updateTimeline(range.start, range.end);
      syncFieldsFromTimeline(range.start, range.end);
    });
    ["dragenter", "dragover"].forEach(function(name) {
      dropZone.addEventListener(name, function(event) {
        event.preventDefault();
        dropZone.classList.add("dragging");
      });
    });
    ["dragleave", "drop"].forEach(function(name) {
      dropZone.addEventListener(name, function(event) {
        event.preventDefault();
        dropZone.classList.remove("dragging");
      });
    });
    dropZone.addEventListener("drop", function(event) {
      if (event.dataTransfer.files.length) {
        fileInput.files = event.dataTransfer.files;
        showSelectedFile(fileInput.files[0]);
      }
    });

    function formatMegabytes(value) {
      if (!Number.isFinite(Number(value))) return "--";
      const mb = Number(value);
      if (mb >= 1024) return (mb / 1024).toFixed(1) + " GB";
      return Math.round(mb) + " MB";
    }

    function updateGpu(gpu) {
      if (!gpu || !gpu.available) {
        gpuUtilization.textContent = "--";
        gpuUtilizationSub.textContent = (gpu && gpu.message) || "等待 GPU 状态";
        gpuMemory.textContent = "--";
        gpuMemorySub.textContent = "已用 / 总量";
        gpuThermal.textContent = "--";
        gpuPower.textContent = "功耗 --";
        gpuName.textContent = "不可用";
        gpuRefresh.textContent = "未读取到 nvidia-smi";
        return;
      }
      const utilization = Number(gpu.utilization_pct);
      gpuUtilization.textContent = Number.isFinite(utilization) ? utilization.toFixed(0) + "%" : "--";
      gpuUtilizationSub.textContent = gpu.count > 1 ? "最高占用 · " + gpu.count + " 张" : "实时采样";
      const used = formatMegabytes(gpu.memory_used_mb);
      const total = formatMegabytes(gpu.memory_total_mb);
      gpuMemory.textContent = used + " / " + total;
      const memoryPct = Number(gpu.memory_total_mb) > 0 ? Number(gpu.memory_used_mb) / Number(gpu.memory_total_mb) * 100 : NaN;
      gpuMemorySub.textContent = Number.isFinite(memoryPct) ? memoryPct.toFixed(0) + "% 已用" : "已用 / 总量";
      const temperature = Number(gpu.temperature_c);
      gpuThermal.textContent = Number.isFinite(temperature) ? temperature.toFixed(0) + "°C" : "--";
      const power = Number(gpu.power_w);
      gpuPower.textContent = Number.isFinite(power) ? "功耗 " + power.toFixed(0) + " W" : "功耗 --";
      gpuName.textContent = gpu.name || "GPU";
      gpuRefresh.textContent = gpu.updated_at ? "更新 " + gpu.updated_at.slice(11) : "实时采样";
    }

    function updatePlan(job, percent) {
      const phaseText = String(job.phase || "");
      let active = 0;
      if (phaseText.includes("ASR") || phaseText.includes("对齐") || (percent >= 5 && percent < 35)) active = 1;
      if (phaseText.includes("视觉") || (percent >= 35 && percent < 90)) active = 2;
      if (phaseText.includes("生成") || phaseText.includes("整理") || percent >= 90) active = 3;
      if (job.status === "done") active = planSteps.length;
      planSteps.forEach(function(step, index) {
        step.classList.toggle("done", index < active || job.status === "done");
        step.classList.toggle("active", index === active && job.status !== "done" && job.status !== "failed");
      });
    }

    function showJob(job) {
      progressCard.hidden = false;
      resultCard.hidden = job.status !== "done" && job.status !== "failed";
      filename.textContent = job.filename || "视频分析任务";
      status.textContent = job.status_label || job.status;
      const percent = Math.max(0, Math.min(100, Number(job.progress) || 0));
      progress.value = percent;
      progressPercent.textContent = percent.toFixed(0) + "%";
      phase.textContent = percent.toFixed(0) + "% · " + (job.phase || "处理中");
      updatePlan(job, percent);
      updateGpu(job.gpu);
      logs.textContent = (job.logs || []).join("\\n");
      logs.scrollTop = logs.scrollHeight;
      if (job.output_dir) {
        outputDir.textContent = job.output_dir;
      }
      files.innerHTML = "";
      (job.files || []).forEach(function(item) {
        const li = document.createElement("li");
        const a = document.createElement("a");
        a.href = item.url;
        a.textContent = item.name;
        a.download = item.name;
        li.appendChild(a);
        files.appendChild(li);
      });
      errorBox.textContent = job.error || "";
      if (job.status === "done" || job.status === "failed") {
        startButton.disabled = false;
        if (timer) {
          clearTimeout(timer);
          timer = null;
        }
      }
    }

    async function poll(jobId) {
      try {
        const response = await fetch("/api/jobs/" + encodeURIComponent(jobId));
        const job = await response.json();
        showJob(job);
        if (job.status !== "done" && job.status !== "failed") {
          timer = setTimeout(function() { poll(jobId); }, 1500);
        }
      } catch (error) {
        errorBox.textContent = String(error);
        timer = setTimeout(function() { poll(jobId); }, 3000);
      }
    }

    form.addEventListener("submit", async function(event) {
      event.preventDefault();
      if (!fileInput.files.length) return;
      startButton.disabled = true;
      progressCard.hidden = false;
      resultCard.hidden = true;
      logs.textContent = "正在上传视频…";
      try {
        const body = new FormData();
        body.append("video", fileInput.files[0]);
        const query = new URLSearchParams();
        if (startSec.value.trim()) query.set("start_sec", startSec.value.trim());
        if (endSec.value.trim()) query.set("end_sec", endSec.value.trim());
        query.set("release_gpu", releaseGpu.checked ? "1" : "0");
        const target = "/api/jobs" + (query.toString() ? "?" + query.toString() : "");
        const response = await fetch(target, { method: "POST", body: body });
        const job = await response.json();
        if (!response.ok) throw new Error(job.error || "创建任务失败");
        showJob(job);
        poll(job.id);
      } catch (error) {
        startButton.disabled = false;
        resultCard.hidden = false;
        errorBox.textContent = String(error);
      }
    });
  </script>
</body>
</html>
"""


def safe_name(value: str, limit: int = 120) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(value or ""))
    value = re.sub(r"\s+", " ", value).strip(" .")
    return value[:limit].rstrip(" ._") or "未命名视频"


def to_windows_path(path: Path) -> str:
    match = re.match(r"^/mnt/([a-zA-Z])(?:/(.*))?$", str(path))
    if not match:
        return str(path)
    rest = (match.group(2) or "").replace("/", "\\")
    return f"{match.group(1).upper()}:\\{rest}" if rest else f"{match.group(1).upper()}:\\"


def save_job_state(job: dict) -> None:
    state = {key: value for key, value in job.items() if key not in {"thread"}}
    (JOB_STATE_DIR / f"{job['id']}.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def update_job(job_id: str, **values: object) -> dict:
    with JOBS_LOCK:
        job = JOBS[job_id]
        job.update(values)
        snapshot = dict(job)
        save_job_state(snapshot)
        return snapshot


def add_log(job_id: str, message: str, progress: int | None = None, phase: str | None = None) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    with JOBS_LOCK:
        job = JOBS[job_id]
        job.setdefault("logs", []).append(f"[{stamp}] {message}")
        job["logs"] = job["logs"][-500:]
        if progress is not None:
            job["progress"] = max(int(job.get("progress", 0)), progress)
        if phase:
            job["phase"] = phase
        snapshot = dict(job)
        save_job_state(snapshot)


def files_for_job(job_id: str) -> list[dict[str, str]]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return []
        output = Path(job["output_path"])
    names = [
        "00_任务参数.md",
        "01_纯视觉分析.md",
        "视觉分析文本汇总.md",
        "视觉片段索引.jsonl",
        "02_纯ASR与时间戳.md",
        "03_代码综合时间线.md",
        "04_最终导演分析.md",
        "分析清单.json",
        "audio_index.md",
        "audio_records.jsonl",
    ]
    result = []
    for name in names:
        if (output / name).is_file():
            result.append({"name": name, "url": f"/api/jobs/{quote(job_id)}/files/{quote(name)}"})
    timeline_dir = output / "02_视频" / "音轨与时间轴"
    for path in (
        sorted(timeline_dir.glob("*.json"))
        + sorted(timeline_dir.glob("*.srt"))
        + sorted(timeline_dir.glob("*.vtt"))
    ):
        result.append(
            {
                "name": str(Path("02_视频") / "音轨与时间轴" / path.name),
                "url": f"/api/jobs/{quote(job_id)}/files/{quote(str(path.relative_to(output)))}",
            }
        )
    return result


def update_progress_from_line(job_id: str, line: str) -> None:
    if "阶段 1/3 完成" in line:
        add_log(job_id, line, 32, "ASR 与 ForcedAligner 完成")
    elif "阶段 1/3" in line:
        add_log(job_id, line, 5, "ASR 与 ForcedAligner")
    elif "加载 ASR" in line:
        add_log(job_id, line, 12, "加载 ASR 模型")
    elif "加载 ForcedAligner" in line:
        add_log(job_id, line, 16, "加载时间对齐模型")
    elif "-> 完成:" in line:
        add_log(job_id, line, 30, "ASR 完成")
    elif "阶段 2/3 完成" in line:
        add_log(job_id, line, 88, "视觉分析完成")
    elif "阶段 2/3" in line:
        add_log(job_id, line, 35, "视觉分析")
    elif "视觉进度" in line:
        match = re.search(r"视觉进度\s*[=:：]\s*(\d+)%", line)
        progress = int(match.group(1)) if match else 35
        add_log(job_id, line, min(88, max(35, progress)), "视觉分析：连续画面")
    elif "20秒上下文完成" in line:
        with JOBS_LOCK:
            current = int(JOBS[job_id].get("progress", 35))
        add_log(job_id, line, min(86, current + 2), "视觉分析：20 秒批次")
    elif "高清逐秒/20秒上下文分析完成" in line:
        add_log(job_id, line, 88, "视觉分析完成")
    elif "释放 GPU 显存" in line:
        add_log(job_id, line, 100, "释放 GPU 显存")
    elif "GPU 显存已释放" in line:
        add_log(job_id, line, 100, "GPU 显存已释放")
    elif "阶段 3/3" in line:
        add_log(job_id, line, 90, "生成四份文档")
    elif "已生成：" in line:
        add_log(job_id, line, 96, "整理输出")
    else:
        add_log(job_id, line)


def trim_video(source: Path, target: Path, start_sec: float, end_sec: float | None, job_id: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    source_mount = f"/data/source/{source.name}"
    work_mount = "/data/work/analysis_input.mp4"
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        f"media-analysis-trim-{job_id}",
        "-v",
        f"{source.parent}:/data/source:ro",
        "-v",
        f"{target.parent}:/data/work",
        "--entrypoint",
        "ffmpeg",
        ASR_IMAGE,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        source_mount,
    ]
    if end_sec is not None:
        command.extend(["-t", f"{end_sec - start_sec:.3f}"])
    command.extend(
        [
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            work_mount,
        ]
    )
    run(command, timeout=None)
    if not target.exists() or target.stat().st_size == 0:
        raise RuntimeError("视频截取失败，结果文件为空")


def run_job(
    job_id: str,
    source: Path,
    original: str,
    output: Path,
    start_sec: float,
    end_sec: float | None,
    release_gpu_after: bool,
) -> None:
    if ANALYSIS_LOCK.locked():
        add_log(job_id, "已有任务占用 GPU，当前任务进入队列。", 0, "排队等待 GPU")
    with ANALYSIS_LOCK:
        add_log(job_id, "已获得 GPU 分析资源，按计划开始执行。", 1, "准备中")
        _run_job(job_id, source, original, output, start_sec, end_sec, release_gpu_after)


def _run_job(
    job_id: str,
    source: Path,
    original: str,
    output: Path,
    start_sec: float,
    end_sec: float | None,
    release_gpu_after: bool,
) -> None:
    add_log(job_id, "任务已创建，准备调用固定分析流程。", 1, "准备中")
    try:
        analysis_source = source
        if start_sec > 0 or end_sec is not None:
            work_dir = WEBUI_STATE / "jobs" / job_id
            analysis_source = work_dir / original
            add_log(
                job_id,
                f"正在截取原视频 {start_sec:.2f}s–{end_sec:.2f}s"
                if end_sec is not None
                else f"正在截取原视频 {start_sec:.2f}s–结尾",
                3,
                "截取分析区间",
            )
            trim_video(source, analysis_source, start_sec, end_sec, job_id)
            add_log(job_id, f"分析片段已生成：{analysis_source.name}", 5, "准备分析")
        else:
            work_dir = WEBUI_STATE / "jobs" / job_id
            work_dir.mkdir(parents=True, exist_ok=True)
            analysis_source = work_dir / original
            try:
                analysis_source.hardlink_to(source)
            except OSError:
                shutil.copy2(source, analysis_source)
            add_log(job_id, "未设置截取区间，将分析整部视频。", 3, "准备分析")

        end_label = f"{end_sec:.3f}" if end_sec is not None else "视频结尾"
        (output / "00_任务参数.md").write_text(
            "\n".join(
                [
                    "# 任务参数",
                    "",
                    f"- 原始上传文件：{original}",
                    f"- 原始上传路径：{source}",
                    f"- 分析起始秒：{start_sec:.3f}",
                    f"- 分析结束秒：{end_label}",
                    f"- 分析完成后释放显存：{'是' if release_gpu_after else '否'}",
                    "- 输出时间码口径：以本次分析片段为 00:00 起点；原始视频区间保留在本文件。",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        add_log(job_id, f"准备阶段异常：{type(exc).__name__}: {exc}")
        update_job(job_id, status="failed", status_label="失败", error=str(exc))
        return
    command = [
        "python3",
        str(PIPELINE),
        "--source",
        str(analysis_source),
        "--output",
        str(output),
    ]
    command.append("--release-gpu-after" if release_gpu_after else "--keep-gpu-after")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        update_job(job_id, pid=process.pid, status="running", status_label="分析中")
        assert process.stdout is not None
        for raw in process.stdout:
            line = raw.rstrip()
            if line:
                update_progress_from_line(job_id, line)
        return_code = process.wait()
        if return_code:
            add_log(job_id, f"分析进程退出，代码 {return_code}。")
            update_job(job_id, status="failed", status_label="失败", error=f"分析进程退出码：{return_code}")
            return
        add_log(job_id, "四份主文档已生成。", 100, "完成")
        update_job(job_id, status="done", status_label="完成", progress=100, files=files_for_job(job_id))
    except Exception as exc:
        add_log(job_id, f"任务异常：{type(exc).__name__}: {exc}")
        update_job(job_id, status="failed", status_label="失败", error=str(exc))


def create_upload(handler: BaseHTTPRequestHandler, job_id: str) -> tuple[str, Path]:
    content_type = handler.headers.get("Content-Type", "")
    match = re.search(r"boundary=([^;]+)", content_type)
    if "multipart/form-data" not in content_type or not match:
        raise ValueError("请求不是 multipart/form-data")
    try:
        content_length = int(handler.headers.get("Content-Length", "0"))
    except ValueError as exc:
        raise ValueError("缺少有效的 Content-Length") from exc
    if content_length <= 0 or content_length > MAX_UPLOAD_BYTES:
        raise ValueError("上传文件为空或超过 20GB 限制")

    boundary = match.group(1).strip().strip('"').encode("utf-8")
    first_boundary = b"--" + boundary
    consumed = 0
    first_line = handler.rfile.readline()
    consumed += len(first_line)
    if first_line.rstrip(b"\r\n") != first_boundary:
        raise ValueError("无法读取上传边界")

    headers: list[bytes] = []
    while True:
        line = handler.rfile.readline()
        consumed += len(line)
        if line in (b"\r\n", b"\n", b""):
            break
        headers.append(line.rstrip(b"\r\n"))
    header_text = b"\r\n".join(headers).decode("utf-8", errors="replace")
    filename_match = re.search(r'filename="([^"]*)"', header_text, flags=re.IGNORECASE)
    original = safe_name(filename_match.group(1) if filename_match else "上传视频.mp4")
    suffix = Path(original).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError("只支持常见视频格式：MP4、MOV、MKV、AVI、WebM 等")

    target = UPLOAD_DIR / f"{job_id}_{original}"
    marker = b"\r\n--" + boundary
    buffer = bytearray()
    remaining = content_length - consumed
    with target.open("wb") as stream:
        while remaining > 0:
            chunk = handler.rfile.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ValueError("上传过程中连接中断")
            consumed += len(chunk)
            remaining -= len(chunk)
            buffer.extend(chunk)
            position = buffer.find(marker)
            if position >= 0:
                stream.write(buffer[:position])
                while remaining > 0:
                    tail = handler.rfile.read(min(1024 * 1024, remaining))
                    if not tail:
                        break
                    remaining -= len(tail)
                break
            keep = len(marker) + 4
            if len(buffer) > keep:
                stream.write(buffer[:-keep])
                del buffer[:-keep]
        else:
            raise ValueError("上传数据中没有结束边界")
    if not target.exists() or target.stat().st_size == 0:
        raise ValueError("上传文件为空")
    return original, target


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def send_bytes(self, payload: bytes, content_type: str, status: int = 200, disposition: str | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        if disposition:
            self.send_header("Content-Disposition", disposition)
        self.end_headers()
        self.wfile.write(payload)

    def send_json(self, value: dict, status: int = 200) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_bytes(payload, "application/json; charset=utf-8", status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/assets/brand/yegou-studio-logo.png":
            logo = ASSET_ROOT / "brand" / "yegou-studio-logo.png"
            if not logo.is_file():
                self.send_json({"error": "品牌 logo 不存在"}, 404)
                return
            self.send_bytes(logo.read_bytes(), "image/png")
            return
        match = re.fullmatch(r"/api/jobs/([^/]+)", parsed.path)
        if match:
            job_id = unquote(match.group(1))
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if job:
                    result = dict(job)
                else:
                    state_path = JOB_STATE_DIR / f"{job_id}.json"
                    result = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
            if not result:
                self.send_json({"error": "任务不存在"}, 404)
                return
            result["files"] = files_for_job(job_id)
            result["gpu"] = gpu_status()
            result.pop("thread", None)
            self.send_json(result)
            return

        match = re.fullmatch(r"/api/jobs/([^/]+)/files/(.+)", parsed.path)
        if match:
            job_id = unquote(match.group(1))
            relative = Path(unquote(match.group(2)))
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                if not job:
                    state_path = JOB_STATE_DIR / f"{job_id}.json"
                    job = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else None
            if not job:
                self.send_json({"error": "任务不存在"}, 404)
                return
            output = Path(job["output_path"]).resolve()
            target = (output / relative).resolve()
            if output not in target.parents or not target.is_file():
                self.send_json({"error": "文件不存在"}, 404)
                return
            data = target.read_bytes()
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self.send_bytes(
                data,
                content_type,
                disposition=f"attachment; filename*=UTF-8''{quote(target.name)}",
            )
            return
        self.send_json({"error": "Not Found"}, 404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/jobs":
            self.send_json({"error": "Not Found"}, 404)
            return
        job_id = uuid.uuid4().hex[:12]
        target: Path | None = None
        try:
            query = parse_qs(parsed.query)
            raw_start = query.get("start_sec", [""])[0].strip()
            raw_end = query.get("end_sec", [""])[0].strip()
            raw_release = query.get("release_gpu", ["1"])[0].strip().lower()
            start_sec = float(raw_start) if raw_start else 0.0
            end_sec = float(raw_end) if raw_end else None
            release_gpu_after = raw_release not in {"0", "false", "no", "off"}
            if not math.isfinite(start_sec) or start_sec < 0:
                raise ValueError("开始秒必须是大于等于 0 的数字")
            if end_sec is not None and (not math.isfinite(end_sec) or end_sec <= start_sec):
                raise ValueError("结束秒必须大于开始秒")
            original, target = create_upload(self, job_id)
            stem = safe_name(Path(original).stem, 80)
            range_label = ""
            if start_sec > 0 or end_sec is not None:
                range_label = f"__片段_{start_sec:.2f}-{end_sec:.2f}" if end_sec is not None else f"__片段_{start_sec:.2f}-结尾"
            output = DEFAULT_OUTPUT_ROOT / f"{stem}__视频分析{range_label}__{datetime.now().strftime('%Y%m%d_%H%M%S')}_{job_id[:8]}"
            output.mkdir(parents=True, exist_ok=True)
            job = {
                "id": job_id,
                "filename": original,
                "source_path": str(target),
                "output_path": str(output),
                "output_dir": to_windows_path(output),
                "status": "queued",
                "status_label": "排队中",
                "phase": "等待启动",
                "progress": 0,
                "logs": [],
                "start_sec": start_sec,
                "end_sec": end_sec,
                "release_gpu_after": release_gpu_after,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "files": [],
            }
            with JOBS_LOCK:
                JOBS[job_id] = job
                save_job_state(job)
            thread = threading.Thread(
                target=run_job,
                args=(job_id, target, original, output, start_sec, end_sec, release_gpu_after),
                daemon=True,
            )
            with JOBS_LOCK:
                JOBS[job_id]["thread"] = thread
            thread.start()
            response = {key: value for key, value in JOBS[job_id].items() if key != "thread"}
            response["gpu"] = gpu_status()
            self.send_json(response, 202)
        except Exception as exc:
            if target and target.exists():
                target.unlink()
            self.send_json({"error": str(exc)}, 400)


def main() -> None:
    parser = argparse.ArgumentParser(description="视频导演拉片分析 WebUI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7867)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"视频分析 WebUI：http://localhost:{args.port}", flush=True)
    print(f"默认输出目录：{DEFAULT_OUTPUT_ROOT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

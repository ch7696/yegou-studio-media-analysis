#!/usr/bin/env python3
"""Small dependency-free WebUI for the fixed media-analysis pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import mimetypes
import os
import queue
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
SUBTITLE_PIPELINE = REPO_ROOT / "subtitle_extraction_pipeline.py"
ASSET_ROOT = REPO_ROOT / "assets"
WEBUI_STATE = Path(os.environ.get("WEBUI_STATE", str(REPO_ROOT / "state")))
UPLOAD_DIR = WEBUI_STATE / "uploads"
JOB_STATE_DIR = WEBUI_STATE / "jobs"
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("MEDIA_OUTPUT_ROOT", str(REPO_ROOT / "outputs")))
MINICPM_CONTAINER = os.environ.get("MINICPM_CONTAINER", "vision-minicpm")
MODEL_START_SCRIPT = REPO_ROOT / "scripts" / "run_model.sh"
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
BATCHES: dict[str, dict] = {}
JOBS_LOCK = threading.RLock()
JOB_QUEUE: queue.Queue[dict[str, object]] = queue.Queue()
QUEUE_THREAD: threading.Thread | None = None
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
MODEL_RESTART_LOCK = threading.RLock()
MODEL_RESTART_THREAD: threading.Thread | None = None
MODEL_RESTART_STATE: dict[str, object] = {
    "status": "idle",
    "message": "视觉模型待命",
    "logs": [],
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


def model_restart_snapshot() -> dict[str, object]:
    """Return the last vLLM restart state for the progress panel."""

    with MODEL_RESTART_LOCK:
        value = dict(MODEL_RESTART_STATE)
        value["logs"] = list(MODEL_RESTART_STATE.get("logs", []))
        return value


def _append_model_restart_log(message: str) -> None:
    stamp = datetime.now().strftime("%H:%M:%S")
    with MODEL_RESTART_LOCK:
        logs = list(MODEL_RESTART_STATE.get("logs", []))
        logs.append(f"[{stamp}] {message}")
        MODEL_RESTART_STATE["logs"] = logs[-80:]
        MODEL_RESTART_STATE["message"] = message


def _job_waiting_for_visual_model(job: dict[str, object]) -> bool:
    if str(job.get("status", "")) != "running":
        return False
    phase = str(job.get("phase", ""))
    if any(marker in phase for marker in ("视觉分析", "识别字幕", "读取字幕", "高清逐秒")) and "等待" not in phase:
        return False
    recent_logs = " ".join(str(item) for item in list(job.get("logs", []))[-20:])
    text = f"{phase} {recent_logs}"
    return any(
        marker in text
        for marker in (
            "等待 MiniCPM",
            "等待视觉模型",
            "等待视觉服务",
            "API 就绪",
            "启动视觉模型",
            "视觉模型服务",
        )
    )


def _model_restart_worker() -> None:
    """Restart/start the configured vLLM container without touching the job queue."""

    global MODEL_RESTART_THREAD
    try:
        _append_model_restart_log(f"准备重启视觉模型容器：{MINICPM_CONTAINER}")
        inspect = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", MINICPM_CONTAINER],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        if inspect.stdout.strip() == "true":
            stop = subprocess.run(
                ["docker", "stop", MINICPM_CONTAINER],
                capture_output=True,
                text=True,
                check=False,
                timeout=45,
            )
            if stop.returncode:
                raise RuntimeError((stop.stderr or stop.stdout or "停止视觉模型容器失败").strip())
            _append_model_restart_log("旧的视觉模型容器已停止，开始重新启动。")
        else:
            _append_model_restart_log("视觉模型容器当前未运行，直接执行启动流程。")

        process = subprocess.Popen(
            ["bash", str(MODEL_START_SCRIPT)],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for raw in process.stdout:
            line = raw.rstrip()
            if line:
                _append_model_restart_log(line)
        return_code = process.wait()
        with MODEL_RESTART_LOCK:
            if return_code:
                MODEL_RESTART_STATE["status"] = "failed"
                MODEL_RESTART_STATE["message"] = f"视觉模型启动失败，退出码 {return_code}"
            else:
                MODEL_RESTART_STATE["status"] = "ready"
                MODEL_RESTART_STATE["message"] = "vLLM API 已就绪"
    except Exception as exc:
        with MODEL_RESTART_LOCK:
            MODEL_RESTART_STATE["status"] = "failed"
            MODEL_RESTART_STATE["message"] = f"视觉模型启动失败：{type(exc).__name__}: {exc}"
        _append_model_restart_log(MODEL_RESTART_STATE["message"])
    finally:
        with MODEL_RESTART_LOCK:
            MODEL_RESTART_THREAD = None


def _model_release_worker() -> None:
    """Stop only the vLLM container so the WebUI process stays online."""

    global MODEL_RESTART_THREAD
    try:
        _append_model_restart_log(f"准备释放视觉模型显存：{MINICPM_CONTAINER}")
        inspect = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", MINICPM_CONTAINER],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        if inspect.stdout.strip() == "true":
            stop = subprocess.run(
                ["docker", "stop", MINICPM_CONTAINER],
                capture_output=True,
                text=True,
                check=False,
                timeout=45,
            )
            if stop.returncode:
                raise RuntimeError((stop.stderr or stop.stdout or "停止视觉模型容器失败").strip())
            _append_model_restart_log("视觉模型容器已停止，GPU 显存已释放。")
        else:
            _append_model_restart_log("视觉模型容器当前未运行，GPU 显存已经处于释放状态。")
        with MODEL_RESTART_LOCK:
            MODEL_RESTART_STATE["status"] = "released"
            MODEL_RESTART_STATE["message"] = "显存已释放，WebUI 保持在线"
    except Exception as exc:
        with MODEL_RESTART_LOCK:
            MODEL_RESTART_STATE["status"] = "failed"
            MODEL_RESTART_STATE["message"] = f"释放显存失败：{type(exc).__name__}: {exc}"
        _append_model_restart_log(str(MODEL_RESTART_STATE["message"]))
    finally:
        with MODEL_RESTART_LOCK:
            MODEL_RESTART_THREAD = None


def start_model_restart() -> dict[str, object]:
    """Start one asynchronous vLLM restart; existing analysis jobs remain queued/running."""

    global MODEL_RESTART_THREAD
    with JOBS_LOCK:
        active_jobs = [dict(job) for job in JOBS.values() if job.get("status") == "running"]
    if active_jobs and not any(_job_waiting_for_visual_model(job) for job in active_jobs):
        raise RuntimeError("当前视觉模型正在处理画面，请等它进入等待阶段后再重启。")

    with MODEL_RESTART_LOCK:
        if MODEL_RESTART_THREAD and MODEL_RESTART_THREAD.is_alive():
            return model_restart_snapshot()
        MODEL_RESTART_STATE.clear()
        MODEL_RESTART_STATE.update(
            {
                "status": "starting",
                "message": "正在启动 vLLM，当前任务队列保持不变。",
                "logs": [],
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        MODEL_RESTART_THREAD = threading.Thread(
            target=_model_restart_worker,
            name="vllm-restart",
            daemon=True,
        )
        MODEL_RESTART_THREAD.start()
        return model_restart_snapshot()


def release_model_memory() -> dict[str, object]:
    """Release vLLM memory without stopping the WebUI or cancelling queued jobs."""

    global MODEL_RESTART_THREAD
    with JOBS_LOCK:
        active_jobs = [dict(job) for job in JOBS.values() if job.get("status") == "running"]
    if active_jobs or ANALYSIS_LOCK.locked():
        raise RuntimeError("当前有分析任务正在执行，请等任务完成后再释放显存。")

    with MODEL_RESTART_LOCK:
        if MODEL_RESTART_THREAD and MODEL_RESTART_THREAD.is_alive():
            raise RuntimeError("视觉模型正在启动或释放中，请稍候。")
        MODEL_RESTART_STATE.clear()
        MODEL_RESTART_STATE.update(
            {
                "status": "releasing",
                "message": "正在释放显存，WebUI 保持在线。",
                "logs": [],
                "started_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        MODEL_RESTART_THREAD = threading.Thread(
            target=_model_release_worker,
            name="vllm-release",
            daemon=True,
        )
        MODEL_RESTART_THREAD.start()
        return model_restart_snapshot()


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
    .hero-actions { display: flex; align-items: center; gap: 9px; }
    .top-action { width: auto; margin: 0; padding: 9px 12px; border: 1px solid #3e5d9b; border-radius: 999px; background: #1a2e5b; color: #b9c9ff; font-size: 12px; font-weight: 700; box-shadow: none; }
    .top-action:hover { border-color: var(--cyan); background: #21396d; color: var(--cyan); box-shadow: none; transform: none; }
    .top-action:disabled { background: #263653; color: #8293b0; cursor: wait; }
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
    #workbench-view { display: flex; flex-direction: column; }
    #workbench-view[hidden] { display: none; }
    .workbench-nav { order: 0; }
    #progress-card { order: 1; display: flex; flex-direction: column; }
    .studio-grid { order: 2; }
    #result-card { order: 3; }
    .pipeline-card { order: 4; }
    .studio-footer { order: 5; }
    #progress-card .row { order: 1; }
    #progress-card .progress-head { order: 2; }
    #progress-card > progress { order: 3; }
    #progress-card > #phase { order: 4; }
    #progress-card > .model-control { order: 5; }
    #progress-card > .log-head { order: 6; }
    #progress-card > #logs { order: 7; }
    #progress-card > .batch-queue-card { order: 8; }
    #progress-card > .run-plan { order: 9; }
    #progress-card > .telemetry-grid { order: 10; }
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
    .mode-option { margin-top: 20px; }
    .mode-option label { margin-bottom: 8px; color: #9aabc7; font-size: 13px; }
    .mode-option select { width: 100%; padding: 11px 12px; border: 1px solid #2d4268; border-radius: 10px; outline: none; background: #0b1528; color: var(--ink); font-size: 14px; transition: border .2s, box-shadow .2s; }
    .mode-option select:focus { border-color: #7393f6; box-shadow: 0 0 0 4px #3568f233; }
    .mode-option .small { display: block; margin-top: 7px; }
    .home-view { margin-top: 20px; }
    .entry-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
    .entry-card { display: grid; grid-template-columns: auto 1fr auto; align-items: center; gap: 16px; min-height: 150px; margin: 0; padding: 22px; border: 1px solid #2d4268; border-radius: 17px; background: linear-gradient(135deg, #12213d, #0d182d); color: var(--ink); text-align: left; box-shadow: 0 14px 34px #00000026; }
    .entry-card:hover { border-color: #7393f6; background: linear-gradient(135deg, #172b51, #111e38); }
    .entry-icon { display: grid; place-items: center; width: 44px; height: 44px; border: 1px solid #6f96ff77; border-radius: 13px; background: #3656a955; color: var(--cyan); font: 800 12px ui-monospace, monospace; }
    .entry-card.subtitle .entry-icon { border-color: #62e3ff77; background: #1a648055; color: #8ff0ff; }
    .entry-card.batch .entry-icon { border-style: dashed; }
    .entry-card.director-batch .entry-icon { border-color: #b79cff88; background: #684e9b44; color: #d6c7ff; }
    .entry-card.subtitle-batch .entry-icon { border-color: #72d8c688; background: #237d7044; color: #a1f5dd; }
    .entry-copy strong, .entry-copy span { display: block; }
    .entry-copy strong { margin-bottom: 7px; font-size: 17px; }
    .entry-copy span { color: var(--muted); font-size: 12px; line-height: 1.6; }
    .entry-arrow { color: var(--cyan); font-size: 24px; }
    .home-note { margin: 14px 2px 0; color: var(--muted); font-size: 12px; line-height: 1.6; }
    .workbench-nav { display: flex; align-items: center; justify-content: space-between; gap: 14px; margin: 20px 0 14px; }
    .back-home { width: auto; margin: 0; padding: 8px 13px; border: 1px solid #33486e; background: #111f39; color: #9eafd0; font-size: 12px; box-shadow: none; }
    .back-home:hover { border-color: #6f96ff; background: #182c50; color: var(--cyan); box-shadow: none; }
    .entry-label { color: #8192b0; font-size: 12px; }
    .release-option { display: flex; align-items: flex-start; gap: 10px; margin-top: 16px; padding: 12px 13px; border: 1px solid #263b60; border-radius: 12px; background: #0e1a30; cursor: pointer; }
    .release-option input { width: 16px; height: 16px; flex: 0 0 auto; margin: 2px 0 0; accent-color: var(--blue); }
    .release-option strong, .release-option small { display: block; }
    .release-option strong { font-size: 12px; }
    .release-option small { margin-top: 4px; color: #8293b0; font-size: 11px; line-height: 1.5; }
    .batch-help { margin: 12px 0 0; padding: 10px 12px; border: 1px dashed #42608c; border-radius: 10px; color: #8fa4c4; font-size: 12px; line-height: 1.5; }
    .batch-source-actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
    .batch-source-actions .mini-button { flex: 0 0 auto; }
    .batch-builder { margin-top: 13px; padding: 13px; border: 1px solid #2b4269; border-radius: 13px; background: #0d192e; }
    .batch-builder-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }
    .batch-builder-head strong { color: var(--ink); font-size: 13px; }
    .batch-builder-head span { color: #7f94b5; font: 11px ui-monospace, monospace; }
    .batch-builder-copy { margin: 5px 0 10px; color: #7f94b5; font-size: 11px; line-height: 1.5; }
    .batch-builder-actions { display: flex; flex-wrap: wrap; gap: 7px; }
    .batch-builder-actions .mini-button { flex: 0 0 auto; }
    .batch-file-list { display: grid; gap: 6px; max-height: 330px; margin-top: 10px; overflow: auto; }
    .batch-file-item { display: grid; grid-template-columns: 18px 29px minmax(0, 1fr) auto; align-items: center; gap: 8px; padding: 8px 9px; border: 1px solid #294064; border-radius: 9px; background: #0f1c32; color: #aebdd7; font-size: 11px; cursor: grab; }
    .batch-file-item:active { cursor: grabbing; }
    .batch-file-item.dragging { opacity: .45; border-color: #6f96ff; }
    .batch-file-handle { color: #7185a7; font-size: 15px; line-height: 1; }
    .batch-file-order { color: #8298c1; font: 700 10px ui-monospace, monospace; }
    .batch-file-main { min-width: 0; }
    .batch-file-name, .batch-file-path { display: block; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .batch-file-name { color: #c7d4e9; }
    .batch-file-path { margin-top: 3px; color: #7185a7; font-size: 10px; }
    .batch-file-actions { display: flex; align-items: center; gap: 4px; }
    .batch-file-actions button { width: auto; min-width: 25px; margin: 0; padding: 4px 6px; border: 1px solid #33486e; border-radius: 7px; background: #111f39; color: #9eafd0; font-size: 12px; line-height: 1; box-shadow: none; }
    .batch-file-actions button:hover { border-color: #6f96ff; background: #182c50; color: var(--cyan); box-shadow: none; transform: none; }
    .batch-file-actions button:disabled { background: transparent; color: #4a5b79; cursor: default; }
    .batch-empty { padding: 14px 8px 5px; color: #7185a7; font-size: 11px; text-align: center; }
    .batch-queue-card { margin-top: 15px; padding-top: 14px; border-top: 1px solid #263b60; }
    .batch-queue-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 9px; color: #8094b3; font-size: 12px; }
    .batch-queue-head strong { color: var(--ink); font-size: 13px; }
    .batch-queue-copy { margin: -3px 0 10px; color: #7185a7; font-size: 11px; line-height: 1.5; }
    .batch-queue-items { display: grid; gap: 6px; max-height: 360px; overflow: auto; }
    .batch-job-item { display: grid; grid-template-columns: 28px minmax(0, 1fr) auto auto; align-items: center; gap: 9px; padding: 9px 10px; border: 1px solid #263b60; border-radius: 9px; background: #0f1b31; }
    .batch-job-index { color: #8298c1; font: 700 10px ui-monospace, monospace; }
    .batch-job-name { min-width: 0; overflow: hidden; color: #c7d4e9; font-size: 12px; text-overflow: ellipsis; white-space: nowrap; }
    .batch-job-status { color: #8fa3c1; font-size: 11px; white-space: nowrap; }
    .batch-job-controls { display: flex; gap: 4px; }
    .batch-job-controls button { width: auto; min-width: 24px; margin: 0; padding: 4px 5px; border: 1px solid #33486e; border-radius: 7px; background: #111f39; color: #9eafd0; font-size: 11px; line-height: 1; box-shadow: none; }
    .batch-job-controls button:hover { border-color: #6f96ff; background: #182c50; color: var(--cyan); box-shadow: none; transform: none; }
    .batch-job-controls button:disabled { background: transparent; color: #4a5b79; cursor: default; }
    .batch-job-progress { grid-column: 2 / -1; display: block; height: 4px; overflow: hidden; border-radius: 999px; background: #1c2a45; }
    .batch-job-progress span { display: block; height: 100%; border-radius: inherit; background: linear-gradient(90deg, #4776f7, #62e3ff); transition: width .25s ease; }
    .batch-job-links { grid-column: 2 / -1; display: flex; flex-wrap: wrap; gap: 8px; margin-top: -3px; }
    .batch-job-links a { color: #8fb1ff; font-size: 10px; }
    .batch-job-item.done { border-color: #73b99c; background: #effaf5; }
    .batch-job-item.done .batch-job-name, .batch-job-item.done .batch-job-status { color: #347b62; }
    .batch-job-item.failed { border-color: #e0a2a2; background: #fff5f5; }
    .batch-job-item.failed .batch-job-name, .batch-job-item.failed .batch-job-status { color: #ae5555; }
    .batch-job-item.failed .batch-job-progress span { background: #d98787; }
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
    .model-control { display: flex; align-items: center; justify-content: space-between; gap: 14px; margin-top: 14px; padding: 11px 13px; border: 1px solid #2d4266; border-radius: 12px; background: linear-gradient(145deg, #12213d, #0d182c); }
    .model-control-copy { min-width: 0; }
    .model-control-copy strong, .model-control-copy span { display: block; }
    .model-control-copy strong { font-size: 12px; }
    .model-control-copy span { margin-top: 4px; overflow: hidden; color: #8293b0; font-size: 10px; line-height: 1.4; text-overflow: ellipsis; white-space: nowrap; }
    .model-restart { flex: 0 0 auto; }
    .row { display: flex; justify-content: space-between; gap: 18px; align-items: center; }
    .status { padding: 6px 10px; border: 1px solid #3e5d9b; border-radius: 999px; background: #1a2e5b; color: #a8c0ff; font-size: 12px; font-weight: 700; }
    .error { color: #ff9b9b; white-space: pre-wrap; }
    .log-head { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-top: 15px; }
    .log-head strong { font-size: 15px; }
    .log-live { display: inline-flex; align-items: center; gap: 6px; color: #62e3ff; font: 10px ui-monospace, SFMono-Regular, Consolas, monospace; letter-spacing: .08em; }
    .log-live::before { width: 6px; height: 6px; border-radius: 50%; background: #47c58a; box-shadow: 0 0 9px #47c58a; content: ""; }
    .log-count { padding: 5px 8px; border: 1px solid #304466; border-radius: 999px; color: #91a1bd; font: 10px ui-monospace, SFMono-Regular, Consolas, monospace; }
    pre { margin: 8px 0 0; padding: 15px; min-height: 230px; max-height: 520px; overflow: auto; background: #111a2d; color: #dbeafe; border-radius: 12px; font: 13px/1.6 ui-monospace, SFMono-Regular, Consolas, monospace; white-space: pre-wrap; }
    code { display: block; padding: 10px 12px; background: #0d182c; border: 1px solid #243758; border-radius: 8px; overflow-wrap: anywhere; color: #b9c8e2; }
    ul { padding-left: 20px; line-height: 1.9; }
    #files { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 8px; padding: 0; margin: 12px 0 0; list-style: none; }
    #files li { min-width: 0; padding: 10px 12px; border: 1px solid #263b60; border-radius: 10px; background: #0e1a30; }
    #files a { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    #files a::before { content: "↓  "; color: #47c58a; }
    .result-head { display: flex; align-items: flex-end; justify-content: space-between; gap: 14px; }
    .result-head h2 { margin: 0; }
    .output-count { color: #47c58a; font: 800 20px/1 ui-monospace, SFMono-Regular, Consolas, monospace; }
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
    @media (max-width: 720px) { main { width: min(100% - 24px, 1240px); padding-top: 26px; } .hero-row { align-items: flex-start; flex-direction: column; } .brand-lockup { align-items: flex-start; gap: 14px; } .brand-logo { width: 96px; height: 96px; border-radius: 19px; } .hero-stage { align-self: center; margin-top: -8px; } .entry-grid { grid-template-columns: 1fr; } .flow { grid-template-columns: 1fr 1fr; } .preview-top, .timeline-actions { align-items: flex-start; flex-direction: column; } .preview-meta { justify-content: flex-start; } .quick-actions { flex-wrap: wrap; } .studio-sidebar { grid-template-columns: 1fr; } .run-plan { grid-template-columns: repeat(2, minmax(0, 1fr)); } .telemetry-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
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
    .top-action { border-color: #c5dce8; background: #ffffffd9; color: #4d7c96; box-shadow: 0 4px 15px #5d96b312; }
    .top-action:hover { border-color: #83bdd8; background: #eaf7fc; color: #2b7eae; box-shadow: none; }
    .top-action:disabled { background: #eef5f8; color: #91a9b7; }
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
    .mode-option label { color: #58778e; }
    .mode-option select { border-color: #cfe2ec; background: #fbfdff; color: #234761; }
    .mode-option select:focus { border-color: #70afd0; box-shadow: 0 0 0 4px #3988c21c; }
    .entry-card { border-color: #cfe2ec; background: linear-gradient(135deg, #f8fcff, #edf7fc); color: #17354f; box-shadow: 0 12px 28px #4d83a514; }
    .entry-card:hover { border-color: #70afd0; background: linear-gradient(135deg, #f0faff, #e6f5fb); }
    .entry-icon { border-color: #acd3e4; background: #e6f5fa; color: #3988b5; }
    .entry-card.subtitle .entry-icon { border-color: #a8dce5; background: #e8f8f9; color: #3e98a7; }
    .entry-card.director-batch .entry-icon { border-color: #c8c9eb; background: #f0f0fc; color: #7478b4; }
    .entry-card.subtitle-batch .entry-icon { border-color: #a8d8cc; background: #eefaf6; color: #428d78; }
    .entry-copy span, .home-note { color: #718aa0; }
    .entry-arrow { color: #3988c2; }
    .back-home { border-color: #cde1eb; background: #f5fbfe; color: #54758b; }
    .back-home:hover { border-color: #83bdd8; background: #eaf7fc; color: #2b7eae; }
    .entry-label { color: #718aa0; }
    .release-option { border-color: #d6e7ef; background: #f8fcfe; }
    .release-option small { color: #7891a3; }
    .batch-help { border-color: #b9d5e2; color: #6f899d; background: #f7fcfe; }
    .batch-source-actions .mini-button, .batch-builder-actions .mini-button { border-color: #cde1eb; background: #f5fbfe; color: #54758b; }
    .batch-source-actions .mini-button:hover, .batch-builder-actions .mini-button:hover { border-color: #83bdd8; background: #eaf7fc; color: #2b7eae; }
    .batch-builder { border-color: #d6e7ef; background: #f7fcfe; }
    .batch-builder-head strong { color: #244b66; }
    .batch-builder-head span, .batch-builder-copy { color: #7891a3; }
    .batch-file-item { border-color: #d6e7ef; background: #f7fcfe; color: #4e7187; }
    .batch-file-item.dragging { border-color: #70afd0; }
    .batch-file-handle, .batch-file-order, .batch-file-path, .batch-empty { color: #88a0af; }
    .batch-file-name { color: #365e75; }
    .batch-file-actions button, .batch-job-controls button { border-color: #cde1eb; background: #f5fbfe; color: #54758b; }
    .batch-file-actions button:hover, .batch-job-controls button:hover { border-color: #83bdd8; background: #eaf7fc; color: #2b7eae; }
    .batch-file-actions button:disabled, .batch-job-controls button:disabled { background: transparent; color: #b1c3cd; }
    .batch-queue-card { border-top-color: #d6e7ef; }
    .batch-queue-head { color: #7891a3; }
    .batch-queue-head strong { color: #244b66; }
    .batch-queue-copy { color: #7891a3; }
    .batch-job-item { border-color: #d6e7ef; background: #f7fcfe; }
    .batch-job-index { color: #7797ad; }
    .batch-job-name { color: #365e75; }
    .batch-job-status { color: #7891a3; }
    .batch-job-progress { background: #deedf4; }
    .batch-job-progress span { background: linear-gradient(90deg, #58abd0, #829fdd); }
    .batch-job-links a { color: #3b86b6; }
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
    .model-control { border-color: #d6e7ef; background: linear-gradient(145deg, #f8fcfe, #eef8fc); }
    .model-control-copy span { color: #7891a3; }
    .log-head strong { color: #244b66; }
    .log-live { color: #3988b5; }
    .log-count { border-color: #d6e7ef; color: #7891a3; background: #f8fcfe; }
    .status { border-color: #b8d8e8; background: #edf8fc; color: #327baa; }
    .error { color: #c45555; }
    pre { background: #eef7fb; border: 1px solid #d6e8f0; color: #355a72; }
    code { background: #f3f9fc; border-color: #d6e8f0; color: #4c6f84; }
    #files li { border-color: #d6e7ef; background: #f8fcfe; }
    .output-count { color: #48a984; }
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

    /* Reference console layout: navigation, task workspace, status and output. */
    body { background: #f7fbff; color: #17354f; }
    body::before { display: none; }
    main.app-shell { width: 100%; max-width: none; min-height: 100vh; margin: 0; padding: 0; display: grid; grid-template-columns: 214px minmax(0, 1fr); grid-template-rows: 86px 1fr; overflow: hidden; background: #f9fcff; }
    .hero { grid-column: 1 / -1; grid-row: 1; display: flex; align-items: center; min-width: 0; height: 86px; margin: 0; padding: 0 34px; border-bottom: 1px solid #e6eef6; background: #fff; }
    .eyebrow { display: none; }
    .hero-row { width: 100%; height: 100%; margin: 0; gap: 24px; }
    .brand-lockup { gap: 13px; }
    .brand-logo { width: 52px; height: 52px; border: 0; border-radius: 12px; background: transparent; box-shadow: none; mix-blend-mode: multiply; }
    .brand-name { margin-bottom: 4px; color: #132c49; font: 800 17px/1 ui-sans-serif, system-ui, sans-serif; letter-spacing: -.04em; text-transform: none; }
    h1 { margin: 0; color: #173b61; font-size: 25px; letter-spacing: -.05em; }
    .title-mark { display: none; }
    .hint { display: none; }
    .hero-tools { margin-left: auto; }
    .hero-actions { gap: 9px; }
    .local-pill { padding: 8px 12px; font-size: 11px; }
    .top-action { padding: 8px 12px; font-size: 11px; }

    .app-sidebar { grid-column: 1; grid-row: 2; display: flex; min-height: calc(100vh - 86px); flex-direction: column; padding: 28px 12px 20px 0; border-right: 1px solid #e6eef6; background: #fff; }
    .side-nav { display: grid; gap: 5px; }
    .side-nav-item { width: 100%; min-height: 48px; margin: 0; padding: 12px 18px 12px 24px; display: flex; align-items: center; gap: 15px; border: 0; border-left: 3px solid transparent; border-radius: 0 10px 10px 0; background: transparent; color: #5a6f88; font-size: 14px; font-weight: 550; text-align: left; box-shadow: none; }
    .side-nav-item:hover { border-left-color: #9fc9f4; background: #f2f8fe; color: #287fda; box-shadow: none; transform: none; }
    .side-nav-item.active { border-left-color: #2b88f5; background: #eaf4ff; color: #1879e8; font-weight: 750; box-shadow: none; }
    .nav-icon { width: 23px; height: 23px; flex: 0 0 auto; display: grid; place-items: center; border: 1px solid #d7e4f0; border-radius: 6px; background: #fbfdff; color: #60758f; font-size: 12px; line-height: 1; }
    .side-nav-item.active .nav-icon { border-color: #b9d9fb; background: #d9ecff; color: #1681f2; }
    .sidebar-foot { margin-top: auto; padding: 18px 24px; color: #90a4b8; font: 700 10px/1.6 ui-monospace, SFMono-Regular, Consolas, monospace; letter-spacing: .12em; }
    .sidebar-foot span { display: block; color: #b1c0ce; font-size: 8px; letter-spacing: .18em; }
    .app-content { grid-column: 2; grid-row: 2; min-width: 0; padding: 30px 36px 54px; background: #f9fcff; }
    #home-view { max-width: 1180px; margin: 0 auto; }
    #workbench-view { min-width: 0; }
    #workbench-view[hidden] { display: none; }
    .workbench-nav { display: none; }
    #progress-card { order: 3; width: 100%; max-width: 1360px; margin: 18px auto 0; padding: 0; border: 0; background: transparent; box-shadow: none; }
    .studio-grid { order: 1; width: 100%; max-width: 1360px; margin: 0 auto; grid-template-columns: minmax(0, 1fr) 340px; gap: 28px; }
    .input-card { margin: 0; padding: 0; border: 0; background: transparent; box-shadow: none; }
    .input-card .card-head { margin-bottom: 18px; }
    .input-card .card-head h2 { font-size: 23px; }
    .input-card .tag { display: none; }
    .mode-picker { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 15px; margin-bottom: 22px; }
    .mode-card { width: 100%; min-height: 86px; margin: 0; padding: 15px 18px; display: flex; align-items: center; gap: 13px; border: 1px solid #dce8f3; border-radius: 10px; background: #fff; color: #17354f; text-align: left; box-shadow: 0 3px 13px #4d83a50b; }
    .mode-card:hover { border-color: #8ec3f3; background: #fbfdff; box-shadow: 0 5px 18px #4d83a514; transform: none; }
    .mode-card.selected { border-color: #2b88f5; box-shadow: 0 0 0 1px #2b88f51c, 0 5px 18px #4d83a514; }
    .mode-card-icon { width: 40px; height: 40px; flex: 0 0 auto; display: grid; place-items: center; border: 1px solid #cbe2fb; border-radius: 10px; background: #edf7ff; color: #2085ee; font-size: 21px; }
    .mode-card-icon.subtitle { border-color: #ccecef; background: #effafb; color: #3d9eae; }
    .mode-card > span:nth-child(2) { min-width: 0; }
    .mode-card strong, .mode-card small { display: block; }
    .mode-card strong { color: #173b61; font-size: 15px; }
    .mode-card small { margin-top: 5px; color: #7790a5; font-size: 11px; }
    .mode-radio { width: 18px; height: 18px; flex: 0 0 auto; margin-left: auto; border: 1px solid #c6d4e2; border-radius: 50%; background: #fff; }
    .mode-card.selected .mode-radio { border-color: #2186f5; background: #2186f5; box-shadow: inset 0 0 0 4px #fff; }
    .upload-zone { min-height: 102px; padding: 18px 20px; border: 1px dashed #b9d1e8; border-radius: 10px; background: #fcfeff; box-shadow: none; }
    .upload-zone:hover, .upload-zone.dragging { border-color: #3d96ef; background: #f5fbff; box-shadow: 0 7px 20px #4d93b51a; transform: none; }
    .upload-icon { width: 42px; height: 42px; border-color: #b8d9f8; border-radius: 10px; background: #eaf5ff; color: #2e8bf0; }
    .upload-copy strong { color: #244b66; font-size: 14px; }
    .upload-copy span { color: #86a0b4; font-size: 11px; }
    .upload-action { margin-left: auto; padding: 9px 15px; border: 1px solid #d8e5f1; border-radius: 8px; background: #fff; color: #496b86; font-size: 12px; font-weight: 650; white-space: nowrap; }
    .upload-zone:hover .upload-action { border-color: #9ccbf5; color: #227fd9; }
    .file-name { min-height: 52px; margin-top: 10px; padding: 13px 15px 13px 46px; position: relative; display: flex; align-items: center; border: 1px solid #dce8f2; border-radius: 9px; background: #fff; color: #365e75; font-size: 12px; box-shadow: 0 3px 12px #4d83a509; }
    .file-name::before { position: absolute; left: 15px; color: #3988c2; content: "▣"; font-size: 18px; }
    .batch-source-actions { gap: 7px; }
    .batch-source-actions[hidden], .batch-builder[hidden], .batch-help[hidden], #range-controls[hidden], #preview-panel[hidden], #progress-card[hidden], #result-card[hidden], #home-view[hidden] { display: none !important; }
    .batch-builder { box-shadow: none; }
    #batch-builder .batch-builder-copy { display: none; }
    #mode-help, #range-note, #pipeline-note, .input-card form > p.small { display: none; }
    .range-grid { margin-top: 14px; }
    .release-option { margin-top: 14px; padding: 10px 12px; }
    .release-option small { font-size: 10px; }
    #start { margin-top: 15px; min-height: 48px; border-radius: 9px; font-size: 14px; }
    .studio-sidebar { position: static; display: grid; gap: 24px; }
    .studio-sidebar .card { margin: 0; padding: 0; border: 0; background: transparent; box-shadow: none; }
    .studio-sidebar .side-head { margin-bottom: 13px; }
    .studio-sidebar .side-head h2 { color: #173b61; font-size: 20px; }
    .studio-sidebar .live-badge { font-size: 9px; }
    .model-stack { gap: 9px; }
    .model-row { min-height: 70px; padding: 12px; border-radius: 9px; background: #fff; box-shadow: 0 3px 12px #4d83a50b; }
    .model-info strong { font-size: 12px; }
    .model-info em { font-size: 10px; }
    .gpu-note { display: none; }
    .delivery-card { padding-top: 21px !important; border-top: 1px solid #e4edf5 !important; }
    .delivery-stat { padding: 11px; }
    .delivery-list { gap: 6px; margin-top: 10px; }
    .delivery-item { font-size: 11px; }
    .path-label, .side-path { display: none; }
    .pipeline-card { order: 2; width: 100%; max-width: 1360px; margin: 24px auto 0; padding: 0; border: 0; background: transparent; box-shadow: none; }
    .pipeline-card .card-head { margin-bottom: 12px; }
    .pipeline-card .section-label { display: none; }
    .pipeline-card .card-head h2 { color: #173b61; font-size: 20px; }
    .pipeline-card .pipeline-note { display: none; }
    .flow { grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-top: 0; }
    .flow-item { min-height: 62px; padding: 12px 13px; display: flex; align-items: center; gap: 8px; position: relative; border-color: #dce8f2; border-radius: 9px; background: #fff; box-shadow: 0 3px 12px #4d83a509; }
    .flow-item:not(:last-child)::after { position: absolute; right: -19px; z-index: 2; color: #2f89ed; content: "→"; font-size: 18px; font-weight: 700; }
    .flow-num { width: 28px; height: 28px; flex: 0 0 auto; display: grid; place-items: center; margin: 0; border-radius: 50%; background: #eaf4ff; color: #328ce4; font: 700 10px ui-monospace, monospace; }
    .flow-item strong { margin: 0; color: #365e75; font-size: 12px; white-space: nowrap; }
    .flow-item span:last-child { display: none; }
    #progress-card .row { margin-bottom: 5px; }
    #progress-card .row > div:first-child { min-width: 0; }
    #progress-card .row .section-label { margin-bottom: 4px; }
    #progress-card .row strong { color: #365e75; font-size: 12px; }
    #progress-card .status { padding: 5px 9px; font-size: 10px; }
    #progress-card .progress-head { margin-top: 8px; }
    #progress-card .progress-head strong { font-size: 14px; }
    #progress-card .progress-percent { font-size: 24px; }
    #progress-card > .phase { font-size: 11px; }
    #progress-card > .run-plan { display: none; }
    #progress-card > .telemetry-grid { order: 5; grid-template-columns: repeat(5, minmax(0, 1fr)); margin-top: 14px; }
    .telemetry-item { min-height: 76px; padding: 11px; border-radius: 9px; background: #fff; }
    .telemetry-label { margin-bottom: 6px; font-size: 9px; }
    .telemetry-value { font-size: 15px; }
    .telemetry-sub { margin-top: 5px; font-size: 9px; }
    #progress-card > .log-head { order: 6; margin-top: 18px; }
    #progress-card > #logs { order: 7; min-height: 92px; max-height: 280px; margin-top: 7px; padding: 12px; border-color: #dce8f2; border-radius: 9px; background: #fff; color: #527087; font-size: 11px; }
    #progress-card > .model-control { order: 8; }
    #progress-card > .batch-queue-card { order: 9; }
    #result-card { order: 4; max-width: 1360px; margin: 22px auto 0; }
    .studio-footer { margin-top: 26px; color: #9aabba; }
    .home-view .entry-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .home-note { display: none; }
    @media (max-width: 1080px) {
      main.app-shell { grid-template-columns: 76px minmax(0, 1fr); }
      .app-sidebar { padding-right: 0; }
      .side-nav-item { justify-content: center; padding: 12px; border-radius: 0; }
      .side-nav-item > span:last-child, .sidebar-foot { display: none; }
      .studio-grid { grid-template-columns: minmax(0, 1fr) 290px; gap: 20px; }
      #progress-card > .telemetry-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    }
    @media (max-width: 760px) {
      main.app-shell { display: block; overflow: visible; }
      .hero { height: auto; min-height: 76px; padding: 12px 18px; }
      .hero-row { align-items: center; }
      .brand-logo { width: 42px; height: 42px; }
      .brand-name { font-size: 14px; }
      h1 { font-size: 20px; }
      .hero-tools { width: auto; }
      .hero-actions { flex-wrap: wrap; justify-content: flex-end; }
      .local-pill { display: none; }
      .app-sidebar { min-height: 0; padding: 0; border-right: 0; border-bottom: 1px solid #e6eef6; }
      .side-nav { display: flex; gap: 0; overflow-x: auto; }
      .side-nav-item { min-width: 82px; min-height: 48px; padding: 8px 12px; flex-direction: column; gap: 3px; border-left: 0; border-bottom: 3px solid transparent; border-radius: 0; font-size: 10px; }
      .side-nav-item.active { border-left: 0; border-bottom-color: #2b88f5; }
      .side-nav-item > span:last-child { display: block; }
      .app-content { padding: 22px 16px 40px; }
      .studio-grid { display: block; }
      .studio-sidebar { margin-top: 26px; }
      .mode-picker, .flow { grid-template-columns: 1fr 1fr; }
      .flow-item:not(:last-child)::after { display: none; }
      #progress-card > .telemetry-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 480px) {
      .hero-row { gap: 10px; }
      .brand-copy { min-width: 0; }
      h1 { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .top-action { padding: 7px 9px; }
      .mode-picker { gap: 8px; }
      .mode-card { min-height: 76px; padding: 11px; gap: 8px; }
      .mode-card-icon { width: 32px; height: 32px; font-size: 16px; }
      .mode-card strong { font-size: 12px; }
      .mode-card small { font-size: 9px; }
      .mode-radio { width: 14px; height: 14px; }
      .upload-action { padding: 8px 9px; font-size: 10px; }
      .flow { gap: 8px; }
      .flow-item { padding: 10px 8px; }
      .flow-item strong { font-size: 10px; }
    }
  </style>
</head>
<body>
  <div class="ripple-field" aria-hidden="true">
    <i class="ripple-ring ripple-one"></i>
    <i class="ripple-ring ripple-two"></i>
    <i class="ripple-ring ripple-three"></i>
    <i class="ripple-ring ripple-four"></i>
  </div>
  <main class="app-shell">
    <header class="hero">
      <div class="eyebrow">野构 STUDIO · LOCAL MEDIA WORKBENCH</div>
      <div class="hero-row">
        <div class="brand-lockup">
          <img class="brand-logo" src="/assets/brand/yegou-studio-logo.png" alt="野构 Studio 创意标志">
          <div class="brand-copy">
            <div class="brand-name">野构 Studio</div>
            <h1 id="hero-title">视频媒体分析工作台 <span class="title-mark">MEDIA LAB</span></h1>
            <p id="hero-hint" class="hint">选择一个工作入口，把视频整理成可回看、可下载、可继续加工的资料。</p>
          </div>
        </div>
        <div class="hero-tools">
          <div class="hero-actions">
            <button id="release-vram" class="top-action" type="button" title="只停止视觉模型容器，WebUI 保持在线">释放显存</button>
            <div class="local-pill"><span></span>本地运行</div>
          </div>
        </div>
      </div>
    </header>

    <aside class="app-sidebar">
      <nav class="side-nav" aria-label="工作台导航">
        <button class="side-nav-item active" type="button" data-sidebar-action="new">
          <span class="nav-icon">▶</span><span>新建任务</span>
        </button>
        <button class="side-nav-item" type="button" data-sidebar-action="batch">
          <span class="nav-icon">▦</span><span>批处理</span>
        </button>
        <button class="side-nav-item" type="button" data-sidebar-action="output">
          <span class="nav-icon">□</span><span>输出</span>
        </button>
        <button class="side-nav-item" type="button" data-sidebar-action="models">
          <span class="nav-icon">◇</span><span>模型</span>
        </button>
        <button class="side-nav-item" type="button" data-sidebar-action="settings">
          <span class="nav-icon">⚙</span><span>设置</span>
        </button>
      </nav>
      <div class="sidebar-foot">野构 STUDIO<span>MEDIA LAB</span></div>
    </aside>

    <div class="app-content">

    <section id="home-view" class="home-view">
      <div class="entry-grid">
        <button class="entry-card director" type="button" data-entry-mode="director">
          <span class="entry-icon">01</span>
          <span class="entry-copy"><strong>导演拉片分析</strong><span>完整分析视觉、声音、ASR 和时间线，输出导演拉片资料。</span></span>
          <span class="entry-arrow">→</span>
        </button>
        <button class="entry-card subtitle" type="button" data-entry-mode="subtitle">
          <span class="entry-icon">02</span>
          <span class="entry-copy"><strong>视觉字幕提取</strong><span>让视觉模型读取画面字幕，输出时间轴给其他项目重配音。</span></span>
          <span class="entry-arrow">→</span>
        </button>
        <button class="entry-card batch director-batch" type="button" data-entry-mode="director-batch">
          <span class="entry-icon">03</span>
          <span class="entry-copy"><strong>批量导演拉片</strong><span>一次选择多个视频，按顺序排队完成整片导演拉片。</span></span>
          <span class="entry-arrow">→</span>
        </button>
        <button class="entry-card batch subtitle-batch" type="button" data-entry-mode="subtitle-batch">
          <span class="entry-icon">04</span>
          <span class="entry-copy"><strong>批量字幕提取</strong><span>批量读取画面字幕；单个视频报错会自动跳过并继续。</span></span>
          <span class="entry-arrow">→</span>
        </button>
      </div>
      <p class="home-note">单片可选区间 · 批量可读文件夹、可排序、整片处理。</p>
    </section>

    <div id="workbench-view" hidden>
    <div class="workbench-nav">
      <button id="back-home" class="back-home" type="button">← 返回四个入口</button>
      <span id="entry-label" class="entry-label">导演拉片分析</span>
    </div>
    <div class="studio-grid">
      <div class="studio-main">
    <section class="card input-card">
      <div class="card-head">
        <div><div class="section-label">NEW TASK</div><h2 id="input-title">新建任务</h2></div>
        <span id="mode-tag" class="tag">导演拉片模式</span>
      </div>
      <form id="upload-form">
        <div class="mode-picker" aria-label="分析类型">
          <button id="mode-director" class="mode-card selected" type="button" data-select-mode="director">
            <span class="mode-card-icon">▣</span>
            <span><strong>导演拉片</strong><small>视频镜头分析</small></span>
            <span class="mode-radio"></span>
          </button>
          <button id="mode-subtitle" class="mode-card" type="button" data-select-mode="subtitle">
            <span class="mode-card-icon subtitle">▤</span>
            <span><strong>字幕提取</strong><small>提取视觉字幕</small></span>
            <span class="mode-radio"></span>
          </button>
        </div>
        <label class="upload-zone" id="drop-zone" for="video">
          <input id="video" name="video" type="file" accept="video/*">
          <span class="upload-icon">↑</span>
          <span class="upload-copy"><strong id="upload-title">点击或拖拽视频文件到此处</strong><span id="upload-copy">MP4 / MOV / MKV / AVI</span></span>
          <span class="upload-action">选择文件</span>
        </label>
        <div id="file-name" class="file-name">尚未选择视频</div>
        <div id="batch-source-actions" class="batch-source-actions" hidden>
          <button id="choose-folder" class="mini-button" type="button">选择视频文件夹</button>
          <button id="sort-batch" class="mini-button" type="button">按文件名排序</button>
          <button id="clear-batch" class="mini-button" type="button">清空队列</button>
        </div>
        <input id="folder-video" type="file" accept="video/*" webkitdirectory directory multiple hidden>
        <section id="batch-builder" class="batch-builder" hidden>
          <div class="batch-builder-head"><strong>待处理队列</strong><span id="batch-builder-count">0 个视频</span></div>
          <p class="batch-builder-copy">拖动条目或使用 ↑ ↓ 调整处理顺序；可反复添加文件和文件夹，提交后按此顺序进入 GPU 队列。</p>
          <div id="batch-file-list" class="batch-file-list"></div>
          <div id="batch-empty" class="batch-empty">先选择视频，或读取一个视频文件夹。</div>
        </section>
        <input id="analysis-mode" type="hidden" value="director">
        <span id="mode-help" class="small" hidden>视觉、ASR、对齐与导演拉片</span>
        <p id="batch-help" class="batch-help" hidden>整片 · 可排序 · 失败跳过</p>
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
        <div id="range-controls" class="range-grid">
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
          <span><strong>分析完成后释放显存</strong><small id="release-help">默认停止 MiniCPM-V 容器，释放 GPU；取消勾选可保留热模型以加快下一次任务。</small></span>
        </label>
        <button id="start" type="submit">开始分析</button>
      </form>
      <p id="range-note" class="small">开始秒留空按 0 处理，结束秒留空按视频结尾处理。长视频会分批执行，页面显示的是阶段级大概进度。</p>
    </section>
      </div>
      <aside class="studio-sidebar">
        <section class="card model-card">
          <div class="side-head">
            <div><div class="section-label">LOCAL MODELS</div><h2>本地模型</h2></div>
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
          <div id="gpu-note" class="gpu-note">模型按 ASR → 视觉 → 整理顺序切换，尽量避免多个大模型同时占用显存。</div>
        </section>
        <section class="card delivery-card">
          <div class="side-head">
            <div><div class="section-label">DELIVERY</div><h2>输出</h2></div>
            <span class="tag">V1</span>
          </div>
          <div class="delivery-stat"><strong id="delivery-count">04</strong><span id="delivery-copy">份主文档<br>可下载、可继续加工</span></div>
          <div class="delivery-list">
            <div id="delivery-item-1" class="delivery-item">纯视觉分析</div>
            <div id="delivery-item-2" class="delivery-item">纯 ASR 与时间戳</div>
            <div id="delivery-item-3" class="delivery-item">代码综合时间线</div>
            <div id="delivery-item-4" class="delivery-item">最终导演分析模板</div>
          </div>
          <div class="path-label">DEFAULT OUTPUT</div>
          <div class="side-path">由 MEDIA_OUTPUT_ROOT 配置</div>
        </section>
      </aside>
    </div>

    <section class="card pipeline-card">
      <div class="card-head">
        <div><div class="section-label">PIPELINE</div><h2>处理流程</h2></div>
      </div>
      <div class="flow">
        <div class="flow-item"><span class="flow-num">01</span><strong>截取范围</strong><span>按开始秒和结束秒生成分析片段。</span></div>
        <div class="flow-item"><span class="flow-num">02</span><strong id="flow-step-2-title">声音时间线</strong><span id="flow-step-2-copy">ASR + ForcedAligner 生成旁白和时间戳。</span></div>
        <div class="flow-item"><span class="flow-num">03</span><strong id="flow-step-3-title">视觉拉片</strong><span id="flow-step-3-copy">1 秒高清帧、20 秒联系图、5 秒细节组。</span></div>
        <div class="flow-item"><span class="flow-num">04</span><strong id="flow-step-4-title">整理下载</strong><span id="flow-step-4-copy">输出四份主文档及 SRT/VTT 字幕。</span></div>
      </div>
      <p id="pipeline-note" class="small">前三份是可复核的机器产物；第四份是解释层。没有额外提交最终分析时，第四份会显示待处理模板。</p>
    </section>

    <section id="progress-card" class="card" hidden>
      <div class="row">
        <div><div class="section-label">SYSTEM STATUS</div><strong id="filename">等待任务</strong></div>
        <span id="status" class="status">等待中</span>
      </div>
      <div class="progress-head">
        <div><div class="section-label">LIVE PROCESS</div><strong>处理进度</strong></div>
        <span id="progress-percent" class="progress-percent">0%</span>
      </div>
      <progress id="progress" value="0" max="100"></progress>
      <div id="phase" class="small">尚未开始</div>
      <div id="model-control" class="model-control" hidden>
        <div class="model-control-copy">
          <div class="section-label">VLLM CONTROL</div>
          <strong>视觉模型服务</strong>
          <span id="model-control-status">模型 API 未就绪，可重新启动。</span>
        </div>
        <button id="restart-vllm" class="mini-button model-restart" type="button">重启 vLLM</button>
      </div>
      <div class="run-plan" aria-label="任务计划">
        <div class="plan-step" data-plan-step="0"><span class="plan-index">01</span><strong>准备与截取</strong><span>生成分析输入</span></div>
        <div class="plan-step" data-plan-step="1"><span class="plan-index">02</span><strong id="plan-title-1">ASR + 对齐</strong><span id="plan-copy-1">语音与词级时间戳</span></div>
        <div class="plan-step" data-plan-step="2"><span class="plan-index">03</span><strong id="plan-title-2">视觉拉片</strong><span id="plan-copy-2">1fps / 20秒 / 5秒</span></div>
        <div class="plan-step" data-plan-step="3"><span class="plan-index">04</span><strong id="plan-title-3">综合打包</strong><span id="plan-copy-3">四份文档与字幕</span></div>
      </div>
      <div class="telemetry-grid">
        <div class="telemetry-item"><span class="telemetry-label">GPU 利用率</span><strong id="gpu-utilization" class="telemetry-value">--</strong><span id="gpu-utilization-sub" class="telemetry-sub">等待采样</span></div>
        <div class="telemetry-item"><span class="telemetry-label">显存占用</span><strong id="gpu-memory" class="telemetry-value">--</strong><span id="gpu-memory-sub" class="telemetry-sub">已用 / 总量</span></div>
        <div class="telemetry-item"><span class="telemetry-label">温度 / 功耗</span><strong id="gpu-thermal" class="telemetry-value">--</strong><span id="gpu-power" class="telemetry-sub">功耗 --</span></div>
        <div class="telemetry-item"><span class="telemetry-label">视觉设备</span><strong id="gpu-name" class="telemetry-value">--</strong><span id="gpu-refresh" class="telemetry-sub">状态等待</span></div>
        <div class="telemetry-item"><span class="telemetry-label">运行状态</span><strong id="runtime-status" class="telemetry-value">正常</strong><span id="runtime-status-sub" class="telemetry-sub">等待任务</span></div>
      </div>
      <div class="log-head"><div><div class="section-label">LIVE LOG</div><strong>运行日志</strong></div><span class="log-live">LIVE</span><span id="log-count" class="log-count">0 行</span></div>
      <pre id="logs"></pre>
      <section id="batch-queue-card" class="batch-queue-card" hidden>
        <div class="batch-queue-head"><strong>批量队列</strong><span id="batch-queue-count">0 / 0</span></div>
        <p class="batch-queue-copy">等待中的视频可以用 ↑ ↓ 调整顺序；当前正在处理的视频不会被移动。</p>
        <div id="batch-queue-items" class="batch-queue-items"></div>
      </section>
    </section>

    <section id="result-card" class="card" hidden>
      <div class="result-head"><div><div class="section-label">OUTPUT</div><h2>输出内容</h2></div><span id="output-count" class="output-count">0</span></div>
      <p>输出目录：</p>
      <code id="output-dir"></code>
      <ul id="files"></ul>
      <p id="error" class="error"></p>
    </section>
    <div class="studio-footer"><span>野构 STUDIO · MEDIA ANALYSIS LAB</span><span>MiniCPM-V / Qwen3-ASR / ForcedAligner</span></div>
    </div>

    </div>
  </main>
  <script>
    const form = document.getElementById("upload-form");
    const fileInput = document.getElementById("video");
    const folderInput = document.getElementById("folder-video");
    const startButton = document.getElementById("start");
    const progressCard = document.getElementById("progress-card");
    const resultCard = document.getElementById("result-card");
    const filename = document.getElementById("filename");
    const status = document.getElementById("status");
    const progress = document.getElementById("progress");
    const progressPercent = document.getElementById("progress-percent");
    const phase = document.getElementById("phase");
    const modelControl = document.getElementById("model-control");
    const modelControlStatus = document.getElementById("model-control-status");
    const restartVllmButton = document.getElementById("restart-vllm");
    const planSteps = Array.from(document.querySelectorAll("[data-plan-step]"));
    const logs = document.getElementById("logs");
    const logCount = document.getElementById("log-count");
    const gpuUtilization = document.getElementById("gpu-utilization");
    const gpuUtilizationSub = document.getElementById("gpu-utilization-sub");
    const gpuMemory = document.getElementById("gpu-memory");
    const gpuMemorySub = document.getElementById("gpu-memory-sub");
    const gpuThermal = document.getElementById("gpu-thermal");
    const gpuPower = document.getElementById("gpu-power");
    const gpuName = document.getElementById("gpu-name");
    const gpuRefresh = document.getElementById("gpu-refresh");
    const runtimeStatus = document.getElementById("runtime-status");
    const runtimeStatusSub = document.getElementById("runtime-status-sub");
    const outputDir = document.getElementById("output-dir");
    const outputCount = document.getElementById("output-count");
    const files = document.getElementById("files");
    const errorBox = document.getElementById("error");
    const startSec = document.getElementById("start-sec");
    const endSec = document.getElementById("end-sec");
    const releaseGpu = document.getElementById("release-gpu");
    const releaseVramButton = document.getElementById("release-vram");
    const analysisMode = document.getElementById("analysis-mode");
    const inputTitle = document.getElementById("input-title");
    const modeTag = document.getElementById("mode-tag");
    const modeHelp = document.getElementById("mode-help");
    const modePickerButtons = Array.from(document.querySelectorAll("[data-select-mode]"));
    const batchHelp = document.getElementById("batch-help");
    const rangeNote = document.getElementById("range-note");
    const releaseHelp = document.getElementById("release-help");
    const uploadTitle = document.getElementById("upload-title");
    const uploadCopy = document.getElementById("upload-copy");
    const batchSourceActions = document.getElementById("batch-source-actions");
    const chooseFolderButton = document.getElementById("choose-folder");
    const sortBatchButton = document.getElementById("sort-batch");
    const clearBatchButton = document.getElementById("clear-batch");
    const batchBuilder = document.getElementById("batch-builder");
    const batchBuilderCount = document.getElementById("batch-builder-count");
    const batchEmpty = document.getElementById("batch-empty");
    const batchFileList = document.getElementById("batch-file-list");
    const batchQueueCard = document.getElementById("batch-queue-card");
    const batchQueueCount = document.getElementById("batch-queue-count");
    const batchQueueItems = document.getElementById("batch-queue-items");
    const homeView = document.getElementById("home-view");
    const workbenchView = document.getElementById("workbench-view");
    const backHome = document.getElementById("back-home");
    const sidebarButtons = Array.from(document.querySelectorAll("[data-sidebar-action]"));
    const entryLabel = document.getElementById("entry-label");
    const entryButtons = Array.from(document.querySelectorAll("[data-entry-mode]"));
    const heroTitle = document.getElementById("hero-title");
    const heroHint = document.getElementById("hero-hint");
    const gpuNote = document.getElementById("gpu-note");
    const deliveryCount = document.getElementById("delivery-count");
    const deliveryCopy = document.getElementById("delivery-copy");
    const deliveryItems = [1, 2, 3, 4].map(function(index) { return document.getElementById("delivery-item-" + index); });
    const flowStep2Title = document.getElementById("flow-step-2-title");
    const flowStep2Copy = document.getElementById("flow-step-2-copy");
    const flowStep3Title = document.getElementById("flow-step-3-title");
    const flowStep3Copy = document.getElementById("flow-step-3-copy");
    const flowStep4Title = document.getElementById("flow-step-4-title");
    const flowStep4Copy = document.getElementById("flow-step-4-copy");
    const pipelineNote = document.getElementById("pipeline-note");
    const planTitles = [1, 2, 3].map(function(index) { return document.getElementById("plan-title-" + index); });
    const planCopies = [1, 2, 3].map(function(index) { return document.getElementById("plan-copy-" + index); });
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
    let batchMode = false;
    let batchPollTimer = null;
    let batchFiles = [];
    let activeBatchId = null;
    let draggedBatchIndex = null;
    let modelPollTimer = null;

    function updateModePresentation() {
      const subtitle = analysisMode.value === "subtitle";
      const singleLabel = subtitle ? "视觉字幕提取" : "导演拉片分析";
      const batchLabel = subtitle ? "批量字幕提取" : "批量导演拉片";
      inputTitle.textContent = "新建任务";
      modeTag.textContent = batchMode ? batchLabel + "模式" : singleLabel + "模式";
      entryLabel.textContent = batchMode ? batchLabel : singleLabel;
      heroTitle.firstChild.textContent = "媒体分析工作台 ";
      heroHint.textContent = batchMode
        ? "文件夹 · 排序 · 队列 · 进度"
        : (subtitle
          ? "读取画面字幕，输出可复核时间轴。"
          : "画面、声音与时间线，一处查看。");
      uploadTitle.textContent = batchMode ? "点击或拖拽多个视频到此处" : "点击或拖拽视频文件到此处";
      uploadCopy.textContent = batchMode
        ? "可连续添加文件或读取文件夹，整理好顺序后一次提交。"
        : "支持 MP4、MOV、MKV、AVI、WebM 等常见格式";
      fileInput.multiple = batchMode;
      fileInput.required = !batchMode;
      batchSourceActions.hidden = !batchMode;
      batchBuilder.hidden = !batchMode;
      batchHelp.hidden = !batchMode;
      document.getElementById("range-controls").hidden = batchMode;
      rangeNote.textContent = batchMode
        ? "批量模式不使用时间区间设置，每个视频都会从 00:00 分析到结尾。"
        : "开始秒留空按 0 处理，结束秒留空按视频结尾处理。长视频会分批执行，页面显示的是阶段级大概进度。";
      releaseHelp.textContent = batchMode
        ? "批量任务会在全部视频完成后统一释放显存；取消勾选可保留热模型。"
        : "默认停止 MiniCPM-V 容器，释放 GPU；取消勾选可保留热模型以加快下一次任务。";
      gpuNote.textContent = subtitle
        ? (batchMode ? "批量字幕任务逐个调用 MiniCPM-V，单个失败不会中断队列。" : "字幕模式只调用 MiniCPM-V，不启动 ASR，完成后可释放显存。")
        : (batchMode ? "批量任务按顺序切换 ASR 与视觉模型，避免多个大模型同时占用显存。" : "模型按 ASR → 视觉 → 整理顺序切换，尽量避免多个大模型同时占用显存。");
      modeHelp.textContent = subtitle
        ? (batchMode ? "逐项读取画面字幕 · 输出 JSON / SRT / VTT" : "画面字幕 · JSON / SRT / VTT")
        : (batchMode ? "逐项处理视觉、ASR、对齐与导演拉片" : "视觉、ASR、对齐与导演拉片");
      modeHelp.hidden = false;
      modePickerButtons.forEach(function(button) {
        button.classList.toggle("selected", button.dataset.selectMode === analysisMode.value);
      });
      startButton.textContent = batchMode ? (subtitle ? "加入批量字幕队列" : "加入批量分析队列") : (subtitle ? "开始提取字幕" : "开始分析");
      if (batchMode) {
        renderBatchFiles();
      } else if (!startButton.disabled) {
        startButton.disabled = false;
      }
      flowStep2Title.textContent = subtitle ? "抽取画面" : "声音时间线";
      flowStep2Copy.textContent = subtitle ? "每秒保留一张可复核的字幕证据帧。" : "ASR + ForcedAligner 生成旁白和时间戳。";
      flowStep3Title.textContent = subtitle ? "读取字幕" : "视觉拉片";
      flowStep3Copy.textContent = subtitle ? "视觉模型逐帧判断对白/旁白字幕，排除水印和场景文字。" : "1 秒高清帧、20 秒联系图、5 秒细节组。";
      flowStep4Title.textContent = subtitle ? "生成时间轴" : "整理下载";
      flowStep4Copy.textContent = subtitle ? "输出字幕 JSON、SRT、VTT 和原始模型结果。" : "输出四份主文档及 SRT/VTT 字幕。";
      pipelineNote.textContent = subtitle
        ? (batchMode ? "每个视频独立生成字幕时间轴；当前视频失败时，队列会继续处理下一个视频。" : "字幕边界依据每秒画面采样估计；JSON 保留逐帧观察，便于在重配音项目中复核和微调。")
        : (batchMode ? "每个视频独立生成四份主文档；批量队列会记录每个视频的成功或失败状态。" : "前三份是可复核的机器产物；第四份是解释层。没有额外提交最终分析时，第四份会显示待处理模板。");
      if (batchMode) {
        resetPreview();
      }
      if (subtitle) {
        deliveryCount.textContent = "06";
        deliveryCopy.innerHTML = "份字幕交付文件<br>可下载、可继续配音";
        ["字幕时间轴 JSON", "字幕时间轴 SRT / VTT", "字幕提取报告", "原始结果与分析清单"].forEach(function(text, index) {
          deliveryItems[index].textContent = text;
        });
        planTitles[0].textContent = "视觉抽帧";
        planCopies[0].textContent = "每秒字幕证据帧";
        planTitles[1].textContent = "字幕识别";
        planCopies[1].textContent = "逐帧 JSON 判断";
        planTitles[2].textContent = "生成时间轴";
        planCopies[2].textContent = "JSON / SRT / VTT";
      } else {
        deliveryCount.textContent = "04";
        deliveryCopy.innerHTML = "份主文档<br>可下载、可继续加工";
        ["纯视觉分析", "纯 ASR 与时间戳", "代码综合时间线", "最终导演分析模板"].forEach(function(text, index) {
          deliveryItems[index].textContent = text;
        });
        planTitles[0].textContent = "ASR + 对齐";
        planCopies[0].textContent = "语音与词级时间戳";
        planTitles[1].textContent = "视觉拉片";
        planCopies[1].textContent = "1fps / 20秒 / 5秒";
        planTitles[2].textContent = "综合打包";
        planCopies[2].textContent = "四份文档与字幕";
      }
    }

    function enterMode(mode, updateUrl = true) {
      const previousBatchMode = batchMode;
      const previousAnalysisMode = analysisMode.value;
      batchMode = mode.endsWith("-batch");
      const baseMode = batchMode ? mode.slice(0, -6) : mode;
      analysisMode.value = baseMode === "subtitle" ? "subtitle" : "director";
      if (previousBatchMode !== batchMode || previousAnalysisMode !== analysisMode.value) {
        fileInput.value = "";
        folderInput.value = "";
        batchFiles = [];
        showSelectedFiles();
      }
      activeBatchId = null;
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
      if (batchPollTimer) {
        clearTimeout(batchPollTimer);
        batchPollTimer = null;
      }
      progressCard.hidden = false;
      resultCard.hidden = true;
      batchQueueCard.hidden = true;
      errorBox.textContent = "";
      updateModePresentation();
      setSidebarActive(batchMode ? "batch" : "new");
      homeView.hidden = true;
      workbenchView.hidden = false;
      if (updateUrl) history.pushState(null, "", "#" + (batchMode ? analysisMode.value + "-batch" : analysisMode.value));
      window.scrollTo({ top: 0, behavior: "smooth" });
    }

    function showHome(updateUrl = true) {
      homeView.hidden = false;
      workbenchView.hidden = true;
      batchMode = false;
      fileInput.value = "";
      folderInput.value = "";
      batchFiles = [];
      setSidebarActive("new");
      renderBatchFiles();
      resetPreview();
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
      if (batchPollTimer) {
        clearTimeout(batchPollTimer);
        batchPollTimer = null;
      }
      heroTitle.firstChild.textContent = "媒体分析工作台 ";
      heroHint.textContent = "选择一个工作入口，把视频整理成可回看、可下载、可继续加工的资料。";
      if (updateUrl) history.pushState(null, "", window.location.pathname + window.location.search);
      window.scrollTo({ top: 0, behavior: "smooth" });
    }

    function syncRoute() {
      const mode = window.location.hash.replace("#", "");
      if (["director", "subtitle", "director-batch", "subtitle-batch"].includes(mode)) enterMode(mode, false);
      else enterMode("director", false);
    }

    function setSidebarActive(action) {
      sidebarButtons.forEach(function(button) {
        button.classList.toggle("active", button.dataset.sidebarAction === action);
      });
    }

    function focusWorkbenchPanel(selector) {
      if (workbenchView.hidden) enterMode(analysisMode.value, false);
      window.setTimeout(function() {
        const target = document.querySelector(selector);
        if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 0);
    }

    entryButtons.forEach(function(button) {
      button.addEventListener("click", function() { enterMode(button.dataset.entryMode); });
    });
    modePickerButtons.forEach(function(button) {
      button.addEventListener("click", function() {
        enterMode(batchMode ? button.dataset.selectMode + "-batch" : button.dataset.selectMode);
      });
    });
    sidebarButtons.forEach(function(button) {
      button.addEventListener("click", function() {
        const action = button.dataset.sidebarAction;
        if (action === "new") enterMode("director");
        else if (action === "batch") enterMode(analysisMode.value + "-batch");
        else if (action === "output") focusWorkbenchPanel("#result-card");
        else if (action === "models") focusWorkbenchPanel(".model-card");
        else if (action === "settings") focusWorkbenchPanel(".release-option");
        if (action !== "batch" && action !== "new") setSidebarActive(action);
      });
    });
    backHome.addEventListener("click", function() { showHome(); });
    window.addEventListener("popstate", syncRoute);
    syncRoute();

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

    const batchVideoSuffixes = [".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".wmv", ".flv"];

    function isBatchVideo(file) {
      if (!file) return false;
      const name = String(file.name || "").toLowerCase();
      return String(file.type || "").startsWith("video/") || batchVideoSuffixes.some(function(suffix) { return name.endsWith(suffix); });
    }

    function batchFilePath(file) {
      return String(file.webkitRelativePath || file.name || "视频");
    }

    function batchFileKey(file) {
      return batchFilePath(file) + "|" + file.size + "|" + file.lastModified;
    }

    function moveBatchFile(index, offset) {
      const target = index + offset;
      if (index < 0 || target < 0 || target >= batchFiles.length) return;
      const moved = batchFiles.splice(index, 1)[0];
      batchFiles.splice(target, 0, moved);
      renderBatchFiles();
    }

    function removeBatchFile(index) {
      if (index < 0 || index >= batchFiles.length) return;
      batchFiles.splice(index, 1);
      renderBatchFiles();
    }

    function renderBatchFiles() {
      batchFileList.innerHTML = "";
      if (!batchMode) {
        batchBuilder.hidden = true;
        return;
      }
      batchBuilder.hidden = false;
      batchBuilderCount.textContent = batchFiles.length + " 个视频";
      batchEmpty.hidden = batchFiles.length > 0;
      sortBatchButton.disabled = batchFiles.length < 2;
      clearBatchButton.disabled = batchFiles.length === 0;
      startButton.disabled = batchFiles.length === 0;
      if (!batchFiles.length) {
        fileName.textContent = "尚未加入视频";
        return;
      }
      const totalSize = batchFiles.reduce(function(sum, file) { return sum + file.size; }, 0);
      fileName.textContent = "队列中 " + batchFiles.length + " 个视频 · " + formatFileSize(totalSize);
      batchFiles.forEach(function(file, index) {
        const item = document.createElement("div");
        item.className = "batch-file-item";
        item.draggable = true;
        item.dataset.index = String(index);
        item.addEventListener("dragstart", function(event) {
          draggedBatchIndex = index;
          item.classList.add("dragging");
          if (event.dataTransfer) event.dataTransfer.effectAllowed = "move";
        });
        item.addEventListener("dragover", function(event) {
          event.preventDefault();
          if (event.dataTransfer) event.dataTransfer.dropEffect = "move";
        });
        item.addEventListener("drop", function(event) {
          event.preventDefault();
          const from = draggedBatchIndex;
          if (from === null || from === index || from < 0 || from >= batchFiles.length) return;
          const moved = batchFiles.splice(from, 1)[0];
          const destination = from < index ? index - 1 : index;
          batchFiles.splice(destination, 0, moved);
          draggedBatchIndex = null;
          renderBatchFiles();
        });
        item.addEventListener("dragend", function() {
          draggedBatchIndex = null;
          item.classList.remove("dragging");
        });

        const handle = document.createElement("span");
        handle.className = "batch-file-handle";
        handle.textContent = "⠿";
        handle.title = "拖动调整顺序";
        const order = document.createElement("span");
        order.className = "batch-file-order";
        order.textContent = String(index + 1).padStart(2, "0");
        const main = document.createElement("span");
        main.className = "batch-file-main";
        const name = document.createElement("span");
        name.className = "batch-file-name";
        name.textContent = file.name || "视频";
        const path = document.createElement("span");
        path.className = "batch-file-path";
        path.textContent = batchFilePath(file) + " · " + formatFileSize(file.size);
        main.appendChild(name);
        main.appendChild(path);
        const actions = document.createElement("span");
        actions.className = "batch-file-actions";
        [["↑", "移到上一位", -1], ["↓", "移到下一位", 1]].forEach(function(entry) {
          const button = document.createElement("button");
          button.type = "button";
          button.textContent = entry[0];
          button.title = entry[1];
          button.disabled = entry[2] < 0 ? index === 0 : index === batchFiles.length - 1;
          button.addEventListener("click", function() { moveBatchFile(index, entry[2]); });
          actions.appendChild(button);
        });
        const remove = document.createElement("button");
        remove.type = "button";
        remove.textContent = "×";
        remove.title = "从队列移除";
        remove.addEventListener("click", function() { removeBatchFile(index); });
        actions.appendChild(remove);
        item.appendChild(handle);
        item.appendChild(order);
        item.appendChild(main);
        item.appendChild(actions);
        batchFileList.appendChild(item);
      });
    }

    function addBatchFiles(fileList, fromFolder) {
      const incoming = Array.from(fileList || []).filter(isBatchVideo);
      if (fromFolder) {
        incoming.sort(function(left, right) { return batchFilePath(left).localeCompare(batchFilePath(right), undefined, { numeric: true, sensitivity: "base" }); });
      }
      const existing = new Set(batchFiles.map(batchFileKey));
      let skipped = 0;
      incoming.forEach(function(file) {
        const key = batchFileKey(file);
        if (existing.has(key)) {
          skipped += 1;
          return;
        }
        existing.add(key);
        batchFiles.push(file);
      });
      renderBatchFiles();
      if (!incoming.length && fileList && fileList.length) {
        fileName.textContent = "没有找到支持的视频文件";
      } else if (skipped > 0) {
        fileName.textContent += " · 已忽略重复 " + skipped + " 个";
      }
    }

    function showSelectedFiles() {
      startSec.value = "";
      endSec.value = "";
      if (batchMode) {
        renderBatchFiles();
        resetPreview();
        return;
      }
      const selected = Array.from(fileInput.files || []);
      if (!selected.length) {
        fileName.textContent = "尚未选择视频";
        resetPreview();
        return;
      }
      const file = selected[0];
      fileName.textContent = "已选择： " + file.name + " · " + formatFileSize(file.size);
      loadPreview(file);
    }

    fileInput.addEventListener("change", function() {
      if (batchMode) {
        addBatchFiles(fileInput.files, false);
        fileInput.value = "";
      } else {
        showSelectedFiles();
      }
    });
    chooseFolderButton.addEventListener("click", function() { folderInput.click(); });
    folderInput.addEventListener("change", function() {
      addBatchFiles(folderInput.files, true);
      folderInput.value = "";
    });
    sortBatchButton.addEventListener("click", function() {
      batchFiles.sort(function(left, right) { return batchFilePath(left).localeCompare(batchFilePath(right), undefined, { numeric: true, sensitivity: "base" }); });
      renderBatchFiles();
    });
    clearBatchButton.addEventListener("click", function() {
      batchFiles = [];
      fileInput.value = "";
      folderInput.value = "";
      renderBatchFiles();
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
        if (batchMode) {
          addBatchFiles(event.dataTransfer.files, false);
        } else {
          fileInput.files = event.dataTransfer.files;
          showSelectedFiles();
        }
      }
    });

    function updateVramButton(state) {
      const modelState = state || {};
      const busy = modelState.status === "releasing" || modelState.status === "starting";
      releaseVramButton.disabled = busy;
      releaseVramButton.textContent = modelState.status === "releasing" ? "释放中…" : "释放显存";
      releaseVramButton.title = modelState.message || "只停止视觉模型容器，WebUI 保持在线";
    }

    function scheduleModelStatePoll(delay) {
      if (modelPollTimer) clearTimeout(modelPollTimer);
      modelPollTimer = setTimeout(refreshModelState, delay || 1000);
    }

    async function refreshModelState() {
      try {
        const response = await fetch("/api/model");
        const state = await response.json();
        updateVramButton(state);
        if (state.status === "starting" || state.status === "releasing") scheduleModelStatePoll(1000);
        else modelPollTimer = null;
      } catch (_error) {
        scheduleModelStatePoll(3000);
      }
    }

    releaseVramButton.addEventListener("click", async function() {
      if (!window.confirm("只停止视觉模型并释放显存，WebUI 会保持在线。继续吗？")) return;
      updateVramButton({ status: "releasing", message: "正在释放显存，WebUI 保持在线。" });
      try {
        const response = await fetch("/api/model/release", { method: "POST" });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "释放显存失败");
        updateVramButton(payload);
        scheduleModelStatePoll(500);
      } catch (error) {
        updateVramButton({ status: "failed", message: String(error) });
        window.alert("释放显存失败：" + String(error));
      }
    });

    restartVllmButton.addEventListener("click", async function() {
      if (!window.confirm("只重新启动视觉模型 vLLM，不会取消当前任务或清空队列。继续吗？")) return;
      restartVllmButton.disabled = true;
      modelControlStatus.textContent = "正在请求重启，当前任务队列保持不变。";
      try {
        const response = await fetch("/api/model/restart", { method: "POST" });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "vLLM 重启失败");
        updateModelControl({ status: "running", phase: "等待视觉模型", logs: [] }, payload);
      } catch (error) {
        restartVllmButton.disabled = false;
        modelControlStatus.textContent = "重启请求失败：" + String(error);
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

    function updateRuntimeStatus(job) {
      if (!job) {
        runtimeStatus.textContent = "正常";
        runtimeStatusSub.textContent = "等待任务";
        return;
      }
      if (job.status === "done") {
        runtimeStatus.textContent = "已完成";
        runtimeStatusSub.textContent = "输出已生成";
      } else if (job.status === "failed") {
        runtimeStatus.textContent = "需处理";
        runtimeStatusSub.textContent = "当前任务失败";
      } else if (job.status === "queued") {
        runtimeStatus.textContent = "排队中";
        runtimeStatusSub.textContent = "等待 GPU";
      } else {
        runtimeStatus.textContent = "运行中";
        runtimeStatusSub.textContent = job.phase || "处理中";
      }
    }

    function updatePlan(job, percent) {
      const phaseText = String(job.phase || "");
      let active = 0;
      if (job.mode === "subtitle") {
        if (phaseText.includes("抽帧") || phaseText.includes("证据") || (percent >= 5 && percent < 35)) active = 1;
        if (phaseText.includes("字幕") || (percent >= 35 && percent < 90)) active = 2;
        if (phaseText.includes("生成") || phaseText.includes("整理") || percent >= 90) active = 3;
      } else {
        if (phaseText.includes("ASR") || phaseText.includes("对齐") || (percent >= 5 && percent < 35)) active = 1;
        if (phaseText.includes("视觉") || (percent >= 35 && percent < 90)) active = 2;
        if (phaseText.includes("生成") || phaseText.includes("整理") || percent >= 90) active = 3;
      }
      if (job.status === "done") active = planSteps.length;
      planSteps.forEach(function(step, index) {
        step.classList.toggle("done", index < active || job.status === "done");
        step.classList.toggle("active", index === active && job.status !== "done" && job.status !== "failed");
      });
    }

    function jobWaitingForVisualModel(job) {
      if (!job || job.status !== "running") return false;
      const phaseText = String(job.phase || "");
      if (["视觉分析", "识别字幕", "读取字幕", "高清逐秒"].some(function(marker) { return phaseText.includes(marker); }) && !phaseText.includes("等待")) return false;
      const recentLogs = (job.logs || []).slice(-20).join(" ");
      const text = phaseText + " " + recentLogs;
      return ["等待 MiniCPM", "等待视觉模型", "等待视觉服务", "API 就绪", "启动视觉模型", "视觉模型服务"]
        .some(function(marker) { return text.includes(marker); });
    }

    function updateModelControl(job, modelState) {
      const state = modelState || (job && job.model_restart) || {};
      updateVramButton(state);
      const waiting = jobWaitingForVisualModel(job);
      const restarting = state.status === "starting";
      const failed = state.status === "failed";
      const ready = state.status === "ready";
      modelControl.hidden = !(waiting || restarting || failed);
      if (modelControl.hidden) return;
      restartVllmButton.disabled = restarting || ready;
      restartVllmButton.textContent = restarting ? "启动中…" : (failed ? "再次启动" : (ready ? "已启动" : "重启 vLLM"));
      if (restarting) {
        modelControlStatus.textContent = state.message || "正在启动 vLLM，当前任务队列保持不变。";
      } else if (failed) {
        modelControlStatus.textContent = state.message || "视觉模型启动失败，可再次尝试。";
      } else if (ready) {
        modelControlStatus.textContent = state.message || "vLLM API 已就绪，任务会继续。";
      } else {
        modelControlStatus.textContent = "视觉模型 API 未就绪，可重新启动。";
      }
    }

    function jobLogLines(job) {
      const lines = Array.isArray(job && job.logs) ? job.logs.slice() : [];
      const modelState = (job && job.model_restart) || {};
      const modelLogs = Array.isArray(modelState.logs) ? modelState.logs : [];
      if (modelLogs.length) {
        lines.push("", "— vLLM 服务控制 —");
        lines.push.apply(lines, modelLogs);
      }
      return lines;
    }

    function showJob(job) {
      progressCard.hidden = false;
      batchQueueCard.hidden = true;
      resultCard.hidden = job.status !== "done" && job.status !== "failed";
      filename.textContent = (job.filename || "视频分析任务") + (job.mode === "subtitle" ? " · 视觉字幕提取" : "");
      status.textContent = job.status_label || job.status;
      const percent = Math.max(0, Math.min(100, Number(job.progress) || 0));
      progress.value = percent;
      progressPercent.textContent = percent.toFixed(0) + "%";
      phase.textContent = percent.toFixed(0) + "% · " + (job.phase || "处理中");
      updatePlan(job, percent);
      updateGpu(job.gpu);
      updateRuntimeStatus(job);
      updateModelControl(job, job.model_restart);
      const logLines = jobLogLines(job);
      logs.textContent = logLines.join("\\n");
      logCount.textContent = logLines.length + " 行";
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
      outputCount.textContent = String((job.files || []).length);
      errorBox.textContent = job.error || "";
      if (job.status === "done" || job.status === "failed") {
        startButton.disabled = false;
        if (timer) {
          clearTimeout(timer);
          timer = null;
        }
      }
    }

    function batchStatusLabel(job) {
      if (job.status === "done") return "完成";
      if (job.status === "failed" || job.status === "cancelled") return "已跳过";
      if (job.status === "running") return (Number(job.progress) || 0).toFixed(0) + "%";
      return "排队中";
    }

    async function moveSubmittedBatchJob(batch, jobId, offset) {
      const jobs = Array.isArray(batch.jobs) ? batch.jobs : [];
      const pending = jobs.filter(function(job) { return job.status === "queued"; }).map(function(job) { return String(job.id); });
      const currentIndex = pending.indexOf(String(jobId));
      const targetIndex = currentIndex + offset;
      if (currentIndex < 0 || targetIndex < 0 || targetIndex >= pending.length) return;
      const moved = pending.splice(currentIndex, 1)[0];
      pending.splice(targetIndex, 0, moved);
      const pendingSet = new Set(pending);
      let cursor = 0;
      const requested = jobs.map(function(job) {
        const id = String(job.id);
        return pendingSet.has(id) ? pending[cursor++] : id;
      });
      try {
        const response = await fetch("/api/batches/" + encodeURIComponent(batch.id) + "/reorder", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ job_ids: requested }),
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "队列调整失败");
        showBatch(payload);
      } catch (error) {
        errorBox.textContent = "调整队列失败：" + String(error);
      }
    }

    function showBatch(batch) {
      activeBatchId = batch.id || activeBatchId;
      progressCard.hidden = false;
      resultCard.hidden = true;
      batchQueueCard.hidden = false;
      const total = Number(batch.total) || 0;
      const completed = Number(batch.completed) || 0;
      filename.textContent = (batch.mode_label || "批量任务") + " · " + total + " 个视频";
      status.textContent = batch.status_label || batch.status;
      const percent = Math.max(0, Math.min(100, Number(batch.progress) || 0));
      progress.value = percent;
      progressPercent.textContent = percent.toFixed(0) + "%";
      phase.textContent = percent.toFixed(0) + "% · " + (batch.phase || "等待任务队列");
      updatePlan({ mode: batch.mode, status: batch.status === "done" || batch.status === "partial" ? "done" : batch.status, progress: percent, phase: batch.phase }, percent);
      updateGpu(batch.gpu);
      updateRuntimeStatus({ status: batch.status === "done" || batch.status === "partial" ? "done" : batch.status, phase: batch.phase });
      batchQueueCount.textContent = completed + " / " + total + " 已完成 · " + (Number(batch.failed) || 0) + " 个失败";
      batchQueueItems.innerHTML = "";
      const jobs = Array.isArray(batch.jobs) ? batch.jobs : [];
      updateModelControl(
        jobs.find(function(job) { return job.status === "running"; }) || { status: batch.status, phase: batch.phase, logs: [] },
        batch.model_restart,
      );
      const pendingJobs = jobs.filter(function(job) { return job.status === "queued"; });
      jobs.forEach(function(job, index) {
        const item = document.createElement("div");
        item.className = "batch-job-item " + (job.status === "done" ? "done" : (job.status === "failed" || job.status === "cancelled" ? "failed" : ""));
        const order = document.createElement("span");
        order.className = "batch-job-index";
        order.textContent = String(index + 1).padStart(2, "0");
        const name = document.createElement("span");
        name.className = "batch-job-name";
        name.title = job.filename || "视频";
        name.textContent = job.filename || "视频";
        const state = document.createElement("span");
        state.className = "batch-job-status";
        state.textContent = job.status === "queued" && job.queue_position ? "排队 · 全局第 " + job.queue_position : batchStatusLabel(job);
        item.appendChild(order);
        item.appendChild(name);
        item.appendChild(state);
        if (job.status === "queued") {
          const controls = document.createElement("span");
          controls.className = "batch-job-controls";
          [["↑", "上移", -1], ["↓", "下移", 1]].forEach(function(entry) {
            const button = document.createElement("button");
            button.type = "button";
            button.textContent = entry[0];
            button.title = entry[1];
            const pendingIndex = pendingJobs.findIndex(function(item) { return String(item.id) === String(job.id); });
            button.disabled = entry[2] < 0 ? pendingIndex === 0 : pendingIndex === pendingJobs.length - 1;
            button.addEventListener("click", function() { moveSubmittedBatchJob(batch, job.id, entry[2]); });
            controls.appendChild(button);
          });
          item.appendChild(controls);
        }
        const meter = document.createElement("span");
        meter.className = "batch-job-progress";
        const meterFill = document.createElement("span");
        const meterPercent = job.status === "done" || job.status === "failed" || job.status === "cancelled"
          ? 100
          : Math.max(0, Math.min(100, Number(job.progress) || 0));
        meterFill.style.width = meterPercent + "%";
        meter.appendChild(meterFill);
        item.appendChild(meter);
        if (job.status === "done" && Array.isArray(job.files) && job.files.length) {
          const links = document.createElement("div");
          links.className = "batch-job-links";
          job.files.slice(0, 4).forEach(function(file) {
            const link = document.createElement("a");
            link.href = file.url;
            link.download = file.name;
            link.textContent = file.name.split("/").pop();
            links.appendChild(link);
          });
          item.appendChild(links);
        }
        batchQueueItems.appendChild(item);
      });
      const logLines = [
        "批量任务：" + (batch.mode_label || "视频分析"),
        "进度：" + completed + " / " + total + "，失败：" + (Number(batch.failed) || 0) + "（失败视频自动跳过）",
      ];
      jobs.forEach(function(job, index) {
        logLines.push(String(index + 1).padStart(2, "0") + " · " + (job.filename || "视频") + " · " + batchStatusLabel(job));
        if (job.error) logLines.push("   错误：" + job.error);
      });
      const modelLogs = batch.model_restart && Array.isArray(batch.model_restart.logs) ? batch.model_restart.logs : [];
      if (modelLogs.length) logLines.push("", "— vLLM 服务控制 —", ...modelLogs);
      logs.textContent = logLines.join("\\n");
      logCount.textContent = logLines.length + " 行";
      logs.scrollTop = logs.scrollHeight;
      errorBox.textContent = Number(batch.failed) > 0 ? (Number(batch.failed) + " 个视频处理失败，已跳过，其余任务继续执行。") : "";
      if (batch.status === "done" || batch.status === "partial") {
        startButton.disabled = false;
        if (batchPollTimer) {
          clearTimeout(batchPollTimer);
          batchPollTimer = null;
        }
      }
      if (batchMode && !batchFiles.length) startButton.disabled = true;
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

    async function pollBatch(batchId) {
      try {
        const response = await fetch("/api/batches/" + encodeURIComponent(batchId));
        const batch = await response.json();
        if (!response.ok) throw new Error(batch.error || "读取批量任务失败");
        showBatch(batch);
        if (batch.status !== "done" && batch.status !== "partial") {
          batchPollTimer = setTimeout(function() { pollBatch(batchId); }, 1500);
        }
      } catch (error) {
        errorBox.textContent = String(error);
        batchPollTimer = setTimeout(function() { pollBatch(batchId); }, 3000);
      }
    }

    form.addEventListener("submit", async function(event) {
      event.preventDefault();
      if (batchMode ? !batchFiles.length : !fileInput.files.length) return;
      startButton.disabled = true;
      progressCard.hidden = false;
      resultCard.hidden = true;
      batchQueueCard.hidden = !batchMode;
      logs.textContent = "正在上传视频…";
      logCount.textContent = "1 行";
      try {
        const body = new FormData();
        const filesToUpload = batchMode ? batchFiles : Array.from(fileInput.files);
        filesToUpload.forEach(function(file) { body.append("video", file, file.name); });
        const query = new URLSearchParams();
        if (!batchMode && startSec.value.trim()) query.set("start_sec", startSec.value.trim());
        if (!batchMode && endSec.value.trim()) query.set("end_sec", endSec.value.trim());
        query.set("mode", analysisMode.value);
        query.set("release_gpu", releaseGpu.checked ? "1" : "0");
        const target = (batchMode ? "/api/batches" : "/api/jobs") + (query.toString() ? "?" + query.toString() : "");
        const response = await fetch(target, { method: "POST", body: body });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "创建任务失败");
        if (batchMode) {
          batchFiles = [];
          fileInput.value = "";
          folderInput.value = "";
          renderBatchFiles();
          showBatch(payload);
          pollBatch(payload.id);
        } else {
          showJob(payload);
          poll(payload.id);
        }
      } catch (error) {
        startButton.disabled = false;
        resultCard.hidden = false;
        errorBox.textContent = String(error);
      }
    });
    refreshModelState();
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


def save_batch_state(batch: dict) -> None:
    (JOB_STATE_DIR / f"batch_{batch['id']}.json").write_text(
        json.dumps(batch, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_batch_state(batch_id: str) -> dict | None:
    state_path = JOB_STATE_DIR / f"batch_{batch_id}.json"
    if not state_path.is_file():
        return None
    try:
        value = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _job_from_state(job_id: str) -> dict | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is not None:
            return dict(job)
    state_path = JOB_STATE_DIR / f"{job_id}.json"
    if not state_path.is_file():
        return None
    try:
        value = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def pending_queue_tasks() -> list[dict[str, object]]:
    """Copy waiting tasks without touching the queue's unfinished-task count."""

    with JOB_QUEUE.mutex:
        return [dict(task) for task in list(JOB_QUEUE.queue)]


def pending_queue_snapshot() -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for position, task in enumerate(pending_queue_tasks(), start=1):
        job_id = str(task.get("job_id", ""))
        job = _job_from_state(job_id) or {}
        result.append(
            {
                "position": position,
                "job_id": job_id,
                "filename": str(job.get("filename", task.get("original", "视频"))),
                "status": str(job.get("status", "queued")),
                "batch_id": str(task.get("batch_id", "")),
            }
        )
    return result


def reorder_batch_queue(batch_id: str, requested_job_ids: list[str]) -> dict | None:
    """Reorder only waiting items from one batch while preserving global FIFO slots."""

    with JOBS_LOCK:
        batch = BATCHES.get(batch_id)
        if batch is None:
            batch = load_batch_state(batch_id)
            if batch is None:
                return None
            BATCHES[batch_id] = batch
        original_ids = [str(item) for item in batch.get("job_ids", [])]
        known_ids = set(original_ids)
        requested = [str(item) for item in requested_job_ids]
        if len(requested) != len(set(requested)) or any(item not in known_ids for item in requested):
            raise ValueError("队列顺序中包含无效或重复的视频任务")

    with JOB_QUEUE.mutex:
        pending = list(JOB_QUEUE.queue)
        batch_slots = [
            index
            for index, task in enumerate(pending)
            if str(task.get("batch_id", "")) == batch_id
        ]
        pending_batch_ids = [str(pending[index].get("job_id", "")) for index in batch_slots]
        pending_batch_set = set(pending_batch_ids)
        requested_pending = [item for item in requested if item in pending_batch_set]
        requested_pending.extend(item for item in pending_batch_ids if item not in requested_pending)
        task_by_id = {
            str(task.get("job_id", "")): task
            for task in pending
            if str(task.get("batch_id", "")) == batch_id
        }
        reordered = list(pending)
        for slot, job_id in zip(batch_slots, requested_pending):
            reordered[slot] = task_by_id[job_id]
        JOB_QUEUE.queue.clear()
        JOB_QUEUE.queue.extend(reordered)

    # Keep completed/running entries anchored in the visible history and move
    # only the pending portion of the batch. This makes the UI match execution.
    with JOBS_LOCK:
        current = BATCHES.get(batch_id)
        if current is None:
            return None
        pending_set = set(pending_batch_ids)
        pending_iter = iter(requested_pending)
        ordered_ids: list[str] = []
        for job_id in original_ids:
            if job_id in pending_set:
                ordered_ids.append(next(pending_iter))
            else:
                ordered_ids.append(job_id)
        current["job_ids"] = ordered_ids
        current["queue_order"] = requested_pending
        for index, job_id in enumerate(ordered_ids, start=1):
            job = JOBS.get(job_id)
            if job and job_id in pending_set:
                job["batch_index"] = index
                save_job_state(job)
        current["queue_size"] = JOB_QUEUE.qsize()
        save_batch_state(current)
    return batch_snapshot(batch_id)


def refresh_batch(batch_id: str) -> dict | None:
    with JOBS_LOCK:
        batch = BATCHES.get(batch_id)
        if batch is None:
            batch = load_batch_state(batch_id)
            if batch is None:
                return None
            BATCHES[batch_id] = batch
        previous = {
            key: batch.get(key)
            for key in ("status", "status_label", "phase", "progress", "completed", "succeeded", "failed")
        }
        job_ids = [str(item) for item in batch.get("job_ids", [])]
        jobs = []
        for job_id in job_ids:
            job = _job_from_state(job_id)
            if job is not None:
                jobs.append(job)
        statuses = [str(item.get("status", "queued")) for item in jobs]
        total = int(batch.get("total", len(job_ids)) or len(job_ids))
        completed = sum(status in {"done", "failed", "cancelled"} for status in statuses)
        succeeded = sum(status == "done" for status in statuses)
        failed = sum(status in {"failed", "cancelled"} for status in statuses)
        progress_values = [max(0, min(100, int(item.get("progress", 0) or 0))) for item in jobs]
        progress = round(sum(progress_values) / total) if total else 100
        if completed >= total and total:
            batch["status"] = "done" if failed == 0 else "partial"
            batch["status_label"] = "全部完成" if failed == 0 else "部分完成"
            batch["phase"] = "批量任务完成"
            batch["progress"] = 100
            batch["current_job_id"] = ""
        elif any(status == "running" for status in statuses):
            batch["status"] = "running"
            batch["status_label"] = "处理中"
            batch["phase"] = "正在处理队列"
            batch["progress"] = progress
        else:
            batch["status"] = "queued"
            batch["status_label"] = "排队中"
            batch["phase"] = "等待任务队列"
            batch["progress"] = progress
        batch["completed"] = completed
        batch["succeeded"] = succeeded
        batch["failed"] = failed
        batch["queue_size"] = JOB_QUEUE.qsize()
        current = {
            key: batch.get(key)
            for key in ("status", "status_label", "phase", "progress", "completed", "succeeded", "failed")
        }
        if current != previous:
            save_batch_state(batch)
        return dict(batch)


def batch_snapshot(batch_id: str) -> dict | None:
    batch = refresh_batch(batch_id)
    if batch is None:
        return None
    queued_positions = {
        str(item.get("job_id", "")): int(item.get("position", 0))
        for item in pending_queue_snapshot()
        if item.get("job_id")
    }
    jobs: list[dict] = []
    for job_id in batch.get("job_ids", []):
        job = _job_from_state(str(job_id))
        if not job:
            continue
        job.pop("thread", None)
        job["queue_position"] = queued_positions.get(str(job_id))
        job["files"] = files_for_output(Path(str(job.get("output_path", ""))), str(job_id))
        jobs.append(job)
    batch["jobs"] = jobs
    batch["queue_order"] = [
        str(item.get("job_id", ""))
        for item in pending_queue_snapshot()
        if str(item.get("batch_id", "")) == batch_id
    ]
    batch["gpu"] = gpu_status()
    batch["model_restart"] = model_restart_snapshot()
    return batch


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


def files_for_output(output: Path, job_id: str) -> list[dict[str, str]]:
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
        "字幕提取任务信息.md",
        "字幕提取时间轴.json",
        "字幕提取时间轴.srt",
        "字幕提取时间轴.vtt",
        "字幕提取原始结果.jsonl",
        "字幕提取报告.md",
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


def files_for_job(job_id: str) -> list[dict[str, str]]:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return []
        output = Path(job["output_path"])
    return files_for_output(output, job_id)


def update_progress_from_line(job_id: str, line: str) -> None:
    if "阶段 1/2" in line:
        add_log(job_id, line, 25, "视觉字幕：抽取证据帧")
    elif "已抽取" in line and "字幕证据帧" in line:
        add_log(job_id, line, 25, "视觉字幕：抽取证据帧")
    elif "字幕批次完成" in line:
        match = re.search(r"进度\s*[=:：]\s*(\d+)%", line)
        batch_progress = int(match.group(1)) if match else 0
        add_log(job_id, line, min(90, 25 + round(batch_progress * 0.65)), "视觉字幕：识别字幕")
    elif "阶段 2/2" in line and "字幕" in line:
        add_log(job_id, line, 94, "视觉字幕：生成时间轴")
    elif "阶段 1/3 完成" in line:
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
    mode: str,
    release_gpu_after: bool,
) -> None:
    if ANALYSIS_LOCK.locked():
        add_log(job_id, "已有任务占用 GPU，当前任务进入队列。", 0, "排队等待 GPU")
    with ANALYSIS_LOCK:
        add_log(job_id, "已获得 GPU 分析资源，按计划开始执行。", 1, "准备中")
        _run_job(job_id, source, original, output, start_sec, end_sec, mode, release_gpu_after)


def _run_job(
    job_id: str,
    source: Path,
    original: str,
    output: Path,
    start_sec: float,
    end_sec: float | None,
    mode: str,
    release_gpu_after: bool,
) -> None:
    mode_label = "视觉字幕提取变体" if mode == "subtitle" else "固定导演拉片流程"
    add_log(job_id, f"任务已创建，准备调用{mode_label}。", 1, "准备中")
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
        job_config = _job_from_state(job_id) or {}
        report_release_gpu = bool(job_config.get("release_gpu_after", release_gpu_after))
        (output / "00_任务参数.md").write_text(
            "\n".join(
                [
                    "# 任务参数",
                    "",
                    f"- 原始上传文件：{original}",
                    f"- 原始上传路径：{source}",
                    f"- 分析起始秒：{start_sec:.3f}",
                    f"- 分析结束秒：{end_label}",
                    f"- 分析模式：{'视觉字幕提取（用于后续重配音）' if mode == 'subtitle' else '导演拉片（视觉 + ASR + ForcedAligner）'}",
                    f"- 分析完成后释放显存：{'是' if report_release_gpu else '否'}",
                    "- 输出时间码口径：字幕提取模式输出原视频时间；导演拉片模式的报告时间码以分析片段为 00:00 起点。",
                    "",
                ]
            ),
            encoding="utf-8",
        )
    except Exception as exc:
        add_log(job_id, f"准备阶段异常：{type(exc).__name__}: {exc}")
        update_job(job_id, status="failed", status_label="失败", error=str(exc))
        return
    if mode == "subtitle":
        command = [
            "python3",
            str(SUBTITLE_PIPELINE),
            "--source",
            str(analysis_source),
            "--output",
            str(output),
            "--time-offset",
            f"{start_sec:.3f}",
        ]
    else:
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
        add_log(job_id, "视觉字幕时间轴已生成。" if mode == "subtitle" else "四份主文档已生成。", 100, "完成")
        update_job(job_id, status="done", status_label="完成", progress=100, files=files_for_job(job_id))
    except Exception as exc:
        add_log(job_id, f"任务异常：{type(exc).__name__}: {exc}")
        update_job(job_id, status="failed", status_label="失败", error=str(exc))


def stop_visual_service() -> None:
    try:
        subprocess.run(
            ["docker", "stop", MINICPM_CONTAINER],
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return


def finish_batch_item(batch_id: str, job_id: str) -> None:
    job = _job_from_state(job_id)
    if job and job.get("status") in {"failed", "cancelled"}:
        add_log(job_id, "当前视频分析失败，已跳过；批量队列继续处理下一个视频。")
    batch = refresh_batch(batch_id)
    if not batch or int(batch.get("completed", 0)) < int(batch.get("total", 0)):
        return
    should_release = False
    with JOBS_LOCK:
        current = BATCHES.get(batch_id)
        if current and current.get("release_gpu_after") and not current.get("gpu_released"):
            current["gpu_released"] = True
            current["phase"] = "正在释放 GPU 显存"
            save_batch_state(current)
            should_release = True
    if should_release:
        add_log(job_id, "批量任务已完成，正在释放 GPU 显存。", 100, "释放 GPU 显存")
        stop_visual_service()
        with JOBS_LOCK:
            current = BATCHES.get(batch_id)
            if current:
                current["phase"] = "批量任务完成"
                save_batch_state(current)


def analysis_queue_worker() -> None:
    while True:
        task = JOB_QUEUE.get()
        job_id = str(task.get("job_id", ""))
        batch_id = str(task.get("batch_id", "")) or None
        try:
            if batch_id:
                job = _job_from_state(job_id) or {}
                add_log(
                    job_id,
                    f"批量队列开始第 {job.get('batch_index', '?')} / {job.get('batch_total', '?')} 个视频。",
                    0,
                    "批量队列处理中",
                )
                with JOBS_LOCK:
                    batch = BATCHES.get(batch_id)
                    if batch:
                        batch["current_job_id"] = job_id
                        batch["status"] = "running"
                        batch["status_label"] = "处理中"
                        batch["phase"] = "正在处理队列"
                        save_batch_state(batch)
            run_job(
                job_id,
                Path(str(task["source"])),
                str(task["original"]),
                Path(str(task["output"])),
                float(task.get("start_sec", 0.0)),
                task.get("end_sec"),
                str(task.get("mode", "director")),
                bool(task.get("release_gpu_after", True)),
            )
        except Exception as exc:
            add_log(job_id, f"队列任务异常：{type(exc).__name__}: {exc}")
            if job_id in JOBS:
                update_job(job_id, status="failed", status_label="失败", error=str(exc))
        finally:
            final_job = _job_from_state(job_id)
            if final_job and final_job.get("status") not in {"done", "failed", "cancelled"}:
                update_job(job_id, status="failed", status_label="失败", error="任务未正常结束")
            if batch_id:
                finish_batch_item(batch_id, job_id)
            JOB_QUEUE.task_done()


def start_queue_worker() -> None:
    global QUEUE_THREAD
    if QUEUE_THREAD and QUEUE_THREAD.is_alive():
        return
    QUEUE_THREAD = threading.Thread(
        target=analysis_queue_worker,
        name="media-analysis-queue",
        daemon=True,
    )
    QUEUE_THREAD.start()


class MultipartReader:
    def __init__(self, handler: BaseHTTPRequestHandler, content_length: int, boundary: bytes) -> None:
        self.handler = handler
        self.remaining = content_length
        self.boundary = boundary
        self.pending = bytearray()

    def read(self, size: int) -> bytes:
        if self.pending:
            data = bytes(self.pending[:size])
            del self.pending[:size]
            return data
        if self.remaining <= 0:
            return b""
        data = self.handler.rfile.read(min(size, self.remaining))
        if not data:
            raise ValueError("上传过程中连接中断")
        self.remaining -= len(data)
        return data

    def readline(self, max_size: int = 128 * 1024) -> bytes:
        line = bytearray()
        while len(line) < max_size:
            chunk = self.read(4096)
            if not chunk:
                return bytes(line)
            newline = chunk.find(b"\n")
            if newline < 0:
                line.extend(chunk)
                continue
            line.extend(chunk[: newline + 1])
            self.pending[:0] = chunk[newline + 1 :]
            return bytes(line)
        raise ValueError("multipart 请求头过长")

    def stream_part(self, target: Path | None, marker: bytes) -> bool:
        buffer = bytearray()
        stream = target.open("wb") if target else None
        try:
            while True:
                chunk = self.read(1024 * 1024)
                if not chunk:
                    raise ValueError("上传数据中没有结束边界")
                buffer.extend(chunk)
                position = buffer.find(marker)
                if position < 0:
                    keep = len(marker) + 2
                    if len(buffer) > keep:
                        if stream:
                            stream.write(buffer[:-keep])
                        del buffer[:-keep]
                    continue
                if stream:
                    stream.write(buffer[:position])
                tail = bytearray(buffer[position + len(marker) :])
                while len(tail) < 2:
                    extra = self.read(4096)
                    if not extra:
                        raise ValueError("multipart 边界不完整")
                    tail.extend(extra)
                if tail.startswith(b"--"):
                    rest = tail[2:]
                    if rest.startswith(b"\r\n"):
                        rest = rest[2:]
                    elif rest.startswith(b"\n"):
                        rest = rest[1:]
                    self.pending[:0] = rest
                    return True
                if tail.startswith(b"\r\n"):
                    self.pending[:0] = tail[2:]
                    return False
                if tail.startswith(b"\n"):
                    self.pending[:0] = tail[1:]
                    return False
                raise ValueError("无法解析 multipart 边界")
        finally:
            if stream:
                stream.close()


def create_uploads(handler: BaseHTTPRequestHandler, upload_id: str) -> list[tuple[str, Path]]:
    content_type = handler.headers.get("Content-Type", "")
    match = re.search(r"boundary=(?:\"([^\"]+)\"|([^;]+))", content_type)
    if "multipart/form-data" not in content_type or not match:
        raise ValueError("请求不是 multipart/form-data")
    try:
        content_length = int(handler.headers.get("Content-Length", "0"))
    except ValueError as exc:
        raise ValueError("缺少有效的 Content-Length") from exc
    if content_length <= 0 or content_length > MAX_UPLOAD_BYTES:
        raise ValueError("上传文件为空或超过 20GB 限制")

    boundary = (match.group(1) or match.group(2) or "").strip().encode("utf-8")
    reader = MultipartReader(handler, content_length, boundary)
    first_line = reader.readline().rstrip(b"\r\n")
    first_boundary = b"--" + boundary
    if first_line != first_boundary:
        raise ValueError("无法读取上传边界")

    marker = b"\r\n--" + boundary
    uploads: list[tuple[str, Path]] = []
    try:
        part_index = 0
        closed = False
        while not closed:
            headers: dict[str, str] = {}
            while True:
                line = reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                if b":" in line:
                    key, value = line.rstrip(b"\r\n").split(b":", 1)
                    headers[key.decode("latin-1").strip().lower()] = value.decode("utf-8", errors="replace").strip()
            if not headers:
                raise ValueError("multipart 文件头为空")
            disposition = headers.get("content-disposition", "")
            filename_match = re.search(r'filename="([^"]*)"', disposition, flags=re.IGNORECASE)
            if not filename_match:
                filename_match = re.search(r"filename=([^;]+)", disposition, flags=re.IGNORECASE)
            original = safe_name(filename_match.group(1).strip() if filename_match else "") if filename_match else ""
            target: Path | None = None
            if original:
                suffix = Path(original).suffix.lower()
                if suffix not in ALLOWED_SUFFIXES:
                    raise ValueError("只支持常见视频格式：MP4、MOV、MKV、AVI、WebM 等")
                target = UPLOAD_DIR / f"{upload_id}_{part_index:04d}_{original}"
                part_index += 1
            closed = reader.stream_part(target, marker)
            if target:
                if not target.exists() or target.stat().st_size == 0:
                    raise ValueError("上传文件为空")
                uploads.append((original, target))
        if not uploads:
            raise ValueError("未找到视频文件")
        return uploads
    except Exception:
        for _original, target in uploads:
            target.unlink(missing_ok=True)
        raise


def output_path_for(original: str, mode: str, start_sec: float, end_sec: float | None, job_id: str) -> Path:
    stem = safe_name(Path(original).stem, 80)
    range_label = ""
    if start_sec > 0 or end_sec is not None:
        range_label = (
            f"__片段_{start_sec:.2f}-{end_sec:.2f}"
            if end_sec is not None
            else f"__片段_{start_sec:.2f}-结尾"
        )
    output_label = "字幕提取" if mode == "subtitle" else "视频分析"
    return DEFAULT_OUTPUT_ROOT / (
        f"{stem}__{output_label}{range_label}__"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{job_id[:8]}"
    )


def make_job(
    job_id: str,
    original: str,
    source: Path,
    output: Path,
    start_sec: float,
    end_sec: float | None,
    mode: str,
    release_gpu_after: bool,
    batch_id: str | None = None,
    batch_index: int = 0,
    batch_total: int = 1,
) -> dict:
    job = {
        "id": job_id,
        "filename": original,
        "source_path": str(source),
        "output_path": str(output),
        "output_dir": to_windows_path(output),
        "status": "queued",
        "status_label": "排队中",
        "phase": "等待任务队列",
        "progress": 0,
        "logs": [],
        "start_sec": start_sec,
        "end_sec": end_sec,
        "mode": mode,
        "release_gpu_after": release_gpu_after,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "files": [],
    }
    if batch_id:
        job.update(
            {
                "batch_id": batch_id,
                "batch_index": batch_index,
                "batch_total": batch_total,
            }
        )
    return job


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
        if parsed.path == "/api/model":
            self.send_json(model_restart_snapshot())
            return
        if parsed.path == "/api/queue":
            with JOBS_LOCK:
                active = next(
                    (dict(item) for item in JOBS.values() if item.get("status") == "running"),
                    None,
                )
                queued = sum(item.get("status") == "queued" for item in JOBS.values())
            self.send_json(
                {
                    "queue_size": JOB_QUEUE.qsize(),
                    "queued_jobs": queued,
                    "queue": pending_queue_snapshot(),
                    "active_job": active,
                    "gpu": gpu_status(),
                    "model_restart": model_restart_snapshot(),
                }
            )
            return
        match = re.fullmatch(r"/api/batches/([^/]+)", parsed.path)
        if match:
            batch_id = unquote(match.group(1))
            result = batch_snapshot(batch_id)
            if not result:
                self.send_json({"error": "批量任务不存在"}, 404)
                return
            self.send_json(result)
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
                result["model_restart"] = model_restart_snapshot()
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
        if parsed.path == "/api/model/release":
            try:
                self.send_json(release_model_memory(), 202)
            except RuntimeError as exc:
                self.send_json({"error": str(exc), "model_restart": model_restart_snapshot()}, 409)
            return
        if parsed.path == "/api/model/restart":
            try:
                self.send_json(start_model_restart(), 202)
            except RuntimeError as exc:
                self.send_json({"error": str(exc), "model_restart": model_restart_snapshot()}, 409)
            return
        reorder_match = re.fullmatch(r"/api/batches/([^/]+)/reorder", parsed.path)
        if reorder_match:
            batch_id = unquote(reorder_match.group(1))
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > 2 * 1024 * 1024:
                    raise ValueError("队列顺序请求无效")
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                requested = payload.get("job_ids") if isinstance(payload, dict) else None
                if not isinstance(requested, list):
                    raise ValueError("缺少 job_ids 队列顺序")
                result = reorder_batch_queue(batch_id, [str(item) for item in requested])
                if result is None:
                    self.send_json({"error": "批量任务不存在"}, 404)
                else:
                    self.send_json(result)
            except (ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
                self.send_json({"error": str(exc)}, 400)
            return
        is_batch = parsed.path == "/api/batches"
        if parsed.path not in {"/api/jobs", "/api/batches"}:
            self.send_json({"error": "Not Found"}, 404)
            return
        request_id = uuid.uuid4().hex[:12]
        targets: list[Path] = []
        try:
            query = parse_qs(parsed.query)
            raw_start = query.get("start_sec", [""])[0].strip()
            raw_end = query.get("end_sec", [""])[0].strip()
            mode = query.get("mode", ["director"])[0].strip().lower()
            raw_release = query.get("release_gpu", ["1"])[0].strip().lower()
            release_gpu_after = raw_release not in {"0", "false", "no", "off"}
            if mode not in {"director", "subtitle"}:
                raise ValueError("分析模式无效")
            if is_batch:
                # 批量入口默认整片分析，避免把单视频的时间区间误套到整批素材上。
                start_sec = 0.0
                end_sec = None
            else:
                start_sec = float(raw_start) if raw_start else 0.0
                end_sec = float(raw_end) if raw_end else None
                if not math.isfinite(start_sec) or start_sec < 0:
                    raise ValueError("开始秒必须是大于等于 0 的数字")
                if end_sec is not None and (not math.isfinite(end_sec) or end_sec <= start_sec):
                    raise ValueError("结束秒必须大于开始秒")
            uploads = create_uploads(self, request_id)
            targets = [target for _original, target in uploads]
            if not is_batch and len(uploads) != 1:
                raise ValueError("单视频入口只能上传一个视频，请使用批量入口")
            if is_batch:
                batch_id = request_id
                batch = {
                    "id": batch_id,
                    "kind": "batch",
                    "mode": mode,
                    "mode_label": "视觉字幕提取" if mode == "subtitle" else "导演拉片分析",
                    "status": "queued",
                    "status_label": "排队中",
                    "phase": "等待任务队列",
                    "progress": 0,
                    "total": len(uploads),
                    "completed": 0,
                    "succeeded": 0,
                    "failed": 0,
                    "job_ids": [],
                    "continue_on_error": True,
                    "release_gpu_after": release_gpu_after,
                    "gpu_released": False,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "current_job_id": "",
                }
                with JOBS_LOCK:
                    BATCHES[batch_id] = batch
                    save_batch_state(batch)
                for index, (original, target) in enumerate(uploads, start=1):
                    child_id = uuid.uuid4().hex[:12]
                    output = output_path_for(original, mode, start_sec, end_sec, child_id)
                    output.mkdir(parents=True, exist_ok=True)
                    job = make_job(
                        child_id,
                        original,
                        target,
                        output,
                        start_sec,
                        end_sec,
                        mode,
                        release_gpu_after,
                        batch_id=batch_id,
                        batch_index=index,
                        batch_total=len(uploads),
                    )
                    with JOBS_LOCK:
                        JOBS[child_id] = job
                        BATCHES[batch_id]["job_ids"].append(child_id)
                        save_job_state(job)
                    # 批量任务只在整批结束后释放视觉模型，减少重复停启。
                    JOB_QUEUE.put(
                        {
                            "job_id": child_id,
                            "source": target,
                            "original": original,
                            "output": output,
                            "start_sec": start_sec,
                            "end_sec": end_sec,
                            "mode": mode,
                            "release_gpu_after": False,
                            "batch_id": batch_id,
                        }
                    )
                with JOBS_LOCK:
                    save_batch_state(BATCHES[batch_id])
                response = batch_snapshot(batch_id) or {"error": "批量任务创建失败"}
                targets = []
                self.send_json(response, 202)
                return

            original, target = uploads[0]
            job_id = request_id
            output = output_path_for(original, mode, start_sec, end_sec, job_id)
            output.mkdir(parents=True, exist_ok=True)
            job = make_job(job_id, original, target, output, start_sec, end_sec, mode, release_gpu_after)
            with JOBS_LOCK:
                JOBS[job_id] = job
                save_job_state(job)
            JOB_QUEUE.put(
                {
                    "job_id": job_id,
                    "source": target,
                    "original": original,
                    "output": output,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "mode": mode,
                    "release_gpu_after": release_gpu_after,
                }
            )
            response = {key: value for key, value in job.items() if key != "thread"}
            response["queue_size"] = JOB_QUEUE.qsize()
            response["gpu"] = gpu_status()
            targets = []
            self.send_json(response, 202)
        except Exception as exc:
            for target in targets:
                target.unlink(missing_ok=True)
            self.send_json({"error": str(exc)}, 400)


def main() -> None:
    parser = argparse.ArgumentParser(description="视频导演拉片分析 WebUI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7867)
    args = parser.parse_args()
    start_queue_worker()
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

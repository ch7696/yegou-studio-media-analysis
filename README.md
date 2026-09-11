# Media Analysis Vision Studio

一个独立的视频导演拉片分析工作台。上传一个视频后，WebUI 按固定流水线生成视觉证据、ASR/时间戳和代码综合时间线；原项目 `/home/administrator/ragflow-managed` 保持不变。

## 固定工作流

```text
上传视频
  → 可选截取（开始秒 / 结束秒）
  → 阶段 1：Qwen3-ASR + Qwen3-ForcedAligner
  → 阶段 2：MiniCPM-V（1 秒 1 帧、20 秒上下文、5 秒细节组）
  → 阶段 3：代码按时间区间合并视觉与语音
  → 下载四份主文档和原始机器文件
```

ASR 和视觉服务按顺序使用 GPU；同一个 WebUI 进程内的多个任务会排队，避免任务之间互相停止或重启 VLM 容器。默认在任务完成或异常后停止视觉容器并释放显存；WebUI 可以取消这个选项以保留热模型。页面会实时显示阶段、百分比、日志，以及 `nvidia-smi` 的 GPU 利用率、显存、温度和功耗。

## 四份主文档

每个任务目录默认位于 `C:\Users\Administrator\Desktop\media_analysis`（WSL 路径：`/mnt/c/Users/Administrator/Desktop/media_analysis`），包含：

- `01_纯视觉分析.md`：只包含视觉模型的导演拉片文字。
- `02_纯ASR与时间戳.md`：完整转写、分段时间码和模型信息。
- `03_代码综合时间线.md`：代码将 20 秒视觉区间、其中的 5 秒高清细节段与 ForcedAligner 语音区间按相交时间合并，不让模型篡改原文。
- `04_最终导演分析.md`：最终解释层模板；可以下载前三份交给 Codex、其他大模型或人工继续完成。

同时保留 `视觉分析文本汇总.md`、`视觉片段索引.jsonl`、逐秒高清帧、20 秒联系图、ASR JSON/SRT/VTT/Markdown、`audio_records.jsonl` 和 `分析清单.json`，便于复核和二次加工。无音频的视频也会生成四份文档，并在 ASR 文档中明确标记“未识别到语音”。

## 本地启动

先确认 Docker、NVIDIA Container Toolkit 和模型目录可用，然后建立软链接：

```bash
cp .env.example .env
./scripts/link_models.sh
./scripts/run_model.sh
./scripts/run_webui.sh
```

默认打开 <http://localhost:7877>。`.env` 可调整输出目录、WebUI 端口、模型路径、ASR 镜像、`MINICPM_MAX_MODEL_LEN`（默认 `32768`）、GPU 显存比例、每次请求的最大图片数和 `RELEASE_GPU_AFTER_JOB`。

模型目录由 `MODEL_ROOT` 提供，至少需要：

```text
${MODEL_ROOT}/MiniCPM-V-4_5-GPTQ
${MODEL_ROOT}/Qwen3-ASR-0.6B
${MODEL_ROOT}/Qwen3-ForcedAligner-0.6B
```

`scripts/link_models.sh` 只在新仓库的 `models/` 下创建软链接，不复制模型权重；模型权重不会提交 Git。

## 直接运行固定流水线

WebUI 调用的是仓库根目录的 `media_analysis_pipeline.py`。也可以直接运行：

```bash
python3 media_analysis_pipeline.py \
  --source /path/to/video.mp4 \
  --output /path/to/output
```

只处理前 60 秒：

```bash
python3 media_analysis_pipeline.py \
  --source /path/to/video.mp4 \
  --output /path/to/output \
  --max-duration 60
```

已有视觉和 ASR 产物时，可用 `--package-only` 重新生成四份文档；已有最终稿可通过 `--final-source /path/to/final.md` 写入第四份文档。

## 云端部署方向

`deploy/compose.gpu.yml` 提供 MiniCPM-V 的 GPU 服务骨架，默认使用 32k 上下文、单序列和每请求最多 8 张图片，适合 4090/5090 + 96GB RAM 环境。WebUI 目前建议与 Docker 宿主机同节点运行，因为流水线需要按任务启动 ASR worker、切片容器并访问共享模型软链接；`deploy/Dockerfile.web` 已包含完整 Python 流水线文件，后续如把 WebUI 容器化，需要额外挂载 Docker socket、模型卷、输入卷、输出卷，并把 `MODEL_ROOT` 改成容器可见路径。

## 目录说明

```text
app/
  media_analysis_webui.py       WebUI、上传、时间轴、日志、进度和 GPU 监控
  director_detail_analyzer.py   1fps/20秒/5秒视觉分析
  vision_core.py                MiniCPM-V API 和视频工具
media_analysis_pipeline.py      ASR → 视觉 → 四文档打包总控
audio_timeline.py               Qwen3-ASR + ForcedAligner worker
asset_naming.py                 音频产物命名工具
scripts/
  link_models.sh                创建模型软链接
  run_model.sh                  启动 MiniCPM-V
  run_webui.sh                  启动 WebUI
deploy/                         GPU compose 和 WebUI 镜像骨架
```

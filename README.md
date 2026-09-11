# Media Analysis Vision Studio

一个独立的纯视觉视频导演拉片 WebUI。它从视频生成逐秒高清帧、20 秒上下文联系图、5 秒细节帧和带时间线的视觉分析报告。

本仓库是从本机托管目录整理出的独立副本；原目录 `/home/administrator/ragflow-managed` 不属于本项目，也不会被修改。

## 当前流程

```text
视频上传 → 可视时间轴截取 → 1fps 高清帧
                         → 20 秒联系图上下文
                         → 5 秒细节分析
                         → 纯视觉报告与证据索引
```

当前唯一模型：`MiniCPM-V-4_5-GPTQ`。

## 本地启动

先建立模型软链接：

```bash
./scripts/link_models.sh
```

启动 MiniCPM-V 服务：

```bash
./scripts/run_model.sh
```

另开终端启动 WebUI：

```bash
./scripts/run_webui.sh
```

浏览器打开 <http://localhost:7877>（如需使用其他端口，修改 `.env`）。

## 输出

默认输出由 `MEDIA_OUTPUT_ROOT` 控制。每个任务包含：

- `01_纯视觉分析.md`
- `视觉分析文本汇总.md`
- `视觉片段索引.jsonl`
- `01_每秒高清帧/`
- `02_20秒联系图/`
- `03_模型文字分析/`
- `分析清单.json`

视频、任务状态、模型权重和生成结果均不提交到 Git。

## 云端方向

云端使用 4090/5090 GPU 时，可以让 MiniCPM-V 常驻，不需要 ASR、RAGFlow、OCR、Embedding 或 CosyVoice。`deploy/compose.gpu.yml` 提供视觉模型服务的 GPU 启动骨架；WebUI 当前仍按宿主机模式运行，便于复用 Docker 中的 FFmpeg 和模型容器。正式云部署时，把 `MODEL_ROOT` 换成持久化模型卷，并为 WebUI 配置独立的输入、输出和状态卷。

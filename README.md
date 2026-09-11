# 野构 Studio · 视频导演拉片分析工作台

<p align="center">
  <img src="assets/brand/yegou-studio-logo.png" width="260" alt="野构 Studio 创意工作室标志">
</p>

<p align="center"><strong>视频导演拉片分析工作台</strong><br>面向视频创作、研究与内容生产的视觉与声音时间线分析工具。</p>

## 项目简介

本项目用于对视频进行视觉镜头分析、语音识别、强制对齐与时间线整合，生成可复核、可下载的导演拉片资料。系统提供 WebUI 操作界面，支持视频上传、区间截取、任务进度与 GPU 状态查看。

## 核心能力

- 按秒提取高清视频帧，并以连续时间上下文进行视觉分析。
- 使用 ASR 与 ForcedAligner 生成带时间戳的语音文本。
- 合并视觉信息与语音时间线，形成统一的镜头分析记录。
- 输出 Markdown、JSONL、SRT、VTT 及逐帧图像等复核资料。
- 支持任务进度、运行日志、GPU 状态与分析完成后的显存释放。

## 分析输出

每个任务生成以下主要文档：

| 文件 | 内容 |
| --- | --- |
| `01_纯视觉分析.md` | 视觉模型生成的时间线分析 |
| `02_纯ASR与时间戳.md` | 语音识别、分段文本与时间戳 |
| `03_代码综合时间线.md` | 视觉信息与语音时间线的程序化合并结果 |
| `04_最终导演分析.md` | 面向后续创作与研究的综合分析文档 |

任务目录同时保留原始帧、联系图、索引文件及 ASR 中间结果，便于复核和二次处理。

## 使用方式

运行环境需要 Docker、NVIDIA Container Toolkit 及本地模型文件。首次使用时执行：

```bash
cp .env.example .env
./scripts/link_models.sh
./scripts/run_model.sh
./scripts/run_webui.sh
```

启动后访问 <http://localhost:7877>。模型目录由 `MODEL_ROOT` 配置，目录结构如下：

```text
${MODEL_ROOT}/MiniCPM-V-4_5-GPTQ
${MODEL_ROOT}/Qwen3-ASR-0.6B
${MODEL_ROOT}/Qwen3-ForcedAligner-0.6B
```

也可以直接运行完整流水线：

```bash
python3 media_analysis_pipeline.py \
  --source /path/to/video.mp4 \
  --output /path/to/output
```

使用 `--max-duration 60` 可限制分析时长；使用 `--package-only` 可基于已有产物重新生成文档。

## 项目结构

```text
app/                         WebUI 与视觉分析模块
media_analysis_pipeline.py   分析流程总控
audio_timeline.py            ASR 与 ForcedAligner 流程
scripts/                     模型链接与服务启动脚本
deploy/                      GPU 部署配置
assets/brand/                野构 Studio 品牌素材
```

## 许可证与品牌

本项目源代码采用 [Apache License 2.0](LICENSE)。`assets/brand/` 中的野构 Studio 标识、Logo 及相关品牌素材不随 Apache-2.0 授权，具体说明见 [NOTICE](NOTICE)。

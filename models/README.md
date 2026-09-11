# 模型目录

模型权重不提交到 Git。运行以下命令，会在本目录创建指向本机模型目录的软链接：

```bash
./scripts/link_models.sh
```

默认模型链接：

```text
models/MiniCPM-V-4_5-GPTQ -> ${MODEL_ROOT}/MiniCPM-V-4_5-GPTQ
models/Qwen3-ASR-0.6B -> ${MODEL_ROOT}/Qwen3-ASR-0.6B
models/Qwen3-ForcedAligner-0.6B -> ${MODEL_ROOT}/Qwen3-ForcedAligner-0.6B
```

通过 `MODEL_ROOT` 指定模型根目录。模型权重不提交至代码仓库。

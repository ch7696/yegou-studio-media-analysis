# 本地模型链接

模型权重不提交到 Git。运行以下命令，会在本目录创建指向本机模型目录的软链接：

```bash
./scripts/link_models.sh
```

默认链接：

```text
models/MiniCPM-V-4_5-GPTQ -> /home/administrator/models/MiniCPM-V-4_5-GPTQ
models/Qwen3-ASR-0.6B -> /home/administrator/models/Qwen3-ASR-0.6B
models/Qwen3-ForcedAligner-0.6B -> /home/administrator/models/Qwen3-ForcedAligner-0.6B
```

可以通过 `MODEL_ROOT` 指定另一份模型根目录。云端部署时改用持久化模型卷，不要把权重复制进代码仓库。

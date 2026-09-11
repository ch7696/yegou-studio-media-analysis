#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi
MODEL_LINK="$REPO_ROOT/models/MiniCPM-V-4_5-GPTQ"
MODEL_NAME_VALUE="${MINICPM_MODEL:-MiniCPM-V-4_5-GPTQ}"
CONTAINER_NAME="${MINICPM_CONTAINER:-vision-minicpm}"
PORT_VALUE="${MINICPM_PORT:-8002}"
IMAGE_VALUE="${MINICPM_IMAGE:-swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/vllm/vllm-openai:v0.26.0}"

if [[ ! -L "$MODEL_LINK" ]]; then
  echo "请先运行：$REPO_ROOT/scripts/link_models.sh" >&2
  exit 1
fi
if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" == "true" ]]; then
    echo "MiniCPM-V 已运行：$CONTAINER_NAME"
    exit 0
  fi
  docker start "$CONTAINER_NAME" >/dev/null
  echo "已启动已有容器：$CONTAINER_NAME"
  exit 0
fi

docker run -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc=host \
  --shm-size=8g \
  -p "$PORT_VALUE:8000" \
  -e HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  -e VLLM_USE_V2_MODEL_RUNNER=0 \
  -v "$MODEL_LINK:/models/minicpm:ro" \
  "$IMAGE_VALUE" \
  --model /models/minicpm \
  --served-model-name "$MODEL_NAME_VALUE" \
  --trust-remote-code \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.70 \
  --max-num-seqs 1 \
  --limit-mm-per-prompt '{"image":1}' \
  --enforce-eager \
  --no-enable-prefix-caching \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key x

echo "MiniCPM-V 容器已启动：$CONTAINER_NAME，API 端口：$PORT_VALUE"

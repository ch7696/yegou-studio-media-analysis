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
MAX_MODEL_LEN_VALUE="${MINICPM_MAX_MODEL_LEN:-32768}"
GPU_MEMORY_VALUE="${MINICPM_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_IMAGES_VALUE="${MINICPM_MAX_IMAGES:-8}"
HEALTH_URL_VALUE="${MINICPM_HEALTH_URL:-http://127.0.0.1:${PORT_VALUE}/health}"
STARTUP_TIMEOUT_VALUE="${MINICPM_STARTUP_TIMEOUT:-600}"

if [[ ! -L "$MODEL_LINK" ]]; then
  echo "请先运行：$REPO_ROOT/scripts/link_models.sh" >&2
  exit 1
fi

wait_for_api() {
  local elapsed=0
  echo "等待 MiniCPM-V API 就绪：$HEALTH_URL_VALUE"
  while (( elapsed < STARTUP_TIMEOUT_VALUE )); do
    if curl -fsS --max-time 3 "$HEALTH_URL_VALUE" >/dev/null 2>&1; then
      echo "MiniCPM-V API 已就绪：$HEALTH_URL_VALUE"
      return 0
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  echo "等待 MiniCPM-V API 超时（${STARTUP_TIMEOUT_VALUE}s）：$HEALTH_URL_VALUE" >&2
  docker logs --tail 100 "$CONTAINER_NAME" >&2 || true
  return 1
}

if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  if [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME")" == "true" ]]; then
    wait_for_api
    echo "MiniCPM-V 已运行：$CONTAINER_NAME"
    exit 0
  fi
  docker start "$CONTAINER_NAME" >/dev/null
  echo "已启动已有容器：$CONTAINER_NAME"
  wait_for_api
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
  --max-model-len "$MAX_MODEL_LEN_VALUE" \
  --gpu-memory-utilization "$GPU_MEMORY_VALUE" \
  --max-num-seqs 1 \
  --limit-mm-per-prompt "{\"image\":${MAX_IMAGES_VALUE}}" \
  --enforce-eager \
  --no-enable-prefix-caching \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key x

wait_for_api
echo "MiniCPM-V 容器已启动：$CONTAINER_NAME，API 端口：$PORT_VALUE"

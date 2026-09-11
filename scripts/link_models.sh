#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi
MODEL_ROOT_VALUE="${MODEL_ROOT:-${REPO_ROOT}/../models}"

link_model() {
  local model_name="$1"
  local source="${MODEL_ROOT_VALUE}/${model_name}"
  local target="${REPO_ROOT}/models/${model_name}"

  if [[ ! -d "$source" ]]; then
    echo "模型目录不存在：$source" >&2
    exit 1
  fi
  if [[ -e "$target" && ! -L "$target" ]]; then
    echo "目标已存在且不是软链接：$target" >&2
    exit 1
  fi
  if [[ -L "$target" ]]; then
    local current
    current="$(readlink "$target")"
    if [[ "$current" != "$source" ]]; then
      echo "已有软链接指向其他目录：$target -> $current" >&2
      exit 1
    fi
  else
    ln -s "$source" "$target"
  fi
  echo "模型软链接：$target -> $source"
}

link_model "MiniCPM-V-4_5-GPTQ"
link_model "Qwen3-ASR-0.6B"
link_model "Qwen3-ForcedAligner-0.6B"

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi
MODEL_ROOT_VALUE="${MODEL_ROOT:-/home/administrator/models}"
SOURCE="${MODEL_ROOT_VALUE}/MiniCPM-V-4_5-GPTQ"
TARGET="${REPO_ROOT}/models/MiniCPM-V-4_5-GPTQ"

if [[ ! -d "$SOURCE" ]]; then
  echo "模型目录不存在：$SOURCE" >&2
  exit 1
fi
if [[ -e "$TARGET" && ! -L "$TARGET" ]]; then
  echo "目标已存在且不是软链接：$TARGET" >&2
  exit 1
fi
if [[ -L "$TARGET" ]]; then
  CURRENT="$(readlink "$TARGET")"
  if [[ "$CURRENT" != "$SOURCE" ]]; then
    echo "已有软链接指向其他目录：$TARGET -> $CURRENT" >&2
    exit 1
  fi
else
  ln -s "$SOURCE" "$TARGET"
fi

echo "模型软链接：$TARGET -> $SOURCE"

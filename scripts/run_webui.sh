#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi
exec python3 app/media_analysis_webui.py --host "${WEBUI_HOST:-0.0.0.0}" --port "${WEBUI_PORT:-7877}"

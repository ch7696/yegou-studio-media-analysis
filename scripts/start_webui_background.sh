#!/usr/bin/env bash
set -u

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_file="${YEGOU_WEBUI_LOG:-/tmp/yegou-media-analysis-webui.log}"

if ! touch "$log_file" 2>/dev/null; then
    printf '[YEGOU] cannot create log file: %s\n' "$log_file" >&2
    exit 1
fi

printf '[YEGOU] launcher started: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" >> "$log_file"

if ! cd "$repo_root"; then
    printf '[YEGOU] cannot cd to %s\n' "$repo_root" >> "$log_file"
    exit 1
fi

if command -v setsid >/dev/null 2>&1; then
    setsid -f bash ./scripts/run_webui.sh >>"$log_file" 2>&1 < /dev/null
else
    nohup bash ./scripts/run_webui.sh >>"$log_file" 2>&1 < /dev/null &
fi
printf '[YEGOU] WebUI process launch requested\n' >> "$log_file"
exit 0

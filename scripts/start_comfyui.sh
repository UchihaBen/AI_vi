#!/usr/bin/env bash
# Chạy ComfyUI server. Nên để trong tmux/screen hoặc systemd (xem README).
#   COMFY_HOST=0.0.0.0 để mở cho máy khác truy cập (cẩn thận: không có mật khẩu)
#   COMFY_ARGS="--lowvram" nếu vẫn hết VRAM
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMFY_DIR="${COMFY_DIR:-$REPO/comfyui}"
cd "$COMFY_DIR"
exec ./venv/bin/python main.py --listen "${COMFY_HOST:-127.0.0.1}" --port "${COMFY_PORT:-8188}" \
  --preview-method none ${COMFY_ARGS:-}

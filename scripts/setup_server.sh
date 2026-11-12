#!/usr/bin/env bash
# Cài toàn bộ trên Ubuntu 22.04 + NVIDIA GPU: ffmpeg, ComfyUI (+GGUF, Manager), venv cho pipeline.
# Chạy:  bash scripts/setup_server.sh
# Tùy chọn môi trường:
#   COMFY_DIR   thư mục cài ComfyUI (mặc định ./comfyui)
#   PYTHON      trình python dùng tạo venv (mặc định python3; nên >= 3.10)
#   TORCH_INDEX index PyTorch (mặc định cu128; driver cũ hơn 570 thì dùng .../whl/cu124)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMFY_DIR="${COMFY_DIR:-$REPO/comfyui}"
PYTHON="${PYTHON:-python3}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"

echo "==> Gói hệ thống"
sudo apt-get update
sudo apt-get install -y git ffmpeg python3-venv python3-pip fonts-dejavu-core

if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "!! Không thấy nvidia-smi. Cài driver NVIDIA trước (sudo ubuntu-drivers install)." >&2
  exit 1
fi

echo "==> ComfyUI -> $COMFY_DIR"
if [ -d "$COMFY_DIR/.git" ]; then
  git -C "$COMFY_DIR" pull --ff-only
else
  git clone https://github.com/comfyanonymous/ComfyUI "$COMFY_DIR"
fi
[ -d "$COMFY_DIR/venv" ] || "$PYTHON" -m venv "$COMFY_DIR/venv"
CPIP="$COMFY_DIR/venv/bin/pip"
"$CPIP" install --upgrade pip wheel
"$CPIP" install torch torchvision torchaudio --index-url "$TORCH_INDEX"
"$CPIP" install -r "$COMFY_DIR/requirements.txt"

clone_node() {  # url dir
  local dest="$COMFY_DIR/custom_nodes/$2"
  if [ -d "$dest/.git" ]; then git -C "$dest" pull --ff-only; else git clone --depth 1 "$1" "$dest"; fi
  [ -f "$dest/requirements.txt" ] && "$CPIP" install -r "$dest/requirements.txt" || true
}
echo "==> Custom nodes"
clone_node https://github.com/city96/ComfyUI-GGUF ComfyUI-GGUF
clone_node https://github.com/ltdrdata/ComfyUI-Manager ComfyUI-Manager
# cho chế độ dance (Wan 2.2 Animate): load ảnh theo thư mục, DWPose, crop mặt
clone_node https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite ComfyUI-VideoHelperSuite
clone_node https://github.com/Fannovel16/comfyui_controlnet_aux comfyui_controlnet_aux
clone_node https://github.com/kijai/ComfyUI-KJNodes ComfyUI-KJNodes
clone_node https://github.com/kijai/ComfyUI-WanVideoWrapper ComfyUI-WanVideoWrapper

echo "==> venv cho pipeline"
[ -d "$REPO/.venv" ] || "$PYTHON" -m venv "$REPO/.venv"
"$REPO/.venv/bin/pip" install --upgrade pip
"$REPO/.venv/bin/pip" install -r "$REPO/requirements.txt"

[ -f "$REPO/.env" ] || { cp "$REPO/.env.example" "$REPO/.env"; echo "==> Đã tạo .env, nhớ điền TTS_API_KEY"; }

"$COMFY_DIR/venv/bin/python" -c "import torch;print('torch',torch.__version__,'CUDA',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo
echo "Xong. Bước tiếp:"
echo "  .venv/bin/python scripts/download_models.py     # tải model (~35 GB)"
echo "  bash scripts/start_comfyui.sh                   # chạy ComfyUI (để trong tmux)"
echo "  .venv/bin/python -m pipeline check"

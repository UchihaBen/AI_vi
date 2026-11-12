#!/usr/bin/env bash
# Train LoRA SDXL cho một nhân vật bằng kohya-ss/sd-scripts (vừa 12 GB VRAM).
#   bash scripts/train_lora_sdxl.sh <character_id> [epochs]
# Dữ liệu: datasets/<character_id>/img/10_<trigger>/*.png + *.txt (tạo bằng `python -m pipeline charsheet`
# hoặc tự bỏ ảnh vào; mỗi ảnh kèm file .txt caption bắt đầu bằng trigger word).
# Kết quả: comfyui/models/loras/<character_id>_v1.safetensors -> khai báo lora: trong project.yaml
#
# Tắt ComfyUI trước khi train để giải phóng VRAM.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMFY_DIR="${COMFY_DIR:-$REPO/comfyui}"
CHAR="${1:?Cần character_id}"
EPOCHS="${2:-10}"
PYTHON="${PYTHON:-python3}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
SD="$REPO/tools/sd-scripts"
BASE="${BASE_MODEL:-$COMFY_DIR/models/checkpoints/RealVisXL_V5.0_fp16.safetensors}"
DATA="$REPO/datasets/$CHAR/img"

[ -d "$DATA" ] || { echo "Không có $DATA" >&2; exit 1; }
[ -f "$BASE" ] || { echo "Không có model gốc $BASE" >&2; exit 1; }

if [ ! -d "$SD/.git" ]; then
  git clone --depth 1 https://github.com/kohya-ss/sd-scripts "$SD"
fi
if [ ! -d "$SD/venv" ]; then
  "$PYTHON" -m venv "$SD/venv"
  "$SD/venv/bin/pip" install --upgrade pip wheel
  "$SD/venv/bin/pip" install torch torchvision --index-url "$TORCH_INDEX"
  (cd "$SD" && ./venv/bin/pip install -r requirements.txt)
  "$SD/venv/bin/pip" install bitsandbytes
fi

N=$(find "$DATA" -type f \( -name '*.png' -o -name '*.jpg' -o -name '*.jpeg' -o -name '*.webp' \) | wc -l)
echo "==> $N ảnh trong $DATA, $EPOCHS epoch"
[ "$N" -ge 10 ] || echo "!! Nên có ít nhất 15-30 ảnh cùng một khuôn mặt để LoRA ổn định"

cd "$SD"
./venv/bin/accelerate launch --num_processes 1 --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
  sdxl_train_network.py \
  --pretrained_model_name_or_path "$BASE" \
  --train_data_dir "$DATA" \
  --output_dir "$COMFY_DIR/models/loras" \
  --output_name "${CHAR}_v1" \
  --save_model_as safetensors \
  --resolution 1024,1024 --enable_bucket --min_bucket_reso 512 --max_bucket_reso 1536 \
  --caption_extension .txt \
  --network_module networks.lora --network_dim 32 --network_alpha 16 \
  --network_train_unet_only --cache_text_encoder_outputs \
  --learning_rate 1e-4 --unet_lr 1e-4 --lr_scheduler cosine --lr_warmup_steps 50 \
  --optimizer_type AdamW8bit \
  --max_train_epochs "$EPOCHS" --save_every_n_epochs 2 \
  --train_batch_size 1 --mixed_precision bf16 --save_precision fp16 --no_half_vae \
  --gradient_checkpointing --cache_latents --sdpa --seed 42

echo "Xong: $COMFY_DIR/models/loras/${CHAR}_v1.safetensors (các bản -0000xx là checkpoint giữa chừng, thử để chọn bản đẹp nhất)"

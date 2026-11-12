# AI Video Pipeline (local, RTX 4070 12GB)

Pipeline tạo video ngắn 10–25 giây bằng AI, chạy local trên Ubuntu với ComfyUI. Có hai chế độ:

| Chế độ | Lệnh | Đầu vào | Công nghệ |
|---|---|---|---|
| **Dance**: nhân vật nhảy theo video mẫu | `python -m pipeline dance` | 1 ảnh nhân vật + 1 video nhảy | Wan 2.2 Animate 14B (GGUF) + DWPose |
| **Phim**: nhiều shot, có thoại | `python -m pipeline run` | `project.yaml` (kịch bản theo shot) | SDXL + LoRA → Wan 2.2 I2V 14B → VieNeu TTS → FFmpeg |

Giọng nói dùng VieNeu TTS qua API (`TTS_BASE_URL`). Phần này không tốn GPU của server.

```
pipeline/
  cli.py        lệnh: check, voices, tts, plan, run, dance, charsheet
  comfy.py      client ComfyUI API (queue → chờ → tải kết quả)
  workflows.py  workflow ComfyUI dựng bằng code (SDXL, Wan I2V/FLF2V, Wan Animate)
  stages.py     các bước của chế độ phim: keyframes → video → voice → assemble
  dance.py      chế độ dance (chia đoạn, nối chuyển động)
  tts.py        client VieNeu TTS
  media.py      FFmpeg: ghép, chèn thoại, nhạc nền, phụ đề
  llm.py        truyện → project.yaml nháp (LLM tương thích OpenAI)
config/default.yaml   toàn bộ tham số (ghi đè bằng config/local.yaml)
scripts/              cài server, tải model, chạy ComfyUI, train LoRA
projects/             project mẫu: demo/ (phim), dance_demo/ (nhảy)
```

---

## 1. Cài đặt trên server

Yêu cầu: Ubuntu 22.04, driver NVIDIA (`nvidia-smi` chạy được), khoảng 60 GB ổ trống, RAM từ 32 GB (khuyên dùng 64 GB vì model 14B phải tràn sang RAM).

```bash
git clone <repo-của-bạn> ai-video && cd ai-video
bash scripts/setup_server.sh              # ffmpeg, ComfyUI, custom nodes, venv
nano .env                                  # điền TTS_API_KEY
.venv/bin/python scripts/download_models.py               # tất cả (~45 GB)
#   hoặc chỉ tải phần cần:  --only dance  |  --only video  |  --only image
```

Chạy ComfyUI ở một cửa sổ riêng, luôn để nó chạy:

```bash
tmux new -s comfy 'bash scripts/start_comfyui.sh'
```

Kiểm tra mọi thứ:

```bash
.venv/bin/python -m pipeline check
```

Lệnh `check` báo `[OK]` hoặc `[LỖI]` cho từng mục: ffmpeg, ComfyUI, từng node, từng file model và TTS key.

> Driver NVIDIA cũ hơn 570: chạy `TORCH_INDEX=https://download.pytorch.org/whl/cu124 bash scripts/setup_server.sh`.
> ComfyUI và pipeline phải chạy trên **cùng một máy**, vì chế độ dance truyền đường dẫn thư mục frame cho ComfyUI.

---

## 2. Chế độ Dance: nhân vật nhảy TikTok

1. Bỏ file vào `projects/dance_demo/assets/`:
   - `character.png`: ảnh nhân vật **toàn thân**, đứng thẳng, thấy rõ mặt và trang phục, khung dọc.
   - `dance.mp4`: video nhảy mẫu có một người, toàn thân trong khung, camera đứng yên.
2. Sửa `projects/dance_demo/dance.yaml` (start, duration, prompt).
3. Chạy:

```bash
.venv/bin/python -m pipeline dance projects/dance_demo/dance.yaml
```

Pipeline thực hiện lần lượt:

1. Chuẩn hóa video mẫu về 16 fps, 480x832.
2. Tách khung xương (DWPose) và khuôn mặt.
3. Sinh từng đoạn 77 frame (khoảng 4,8 giây). Mỗi đoạn nối tiếp đoạn trước bằng 5 frame chồng lên nhau, nên chuyển động liền mạch.
4. Ghép các đoạn, nội suy lên 24 fps, upscale lên 720x1280 và gắn nhạc của video mẫu.

Kết quả nằm ở `outputs/dance_demo/final.mp4`.

Nếu bị dừng giữa chừng, chạy lại đúng lệnh cũ: các đoạn đã xong sẽ được bỏ qua. Muốn làm lại từ đầu thì thêm `--force`.

**Mẹo để ra đẹp**
- Tư thế trong ảnh nhân vật nên giống frame đầu của video mẫu (cùng đứng thẳng, cùng khoảng cách tới camera).
- Video mẫu có nền gọn, quần áo không quá rộng, động tác không bị che sẽ cho pose chính xác hơn.
- Động tác tay quan trọng: đặt `dance.detect_hand: true` trong `config/local.yaml`.
- Chỉ dùng video mẫu bạn quay hoặc có quyền sử dụng. Nhạc TikTok thường có bản quyền, nên khi đăng có thể cần thay bằng `music:`.

---

## 3. Chế độ Phim: nhiều shot, có thoại

Xem mẫu ở `projects/demo/project.yaml`. Mỗi shot dài khoảng 5 giây, có một hành động, kèm thoại tiếng Việt nếu cần.

```bash
# (tùy chọn) để LLM viết bản nháp từ truyện
.venv/bin/python -m pipeline plan story.txt --seconds 20 --aspect 9:16

# chạy toàn bộ
.venv/bin/python -m pipeline run projects/demo/project.yaml

# hoặc từng bước, để duyệt keyframe trước khi tốn thời gian render video
.venv/bin/python -m pipeline run projects/demo/project.yaml --stages keyframes
#   xem outputs/demo/s01/keyframe_cand0..2.png, chọn bằng keyframe_pick: 1 trong yaml,
#   hoặc chép đè thẳng ảnh mình muốn vào keyframe.png
.venv/bin/python -m pipeline run projects/demo/project.yaml --stages video,voice,assemble
# làm lại riêng một shot
.venv/bin/python -m pipeline run projects/demo/project.yaml --stages video --shots s02 --force
```

Mỗi bước đều bỏ qua phần đã có kết quả. Vì vậy có thể sửa một shot rồi chạy lại `assemble` mà không phải render lại cả phim.

Một số tính năng trong `project.yaml`:
- `end_image_prompt`: đặt cả khung đầu và khung cuối (Wan First-Last-Frame) để khống chế điểm kết thúc của hành động.
- `image`: dùng ảnh có sẵn thay cho ảnh sinh.
- `dialogue`: nhiều câu trong một shot, mỗi câu có `start` (giây). Nếu thoại dài hơn clip, hình sẽ đứng ở frame cuối cho đến khi hết thoại.
- `bgm`: nhạc nền, tự nhỏ lại khi có thoại. `subtitles`: `burn`, `srt` hoặc `none`.

Giọng nói:

```bash
.venv/bin/python -m pipeline voices                    # danh sách giọng
.venv/bin/python -m pipeline tts "Xin chào" --voice <tên> -o test.wav
```

Đặt `voice: <tên>` cho từng nhân vật trong `project.yaml`.

---

## 4. Giữ nhân vật nhất quán (LoRA)

Mô tả bằng chữ thôi thì mỗi shot sẽ ra một khuôn mặt hơi khác. Muốn giữ mặt ổn định, hãy train LoRA cho nhân vật:

```bash
# 1. lấy 15-30 ảnh của CÙNG một nhân vật bỏ vào datasets/<id>/img/10_<trigger>/
#    (mỗi ảnh kèm file .txt caption bắt đầu bằng trigger), hoặc sinh ứng viên:
.venv/bin/python -m pipeline charsheet projects/demo/project.yaml linh --count 40
#    rồi XÓA những ảnh không giống nhau. Sinh từ chữ nên mặt không đồng nhất 100%,
#    phải lọc tay. Ảnh thật hoặc ảnh nhân vật bạn đã có là tốt nhất.
# 2. tắt ComfyUI (Ctrl+C trong tmux) để trả VRAM, rồi train (~1-3 giờ trên 4070)
bash scripts/train_lora_sdxl.sh linh
# 3. khai báo trong project.yaml:  lora: linh_v1.safetensors  /  lora_strength: 0.85
```

Chế độ dance **không cần LoRA**, vì ngoại hình được lấy trực tiếp từ ảnh nhân vật.

---

## 5. Cấu hình

Mọi tham số nằm trong `config/default.yaml`. Để sửa trên server, tạo file `config/local.yaml` (không commit) chỉ chứa phần muốn đổi:

```yaml
video:
  lightning: false        # chất lượng cao hơn, chậm hơn khoảng 5 lần
  sizes: {"9:16": [720, 1280]}
dance:
  chunk_length: 49        # giảm nếu hết VRAM
```

Chạy ComfyUI như một service (tùy chọn):

```ini
# /etc/systemd/system/comfyui.service
[Unit]
Description=ComfyUI
After=network.target
[Service]
User=<user>
WorkingDirectory=/path/to/ai-video
ExecStart=/bin/bash scripts/start_comfyui.sh
Restart=on-failure
[Install]
WantedBy=multi-user.target
```

---

## 6. Thời gian và VRAM (ước lượng cho RTX 4070 12GB)

| Việc | Thời gian ước lượng |
|---|---|
| Keyframe SDXL 1344x768, 3 ảnh | ~1 phút |
| Clip Wan 2.2 I2V 5 giây, 832x480, lightning 4 bước | ~5–10 phút |
| Dance 15 giây (4 đoạn), 480x832 | ~20–40 phút |
| TTS một câu | vài giây (server ngoài) |

**Hết VRAM (OOM)**: lần lượt thử `dance.chunk_length: 49`, chạy ComfyUI với `COMFY_ARGS="--lowvram"`, rồi dùng quant thấp hơn (`download_models.py --quant Q3_K_M` và đổi tên file trong `local.yaml`).

---

## 7. Giấy phép model (kiểm tra lại trước khi dùng thương mại)

- Wan 2.2 (I2V, Animate): Apache 2.0.
- RealVisXL V5 (SDXL): CreativeML OpenRAIL++-M.
- LightX2V LoRA: xem model card trên HuggingFace.
- DWPose: Apache 2.0.

## 8. Trạng thái kiểm thử

- Đã test trong Docker Ubuntu 22.04 (ffmpeg 4.4): ghép shot, crossfade, giữ frame khi thoại dài, nhạc nền có ducking, phụ đề tiếng Việt, logic chia đoạn và nối frame của chế độ dance.
- Tên node và input của ComfyUI đã đối chiếu với mã nguồn hiện tại của ComfyUI và các custom node. Đường dẫn model đã đối chiếu với HuggingFace.
- **Chưa chạy trên GPU thật.** Lần chạy đầu trên server hãy bắt đầu bằng `pipeline check`. Nếu ComfyUI báo lỗi workflow, nội dung lỗi (tên node, input) sẽ được in ra.

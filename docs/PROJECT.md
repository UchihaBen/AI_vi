# Dự án AI Video: ý tưởng, kiến trúc và cách vận hành

Tài liệu tổng quan: dự án làm gì, vì sao chọn những công nghệ này, code chạy thế nào, và quy trình **viết code ở máy cá nhân → đẩy lên Git → server kéo về chạy**.
Hướng dẫn chi tiết từng lệnh có trong [README.md](../README.md).

---

## 1. Mục tiêu

Tạo **video ngắn 10–25 giây** bằng AI, chạy hoàn toàn local trên một máy GPU tầm trung. Video cần đạt ba yêu cầu:

- **Nhân vật nhất quán**: giữ mặt, tóc, trang phục suốt video.
- **Chuyển động chân thực**: dáng đi, nhảy, quay người tự nhiên, ít méo.
- **Tự động hóa**: một lệnh chạy từ đầu vào đến file `final.mp4`, dừng giữa chừng thì chạy tiếp được.

Hai nhu cầu chính:

1. **Nhân vật nhảy TikTok**: có sẵn ảnh nhân vật, muốn nhân vật nhảy theo một video mẫu.
2. **Phim ngắn có thoại**: vài shot, nhân vật nói tiếng Việt, có nhạc nền và phụ đề.

## 2. Phần cứng mục tiêu

| Thành phần | Cấu hình |
|---|---|
| GPU | NVIDIA RTX 4070 **12 GB VRAM** |
| CPU | AMD Ryzen 9 5950X |
| RAM | 62 GB |
| OS | Ubuntu 22.04 |

**Giới hạn thực tế của 12 GB VRAM:**
- Model video 14B chỉ chạy được ở dạng **lượng tử hóa GGUF (Q4/Q5)**. Phần không vừa VRAM được tràn sang RAM, nên 62 GB RAM rất quan trọng.
- Mỗi lần chỉ sinh được khoảng **5 giây ở 480p**. Video dài hơn phải **chia thành nhiều đoạn rồi nối lại**. Độ phân giải cuối được upscale lên 720p.
- Không chạy hai model nặng cùng lúc. Pipeline gọi `/free` của ComfyUI để giải phóng VRAM giữa các bước.

## 3. Ý tưởng cốt lõi

### 3.1. Dùng ảnh làm gốc, không tạo video thẳng từ prompt
Tạo video thẳng từ prompt (text-to-video) cho ra nhân vật khác nhau ở mỗi lần chạy. Dự án luôn đi theo hướng **ảnh → video**: có ảnh nhân vật chuẩn trước, rồi mới cho chuyển động. Ngoại hình nằm trong ảnh, nên giữ được ổn định.

### 3.2. Lấy chuyển động từ người thật
Model tự nghĩ ra động tác nhảy thì thường giả. Chế độ dance dùng **Wan 2.2 Animate**: tách khung xương (DWPose) và khuôn mặt từ một video người thật nhảy, rồi áp lên nhân vật. Chuyển động lấy từ người thật nên trông thật.

### 3.3. Chia video thành đoạn ngắn và nối liền mạch
Mỗi đoạn dài 77 frame (khoảng 4,8 giây). Đoạn sau nhận **5 frame cuối của đoạn trước** làm `continue_motion`, nên chuyển động nối tiếp không bị giật. 5 frame trùng đó bị bỏ đi khi ghép.

### 3.4. Kết quả từng bước được lưu lại
Mỗi bước ghi kết quả ra `outputs/<tên>/…`. Bước nào đã có kết quả thì lần sau bỏ qua (trừ khi chạy với `--force`). Nhờ vậy:
- render hỏng hoặc mất điện thì chạy lại đúng lệnh cũ là đi tiếp;
- có thể duyệt keyframe bằng mắt, thay ảnh khác, rồi mới render video;
- sửa một shot thì chỉ render lại shot đó.

### 3.5. Workflow ComfyUI dựng bằng code
Thay vì lưu file JSON cố định, `pipeline/workflows.py` dựng workflow bằng Python. Khi bật hoặc tắt LoRA, ảnh cuối hay chế độ lightning, đồ thị node thay đổi theo mà không phải sửa JSON bằng tay.

### 3.6. TTS dùng dịch vụ ngoài
Giọng tiếng Việt dùng **VieNeu TTS** qua API `https://api-tts.saokhue.io` (chuẩn OpenAI `/v1/audio/speech`). Phần này không tốn VRAM của server render.

## 4. Kiến trúc

```
                ┌──────────────── máy dev (Windows) ────────────────┐
                │  viết code, sửa project.yaml, test ffmpeg (Docker) │
                └───────────────┬────────────────────────────────────┘
                                │ git push
                                ▼
                     GitHub: UchihaBen/AI_vi
                                │ git pull
                                ▼
┌──────────────────────────── server GPU (Ubuntu) ────────────────────────────┐
│                                                                             │
│   python -m pipeline …  (orchestrator, .venv)                               │
│        │  HTTP :8188                         │  HTTPS                       │
│        ▼                                     ▼                              │
│   ComfyUI (comfyui/, venv riêng)        VieNeu TTS (api-tts.saokhue.io)     │
│     ├─ SDXL / RealVisXL  → keyframe                                         │
│     ├─ Wan 2.2 I2V 14B GGUF → clip 5s                                       │
│     ├─ Wan 2.2 Animate 14B GGUF + DWPose → nhảy theo video                  │
│     └─ models/ (symlink tới HF cache)                                       │
│        │                                                                    │
│        ▼                                                                    │
│   FFmpeg: nội suy fps, upscale, chèn thoại, nhạc nền, phụ đề → final.mp4    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Luồng chế độ Dance
```
character.png ─────────────────────────────┐
dance.mp4 → chuẩn hóa 16fps 480x832 → DWPose (pose + mặt)
                                            ▼
                        Wan 2.2 Animate: đoạn 1 → đoạn 2 → … (nối 5 frame)
                                            ▼
                ghép frame → nội suy 24fps → 720x1280 → gắn nhạc → final.mp4
```

### Luồng chế độ Phim
```
truyện.txt ─(LLM, tùy chọn)→ project.yaml (shot, prompt, thoại)
  → keyframes : SDXL + LoRA nhân vật, sinh 3 ứng viên mỗi shot (duyệt tay)
  → video     : Wan 2.2 I2V (hoặc First-Last-Frame) → clip 5s
  → voice     : VieNeu TTS → line_XX.wav
  → assemble  : thoại đặt đúng giây, crossfade, nhạc nền ducking, phụ đề → final.mp4
```

## 5. Lựa chọn công nghệ

| Công đoạn | Chọn | Lý do | Phương án khác |
|---|---|---|---|
| Điều phối | Python + ComfyUI HTTP API | ComfyUI có hệ sinh thái node lớn nhất, quản lý VRAM tốt | Diffusers thuần |
| Ảnh keyframe | SDXL (RealVisXL V5) | Chạy thoải mái trên 12 GB, train LoRA dễ, ảnh chân thực | FLUX GGUF, Qwen-Image |
| Giữ nhân vật | LoRA riêng cho từng nhân vật (kohya sd-scripts) | Ổn định nhất cho phim nhiều shot | IP-Adapter, PuLID, Qwen-Image-Edit |
| Ảnh → video | Wan 2.2 I2V A14B GGUF + LightX2V 4 bước | Model mở tốt nhất hiện có, Apache 2.0, nhanh nhờ LoRA lightning | LTX-Video (nhanh hơn, kém hơn) |
| Nhảy theo video | Wan 2.2 Animate 14B + DWPose | Chuyển động thật từ video mẫu, giữ mặt nhờ face video | Wan VACE, Wan Fun Control |
| Giọng nói | VieNeu TTS (API riêng) | Đã có sẵn, giọng Việt tốt, không tốn GPU | F5-TTS VI, viXTTS |
| Hậu kỳ | FFmpeg | Ổn định, script được, có sẵn trên Ubuntu | MoviePy, DaVinci Resolve |
| Kịch bản | LLM tương thích OpenAI (Ollama/Qwen3) | Chạy local hoặc qua API | Claude/GPT API |

## 6. Cấu trúc mã nguồn

```
AI_vi/
├─ pipeline/
│  ├─ cli.py         lệnh: check, voices, tts, plan, run, dance, charsheet
│  ├─ settings.py    đọc config/default.yaml → config/local.yaml → .env
│  ├─ project.py     đọc + kiểm tra project.yaml (phim)
│  ├─ stages.py      keyframes → video → voice → assemble (+ charsheet)
│  ├─ dance.py       chế độ dance: chuẩn hóa → pose/mặt → chia đoạn → ghép
│  ├─ workflows.py   workflow ComfyUI dựng bằng code
│  ├─ comfy.py       client ComfyUI: upload, queue, chờ, tải kết quả, free VRAM
│  ├─ tts.py         client VieNeu TTS (retry, WAV/PCM)
│  ├─ media.py       FFmpeg: clip, shot, concat, nhạc nền, phụ đề
│  └─ llm.py         truyện → project.yaml nháp
├─ config/default.yaml    toàn bộ tham số model / chất lượng
├─ scripts/
│  ├─ setup_server.sh     cài ffmpeg, ComfyUI, custom nodes, venv
│  ├─ download_models.py  tải model từ HuggingFace
│  ├─ start_comfyui.sh    chạy ComfyUI
│  └─ train_lora_sdxl.sh  train LoRA nhân vật
├─ projects/
│  ├─ demo/project.yaml       mẫu phim 3 shot có thoại
│  └─ dance_demo/dance.yaml   mẫu nhảy TikTok
└─ docs/PROJECT.md        (tài liệu này)
```

Những thứ **không đưa lên Git** (đã có trong `.gitignore`): `.env` (API key), `config/local.yaml`, `comfyui/`, `outputs/`, `datasets/`, `tools/`, `.venv/`.

---

## 7. Quy trình làm việc: dev → Git → server

### 7.1. Lần đầu

**Máy dev (Windows):**
```bash
git clone https://github.com/UchihaBen/AI_vi.git
cd AI_vi
# sửa code / project → commit → push
git add -A
git commit -m "mô tả thay đổi"
git push
```

**Server (Ubuntu + GPU):**
```bash
git clone https://github.com/UchihaBen/AI_vi.git
cd AI_vi
bash scripts/setup_server.sh                  # ~15-30 phút
nano .env                                      # điền TTS_API_KEY
.venv/bin/python scripts/download_models.py    # ~45 GB, hoặc --only dance
tmux new -s comfy 'bash scripts/start_comfyui.sh'
.venv/bin/python -m pipeline check             # phải toàn [OK]
```

### 7.2. Mỗi lần cập nhật code

```bash
# máy dev
git add -A && git commit -m "..." && git push

# server
cd AI_vi
git pull
bash scripts/setup_server.sh      # chỉ cần khi đổi requirements / custom nodes
# nếu đổi code trong pipeline/ thì không cần khởi động lại ComfyUI
# nếu cập nhật custom nodes thì restart ComfyUI: tmux attach -t comfy → Ctrl+C → chạy lại
```

### 7.3. Đưa ảnh và video đầu vào lên server

File ảnh nhân vật và video mẫu thường nặng, có thể có bản quyền, nên **không nên commit**. Chép thẳng lên server:
```bash
scp character.png dance.mp4 user@server:~/AI_vi/projects/dance_demo/assets/
```

Lấy kết quả về:
```bash
scp user@server:~/AI_vi/outputs/dance_demo/final.mp4 .
```

### 7.4. Cấu hình riêng của server

Không sửa `config/default.yaml` trên server, vì `git pull` sẽ bị xung đột. Tạo `config/local.yaml` (đã nằm trong .gitignore):
```yaml
dance:
  chunk_length: 49
video:
  lightning: true
```

---

## 8. Bảng lệnh nhanh

| Mục đích | Lệnh |
|---|---|
| Kiểm tra môi trường | `.venv/bin/python -m pipeline check` |
| Danh sách giọng | `.venv/bin/python -m pipeline voices` |
| Đọc thử giọng | `.venv/bin/python -m pipeline tts "Xin chào" --voice <tên> -o test.wav` |
| **Nhân vật nhảy** | `.venv/bin/python -m pipeline dance projects/dance_demo/dance.yaml` |
| Nhảy, làm lại từ đầu | `… dance projects/dance_demo/dance.yaml --force` |
| Truyện → kịch bản nháp | `.venv/bin/python -m pipeline plan story.txt --seconds 20 --aspect 9:16` |
| **Phim, chạy hết** | `.venv/bin/python -m pipeline run projects/demo/project.yaml` |
| Phim, chỉ keyframe để duyệt | `… run projects/demo/project.yaml --stages keyframes` |
| Phim, render tiếp | `… run projects/demo/project.yaml --stages video,voice,assemble` |
| Làm lại một shot | `… run projects/demo/project.yaml --stages video --shots s02 --force` |
| Sinh dataset nhân vật | `.venv/bin/python -m pipeline charsheet projects/demo/project.yaml linh --count 40` |
| Train LoRA | `bash scripts/train_lora_sdxl.sh linh` (tắt ComfyUI trước) |
| Tải model quant khác | `.venv/bin/python scripts/download_models.py --quant Q5_K_M` |

## 9. Thời gian ước lượng (RTX 4070, chưa đo thực tế)

| Việc | Ước lượng |
|---|---|
| 3 keyframe SDXL | ~1 phút |
| 1 clip I2V 5 giây 480p | ~5–10 phút |
| Dance 15 giây (4 đoạn) | ~20–40 phút |
| Phim 20 giây (4 shot, tính cả số lần tạo lại để chọn bản đẹp) | ~2–5 giờ |

## 10. Giới hạn đã biết

- Ngón tay khi cầm nắm, hành động rất nhanh, và hai nhân vật chạm nhau vẫn dễ bị lỗi.
- Chế độ dance cần video mẫu có một người, toàn thân, camera đứng yên.
- Ảnh dataset do `charsheet` sinh từ chữ nên không cùng một khuôn mặt 100%, phải lọc tay.
- Pipeline **chưa được chạy trên GPU thật**. Phần FFmpeg và logic chia đoạn đã test trong Docker Ubuntu 22.04.

## 11. Hướng phát triển tiếp

- [ ] Lip-sync cho shot cận mặt (LatentSync / InfiniteTalk).
- [ ] Tự sinh tiếng động khớp hình bằng MMAudio, và nhạc nền bằng ACE-Step.
- [ ] Nội suy fps bằng RIFE thay cho minterpolate của FFmpeg.
- [ ] Giữ nhân vật bằng ảnh tham chiếu (Qwen-Image-Edit / FLUX Kontext) thay vì phải train LoRA.
- [ ] Hàng đợi job và giao diện web nhỏ để gửi job từ xa.
- [ ] Upscale video bằng AI (SeedVR2) thay cho lanczos.

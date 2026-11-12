"""Story -> draft project.yaml using any OpenAI-compatible chat API (Ollama, vLLM, ...)."""
from __future__ import annotations

import json
import re

import requests
import yaml

SYSTEM = """Bạn là đạo diễn và biên kịch phim ngắn AI. Từ câu chuyện của người dùng, hãy chia thành các shot
cho một video {seconds} giây (mỗi shot khoảng 5 giây, tổng {n_min}-{n_max} shot).

Quy tắc:
- image_prompt, motion_prompt, camera, description viết bằng TIẾNG ANH (cho model ảnh/video).
- Lời thoại (dialogue.text) viết bằng TIẾNG VIỆT, tự nhiên, mỗi câu đọc được trong khoảng 4 giây (tối đa ~16 chữ).
- Mỗi shot chỉ MỘT hành động đơn giản, chậm, rõ ràng (quay đầu, bước tới, mỉm cười, cầm cốc...).
  Tránh: đánh nhau, chạy nhanh, nhiều người chạm vào nhau, động tác tay phức tạp.
- image_prompt mô tả khung hình ĐẦU của shot: bố cục, bối cảnh, ánh sáng, tư thế. KHÔNG lặp lại mô tả ngoại hình nhân vật
  (hệ thống tự thêm từ phần characters).
- description của nhân vật phải cố định và cụ thể: tuổi, giới tính, tóc, trang phục, màu sắc.
- character id dùng chữ thường không dấu, ví dụ "linh".

Chỉ trả về JSON đúng schema:
{{
  "name": "ten_project_khong_dau",
  "aspect": "16:9" | "9:16",
  "characters": {{ "<id>": {{"description": "...", "trigger": "<id>_person", "voice": null}} }},
  "shots": [
    {{"id": "s01", "characters": ["<id>"], "image_prompt": "...", "motion_prompt": "...",
      "camera": "slow dolly in", "dialogue": [{{"character": "<id>", "text": "...", "start": 0.4}}]}}
  ]
}}
"""


def plan_project(story: str, cfg: dict, seconds: int = 20, aspect: str | None = None) -> dict:
    n = max(1, round(seconds / 5))
    sys_prompt = SYSTEM.format(seconds=seconds, n_min=max(1, n - 1), n_max=n + 1)
    body = {
        "model": cfg["model"],
        "temperature": cfg.get("temperature", 0.7),
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": sys_prompt}, {"role": "user", "content": story}],
    }
    headers = {"Authorization": f"Bearer {cfg['api_key']}"} if cfg.get("api_key") else {}
    r = requests.post(f"{cfg['base_url'].rstrip('/')}/chat/completions", json=body, headers=headers,
                      timeout=cfg.get("timeout_sec", 600))
    if not r.ok:
        raise RuntimeError(f"LLM lỗi ({r.status_code}): {r.text[:1000]}")
    text = r.json()["choices"][0]["message"]["content"]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()   # qwen3 thinking
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise RuntimeError(f"LLM không trả JSON:\n{text[:1000]}")
    data = json.loads(m.group(0))
    if aspect:
        data["aspect"] = aspect
    return data


def to_yaml(data: dict) -> str:
    header = ("# Bản nháp do LLM tạo. Hãy đọc lại, sửa prompt và lời thoại trước khi chạy.\n"
              "# Thêm lora/lora_strength cho nhân vật sau khi train LoRA, và voice = tên giọng VieNeu.\n")
    return header + yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=110)

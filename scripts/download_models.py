#!/usr/bin/env python3
"""Download required models from HuggingFace into ComfyUI/models (symlinks to the HF cache).

  .venv/bin/python scripts/download_models.py            # Q4_K_M (khuyên dùng cho 12 GB)
  .venv/bin/python scripts/download_models.py --quant Q5_K_M
  .venv/bin/python scripts/download_models.py --only dance   # chỉ model cho nhảy TikTok

Nhớ đổi unet_high/unet_low trong config/local.yaml nếu dùng quant khác Q4_K_M.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def manifest(quant: str) -> list[tuple[str, str, str, str]]:
    """(group, hf_repo, file_in_repo, comfy_models_subdir)"""
    wan22 = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
    gguf = "QuantStack/Wan2.2-I2V-A14B-GGUF"
    return [
        ("video", gguf, f"HighNoise/Wan2.2-I2V-A14B-HighNoise-{quant}.gguf", "unet"),
        ("video", gguf, f"LowNoise/Wan2.2-I2V-A14B-LowNoise-{quant}.gguf", "unet"),
        ("video", wan22, "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders"),
        ("video", wan22, "split_files/vae/wan_2.1_vae.safetensors", "vae"),
        ("video", wan22, "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", "loras"),
        ("video", wan22, "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", "loras"),
        ("image", "SG161222/RealVisXL_V5.0", "RealVisXL_V5.0_fp16.safetensors", "checkpoints"),
        ("dance", "QuantStack/Wan2.2-Animate-14B-GGUF", f"Wan2.2-Animate-14B-{quant}.gguf", "unet"),
        ("dance", "Kijai/WanVideo_comfy", "Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors", "loras"),
        ("dance", wan22, "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors", "text_encoders"),
        ("dance", wan22, "split_files/vae/wan_2.1_vae.safetensors", "vae"),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy-dir", default=os.environ.get("COMFY_DIR", str(REPO / "comfyui")))
    ap.add_argument("--quant", default="Q4_K_M", help="Q3_K_M | Q4_K_M | Q5_K_M | Q6_K | Q8_0")
    ap.add_argument("--only", choices=["video", "image", "dance"],
                    help="video = phim (I2V), image = keyframe SDXL, dance = nhảy theo video mẫu")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(REPO / ".env")
    except ImportError:
        pass
    from huggingface_hub import hf_hub_download

    models = Path(args.comfy_dir) / "models"
    if not models.parent.exists():
        print(f"Không thấy ComfyUI ở {models.parent}. Chạy scripts/setup_server.sh trước.", file=sys.stderr)
        return 1

    for group, repo, fname, sub in manifest(args.quant):
        if args.only and group != args.only:
            continue
        target = models / sub / Path(fname).name
        if target.exists():
            print(f"[có rồi] {target.relative_to(models)}")
            continue
        print(f"[tải]    {repo}/{fname}")
        cached = hf_hub_download(repo_id=repo, filename=fname, token=os.environ.get("HF_TOKEN") or None)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            target.unlink()
        target.symlink_to(Path(cached).resolve())
        print(f"         -> {target.relative_to(models)}")
    print("\nXong.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Command line entry point:  python -m pipeline <command> ..."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .settings import ROOT, load_settings


def cmd_check(args, s) -> int:
    from .comfy import ComfyClient
    from .tts import VieNeuTTS

    ok = True

    def report(good: bool, msg: str):
        nonlocal ok
        ok &= good
        print(("  [OK]   " if good else "  [LỖI]  ") + msg)

    print("FFmpeg")
    for tool in ("ffmpeg", "ffprobe"):
        report(bool(shutil.which(tool)), tool)

    print(f"ComfyUI  {s['comfy']['url']}")
    c = ComfyClient(s["comfy"]["url"])
    try:
        st = c.system_stats()
        dev = (st.get("devices") or [{}])[0]
        report(True, f"đang chạy, GPU: {dev.get('name', '?')}, VRAM {dev.get('vram_total', 0) / 2**30:.1f} GB")
        k, v = s["keyframe"], s["video"]
        need = [
            ("CheckpointLoaderSimple", "ckpt_name", k["checkpoint"]),
            ("UnetLoaderGGUF", "unet_name", v["unet_high"]),
            ("UnetLoaderGGUF", "unet_name", v["unet_low"]),
            ("CLIPLoader", "clip_name", v["text_encoder"]),
            ("VAELoader", "vae_name", v["vae"]),
        ]
        if v.get("lightning"):
            need += [("LoraLoaderModelOnly", "lora_name", v["lora_high"]),
                     ("LoraLoaderModelOnly", "lora_name", v["lora_low"])]
        for node in ("WanImageToVideo", "WanFirstLastFrameToVideo", "UnetLoaderGGUF"):
            report(c.object_info(node) is not None, f"node {node}")
        for node, field, name in need:
            opts = c.input_options(node, field)
            report(bool(opts) and name in opts, f"model {name}")

        print("  -- chế độ dance (Wan 2.2 Animate) --")
        dn = s["dance"]
        for node in ("WanAnimateToVideo", "TrimVideoLatent", "VHS_LoadImagesPath", "DWPreprocessor",
                     "FaceMaskFromPoseKeypoints", "ImageCropByMaskAndResize"):
            report(c.object_info(node) is not None, f"node {node}")
        for node, field, name in (("UnetLoaderGGUF", "unet_name", dn["unet"]),
                                  ("LoraLoaderModelOnly", "lora_name", dn["lora"])):
            opts = c.input_options(node, field)
            report(bool(opts) and name in opts, f"model {name}")
    except Exception as e:  # noqa: BLE001
        report(False, f"không kết nối được: {e}")

    print(f"TTS  {s['tts']['base_url']}")
    t = VieNeuTTS(s["tts"])
    try:
        h = t.health()
        report(h.get("status") == "ok", f"health: model {h.get('model')}, {h.get('sample_rate')} Hz")
        voices = t.voices()
        report(True, f"API key hợp lệ, {len(voices)} giọng")
    except Exception as e:  # noqa: BLE001
        report(False, str(e))
    print("\nSẵn sàng." if ok else "\nCòn lỗi, xem các dòng [LỖI] ở trên.")
    return 0 if ok else 1


def cmd_voices(args, s) -> int:
    from .tts import VieNeuTTS

    for v in VieNeuTTS(s["tts"]).voices():
        extra = " · ".join(str(x) for x in (v.get("gender"), v.get("description")) if x)
        print(f"{v.get('name'):<24} {'(custom) ' if v.get('custom') else ''}{extra}")
    return 0


def cmd_tts(args, s) -> int:
    from .tts import VieNeuTTS

    out = VieNeuTTS(s["tts"]).speech(args.text, Path(args.out), voice=args.voice, speed=args.speed)
    print(out)
    return 0


def cmd_plan(args, s) -> int:
    from .llm import plan_project, to_yaml

    story = Path(args.story).read_text(encoding="utf-8")
    data = plan_project(story, s["llm"], seconds=args.seconds, aspect=args.aspect)
    out = Path(args.out) if args.out else ROOT / "projects" / data.get("name", "draft") / "project.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.force:
        print(f"{out} đã tồn tại (dùng --force để ghi đè)")
        return 1
    out.write_text(to_yaml(data), encoding="utf-8")
    print(f"Đã tạo {out}  ({len(data.get('shots', []))} shot)")
    return 0


def cmd_run(args, s) -> int:
    from .project import load_project
    from .stages import STAGES, Runner

    stages = [x.strip() for x in args.stages.split(",")] if args.stages else list(STAGES)
    bad = [x for x in stages if x not in STAGES]
    if bad:
        print(f"Stage không hợp lệ: {bad}. Chọn trong {STAGES}")
        return 2
    project = load_project(args.project)
    shots = [x.strip() for x in args.shots.split(",")] if args.shots else None
    Runner(project, s, force=args.force, only=shots).run(stages)
    return 0


def cmd_dance(args, s) -> int:
    from .dance import DanceRunner, load_dance

    DanceRunner(load_dance(args.job), s, force=args.force).run()
    return 0


def cmd_charsheet(args, s) -> int:
    from .project import load_project
    from .stages import character_sheet

    out = character_sheet(load_project(args.project), s, args.character, args.count)
    print(f"\nẢnh + caption ở {out}\nXóa những ảnh không giống nhân vật rồi train LoRA: bash scripts/train_lora_sdxl.sh {args.character}")
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap =argparse.ArgumentParser(prog="python -m pipeline", description="AI short film pipeline")
    ap.add_argument("--config", help="file YAML ghi đè config (mặc định config/local.yaml)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="kiểm tra ffmpeg, ComfyUI, model, TTS")
    sub.add_parser("voices", help="liệt kê giọng TTS")

    p = sub.add_parser("tts", help="đọc thử một câu")
    p.add_argument("text")
    p.add_argument("-o", "--out", default="test.wav")
    p.add_argument("--voice")
    p.add_argument("--speed", type=float)

    p = sub.add_parser("plan", help="truyện (txt) -> project.yaml nháp bằng LLM")
    p.add_argument("story")
    p.add_argument("-o", "--out")
    p.add_argument("--seconds", type=int, default=20)
    p.add_argument("--aspect", choices=["16:9", "9:16", "1:1"])
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("run", help="chạy pipeline cho một project")
    p.add_argument("project", help="đường dẫn project.yaml")
    p.add_argument("--stages", help="keyframes,video,voice,assemble (mặc định: tất cả)")
    p.add_argument("--shots", help="chỉ chạy các shot này, vd s01,s03")
    p.add_argument("--force", action="store_true", help="làm lại dù đã có kết quả")

    p = sub.add_parser("dance", help="ảnh nhân vật + video nhảy mẫu -> nhân vật nhảy (Wan 2.2 Animate)")
    p.add_argument("job", help="đường dẫn dance.yaml")
    p.add_argument("--force", action="store_true", help="làm lại từ đầu")

    p = sub.add_parser("charsheet", help="sinh ảnh dataset để train LoRA nhân vật")
    p.add_argument("project")
    p.add_argument("character")
    p.add_argument("--count", type=int, default=40)

    args = ap.parse_args(argv)
    s = load_settings(args.config)
    handler = {"check": cmd_check, "voices": cmd_voices, "tts": cmd_tts, "plan": cmd_plan,
               "run": cmd_run, "dance": cmd_dance, "charsheet": cmd_charsheet}[args.cmd]
    try:
        return handler(args, s)
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # noqa: BLE001
        print(f"\nLỖI: {e}", file=sys.stderr)
        return 1

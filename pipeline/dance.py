"""Make a character image dance like a reference video (Wan 2.2 Animate 14B).

  outputs/<name>/
    drive/frame_*.png      video mẫu đã chuẩn hóa (fps, khung hình)
    drive_audio.wav        nhạc của video mẫu
    pose/ face/            skeleton DWPose + mặt (sinh 1 lần)
    chunks/c00 c01 ...     từng đoạn sinh ra (resume được)
    final.mp4
"""
from __future__ import annotations

import shutil
import zlib
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import media, workflows
from .comfy import ComfyClient
from .project import ASPECTS, ProjectError
from .settings import ROOT
from .stages import log


@dataclass
class DanceJob:
    name: str
    root: Path
    character_image: Path
    drive_video: Path
    aspect: str = "9:16"
    prompt: str | None = None
    start: float = 0.0
    duration: float = 15.0
    use_drive_audio: bool = True
    music: Path | None = None
    seed: int | None = None


def load_dance(path: str | Path) -> DanceJob:
    path = Path(path).resolve()
    d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def rel(key: str, required: bool = True) -> Path | None:
        v = d.get(key)
        if not v:
            if required:
                raise ProjectError(f"thiếu '{key}'")
            return None
        p = Path(v)
        p = p if p.is_absolute() else (path.parent / p).resolve()
        if not p.exists():
            raise ProjectError(f"không thấy file {key}: {p}")
        return p

    job = DanceJob(
        name=str(d.get("name") or path.parent.name), root=path.parent,
        character_image=rel("character_image"), drive_video=rel("drive_video"),
        aspect=str(d.get("aspect", "9:16")), prompt=d.get("prompt"),
        start=float(d.get("start", 0)), duration=float(d.get("duration", 15)),
        use_drive_audio=bool(d.get("use_drive_audio", True)), music=rel("music", required=False),
        seed=d.get("seed"))
    if job.aspect not in ASPECTS:
        raise ProjectError(f"aspect phải là một trong {ASPECTS}")
    if not 1 <= job.duration <= 60:
        raise ProjectError("duration nên trong khoảng 1-60 giây")
    return job


class DanceRunner:
    def __init__(self, job: DanceJob, settings: dict, force: bool = False, out_root: Path | None = None):
        self.j = job
        self.s = settings
        # model chung với video (text encoder, vae) + phần riêng của dance
        self.d = {"text_encoder": settings["video"]["text_encoder"], "vae": settings["video"]["vae"], **settings["dance"]}
        ov = int(self.d["overlap"])
        if (ov - 1) % 4 or (int(self.d["chunk_length"]) - 1) % 4:
            raise ValueError("dance.overlap và dance.chunk_length phải có dạng 4n+1 (vd overlap 5, chunk 77)")
        if ov >= int(self.d["chunk_length"]):
            raise ValueError("dance.overlap phải nhỏ hơn chunk_length")
        self.force = force
        self.out = (out_root or ROOT / "outputs") / job.name
        self.out.mkdir(parents=True, exist_ok=True)
        c = settings["comfy"]
        self.comfy = ComfyClient(c["url"], c.get("timeout_sec", 3600), c.get("poll_sec", 2))
        self.size = tuple(self.d["sizes"][job.aspect])

    def seed(self) -> int:
        return int(self.j.seed) if self.j.seed is not None else zlib.crc32(self.j.name.encode()) & 0x7FFFFFFF

    def run(self) -> Path:
        n = self.prepare()
        self.preprocess(n)
        self.animate(n)
        return self.stitch(n)

    # 1. chuẩn hóa video mẫu
    def prepare(self) -> int:
        log("\n=== chuẩn bị video mẫu ===")
        drive = self.out / "drive"
        if drive.exists() and any(drive.iterdir()) and not self.force:
            n = len(list(drive.glob("frame_*.png")))
            log(f"  đã có {n} frame, bỏ qua")
            return n
        shutil.rmtree(drive, ignore_errors=True)
        drive.mkdir(parents=True)
        w, h = self.size
        fps = int(self.d["fps"])
        media.ff("-ss", f"{self.j.start}", "-t", f"{self.j.duration}", "-i", str(self.j.drive_video),
                 "-vf", f"fps={fps},scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h}",
                 str(drive / "frame_%05d.png"))
        audio = self.out / "drive_audio.wav"
        try:
            media.ff("-ss", f"{self.j.start}", "-t", f"{self.j.duration}", "-i", str(self.j.drive_video),
                     "-vn", "-ac", "2", "-ar", str(media.AUDIO_RATE), str(audio))
        except media.MediaError:
            log("  (video mẫu không có âm thanh)")
        n = len(list(drive.glob("frame_*.png")))
        log(f"  {n} frame, {w}x{h} @ {fps}fps ({n / fps:.1f}s)")
        return n

    # 2. pose + mặt
    def preprocess(self, n: int) -> None:
        log("\n=== tách pose + khuôn mặt (DWPose) ===")
        pose, face = self.out / "pose", self.out / "face"
        if not self.force and len(list(pose.glob("*.png"))) == n and len(list(face.glob("*.png"))) == n:
            log("  đã có, bỏ qua")
            return
        wf = workflows.dance_preprocess(drive_dir=(self.out / "drive").resolve().as_posix(), d=self.d,
                                        prefix=f"pipeline/{self.j.name}/prep")
        entry = self.comfy.wait(self.comfy.queue(wf), "pose/face")
        for node, dest in (("save_pose", pose), ("save_face", face)):
            shutil.rmtree(dest, ignore_errors=True)
            files = sorted(self.comfy.output_files(entry, node), key=lambda f: f["filename"])
            for i, f in enumerate(files, start=1):
                self.comfy.download(f, dest / f"frame_{i:05d}.png")
            log(f"  {dest.name}: {len(files)} frame")
            if len(files) != n:
                log(f"  !! số frame {dest.name} ({len(files)}) khác video mẫu ({n})")
        self.comfy.free()

    # 3. sinh từng đoạn
    def chunk_plan(self, n: int) -> list[int]:
        """Offsets (first new frame index) of each chunk."""
        L, ov = int(self.d["chunk_length"]), int(self.d["overlap"])
        offsets, o = [0], L
        while o < n:
            offsets.append(o)
            o += L - ov
        return offsets

    def animate(self, n: int) -> None:
        offsets = self.chunk_plan(n)
        log(f"\n=== Wan 2.2 Animate: {len(offsets)} đoạn ===")
        w, h = self.size
        L, ov = int(self.d["chunk_length"]), int(self.d["overlap"])
        ref = self.comfy.upload_image(self.j.character_image)
        prompt = self.j.prompt or self.d["prompt"]
        chunks = self.out / "chunks"
        prev: Path | None = None
        for k, off in enumerate(offsets):
            cdir = chunks / f"c{k:02d}"
            if not self.force and cdir.exists() and len(list(cdir.glob("*.png"))) == L:
                log(f"  đoạn {k}: đã có, bỏ qua")
                prev = cdir
                continue
            wf = workflows.wan_animate_chunk(
                d=self.d, positive=prompt, negative=self.s["video"]["negative"], width=w, height=h, length=L,
                offset=off, seed=self.seed(), ref_image=ref,
                pose_dir=(self.out / "pose").resolve().as_posix(), face_dir=(self.out / "face").resolve().as_posix(),
                continue_dir=prev.resolve().as_posix() if prev else None, continue_skip=max(0, L - ov),
                prefix=f"pipeline/{self.j.name}/c{k:02d}")
            log(f"  đoạn {k + 1}/{len(offsets)}: frame {off}…{min(n, off + (L if k == 0 else L - ov))}")
            files = sorted(self.comfy.run(wf, f"chunk {k}", workflows.SAVE_NODE), key=lambda f: f["filename"])
            shutil.rmtree(cdir, ignore_errors=True)
            for i, f in enumerate(files, start=1):
                self.comfy.download(f, cdir / f"frame_{i:05d}.png")
            prev = cdir
        self.comfy.free()

    # 4. ghép
    def stitch(self, n: int) -> Path:
        log("\n=== ghép video ===")
        ov = int(self.d["overlap"])
        frames = self.out / "_frames"
        shutil.rmtree(frames, ignore_errors=True)
        frames.mkdir()
        idx = 0
        for k in range(len(self.chunk_plan(n))):
            imgs = sorted((self.out / "chunks" / f"c{k:02d}").glob("*.png"))
            for img in imgs[(ov if k else 0):]:
                if idx >= n:
                    break
                idx += 1
                shutil.copyfile(img, frames / f"frame_{idx:05d}.png")
        a = self.s["assemble"]
        fps, crf = int(a["output_fps"]), int(a["crf"])
        clip = media.frames_to_clip(frames, int(self.d["fps"]), self.out / "_clip.mp4", fps,
                                    self.s["video"].get("interpolation", "ffmpeg"))
        audio = self.j.music or (self.out / "drive_audio.wav" if self.j.use_drive_audio else None)
        lines = []
        if audio and Path(audio).exists():
            trimmed = self.out / "_audio.wav"   # không để nhạc dài hơn hình
            media.ff("-i", str(audio), "-t", f"{media.duration(clip):.3f}", "-ac", "2", "-ar", str(media.AUDIO_RATE), str(trimmed))
            lines = [(trimmed, 0.0)]
        final = self.out / "final.mp4"
        media.build_shot(clip, lines, final, tuple(a["sizes"][self.j.aspect]), fps, 0.0, crf)
        log(f"\n✔ Xong: {final}  ({media.duration(final):.1f}s)")
        return final

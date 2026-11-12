"""Production stages. Every stage writes into outputs/<project>/<shot>/ and skips
work whose output already exists (use --force to redo), so you can stop, swap a
keyframe by hand, and continue.

  outputs/<project>/
    s01/keyframe_cand0..N.png  keyframe.png  [end_cand*.png end.png]
        frames/  clip.mp4  line_00.wav ...  shot.mp4
    final.mp4  final.srt
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import media, workflows
from .comfy import ComfyClient
from .project import Project, Shot
from .settings import ROOT
from .tts import VieNeuTTS

STAGES = ("keyframes", "video", "voice", "assemble")


def log(msg: str) -> None:
    print(msg, flush=True)


class Runner:
    def __init__(self, project: Project, settings: dict, force: bool = False,
                 only: list[str] | None = None, out_root: Path | None = None):
        self.p = project
        self.s = settings
        self.force = force
        self.only = set(only or [])
        unknown = self.only - {sh.id for sh in project.shots}
        if unknown:
            raise ValueError(f"Không có shot: {', '.join(sorted(unknown))}")
        self.out = (out_root or ROOT / "outputs") / project.name
        self._comfy: ComfyClient | None = None
        self._tts: VieNeuTTS | None = None

    # ---- helpers ----
    @property
    def comfy(self) -> ComfyClient:
        if not self._comfy:
            c = self.s["comfy"]
            self._comfy = ComfyClient(c["url"], c.get("timeout_sec", 3600), c.get("poll_sec", 2))
        return self._comfy

    @property
    def tts(self) -> VieNeuTTS:
        if not self._tts:
            self._tts = VieNeuTTS(self.s["tts"])
        return self._tts

    def shots(self) -> list[Shot]:
        return [s for s in self.p.shots if not self.only or s.id in self.only]

    def dir(self, shot: Shot) -> Path:
        d = self.out / shot.id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _skip(self, path: Path) -> bool:
        return path.exists() and not self.force

    def run(self, stages: list[str]) -> None:
        for st in stages:
            log(f"\n=== {st} ===")
            getattr(self, f"stage_{st}")()

    # ---- 1. keyframes (SDXL + character LoRA) ----
    def _image_prompt(self, shot: Shot, text: str) -> str:
        style = self.p.style or self.s["keyframe"]["style"]
        chars = [self.p.characters[c].prompt() for c in shot.characters]
        return ", ".join(x.strip() for x in [style, *chars, text] if x and x.strip())

    def _loras(self, shot: Shot) -> list[tuple[str, float]]:
        out = []
        for cid in shot.characters:
            c = self.p.characters[cid]
            if c.lora:
                out.append((c.lora, float(c.lora_strength)))
        return out

    def _make_image(self, shot: Shot, tag: str, prompt_text: str, given: str | None) -> Path:
        d = self.dir(shot)
        final = d / f"{tag}.png"
        if given:
            src = self.p.resolve(given)
            if not src or not src.exists():
                raise FileNotFoundError(f"shot {shot.id}: không thấy ảnh {given}")
            shutil.copyfile(src, final)
            log(f"  {shot.id}/{tag}: dùng ảnh có sẵn {given}")
            return final
        if self._skip(final):
            log(f"  {shot.id}/{tag}: đã có, bỏ qua")
            return final

        k = self.s["keyframe"]
        w, h = k["sizes"][self.p.aspect]
        n = max(1, int(k.get("candidates", 1)))
        wf = workflows.sdxl_t2i(
            checkpoint=k["checkpoint"], positive=self._image_prompt(shot, prompt_text),
            negative=", ".join(x for x in (k["negative"], shot.negative) if x),
            width=w, height=h, seed=self.p.seed_for(shot, tag), steps=k["steps"], cfg=k["cfg"],
            sampler=k["sampler"], scheduler=k["scheduler"], loras=self._loras(shot), batch=n,
            prefix=f"pipeline/{self.p.name}/{shot.id}_{tag}")
        log(f"  {shot.id}/{tag}: sinh {n} ảnh ứng viên…")
        files = self.comfy.run(wf, f"{shot.id}/{tag}", workflows.SAVE_NODE)
        for old in d.glob(f"{tag}_cand*.png"):
            old.unlink()
        cands = [self.comfy.download(f, d / f"{tag}_cand{i}.png") for i, f in enumerate(files)]
        pick = min(max(shot.keyframe_pick, 0), len(cands) - 1)
        shutil.copyfile(cands[pick], final)
        log(f"  {shot.id}/{tag}: chọn cand{pick} -> {final.relative_to(self.out)}")
        return final

    def stage_keyframes(self) -> None:
        for shot in self.shots():
            self._make_image(shot, "keyframe", shot.image_prompt, shot.image)
            if shot.end_image or shot.end_image_prompt:
                self._make_image(shot, "end", shot.end_image_prompt or shot.image_prompt, shot.end_image)
        self.comfy.free()

    # ---- 2. image -> video (Wan 2.2 I2V / FLF2V) ----
    def _motion_prompt(self, shot: Shot) -> str:
        chars = [self.p.characters[c].description for c in shot.characters]
        parts = [shot.motion_prompt, shot.camera, *chars, self.s["video"].get("prompt_suffix", "")]
        return ". ".join(x.strip().rstrip(".") for x in parts if x and x.strip())

    def stage_video(self) -> None:
        v = self.s["video"]
        w, h = v["sizes"][self.p.aspect]
        out_fps = int(self.s["assemble"]["output_fps"])
        for shot in self.shots():
            d = self.dir(shot)
            clip = d / "clip.mp4"
            if self._skip(clip):
                log(f"  {shot.id}: clip đã có, bỏ qua")
                continue
            start = d / "keyframe.png"
            if not start.exists():
                raise FileNotFoundError(f"{shot.id}: chưa có keyframe.png, chạy stage keyframes trước")
            end = d / "end.png" if (shot.end_image or shot.end_image_prompt) else None
            if end and not end.exists():
                raise FileNotFoundError(f"{shot.id}: thiếu end.png")

            length = int(shot.length or v["length"])
            wf = workflows.wan22_i2v(
                v=v, positive=self._motion_prompt(shot), negative=v["negative"], width=w, height=h,
                length=length, seed=self.p.seed_for(shot, "video"),
                start_image=self.comfy.upload_image(start),
                end_image=self.comfy.upload_image(end) if end else None,
                prefix=f"pipeline/{self.p.name}/{shot.id}_video")
            mode = "first-last-frame" if end else "i2v"
            log(f"  {shot.id}: Wan 2.2 {mode}, {w}x{h}, {length} frame…")
            files = self.comfy.run(wf, f"{shot.id}/video", workflows.SAVE_NODE)

            frames = d / "frames"
            shutil.rmtree(frames, ignore_errors=True)
            for i, f in enumerate(sorted(files, key=lambda x: x["filename"]), start=1):
                self.comfy.download(f, frames / f"frame_{i:05d}.png")
            media.frames_to_clip(frames, int(v["fps"]), clip, out_fps, v.get("interpolation", "ffmpeg"))
            log(f"  {shot.id}: -> {clip.relative_to(self.out)} ({media.duration(clip):.1f}s)")
        self.comfy.free()

    # ---- 3. voice (VieNeu TTS) ----
    def stage_voice(self) -> None:
        for shot in self.shots():
            d = self.dir(shot)
            for i, ln in enumerate(shot.dialogue):
                wav = d / f"line_{i:02d}.wav"
                if self._skip(wav):
                    continue
                voice = ln.voice or self.p.characters[ln.character].voice
                log(f"  {shot.id} #{i} [{ln.character}/{voice or 'mặc định'}]: {ln.text}")
                self.tts.speech(ln.text, wav, voice=voice, speed=ln.speed)
                log(f"     -> {media.duration(wav):.2f}s")

    # ---- 4. assemble (ffmpeg) ----
    def stage_assemble(self) -> None:
        if self.only:
            log("  (assemble luôn ghép toàn bộ shot của project)")
        a = self.s["assemble"]
        size = tuple(a["sizes"][self.p.aspect])
        fps, crf = int(a["output_fps"]), int(a["crf"])
        crossfade = float(self.p.crossfade if self.p.crossfade is not None else a["crossfade"])

        shot_files, cues, t = [], [], 0.0
        for idx, shot in enumerate(self.p.shots):
            d = self.dir(shot)
            clip = d / "clip.mp4"
            if not clip.exists():
                raise FileNotFoundError(f"{shot.id}: chưa có clip.mp4")
            lines = []
            for i, ln in enumerate(shot.dialogue):
                wav = d / f"line_{i:02d}.wav"
                if not wav.exists():
                    raise FileNotFoundError(f"{shot.id}: thiếu {wav.name}, chạy stage voice")
                lines.append((wav, ln.start))
                cues.append((t + ln.start, t + ln.start + media.duration(wav), ln.text))
            out = d / "shot.mp4"
            dur = media.build_shot(clip, lines, out, size, fps, float(a["tail_after_dialogue"]), crf)
            log(f"  {shot.id}: {dur:.2f}s")
            shot_files.append(out)
            t += dur - (crossfade if idx < len(self.p.shots) - 1 else 0)

        work = self.out / "_work"
        work.mkdir(exist_ok=True)
        cur = media.concat(shot_files, work / "concat.mp4", crossfade, crf)

        bgm = self.p.resolve(self.p.bgm)
        if bgm:
            if not bgm.exists():
                raise FileNotFoundError(f"không thấy nhạc nền {bgm}")
            vol = float(self.p.bgm_volume if self.p.bgm_volume is not None else a["bgm_volume"])
            cur = media.add_music(cur, bgm, work / "music.mp4", vol, bool(a["ducking"]))

        final = self.out / "final.mp4"
        mode = self.p.subtitles or a["subtitles"]
        if cues and mode in ("burn", "srt"):
            srt = media.write_srt(cues, self.out / "final.srt")
            if mode == "burn":
                cur = media.burn_subtitles(cur, srt, work / "subs.mp4", a["subtitle_style"], crf)
        shutil.copyfile(cur, final)
        (self.out / "manifest.json").write_text(json.dumps({
            "project": self.p.name, "duration": round(media.duration(final), 2),
            "shots": [s.id for s in self.p.shots]}, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"\n✔ Xong: {final}  ({media.duration(final):.1f}s)")


# ---- dataset ảnh để train LoRA nhân vật ----
ANGLES = [
    "front view, looking at camera, neutral expression",
    "three-quarter view from the left, slight smile",
    "three-quarter view from the right",
    "side profile view",
    "close-up portrait, soft light",
    "medium shot, waist up, standing",
    "full body shot, standing, plain background",
    "looking over shoulder",
    "laughing, candid",
    "serious expression, dramatic side lighting",
    "sitting on a chair, medium shot",
    "walking, full body",
]


def character_sheet(project: Project, settings: dict, char_id: str, count: int, out_root: Path | None = None) -> Path:
    if char_id not in project.characters:
        raise ValueError(f"không có nhân vật '{char_id}'")
    c = project.characters[char_id]
    k = settings["keyframe"]
    trigger = c.trigger or char_id
    out = (out_root or ROOT / "datasets") / char_id / "img" / f"10_{trigger}"
    out.mkdir(parents=True, exist_ok=True)
    cfg = settings["comfy"]
    comfy = ComfyClient(cfg["url"], cfg.get("timeout_sec", 3600), cfg.get("poll_sec", 2))

    for i in range(count):
        angle = ANGLES[i % len(ANGLES)]
        prompt = f"photo of {trigger}, {c.description}, {angle}, plain neutral background, studio photo, photorealistic"
        wf = workflows.sdxl_t2i(
            checkpoint=k["checkpoint"], positive=prompt, negative=k["negative"], width=1024, height=1024,
            seed=1000 + i, steps=k["steps"], cfg=k["cfg"], sampler=k["sampler"], scheduler=k["scheduler"],
            loras=[], batch=1, prefix=f"pipeline/dataset/{char_id}")
        f = comfy.run(wf, f"{char_id} #{i}", workflows.SAVE_NODE)[0]
        img = comfy.download(f, out / f"{char_id}_{i:03d}.png")
        img.with_suffix(".txt").write_text(f"{trigger}, {c.description}, {angle}", encoding="utf-8")
        log(f"  {img.name}: {angle}")
    comfy.free()
    return out

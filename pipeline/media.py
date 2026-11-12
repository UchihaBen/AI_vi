"""FFmpeg helpers: frames -> clip, shot assembly, concat, music, subtitles."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

AUDIO_RATE = 48000


class MediaError(RuntimeError):
    pass


def require_ffmpeg() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise MediaError(f"Không tìm thấy {tool}. Cài bằng: sudo apt install ffmpeg")


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode != 0:
        raise MediaError(f"ffmpeg lỗi:\n{' '.join(cmd)}\n{p.stderr[-3000:]}")


def ff(*args: str, cwd: Path | None = None) -> None:
    _run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args], cwd=cwd)


def duration(path: Path) -> float:
    p = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True)
    if p.returncode != 0 or not p.stdout.strip():
        raise MediaError(f"Không đọc được thời lượng: {path}\n{p.stderr}")
    return float(p.stdout.strip())


def _vcodec(crf: int) -> list[str]:
    return ["-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p"]


def _acodec() -> list[str]:
    return ["-c:a", "aac", "-b:a", "192k", "-ar", str(AUDIO_RATE), "-ac", "2"]


def frames_to_clip(frames_dir: Path, src_fps: int, out: Path, out_fps: int, interpolation: str, crf: int = 16) -> Path:
    if interpolation == "ffmpeg" and out_fps != src_fps:
        vf = f"minterpolate=fps={out_fps}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1"
    else:
        vf = f"fps={out_fps}"
    ff("-framerate", str(src_fps), "-i", str(frames_dir / "frame_%05d.png"), "-vf", vf, *_vcodec(crf), "-an", str(out))
    return out


def build_shot(clip: Path, lines: list[tuple[Path, float]], out: Path, size: tuple[int, int],
               fps: int, tail: float, crf: int) -> float:
    """Scale/crop the clip, lay dialogue at their offsets, hold last frame if speech runs long.

    Returns the shot duration in seconds.
    """
    w, h = size
    clip_dur = duration(clip)
    need = max([start + duration(wav) + tail for wav, start in lines] or [0.0])
    total = max(clip_dur, need)
    pad = max(0.0, total - clip_dur)

    inputs = ["-i", str(clip)]
    vf = (f"[0:v]scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h},setsar=1,"
          f"fps={fps},tpad=stop_mode=clone:stop_duration={pad:.3f},format=yuv420p[v]")
    parts = [vf]
    if lines:
        labels = []
        for i, (wav, start) in enumerate(lines, start=1):
            inputs += ["-i", str(wav)]
            ms = int(round(start * 1000))
            parts.append(f"[{i}:a]aresample={AUDIO_RATE},aformat=channel_layouts=stereo,adelay={ms}|{ms}[a{i}]")
            labels.append(f"[a{i}]")
        if len(labels) == 1:
            parts.append(f"{labels[0]}apad[a]")
        else:
            parts.append(f"{''.join(labels)}amix=inputs={len(labels)}:normalize=0:duration=longest,apad[a]")
    else:
        inputs += ["-f", "lavfi", "-i", f"anullsrc=r={AUDIO_RATE}:cl=stereo"]
        parts.append("[1:a]anull[a]")

    ff(*inputs, "-filter_complex", ";".join(parts), "-map", "[v]", "-map", "[a]",
       *_vcodec(crf), *_acodec(), "-t", f"{total:.3f}", str(out))
    return total


def concat(shots: list[Path], out: Path, crossfade: float, crf: int) -> Path:
    if len(shots) == 1:
        shutil.copyfile(shots[0], out)
        return out
    if crossfade <= 0:
        lst = out.with_suffix(".txt")
        lst.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in shots), encoding="utf-8")
        ff("-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out))
        return out

    inputs, parts = [], []
    for p in shots:
        inputs += ["-i", str(p)]
    durs = [duration(p) for p in shots]
    vprev, aprev, offset = "[0:v]", "[0:a]", 0.0
    for i in range(1, len(shots)):
        offset += durs[i - 1] - crossfade
        vo, ao = f"[v{i}]", f"[a{i}]"
        parts.append(f"{vprev}[{i}:v]xfade=transition=fade:duration={crossfade}:offset={offset:.3f}{vo}")
        parts.append(f"{aprev}[{i}:a]acrossfade=d={crossfade}{ao}")
        vprev, aprev = vo, ao
    ff(*inputs, "-filter_complex", ";".join(parts), "-map", vprev, "-map", aprev,
       *_vcodec(crf), *_acodec(), str(out))
    return out


def add_music(video: Path, music: Path, out: Path, volume: float, ducking: bool) -> Path:
    total = duration(video)
    fade_st = max(0.0, total - 1.5)
    bg = f"[1:a]aresample={AUDIO_RATE},aformat=channel_layouts=stereo,volume={volume}[bg]"
    if ducking:
        graph = (f"{bg};[0:a]asplit=2[vo][sc];"
                 f"[bg][sc]sidechaincompress=threshold=0.03:ratio=10:attack=15:release=350[bgd];"
                 f"[vo][bgd]amix=inputs=2:normalize=0:duration=first,afade=t=out:st={fade_st:.3f}:d=1.5[a]")
    else:
        graph = f"{bg};[0:a][bg]amix=inputs=2:normalize=0:duration=first,afade=t=out:st={fade_st:.3f}:d=1.5[a]"
    ff("-i", str(video), "-stream_loop", "-1", "-i", str(music), "-filter_complex", graph,
       "-map", "0:v", "-map", "[a]", "-c:v", "copy", *_acodec(), "-t", f"{total:.3f}", str(out))
    return out


def _ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(cues: list[tuple[float, float, str]], out: Path) -> Path:
    body = "\n".join(f"{i}\n{_ts(a)} --> {_ts(b)}\n{text}\n" for i, (a, b, text) in enumerate(cues, start=1))
    out.write_text(body, encoding="utf-8")
    return out


def burn_subtitles(video: Path, srt: Path, out: Path, style: str, crf: int) -> Path:
    # cwd = srt folder so the filter gets a bare filename (no path escaping issues)
    vf = f"subtitles={srt.name}:force_style='{style}'"
    ff("-i", str(video.resolve()), "-vf", vf, *_vcodec(crf), "-c:a", "copy", str(out.resolve()), cwd=srt.parent)
    return out

"""project.yaml -> typed objects, with validation."""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ASPECTS = ("16:9", "9:16", "1:1")


class ProjectError(ValueError):
    pass


@dataclass
class Character:
    id: str
    description: str
    trigger: str = ""            # trigger word của LoRA nhân vật
    lora: str | None = None      # tên file trong ComfyUI/models/loras
    lora_strength: float = 0.8
    voice: str | None = None     # tên giọng VieNeu

    def prompt(self) -> str:
        return ", ".join(p for p in (self.trigger, self.description) if p)


@dataclass
class Line:
    character: str
    text: str
    start: float = 0.3           # giây, tính từ đầu shot
    voice: str | None = None     # ghi đè giọng của nhân vật
    speed: float | None = None


@dataclass
class Shot:
    id: str
    image_prompt: str = ""
    motion_prompt: str = ""
    camera: str = ""
    characters: list[str] = field(default_factory=list)
    end_image_prompt: str | None = None   # có -> dùng First-Last-Frame
    image: str | None = None              # ảnh có sẵn thay cho keyframe sinh
    end_image: str | None = None
    seed: int | None = None
    length: int | None = None             # số frame, 4n+1
    keyframe_pick: int = 0
    negative: str = ""
    dialogue: list[Line] = field(default_factory=list)


@dataclass
class Project:
    name: str
    root: Path                   # thư mục chứa project.yaml
    aspect: str
    characters: dict[str, Character]
    shots: list[Shot]
    style: str | None = None
    bgm: str | None = None
    bgm_volume: float | None = None
    subtitles: str | None = None
    crossfade: float | None = None

    def resolve(self, rel: str | None) -> Path | None:
        if not rel:
            return None
        p = Path(rel)
        return p if p.is_absolute() else (self.root / p).resolve()

    def seed_for(self, shot: Shot, salt: str = "") -> int:
        if shot.seed is not None:
            return int(shot.seed)
        return zlib.crc32(f"{self.name}/{shot.id}/{salt}".encode()) & 0x7FFFFFFF


def load_project(path: str | Path) -> Project:
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    name = str(data.get("name") or path.parent.name)
    aspect = str(data.get("aspect", "16:9"))
    if aspect not in ASPECTS:
        raise ProjectError(f"aspect phải là một trong {ASPECTS}, nhận '{aspect}'")

    chars = {}
    for cid, c in (data.get("characters") or {}).items():
        if not isinstance(c, dict) or not c.get("description"):
            raise ProjectError(f"nhân vật '{cid}' thiếu description")
        chars[cid] = Character(id=cid, **c)

    shots: list[Shot] = []
    seen = set()
    for i, raw in enumerate(data.get("shots") or []):
        raw = dict(raw)
        sid = str(raw.pop("id", f"s{i + 1:02d}"))
        if sid in seen:
            raise ProjectError(f"trùng id shot '{sid}'")
        seen.add(sid)
        lines = [Line(**ln) for ln in raw.pop("dialogue", None) or []]
        shot = Shot(id=sid, dialogue=lines, **raw)
        for cid in shot.characters + [ln.character for ln in lines]:
            if cid not in chars:
                raise ProjectError(f"shot '{sid}' dùng nhân vật chưa khai báo '{cid}'")
        if not shot.image and not shot.image_prompt:
            raise ProjectError(f"shot '{sid}' cần image_prompt hoặc image")
        if shot.length is not None and (shot.length - 1) % 4:
            raise ProjectError(f"shot '{sid}': length phải có dạng 4n+1 (vd 49, 65, 81)")
        shots.append(shot)
    if not shots:
        raise ProjectError("project chưa có shot nào")

    return Project(
        name=name, root=path.parent, aspect=aspect, characters=chars, shots=shots,
        style=data.get("style"), bgm=data.get("bgm"), bgm_volume=data.get("bgm_volume"),
        subtitles=data.get("subtitles"), crossfade=data.get("crossfade"),
    )

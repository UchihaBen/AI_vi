"""Load config/default.yaml, then config/local.yaml (or --config), then env vars."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

_ENV_OVERRIDES = {
    "COMFY_URL": ("comfy", "url"),
    "TTS_BASE_URL": ("tts", "base_url"),
    "TTS_API_KEY": ("tts", "api_key"),
    "LLM_BASE_URL": ("llm", "base_url"),
    "LLM_API_KEY": ("llm", "api_key"),
    "LLM_MODEL": ("llm", "model"),
}


def deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_settings(config_path: str | None = None) -> dict:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    s = _read_yaml(ROOT / "config" / "default.yaml")
    local = Path(config_path) if config_path else ROOT / "config" / "local.yaml"
    if local.exists():
        s = deep_merge(s, _read_yaml(local))
    elif config_path:
        raise FileNotFoundError(config_path)

    for env, (sec, key) in _ENV_OVERRIDES.items():
        val = os.environ.get(env)
        if val:
            s.setdefault(sec, {})[key] = val
    return s

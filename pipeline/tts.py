"""Client for the VieNeu-TTS server (OpenAI-style /v1/audio/speech)."""
from __future__ import annotations

import io
import time
import wave
from pathlib import Path

import requests


class TTSError(RuntimeError):
    pass


class VieNeuTTS:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.url = cfg["base_url"].rstrip("/")
        self.http = requests.Session()
        if cfg.get("api_key"):
            self.http.headers["Authorization"] = f"Bearer {cfg['api_key']}"

    @staticmethod
    def _err(r: requests.Response) -> str:
        try:
            j = r.json()
            return j.get("error", {}).get("message") or str(j.get("detail") or j)
        except ValueError:
            return r.text[:500] or str(r.status_code)

    def health(self) -> dict:
        r = self.http.get(f"{self.url}/health", timeout=15)
        r.raise_for_status()
        return r.json()

    def voices(self) -> list[dict]:
        r = self.http.get(f"{self.url}/v1/voices", timeout=30)
        if not r.ok:
            raise TTSError(f"Không lấy được danh sách giọng ({r.status_code}): {self._err(r)}")
        return r.json().get("data", [])

    def speech(self, text: str, out: Path, voice: str | None = None, speed: float | None = None) -> Path:
        c = self.cfg
        payload = {
            "model": c.get("model", "vieneu-v3-turbo"),
            "input": text,
            "voice": voice or c.get("default_voice"),
            "response_format": "wav",
            "sample_rate": c.get("sample_rate", 48000),
            "temperature": c.get("temperature", 0.8),
            "top_k": c.get("top_k", 25),
            "top_p": c.get("top_p", 0.95),
            "repetition_penalty": c.get("repetition_penalty", 1.2),
        }
        if speed:
            payload["speed"] = speed

        retries = int(c.get("retries", 3))
        for attempt in range(1, retries + 1):
            try:
                r = self.http.post(f"{self.url}/v1/audio/speech", json=payload, timeout=c.get("timeout_sec", 300))
            except requests.RequestException as e:
                if attempt == retries:
                    raise TTSError(f"Không kết nối được TTS: {e}") from e
                time.sleep(2 * attempt)
                continue
            if r.ok:
                break
            if r.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                time.sleep(2 * attempt)
                continue
            raise TTSError(f"TTS lỗi ({r.status_code}): {self._err(r)}")

        data = r.content
        out.parent.mkdir(parents=True, exist_ok=True)
        if data[:4] == b"RIFF":
            out.write_bytes(data)
        else:  # raw PCM s16le mono
            rate = int(r.headers.get("X-Sample-Rate") or payload["sample_rate"])
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(data)
            out.write_bytes(buf.getvalue())
        return out


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())

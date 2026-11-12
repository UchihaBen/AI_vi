"""Minimal ComfyUI HTTP API client (queue prompt -> poll history -> download)."""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import requests


class ComfyError(RuntimeError):
    pass


class ComfyClient:
    def __init__(self, url: str, timeout_sec: int = 3600, poll_sec: float = 2.0):
        self.url = url.rstrip("/")
        self.timeout = timeout_sec
        self.poll = poll_sec
        self.client_id = uuid.uuid4().hex
        self.http = requests.Session()

    # ---- info ----
    def system_stats(self) -> dict:
        r = self.http.get(f"{self.url}/system_stats", timeout=10)
        r.raise_for_status()
        return r.json()

    def object_info(self, node: str) -> dict | None:
        r = self.http.get(f"{self.url}/object_info/{node}", timeout=30)
        if r.status_code != 200:
            return None
        return r.json().get(node)

    def input_options(self, node: str, field: str) -> list[str] | None:
        """List the combo values of a node input (e.g. available model files)."""
        info = self.object_info(node)
        if not info:
            return None
        inputs = {**info.get("input", {}).get("required", {}), **info.get("input", {}).get("optional", {})}
        spec = inputs.get(field)
        if not spec:
            return None
        if isinstance(spec[0], list):                     # old format: [[opts], {...}]
            return spec[0]
        if spec[0] == "COMBO" and len(spec) > 1:          # new format: ["COMBO", {"options": [...]}]
            return spec[1].get("options", [])
        return None

    # ---- jobs ----
    def upload_image(self, path: Path) -> str:
        """Upload to ComfyUI/input and return the name to put in LoadImage."""
        name = f"pl_{uuid.uuid4().hex[:10]}_{path.name}"
        with open(path, "rb") as f:
            r = self.http.post(
                f"{self.url}/upload/image",
                files={"image": (name, f, "image/png")},
                data={"overwrite": "true", "type": "input"},
                timeout=120,
            )
        r.raise_for_status()
        d = r.json()
        return f"{d['subfolder']}/{d['name']}" if d.get("subfolder") else d["name"]

    def queue(self, workflow: dict) -> str:
        r = self.http.post(f"{self.url}/prompt", json={"prompt": workflow, "client_id": self.client_id}, timeout=60)
        if r.status_code != 200:
            raise ComfyError(f"ComfyUI từ chối workflow ({r.status_code}): {r.text[:3000]}")
        return r.json()["prompt_id"]

    def wait(self, prompt_id: str, label: str = "") -> dict:
        t0 = time.time()
        last_print = 0.0
        while True:
            r = self.http.get(f"{self.url}/history/{prompt_id}", timeout=30)
            r.raise_for_status()
            entry = r.json().get(prompt_id)
            if entry:
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    msgs = [m for m in status.get("messages", []) if m and m[0] == "execution_error"]
                    detail = msgs[-1][1] if msgs else status
                    if isinstance(detail, dict):
                        detail = f"{detail.get('node_type')}: {detail.get('exception_message')}"
                    raise ComfyError(f"Lỗi khi chạy workflow {label}: {detail}")
                if status.get("completed", True):
                    return entry
            elapsed = time.time() - t0
            if elapsed > self.timeout:
                raise ComfyError(f"Quá thời gian chờ ({self.timeout}s) cho {label}")
            if elapsed - last_print >= 30:
                print(f"    … đang chạy {label} ({int(elapsed)}s)", flush=True)
                last_print = elapsed
            time.sleep(self.poll)

    @staticmethod
    def output_files(entry: dict, node_id: str | None = None) -> list[dict]:
        files = []
        for nid, out in (entry.get("outputs") or {}).items():
            if node_id and nid != node_id:
                continue
            for items in out.values():
                if isinstance(items, list):
                    files += [it for it in items if isinstance(it, dict) and "filename" in it]
        return [f for f in files if f.get("type", "output") == "output"]

    def download(self, info: dict, dest: Path) -> Path:
        params = {"filename": info["filename"], "subfolder": info.get("subfolder", ""), "type": info.get("type", "output")}
        r = self.http.get(f"{self.url}/view", params=params, timeout=300)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        return dest

    def run(self, workflow: dict, label: str = "", output_node: str | None = None) -> list[dict]:
        pid = self.queue(workflow)
        entry = self.wait(pid, label)
        files = self.output_files(entry, output_node)
        if not files:
            raise ComfyError(f"Workflow {label} chạy xong nhưng không có file đầu ra")
        return files

    def free(self) -> None:
        """Unload models from VRAM (between SDXL and Wan stages)."""
        try:
            self.http.post(f"{self.url}/free", json={"unload_models": True, "free_memory": True}, timeout=30)
        except requests.RequestException:
            pass

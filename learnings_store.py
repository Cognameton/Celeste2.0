"""LearningsStore — append-only capture of corrections and extracted patterns.

Entries live in self/learnings/learnings.jsonl.  The reflector appends here;
entries are promoted once they've been acted on (written to skills or playbook).
"""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _short_id(seed: str) -> str:
    return hashlib.sha1(seed.encode()).hexdigest()[:8]


class LearningsStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._path = self.root / "learnings.jsonl"
        self._lock = threading.Lock()

    # ---- write ----

    def append(self, type: str, content: str, trigger: str = "") -> None:
        now = _now()
        entry: dict[str, Any] = {
            "id": _short_id(now + content),
            "timestamp": now,
            "type": type,       # "correction" | "skill_draft" | "pattern"
            "content": content,
            "trigger": trigger,
            "promoted": False,
        }
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    def promote(self, entry_id: str) -> None:
        entries = self._load_locked()
        for e in entries:
            if e.get("id") == entry_id:
                e["promoted"] = True
        with self._lock:
            with self._path.open("w", encoding="utf-8") as f:
                for e in entries:
                    f.write(json.dumps(e) + "\n")

    # ---- read ----

    def pending(self, type: str | None = None) -> list[dict[str, Any]]:
        return [
            e for e in self._load_locked()
            if not e.get("promoted") and (type is None or e.get("type") == type)
        ]

    def recent(self, n: int = 20) -> list[dict[str, Any]]:
        return self._load_locked()[-n:]

    # ---- internal ----

    def _load_locked(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self._path.exists():
                return []
            raw = self._path.read_text(encoding="utf-8")
        result: list[dict[str, Any]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                result.append(json.loads(line))
            except Exception:
                pass
        return result

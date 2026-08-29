"""Wants — Synthia's persistent goals and curiosities.

Each want is a thing she has decided she would like to know, do, or resolve.
Wants come from three places:

    self           generated during a heartbeat tick
    user-implied   extracted from a user turn (something she noticed)
    operator       given to her directly by Shane

Active wants live in ``self/wants/active.json`` as a list. When a want is
resolved or abandoned it is appended to ``self/wants/completed.jsonl`` and
removed from active. The completed log is append-only history.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _short_id(seed: str) -> str:
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]


@dataclass
class Want:
    id: str
    text: str
    origin: str            # "self" | "user-implied" | "operator"
    priority: int          # 1 (low) .. 5 (high)
    created: str
    last_touched: str
    status: str = "active"  # "active" | "in-progress" | "resolved" | "abandoned"
    notes: list[str] = field(default_factory=list)

    @classmethod
    def new(cls, text: str, origin: str = "self", priority: int = 3) -> "Want":
        ts = _now()
        return cls(
            id=_short_id(text + ts),
            text=text.strip(),
            origin=origin,
            priority=max(1, min(5, int(priority))),
            created=ts,
            last_touched=ts,
        )

    def touch(self, note: str | None = None) -> None:
        self.last_touched = _now()
        if note:
            self.notes.append(f"{self.last_touched}: {note.strip()}")


class WantsStore:
    """Filesystem-backed store for active and completed wants."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.active_path = self.root / "active.json"
        self.completed_path = self.root / "completed.jsonl"
        self._lock = threading.Lock()
        if not self.active_path.exists():
            self.active_path.write_text("[]", encoding="utf-8")

    # ---- read ----

    def load_active(self) -> list[Want]:
        with self._lock:
            raw = json.loads(self.active_path.read_text(encoding="utf-8") or "[]")
        return [Want(**item) for item in raw]

    def get(self, want_id: str) -> Want | None:
        for w in self.load_active():
            if w.id == want_id:
                return w
        return None

    def for_prompt(self, max_items: int = 10) -> str:
        active = self.load_active()
        if not active:
            return ""
        active.sort(key=lambda w: (-w.priority, w.created))
        lines = []
        for w in active[:max_items]:
            lines.append(f"- [{w.id}] (p{w.priority}, {w.status}, from {w.origin}) {w.text}")
        return "\n".join(lines)

    # ---- write ----

    def add(self, text: str, *, origin: str = "self", priority: int = 3) -> Want:
        want = Want.new(text, origin=origin, priority=priority)
        with self._lock:
            active = self._read_active_locked()
            # de-dupe by exact text match
            for existing in active:
                if existing["text"].lower() == want.text.lower():
                    return Want(**existing)
            active.append(asdict(want))
            self._write_active_locked(active)
        return want

    def advance(self, want_id: str, note: str) -> Want | None:
        return self._mutate(want_id, status="in-progress", note=note)

    def resolve(self, want_id: str, outcome: str) -> Want | None:
        return self._complete(want_id, status="resolved", final_note=outcome)

    def abandon(self, want_id: str, reason: str) -> Want | None:
        return self._complete(want_id, status="abandoned", final_note=reason)

    # ---- internals ----

    def _mutate(self, want_id: str, *, status: str, note: str) -> Want | None:
        with self._lock:
            active = self._read_active_locked()
            for item in active:
                if item["id"] == want_id:
                    item["status"] = status
                    item["last_touched"] = _now()
                    if note:
                        item.setdefault("notes", []).append(
                            f"{item['last_touched']}: {note.strip()}"
                        )
                    self._write_active_locked(active)
                    return Want(**item)
        return None

    def _complete(self, want_id: str, *, status: str, final_note: str) -> Want | None:
        with self._lock:
            active = self._read_active_locked()
            for i, item in enumerate(active):
                if item["id"] == want_id:
                    item["status"] = status
                    item["last_touched"] = _now()
                    if final_note:
                        item.setdefault("notes", []).append(
                            f"{item['last_touched']}: {final_note.strip()}"
                        )
                    completed = active.pop(i)
                    self._write_active_locked(active)
                    with self.completed_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(completed, ensure_ascii=False) + "\n")
                    return Want(**completed)
        return None

    def _read_active_locked(self) -> list[dict[str, Any]]:
        return json.loads(self.active_path.read_text(encoding="utf-8") or "[]")

    def _write_active_locked(self, items: list[dict[str, Any]]) -> None:
        self.active_path.write_text(
            json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8"
        )

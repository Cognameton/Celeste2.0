"""PerformanceStore — append-only log of turn outcomes with skill attribution.

Outcomes live in self/performance/outcomes.jsonl.
Each entry records whether the previous assistant turn succeeded, partially
succeeded, or failed, along with which skill slugs were active at the time.
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


class PerformanceStore:
    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._path = self.root / "outcomes.jsonl"
        self._lock = threading.Lock()

    # ---- write ----

    def record(
        self,
        outcome: float,                     # 1.0 success | 0.5 partial | 0.0 failure
        active_skill_slugs: list[str],
        user_snippet: str = "",
        answer_snippet: str = "",
    ) -> None:
        now = _now()
        entry: dict[str, Any] = {
            "id": _short_id(now + str(outcome)),
            "timestamp": now,
            "outcome": outcome,
            "outcome_label": _label(outcome),
            "active_skills": list(active_skill_slugs),
            "user_snippet": user_snippet[:200],
            "answer_snippet": answer_snippet[:200],
        }
        with self._lock:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    # ---- read ----

    def skill_stats(self, slug: str) -> dict[str, Any]:
        """Return {count, confidence, successes, partials, failures} for a skill slug."""
        counts = {"successes": 0, "partials": 0, "failures": 0}
        for entry in self._load():
            if slug in entry.get("active_skills", []):
                val = float(entry.get("outcome", 0.5))
                if val >= 0.9:
                    counts["successes"] += 1
                elif val >= 0.4:
                    counts["partials"] += 1
                else:
                    counts["failures"] += 1
        total = sum(counts.values())
        confidence = (
            (counts["successes"] + 0.5 * counts["partials"]) / total
            if total >= 5 else 1.0
        )
        return {"count": total, "confidence": round(confidence, 3), **counts}

    def recent_summary(self, n: int = 20) -> str:
        """One-line summary of the last n outcomes for use in the heartbeat prompt."""
        entries = self._load()[-n:]
        if not entries:
            return ""
        labels = [_label(float(e.get("outcome", 0.5))) for e in entries]
        s = labels.count("success")
        p = labels.count("partial")
        f = labels.count("failure")
        return f"Last {len(entries)} turns: {s} success, {p} partial, {f} failure"

    def recent(self, n: int = 20) -> list[dict[str, Any]]:
        return self._load()[-n:]

    # ---- internal ----

    def _load(self) -> list[dict[str, Any]]:
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


def _label(outcome: float) -> str:
    if outcome >= 0.9:
        return "success"
    if outcome >= 0.4:
        return "partial"
    return "failure"

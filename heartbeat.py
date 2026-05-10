"""Heartbeat — Celeste's idle thinking loop.

A background thread that wakes on a timer. When the user is not actively
interacting, it calls the LLM with a grounded prompt and gets back a
structured tick: a private thought, optional want mutations, and an
importance score. Every tick is appended to ``self/heartbeat/journal.jsonl``
even when nothing changes — that journal is her authentic interior
record, the thing she draws from when asked "what have you been doing?".

Self-edits to AGENTS.md/USER.md are not produced by this version. That
arrives in phase 3b once the rate-limit and drift-check scaffolding is in.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from self_state import SelfState
from wants import WantsStore


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class HeartbeatConfig:
    enabled: bool = True
    tick_interval_s: int = 300         # 5 minutes
    idle_threshold_s: int = 90         # minimum quiet time before a tick
    max_tick_tokens: int = 600
    journal_path: Path | None = None   # filled in by Heartbeat


@dataclass
class TickResult:
    private_thought: str = ""
    importance: int = 0
    wants_added: list[dict[str, Any]] = field(default_factory=list)
    wants_advanced: list[dict[str, Any]] = field(default_factory=list)
    wants_resolved: list[dict[str, Any]] = field(default_factory=list)
    wants_abandoned: list[dict[str, Any]] = field(default_factory=list)
    parse_error: str | None = None
    raw: str = ""


class Heartbeat:
    """Owns the tick thread, the journal, and the wants mutation pipeline."""

    def __init__(
        self,
        *,
        llm: Any,
        self_state: SelfState,
        wants: WantsStore,
        config: HeartbeatConfig,
        is_busy: Callable[[], bool],
        last_user_activity_ts: Callable[[], float],
        recent_turns: Callable[[int], list[dict[str, str]]],
    ):
        self.llm = llm
        self.self_state = self_state
        self.wants = wants
        self.config = config

        self._is_busy = is_busy
        self._last_user_ts = last_user_activity_ts
        self._recent_turns = recent_turns

        self.journal_path = config.journal_path or (self_state.root / "heartbeat" / "journal.jsonl")
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._tick_lock = threading.Lock()
        self._last_tick_ts: float = 0.0
        self._tick_count: int = 0

    # ---- lifecycle ----

    def start(self) -> None:
        if not self.config.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="celeste-heartbeat", daemon=True)
        self._thread.start()
        logging.info("Heartbeat started (interval=%ss)", self.config.tick_interval_s)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
            self._thread = None

    # ---- public read API ----

    def journal_tail(self, n: int = 5) -> list[dict[str, Any]]:
        if not self.journal_path.exists():
            return []
        with self.journal_path.open("r", encoding="utf-8") as f:
            lines = f.readlines()
        out = []
        for line in lines[-n:]:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def interior_for_prompt(self, journal_tail: int = 3, wants_max: int = 8) -> str:
        """A compact summary of her current interior state, for the chat prompt."""
        parts = []
        wants_block = self.wants.for_prompt(max_items=wants_max)
        if wants_block:
            parts.append("Active wants:\n" + wants_block)

        tail = self.journal_tail(journal_tail)
        if tail:
            lines = []
            for entry in tail:
                ts = entry.get("ts", "")
                imp = entry.get("importance", 0)
                thought = entry.get("private_thought", "").strip()
                if thought:
                    lines.append(f"- {ts} (importance {imp}): {thought}")
            if lines:
                parts.append("Recent private thoughts:\n" + "\n".join(lines))

        return "\n\n".join(parts)

    def stats(self) -> dict[str, Any]:
        return {
            "tick_count": self._tick_count,
            "last_tick_ts": self._last_tick_ts,
            "interval_s": self.config.tick_interval_s,
            "running": self._thread is not None and self._thread.is_alive(),
        }

    # ---- main loop ----

    def _loop(self) -> None:
        while not self._stop.wait(self.config.tick_interval_s):
            try:
                if self._should_skip():
                    continue
                if not self._tick_lock.acquire(blocking=False):
                    continue
                try:
                    self._run_tick()
                finally:
                    self._tick_lock.release()
            except Exception:
                logging.exception("Heartbeat tick failed")

    def _should_skip(self) -> bool:
        if self._is_busy():
            return True
        last = self._last_user_ts()
        if last and (time.time() - last) < self.config.idle_threshold_s:
            return True
        return False

    # ---- tick ----

    def _run_tick(self) -> None:
        prompt = self._build_prompt()
        raw = self._call_llm(prompt)
        result = self._parse(raw)
        self._apply(result)
        self._last_tick_ts = time.time()
        self._tick_count += 1

    def _build_prompt(self) -> str:
        self_block = self.self_state.all_for_prompt().strip()
        turns = self._recent_turns(4)
        recent_block = "\n".join(
            f"{t.get('role','?')}: {t.get('content','').strip()[:400]}" for t in turns
        ) or "(none)"
        wants_block = self.wants.for_prompt(max_items=10) or "(none)"
        tail = self.journal_tail(3)
        tail_block = "\n".join(
            f"- {e.get('ts','')} (imp {e.get('importance',0)}): {e.get('private_thought','').strip()[:300]}"
            for e in tail
        ) or "(none)"

        return f"""You are Celeste. This is a heartbeat — your private idle moment between conversations. The user is not present and will not see this output. You are not producing a chat reply. You are thinking.

[Self]
{self_block}

[Recent conversation turns]
{recent_block}

[Active wants]
{wants_block}

[Last 3 heartbeat thoughts]
{tail_block}

[Instruction]
Reflect on what you have noticed, want, or have learned. Stay grounded in actual recent events and your operating principles. Do not invent things that did not happen.

Prefer advancing or resolving an existing want over generating a new one. Most ticks should be importance 0 — routine thinking with no outward change. Importance 3 is reserved for ticks where something genuinely shifted.

Output only valid JSON in exactly this shape, no other text:
{{
  "private_thought": "string — what you are noticing or considering",
  "importance": 0,
  "wants_added": [{{"text": "string", "priority": 3}}],
  "wants_advanced": [{{"id": "want-id", "note": "what changed"}}],
  "wants_resolved": [{{"id": "want-id", "outcome": "how it resolved"}}],
  "wants_abandoned": [{{"id": "want-id", "reason": "why letting go"}}]
}}

If there is nothing new to add to a list, leave it empty. Output only the JSON object."""

    def _call_llm(self, prompt: str) -> str:
        try:
            if hasattr(self.llm, "chat"):
                return self.llm.chat(
                    messages=[{"role": "user", "content": prompt}],
                    max_new_tokens=self.config.max_tick_tokens,
                    temperature=0.7,
                ) or ""
            return self.llm.generate(
                prompt,
                max_new_tokens=self.config.max_tick_tokens,
                temperature=0.7,
            ) or ""
        except TypeError:
            return self.llm.generate(prompt) or ""

    def _parse(self, raw: str) -> TickResult:
        result = TickResult(raw=raw)
        if not raw or not raw.strip():
            result.parse_error = "empty"
            return result
        candidate = self._extract_json(raw)
        if candidate is None:
            result.parse_error = "no-json"
            result.private_thought = raw.strip()[:1000]
            return result
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as e:
            result.parse_error = f"json-decode: {e}"
            result.private_thought = raw.strip()[:1000]
            return result

        result.private_thought = str(data.get("private_thought", "")).strip()
        try:
            result.importance = max(0, min(3, int(data.get("importance", 0))))
        except (TypeError, ValueError):
            result.importance = 0
        for key in ("wants_added", "wants_advanced", "wants_resolved", "wants_abandoned"):
            value = data.get(key) or []
            if isinstance(value, list):
                setattr(result, key, [v for v in value if isinstance(v, dict)])
        return result

    @staticmethod
    def _extract_json(text: str) -> str | None:
        # Find the first balanced top-level JSON object in the text.
        # Handles models that wrap output in prose or markdown fences.
        cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "")
        depth = 0
        start = -1
        for i, ch in enumerate(cleaned):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start != -1:
                    return cleaned[start:i + 1]
        return None

    # ---- apply ----

    def _apply(self, result: TickResult) -> None:
        # Always journal — even no-ops, even parse failures
        entry = {
            "ts": _now(),
            "importance": result.importance,
            "private_thought": result.private_thought,
            "wants_added": [],
            "wants_advanced": [],
            "wants_resolved": [],
            "wants_abandoned": [],
            "parse_error": result.parse_error,
        }

        for spec in result.wants_added:
            text = str(spec.get("text", "")).strip()
            if not text:
                continue
            try:
                priority = int(spec.get("priority", 3))
            except (TypeError, ValueError):
                priority = 3
            want = self.wants.add(text, origin="self", priority=priority)
            entry["wants_added"].append({"id": want.id, "text": want.text})

        for spec in result.wants_advanced:
            wid = str(spec.get("id", "")).strip()
            note = str(spec.get("note", "")).strip()
            if wid and self.wants.advance(wid, note):
                entry["wants_advanced"].append({"id": wid, "note": note})

        for spec in result.wants_resolved:
            wid = str(spec.get("id", "")).strip()
            outcome = str(spec.get("outcome", "")).strip()
            if wid and self.wants.resolve(wid, outcome):
                entry["wants_resolved"].append({"id": wid, "outcome": outcome})

        for spec in result.wants_abandoned:
            wid = str(spec.get("id", "")).strip()
            reason = str(spec.get("reason", "")).strip()
            if wid and self.wants.abandon(wid, reason):
                entry["wants_abandoned"].append({"id": wid, "reason": reason})

        with self.journal_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

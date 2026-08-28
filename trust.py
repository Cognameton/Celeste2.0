"""Trust ladder — per-channel autonomy tiers, earned from evidence.

Phases 1-8 set trust to maximum: Celeste edited her own operating files
directly. Nova 2.0 set it to minimum: deterministic code owned every write.
This module makes trust a *variable* — per channel, computed from her track
record, and revocable.

Four tiers, low to high:

    observe      logged, nothing applies       (demotion floor)
    propose      applies only on approval
    review       applies immediately, visibly  (INITIAL tier for every channel)
    autonomous   applies silently              (earned, never granted)

Promotion is arithmetic, not vibes: N applied proposals on a channel, over a
minimum wall-clock span at the current tier, with zero reverts and zero drift
flags in that window. Any revert or drift flag demotes one tier immediately.
The operator sits above the ladder and can set any tier directly.

State lives in self/governor/trust.json; evidence in
self/governor/track_record.jsonl. Both are committed by Governor.flush().
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

TIERS: tuple[str, ...] = ("observe", "propose", "review", "autonomous")
INITIAL_TIER = "review"

PROMOTION_MIN_APPLIED = 10      # applied proposals on the channel since `since`
PROMOTION_MIN_SPAN_DAYS = 7     # wall-clock at the current tier

EVENTS = ("applied", "rejected", "revert", "drift_flag", "override", "tier_change")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except Exception:
        return None


def tier_index(tier: str) -> int:
    try:
        return TIERS.index(tier)
    except ValueError:
        return TIERS.index(INITIAL_TIER)


class TrackRecord:
    """Append-only per-channel evidence log: self/governor/track_record.jsonl."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(self, channel: str, event: str, *, target: str = "",
               detail: str = "", proposal_id: str = "") -> dict[str, Any]:
        entry = {
            "ts": _now(),
            "channel": channel,
            "event": event,
            "target": target,
            "detail": detail,
            "proposal_id": proposal_id,
        }
        try:
            with self._lock:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception:
            logging.exception("TrackRecord write failed")
        return entry

    def _load(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            with self._lock:
                raw = self.path.read_text(encoding="utf-8")
        except Exception:
            return []
        out: list[dict[str, Any]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def since(self, channel: str, ts: str) -> list[dict[str, Any]]:
        """Events for one channel at or after `ts`."""
        cutoff = _parse(ts)
        rows = [e for e in self._load() if e.get("channel") == channel]
        if cutoff is None:
            return rows
        kept = []
        for e in rows:
            when = _parse(e.get("ts", ""))
            if when is None or when >= cutoff:
                kept.append(e)
        return kept

    def tail(self, n: int = 50, channel: str | None = None) -> list[dict[str, Any]]:
        rows = self._load()
        if channel:
            rows = [e for e in rows if e.get("channel") == channel]
        return rows[-n:]

    def counts(self, channel: str, ts: str) -> dict[str, int]:
        out: dict[str, int] = {e: 0 for e in EVENTS}
        for e in self.since(channel, ts):
            kind = e.get("event", "")
            if kind in out:
                out[kind] += 1
        return out


class TrustLadder:
    """Owns trust.json and the promote/demote arithmetic."""

    def __init__(self, governor_dir: Path | str, *, channels: tuple[str, ...] = (),
                 on_event: Any = None):
        self.dir = Path(governor_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "trust.json"
        self.track = TrackRecord(self.dir / "track_record.jsonl")
        self._channels = tuple(channels)
        self._on_event = on_event
        self._lock = threading.RLock()
        if not self.path.exists():
            self._write({"version": 1, "channels": {
                c: {"tier": INITIAL_TIER, "since": _now(), "reason": "initial"}
                for c in self._channels
            }})

    # ---- state ----

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": 1, "channels": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return {"version": 1, "channels": {}}
            data.setdefault("channels", {})
            if not isinstance(data["channels"], dict):
                data["channels"] = {}
            return data
        except Exception:
            # Fail toward today's behavior, never toward a lockout.
            logging.exception("trust.json unreadable; defaulting to %s", INITIAL_TIER)
            return {"version": 1, "channels": {}}

    def _write(self, data: dict[str, Any]) -> None:
        try:
            self.path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            logging.exception("trust.json write failed")

    def tier_for(self, channel: str) -> str:
        entry = self._read()["channels"].get(channel)
        if not isinstance(entry, dict):
            return INITIAL_TIER
        tier = entry.get("tier")
        return tier if tier in TIERS else INITIAL_TIER

    def tiers(self) -> dict[str, dict[str, Any]]:
        data = self._read()["channels"]
        out: dict[str, dict[str, Any]] = {}
        for channel in (self._channels or tuple(data.keys())):
            entry = data.get(channel)
            if not isinstance(entry, dict) or entry.get("tier") not in TIERS:
                entry = {"tier": INITIAL_TIER, "since": _now(), "reason": "default"}
            out[channel] = dict(entry)
        return out

    def _since_of(self, channel: str) -> str:
        entry = self._read()["channels"].get(channel)
        if isinstance(entry, dict) and entry.get("since"):
            return str(entry["since"])
        return _now()

    def set_tier(self, channel: str, tier: str, *, reason: str = "",
                 operator: bool = False) -> bool:
        """Set a channel's tier. Returns True if it changed."""
        if tier not in TIERS:
            return False
        with self._lock:
            data = self._read()
            current = data["channels"].get(channel) or {}
            old = current.get("tier") if current.get("tier") in TIERS else INITIAL_TIER
            data["channels"][channel] = {
                "tier": tier,
                "since": _now(),
                "reason": reason or ("operator override" if operator else "computed"),
            }
            self._write(data)
        if operator:
            self.track.record(channel, "override", target=tier,
                              detail=reason or f"{old} -> {tier}")
        if old != tier:
            self.track.record(channel, "tier_change", target=tier,
                              detail=f"{old} -> {tier}: {reason}".strip())
            self._emit(f"Trust • {channel}: {old} → {tier}"
                       + (f" ({reason})" if reason else ""))
            return True
        return False

    # ---- evidence ----

    def note_applied(self, channel: str, proposal_id: str = "", target: str = "") -> None:
        self.track.record(channel, "applied", target=target, proposal_id=proposal_id)

    def note_rejected(self, channel: str, reason: str = "", target: str = "") -> None:
        self.track.record(channel, "rejected", target=target, detail=reason)

    def _demote(self, channel: str, cause: str, detail: str) -> str:
        current = self.tier_for(channel)
        lowered = TIERS[max(0, tier_index(current) - 1)]
        if lowered != current:
            self.set_tier(channel, lowered, reason=f"{cause}: {detail}".strip(": "))
        else:
            # Already at the floor — still resets the clock so the window is honest.
            self.set_tier(channel, current, reason=f"{cause} at floor")
        return lowered

    def record_revert(self, channel: str, *, target: str = "", note: str = "") -> str:
        """An operator undid something on this channel. Demote one tier."""
        self.track.record(channel, "revert", target=target, detail=note)
        return self._demote(channel, "revert", note or target)

    def record_drift_flag(self, channel: str, *, detail: str = "") -> str:
        """Phase 11 will call this. Demote one tier."""
        self.track.record(channel, "drift_flag", detail=detail)
        return self._demote(channel, "drift flag", detail)

    # ---- the arithmetic ----

    def _eligible(self, channel: str) -> bool:
        tier = self.tier_for(channel)
        if tier_index(tier) >= len(TIERS) - 1:
            return False
        since = self._since_of(channel)
        started = _parse(since)
        if started is None:
            return False
        if datetime.now(timezone.utc) - started < timedelta(days=PROMOTION_MIN_SPAN_DAYS):
            return False
        counts = self.track.counts(channel, since)
        if counts["revert"] or counts["drift_flag"]:
            return False
        return counts["applied"] >= PROMOTION_MIN_APPLIED

    def evaluate(self) -> list[dict[str, str]]:
        """Promote every eligible channel one tier. Returns the changes made."""
        changes: list[dict[str, str]] = []
        for channel in (self._channels or tuple(self._read()["channels"].keys())):
            try:
                if not self._eligible(channel):
                    continue
                old = self.tier_for(channel)
                new = TIERS[tier_index(old) + 1]
                counts = self.track.counts(channel, self._since_of(channel))
                if self.set_tier(channel, new,
                                 reason=f"earned: {counts['applied']} applied, "
                                        f"{PROMOTION_MIN_SPAN_DAYS}d clean"):
                    changes.append({"channel": channel, "from": old, "to": new})
            except Exception:
                logging.exception("Trust evaluate failed for channel %s", channel)
        return changes

    def _emit(self, msg: str) -> None:
        if callable(self._on_event):
            try:
                self._on_event(msg)
            except Exception:
                pass

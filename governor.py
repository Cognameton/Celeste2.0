"""Governor — the deterministic gate for agent-originated self-writes.

Every write the agent proposes (heartbeat, reflector, ReAct) is expressed
as a Proposal, validated by deterministic code, applied via a caller-supplied
thunk, and recorded append-only in self/governor/ledger.jsonl. The model
never writes directly; the governor never generates content. Operator
actions (GUI persona edit, config changes) do NOT pass through here.

Phase 9 contract: this is a refactor-with-ledger. Observable behavior is
unchanged — the same writes apply, the same rejections fire, with the same
rejection wording — but every agent-origin write now flows through one gate
and leaves one audit trail.
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from user_model import WRITABLE_SECTIONS

CHANNELS = ("self_edit", "want", "skill", "user_model", "reflection_rule", "tool")

# Gates migrated from heartbeat.py — this module is now their only home.
ALLOWED_SELF_EDIT_FILES = frozenset({"AGENTS.md", "USER.md"})
EDIT_COOLDOWN_S = 6 * 3600              # one self-edit per file per 6 hours
SKILL_PROPOSAL_COOLDOWN_S = 24 * 3600   # one new skill proposal per 24 hours
USER_MODEL_COOLDOWN_S = 2 * 3600        # one user-model update per section per 2 hours
MIN_REASON_LEN = 20                     # reason must be a real sentence, not a label
MAX_PAYLOAD_CHARS = 8000                # generous ceiling; drift checks are tighter

# Rate-limit key used by heartbeat's legacy edit_log.json for skill proposals.
SKILL_RATE_KEY = "__skill_last_proposed__"

_GIT_AUTHOR = ("user.name=Celeste", "user.email=celeste@local")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _short_id(seed: str) -> str:
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8]


@dataclass
class Proposal:
    id: str
    ts: str
    channel: str
    origin: str        # "heartbeat" | "reflector" | "react"
    action: str
    target: str
    payload: dict
    evidence: str = ""

    @classmethod
    def new(cls, *, channel: str, origin: str, action: str, target: str,
            payload: dict, evidence: str = "") -> "Proposal":
        ts = _now()
        return cls(
            id=_short_id(f"{channel}|{target}|{ts}|{action}"),
            ts=ts,
            channel=channel,
            origin=origin,
            action=action,
            target=target,
            payload=payload or {},
            evidence=evidence,
        )


@dataclass
class Decision:
    proposal_id: str
    verdict: str            # "applied" | "rejected" | "error"
    reason: str = ""
    validator: str = ""
    result: Any = None      # apply_fn's return value; never serialized

    @property
    def applied(self) -> bool:
        return self.verdict == "applied"


Validator = Callable[[Proposal], "str | None"]   # None = pass, str = rejection reason


# ---- rate-limit keying (must reproduce heartbeat's edit_log keys exactly) ----

def _rate_key(p: Proposal) -> str | None:
    if p.channel == "self_edit":
        return p.target
    if p.channel == "skill" and p.action == "create":
        return SKILL_RATE_KEY
    if p.channel == "user_model" and p.action == "upsert":
        return f"__user_model_{p.target}__"
    return None


def _cooldown_for(p: Proposal) -> int:
    if p.channel == "skill":
        return SKILL_PROPOSAL_COOLDOWN_S
    if p.channel == "user_model":
        return USER_MODEL_COOLDOWN_S
    return EDIT_COOLDOWN_S


def _rate_limit_message(p: Proposal) -> str:
    # Legacy wording, preserved so journal history stays comparable.
    if p.channel == "skill":
        return "rate-limited (24h)"
    return "rate-limited"


class Governor:
    """Single gate for agent-originated writes: validate, apply, record."""

    def __init__(self, self_root: Path | str, *,
                 on_event: Callable[[str], None] | None = None,
                 tool_risk_lookup: Callable[[str], "str | None"] | None = None):
        self.root = Path(self_root)
        self._on_event = on_event
        # Set by Agent once the Executor exists; None means the tool channel
        # is not independently checked here (Executor still enforces).
        self.tool_risk_lookup = tool_risk_lookup

        self.dir = self.root / "governor"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.dir / "ledger.jsonl"
        self.rate_log_path = self.dir / "rate_log.json"
        self._lock = threading.Lock()

        # Seed cooldown state from heartbeat's legacy edit_log.json on first run.
        if not self.rate_log_path.exists():
            legacy = self.root / "heartbeat" / "edit_log.json"
            seed: dict[str, str] = {}
            if legacy.exists():
                try:
                    loaded = json.loads(legacy.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        seed = {str(k): str(v) for k, v in loaded.items()}
                except Exception:
                    seed = {}
            self._write_rate_log(seed)

    # ---- rate log ----

    def _read_rate_log(self) -> dict[str, str]:
        if not self.rate_log_path.exists():
            return {}
        try:
            data = json.loads(self.rate_log_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write_rate_log(self, log: dict[str, str]) -> None:
        self.rate_log_path.write_text(
            json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _cooldown_ok(self, key: str, cooldown_s: int) -> bool:
        last_str = self._read_rate_log().get(key)
        if not last_str:
            return True
        try:
            last_ts = datetime.fromisoformat(str(last_str).replace("Z", "+00:00"))
            elapsed = (datetime.now(timezone.utc) - last_ts).total_seconds()
            return elapsed >= cooldown_s
        except Exception:
            return True

    # ---- built-in validators ----

    def v_channel(self, p: Proposal) -> str | None:
        if p.channel not in CHANNELS:
            return f"unknown channel '{p.channel}'"
        return None

    def v_payload_size(self, p: Proposal) -> str | None:
        try:
            size = len(json.dumps(p.payload, ensure_ascii=False, default=str))
        except Exception:
            size = len(str(p.payload))
        if size > MAX_PAYLOAD_CHARS:
            return f"payload too large ({size} chars > {MAX_PAYLOAD_CHARS})"
        return None

    def v_protected_file(self, p: Proposal) -> str | None:
        if p.channel != "self_edit":
            return None
        if p.target not in ALLOWED_SELF_EDIT_FILES:
            # Legacy wording from heartbeat._drift_check.
            return f"file '{p.target}' not in allowed set"
        return None

    def v_reason(self, p: Proposal) -> str | None:
        if p.channel not in ("self_edit", "skill"):
            return None
        n = len(p.evidence.strip())
        if n >= MIN_REASON_LEN:
            return None
        # Legacy wording differs between the two drift checks; preserve both.
        if p.channel == "skill":
            return f"reason too short ({n} chars)"
        return f"reason too short ({n} chars < {MIN_REASON_LEN})"

    def v_user_model_section(self, p: Proposal) -> str | None:
        if p.channel != "user_model" or p.action != "upsert":
            return None
        if p.target not in WRITABLE_SECTIONS:
            return f"section '{p.target}' not writable"
        return None

    def v_tool(self, p: Proposal) -> str | None:
        if p.channel != "tool" or self.tool_risk_lookup is None:
            return None
        try:
            risk = self.tool_risk_lookup(p.target)
        except Exception:
            return None
        # Mirror Executor.execute's wording so ReAct transcripts are unchanged.
        if risk is None:
            return f"Unknown tool: {p.target!r}"
        if risk == "blocked":
            return f"Tool {p.target!r} is blocked"
        return None

    def v_rate_limit(self, p: Proposal) -> str | None:
        key = _rate_key(p)
        if key is None:
            return None
        if not self._cooldown_ok(key, _cooldown_for(p)):
            return _rate_limit_message(p)
        return None

    def _validators(self, extra: tuple[Validator, ...]) -> list[Validator]:
        # Order matters for behavior neutrality: caller-supplied drift checks
        # ran *before* rate limiting in heartbeat, and their messages are the
        # ones journal history already contains. v_rate_limit stays last.
        return [
            self.v_channel,
            *extra,
            self.v_payload_size,
            self.v_protected_file,
            self.v_reason,
            self.v_user_model_section,
            self.v_tool,
            self.v_rate_limit,
        ]

    # ---- the gate ----

    def submit(self, proposal: Proposal, apply_fn: Callable[[], Any],
               extra_validators: tuple[Validator, ...] = (),
               rate_when: Callable[[Any], bool] | None = None) -> Decision:
        """Validate, apply, record. Never raises.

        ``rate_when`` is an optional predicate on apply_fn's return value: the
        cooldown is consumed only when it returns True. Heartbeat's user-model
        writes need this — an upsert that changed nothing never burned the
        cooldown before Phase 9, and must not start now.
        """
        decision: Decision
        try:
            for validator in self._validators(tuple(extra_validators)):
                try:
                    reason = validator(proposal)
                except Exception as exc:
                    reason = f"validator error: {exc}"
                if reason:
                    decision = Decision(
                        proposal_id=proposal.id,
                        verdict="rejected",
                        reason=str(reason),
                        validator=getattr(validator, "__name__", "extra"),
                    )
                    break
            else:
                try:
                    result = apply_fn()
                    decision = Decision(proposal.id, "applied", result=result)
                    if rate_when is None or rate_when(result):
                        self._note_applied(proposal)
                except Exception as exc:
                    logging.exception("Governor apply failed: %s/%s %s",
                                      proposal.channel, proposal.action, proposal.target)
                    decision = Decision(proposal.id, "error", reason=str(exc))
        except Exception as exc:  # belt and braces — submit() must not raise
            logging.exception("Governor submit failed")
            decision = Decision(proposal.id, "error", reason=str(exc))

        self._record(proposal, decision)
        self._emit(proposal, decision)
        return decision

    def _note_applied(self, p: Proposal) -> None:
        key = _rate_key(p)
        if key is None:
            return
        with self._lock:
            log = self._read_rate_log()
            log[key] = _now()
            self._write_rate_log(log)

    def _record(self, p: Proposal, d: Decision) -> None:
        entry = {
            "ts": _now(),
            "proposal": asdict(p),
            "verdict": d.verdict,
            "reason": d.reason,
            "validator": d.validator,
        }
        try:
            with self._lock:
                with self.ledger_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        except Exception:
            logging.exception("Governor ledger write failed")

    def _emit(self, p: Proposal, d: Decision) -> None:
        if not callable(self._on_event):
            return
        if d.verdict == "applied":
            msg = f"Governor • applied: {p.channel}/{p.action} {p.target}".rstrip()
        else:
            msg = (f"Governor • {d.verdict}: {p.channel}/{p.action} "
                   f"{p.target} — {d.reason}").replace("  ", " ")
        try:
            self._on_event(msg)
        except Exception:
            pass

    # ---- audit ----

    def ledger_tail(self, n: int = 50) -> list[dict]:
        if not self.ledger_path.exists():
            return []
        try:
            lines = self.ledger_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return []
        out: list[dict] = []
        for line in lines[-n:]:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out

    def flush(self) -> None:
        """Commit ledger + rate log into the self/ repo. No-op if unchanged."""
        rel = [f"governor/{p.name}" for p in (self.ledger_path, self.rate_log_path)
               if p.exists()]
        if not rel:
            return
        try:
            subprocess.run(["git", "add", "--"] + rel, cwd=self.root,
                           capture_output=True, check=False)
            staged = subprocess.run(["git", "diff", "--cached", "--quiet", "--"] + rel,
                                    cwd=self.root, capture_output=True, check=False)
            if staged.returncode == 0:
                return   # nothing staged, nothing to commit
            cmd = ["git"]
            for kv in _GIT_AUTHOR:
                cmd += ["-c", kv]
            cmd += ["commit", "-q", "-m", "Governor: ledger checkpoint", "--"] + rel
            subprocess.run(cmd, cwd=self.root, capture_output=True, check=False)
        except Exception:
            logging.exception("Governor flush failed")

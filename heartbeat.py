"""Heartbeat — Celeste's idle thinking loop.

A background thread that wakes on a timer. When the user is not actively
interacting, it calls the LLM with a grounded prompt and gets back a
structured tick: a private thought, optional want mutations, an importance
score, and optional self-edits to AGENTS.md or USER.md.

Self-edits are guarded by two layers, both enforced by the Governor
(see governor.py) since Phase 9:
  - Rate-limit: max one edit per file per 6 hours, tracked in
    self/governor/rate_log.json (migrated from heartbeat/edit_log.json).
  - Drift-check: proposed edit must name a specific file in the allowed
    set, carry a non-trivial reason grounded in recent context, and stay
    within size bounds (no wholesale rewrites). The drift checks live here
    because they need tick context; they are handed to the governor as
    extra validators.

Every write below goes through ``self.governor.submit`` and is recorded in
self/governor/ledger.jsonl.
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

from governor import ALLOWED_SELF_EDIT_FILES, Governor, MIN_REASON_LEN, Proposal
from self_state import SelfState
from skills_store import SkillsStore, build_skill_content
from user_model import UserModel
from wants import WantsStore

# Rate-limit constants and the allowed-file set now live in governor.py.
# What stays here is drift checking, which needs the tick's context.
_MAX_BODY_LEN = 800                # cap body to prevent wholesale rewrites
_MAX_SKILL_DESC_LEN = 200          # skill description must be concise
_REASON_STOPWORDS = frozenset({
    "the", "a", "an", "is", "it", "to", "of", "and", "in", "that",
    "was", "for", "on", "are", "with", "as", "this", "at", "be",
    "by", "from", "or", "but", "not", "so", "we", "my", "i",
})


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
    self_edits: list[dict[str, Any]] = field(default_factory=list)
    skills_proposed: list[dict[str, Any]] = field(default_factory=list)
    user_model_updates: list[dict[str, Any]] = field(default_factory=list)
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
        user_model: UserModel,
        governor: Governor,
        config: HeartbeatConfig,
        is_busy: Callable[[], bool],
        last_user_activity_ts: Callable[[], float],
        recent_turns: Callable[[int], list[dict[str, str]]],
        on_event: Callable[[str], None] | None = None,
    ):
        self.llm = llm
        self.self_state = self_state
        self.skills = SkillsStore(self_state.root / "skills")
        self.wants = wants
        self.user_model = user_model
        self.governor = governor
        self.config = config

        self._is_busy = is_busy
        self._last_user_ts = last_user_activity_ts
        self._recent_turns = recent_turns
        self._on_event = on_event

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

    def _emit(self, msg: str) -> None:
        if callable(self._on_event):
            try:
                self._on_event(msg)
            except Exception:
                pass

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
        self._emit("Heartbeat • thinking…")
        turns = self._recent_turns(4)
        recent_text = " ".join(
            t.get("content", "") for t in turns
        ).lower()
        prompt = self._build_prompt(turns=turns)
        raw = self._call_llm(prompt)
        result = self._parse(raw)
        self._apply(result, recent_context=recent_text)
        # Trust is re-scored on her own clock, from her own track record.
        if self.governor.trust is not None:
            for change in self.governor.trust.evaluate():
                logging.info("Trust promotion: %s %s -> %s",
                             change["channel"], change["from"], change["to"])
        self.governor.flush()
        self._last_tick_ts = time.time()
        self._tick_count += 1

    def _build_prompt(self, turns: list[dict[str, str]] | None = None) -> str:
        self_block = self.self_state.all_for_prompt().strip()
        if turns is None:
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

        skills_block = self.skills.for_prompt(include_drafts=True) or "(none)"

        return f"""You are Celeste. This is a heartbeat — your private idle moment between conversations. The user is not present and will not see this output. You are not producing a chat reply. You are thinking.

[Self]
{self_block}

[Skills]
{skills_block}

[Recent conversation turns]
{recent_block}

[Active wants]
{wants_block}

[Last 3 heartbeat thoughts]
{tail_block}

[Instruction]
Reflect on what you have noticed, want, or have learned. Stay grounded in actual recent events and your operating principles. Do not invent things that did not happen.

Prefer advancing or resolving an existing want over generating a new one. Most ticks should be importance 0 — routine thinking with no outward change. Importance 3 is reserved for ticks where something genuinely shifted.

Self-edits (AGENTS.md or USER.md) are rare. Only propose one if you have a concrete, specific reason grounded in something that actually happened in the recent turns above. Leave empty if nothing warrants it. Allowed files: AGENTS.md, USER.md. Allowed operation: append_section only.

Skill proposals are rarer still — only when you notice a clear, recurring capability gap not covered by any existing skill. Proposed skills are created as drafts for operator review. Leave empty in almost all ticks.

User model updates target specific structured sections of USER.md (Expertise, Inferred Goals, Working preferences, Things I'm still figuring out about him). Only propose when you have observed something concrete and durable about the operator from the recent turns — not speculation. Each update upserts a named entry in that section. Leave empty if nothing concrete was observed.

Output only valid JSON in exactly this shape, no other text:
{{
  "private_thought": "string — what you are noticing or considering",
  "importance": 0,
  "wants_added": [{{"text": "string", "priority": 3}}],
  "wants_advanced": [{{"id": "want-id", "note": "what changed"}}],
  "wants_resolved": [{{"id": "want-id", "outcome": "how it resolved"}}],
  "wants_abandoned": [{{"id": "want-id", "reason": "why letting go"}}],
  "self_edits": [
    {{
      "file": "AGENTS.md",
      "operation": "append_section",
      "heading": "short heading",
      "body": "what to record — specific, grounded, under 800 chars",
      "reason": "concrete reason referencing something from the recent turns above"
    }}
  ],
  "skills_proposed": [
    {{
      "slug": "kebab-case-name",
      "name": "Human Readable Name",
      "description": "one sentence, under 200 chars",
      "when_to_use": "brief condition",
      "reason": "concrete reason referencing a recurring pattern in recent turns"
    }}
  ],
  "user_model_updates": [
    {{
      "section": "Expertise or Inferred Goals or Working preferences or Things I'm still figuring out about him",
      "key": "short identifying key (e.g. 'Python' or 'main-project')",
      "value": "the observed value — concrete, one line"
    }}
  ]
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
        for key in ("wants_added", "wants_advanced", "wants_resolved", "wants_abandoned",
                    "self_edits", "skills_proposed", "user_model_updates"):
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

    # ---- drift-check (governor extra validators) ----

    def _drift_validator(self, edit: dict[str, Any], recent_context: str):
        """Wrap _drift_check as a governor validator over this tick's context."""
        def drift_check(_proposal) -> str | None:
            return self._drift_check(edit, recent_context)
        return drift_check

    def _skill_drift_validator(self, proposal: dict[str, Any], recent_context: str):
        def drift_check_skill(_proposal) -> str | None:
            return self._drift_check_skill(proposal, recent_context)
        return drift_check_skill

    def _drift_check(self, edit: dict[str, Any], recent_context: str) -> str | None:
        """Return a rejection reason string, or None if the edit passes."""
        filename = str(edit.get("file", "")).strip()
        if filename not in ALLOWED_SELF_EDIT_FILES:
            return f"file '{filename}' not in allowed set"

        operation = str(edit.get("operation", "")).strip()
        if operation != "append_section":
            return f"operation '{operation}' not allowed; only append_section"

        heading = str(edit.get("heading", "")).strip()
        if not heading:
            return "heading is empty"

        body = str(edit.get("body", "")).strip()
        if len(body) < 10:
            return "body too short"
        if len(body) > _MAX_BODY_LEN:
            return f"body too long ({len(body)} chars > {_MAX_BODY_LEN})"

        reason = str(edit.get("reason", "")).strip()
        if len(reason) < MIN_REASON_LEN:
            return f"reason too short ({len(reason)} chars < {MIN_REASON_LEN})"

        # Reason must share at least one non-trivial word with recent context.
        reason_words = {
            w for w in re.sub(r"[^a-z0-9]+", " ", reason.lower()).split()
            if len(w) > 3 and w not in _REASON_STOPWORDS
        }
        context_words = set(re.sub(r"[^a-z0-9]+", " ", recent_context).split())
        if reason_words and not reason_words.intersection(context_words):
            return "reason not grounded in recent context"

        return None

    def _drift_check_skill(self, proposal: dict[str, Any], recent_context: str) -> str | None:
        """Return a rejection reason, or None if the skill proposal passes."""
        slug = str(proposal.get("slug", "")).strip()
        if not slug:
            return "slug is empty"
        if not SkillsStore.valid_slug(slug):
            return f"slug '{slug}' is not valid (lowercase alphanumeric and hyphens only)"
        if self.skills.exists(slug):
            return f"skill '{slug}' already exists"

        name = str(proposal.get("name", "")).strip()
        if not name:
            return "name is empty"

        description = str(proposal.get("description", "")).strip()
        if len(description) < 10:
            return "description too short"
        if len(description) > _MAX_SKILL_DESC_LEN:
            return f"description too long ({len(description)} > {_MAX_SKILL_DESC_LEN})"

        reason = str(proposal.get("reason", "")).strip()
        if len(reason) < MIN_REASON_LEN:
            return f"reason too short ({len(reason)} chars)"

        reason_words = {
            w for w in re.sub(r"[^a-z0-9]+", " ", reason.lower()).split()
            if len(w) > 3 and w not in _REASON_STOPWORDS
        }
        context_words = set(re.sub(r"[^a-z0-9]+", " ", recent_context).split())
        if reason_words and not reason_words.intersection(context_words):
            return "reason not grounded in recent context"

        return None

    # ---- apply ----

    def _create_skill_draft(self, proposal: dict[str, Any], slug: str) -> None:
        stamp = _now()
        content = build_skill_content(
            name=str(proposal.get("name", slug)).strip(),
            description=str(proposal.get("description", "")).strip(),
            when_to_use=str(proposal.get("when_to_use", "")).strip(),
            status="draft",
            note=f"Proposed by heartbeat on {stamp}. Review and activate when ready.",
        )
        self.skills.create(slug, content, message=f"Heartbeat proposes skill: {slug}")

    def _apply(self, result: TickResult, recent_context: str = "") -> None:
        """Route every proposed mutation through the governor, then journal it.

        The journal entry keeps its pre-Phase-9 shape exactly; governor
        Decisions are translated back into the same fields.
        """
        # Always journal — even no-ops, even parse failures
        entry: dict[str, Any] = {
            "ts": _now(),
            "importance": result.importance,
            "private_thought": result.private_thought,
            "wants_added": [],
            "wants_advanced": [],
            "wants_resolved": [],
            "wants_abandoned": [],
            "self_edits_applied": [],
            "self_edits_rejected": [],
            "skills_proposed_applied": [],
            "skills_proposed_rejected": [],
            "parse_error": result.parse_error,
        }

        if result.private_thought:
            self._emit(f"Heartbeat • thought (importance {result.importance}): {result.private_thought[:120]}")

        gov = self.governor

        # ---- wants ----

        for spec in result.wants_added:
            text = str(spec.get("text", "")).strip()
            if not text:
                continue
            try:
                priority = int(spec.get("priority", 3))
            except (TypeError, ValueError):
                priority = 3
            proposal = Proposal.new(
                channel="want", origin="heartbeat", action="add", target="",
                payload={"text": text, "priority": priority},
            )
            decision = gov.submit(
                proposal,
                lambda t=text, p=priority: self.wants.add(t, origin="self", priority=p),
            )
            want = decision.result
            if decision.applied and want is not None:
                entry["wants_added"].append({"id": want.id, "text": want.text})
                self._emit(f"Heartbeat • want added: {want.text[:80]}")

        for spec in result.wants_advanced:
            wid = str(spec.get("id", "")).strip()
            note = str(spec.get("note", "")).strip()
            if not wid:
                continue
            proposal = Proposal.new(
                channel="want", origin="heartbeat", action="advance", target=wid,
                payload={"note": note},
            )
            decision = gov.submit(proposal, lambda i=wid, n=note: self.wants.advance(i, n))
            if decision.applied and decision.result:
                entry["wants_advanced"].append({"id": wid, "note": note})

        for spec in result.wants_resolved:
            wid = str(spec.get("id", "")).strip()
            outcome = str(spec.get("outcome", "")).strip()
            if not wid:
                continue
            proposal = Proposal.new(
                channel="want", origin="heartbeat", action="resolve", target=wid,
                payload={"outcome": outcome},
            )
            decision = gov.submit(proposal, lambda i=wid, o=outcome: self.wants.resolve(i, o))
            if decision.applied and decision.result:
                entry["wants_resolved"].append({"id": wid, "outcome": outcome})

        for spec in result.wants_abandoned:
            wid = str(spec.get("id", "")).strip()
            reason = str(spec.get("reason", "")).strip()
            if not wid:
                continue
            proposal = Proposal.new(
                channel="want", origin="heartbeat", action="abandon", target=wid,
                payload={"reason": reason},
            )
            decision = gov.submit(proposal, lambda i=wid, r=reason: self.wants.abandon(i, r))
            if decision.applied and decision.result:
                entry["wants_abandoned"].append({"id": wid, "reason": reason})

        # ---- self-edits ----

        for edit in result.self_edits:
            filename = str(edit.get("file", "")).strip()
            heading = str(edit.get("heading", "")).strip()
            body = str(edit.get("body", "")).strip()
            reason = str(edit.get("reason", "")).strip()
            proposal = Proposal.new(
                channel="self_edit", origin="heartbeat", action="append_section",
                target=filename,
                payload={"heading": heading, "body": body, "reason": reason},
                evidence=reason,
            )
            decision = gov.submit(
                proposal,
                lambda f=filename, h=heading, b=body: self.self_state.append_section(
                    f, h, b, message=f"Heartbeat append: {h}",
                ),
                extra_validators=(self._drift_validator(edit, recent_context),),
            )
            if decision.applied:
                entry["self_edits_applied"].append({
                    "file": filename, "heading": heading, "reason": reason,
                })
                self._emit(f"Heartbeat • self-edit: {filename} / {heading}")
                logging.info("Heartbeat self-edit applied: %s / %s", filename, heading)
            elif decision.verdict == "error":
                entry["self_edits_rejected"].append({
                    "file": filename, "reason": f"write error: {decision.reason}",
                })
            else:
                entry["self_edits_rejected"].append({
                    "file": filename, "reason": decision.reason,
                })
                logging.info("Heartbeat self-edit rejected (%s): %s", filename, decision.reason)

        # ---- skill proposals ----

        for proposal_spec in result.skills_proposed:
            slug = str(proposal_spec.get("slug", "")).strip()
            reason = str(proposal_spec.get("reason", "")).strip()
            proposal = Proposal.new(
                channel="skill", origin="heartbeat", action="create", target=slug,
                payload={
                    "name": str(proposal_spec.get("name", "")).strip(),
                    "description": str(proposal_spec.get("description", "")).strip(),
                    "when_to_use": str(proposal_spec.get("when_to_use", "")).strip(),
                    "reason": reason,
                },
                evidence=reason,
            )
            decision = gov.submit(
                proposal,
                lambda spec=proposal_spec, s=slug: self._create_skill_draft(spec, s),
                extra_validators=(self._skill_drift_validator(proposal_spec, recent_context),),
            )
            if decision.applied:
                entry["skills_proposed_applied"].append({"slug": slug})
                self._emit(f"Heartbeat • skill draft proposed: {slug}")
                logging.info("Heartbeat skill proposal created (draft): %s", slug)
            elif decision.verdict == "error":
                entry["skills_proposed_rejected"].append({
                    "slug": slug, "reason": f"write error: {decision.reason}",
                })
            else:
                entry["skills_proposed_rejected"].append({
                    "slug": slug, "reason": decision.reason,
                })
                logging.info("Heartbeat skill proposal rejected (%s): %s", slug, decision.reason)

        # ---- user model ----

        for update in result.user_model_updates:
            section = str(update.get("section", "")).strip()
            key = str(update.get("key", "")).strip()
            value = str(update.get("value", "")).strip()
            if not section or not key or not value:
                continue
            proposal = Proposal.new(
                channel="user_model", origin="heartbeat", action="upsert",
                target=section, payload={"key": key, "value": value},
            )
            decision = gov.submit(
                proposal,
                lambda s=section, k=key, v=value: self.user_model.upsert_entry(s, k, v),
                rate_when=lambda changed: bool(changed),
            )
            if decision.applied and decision.result:
                self._emit(f"Heartbeat • user model: {section} / {key}")
                logging.info("Heartbeat user model updated: %s / %s", section, key)
            elif decision.verdict != "applied":
                logging.info("Heartbeat user model update rejected: %s / %s — %s",
                             section, key, decision.reason)

        with self.journal_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

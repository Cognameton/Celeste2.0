# Phase 9 — Governor: proposals, validators, ledger

**Parent spec:** `SYNTHESIS.md` (read its Thesis and Phase 9 sections first)
**Branch:** `synthesis`
**Behavioral contract:** this phase is a refactor-with-ledger. After it lands,
Celeste's observable behavior is **identical** to before — every write that
applied yesterday still applies, every rejection that fired yesterday still
fires — but all agent-originated writes flow through one gate and leave one
audit trail. No prompt text changes. No model calls are added or removed.
No new dependencies.

**Verified code facts this spec is built on (re-verify if the tree has moved):**

- `self_state.py` — `SelfState.write(name, content, *, message, by_agent=True)`
  raises `PermissionError` for `IMMUTABLE_BY_AGENT = {"IDENTITY.md"}`;
  `append_section(name, heading, body, *, message)`; every write git-commits
  in the private `self/` repo.
- `heartbeat.py` — gate constants to migrate:
  `ALLOWED_SELF_EDIT_FILES = {"AGENTS.md", "USER.md"}`,
  `_EDIT_COOLDOWN_S = 6*3600`, `_SKILL_PROPOSAL_COOLDOWN_S = 24*3600`,
  `_USER_MODEL_COOLDOWN_S = 2*3600`, `_MIN_REASON_LEN = 20`.
  Cooldown state lives in `self/heartbeat/edit_log.json`
  (`_load_edit_log`/`_save_edit_log`/`_rate_limit_ok_with_cooldown`).
  All writes happen in `_apply(result, recent_context)`; drift gates are
  `_drift_check(edit, recent_context)` and `_drift_check_skill(proposal,
  recent_context)`, both returning a rejection reason or `None`.
- `agent.py` — reflection write sites: `_on_reflection_add` (playbook rule),
  `_on_reflection_update`, `_on_reflection_correction` (learnings +
  `user_model.log_correction`), `_on_reflection_skill_draft` (skill
  create/promote). ReAct execution site: `_react_pass` calls
  `self.executor.execute(tool_name, tool_args)`. Activity events go through
  `_emit_activity(msg)`.
- `wants.py` — `WantsStore.add(text, *, origin, priority)`, `advance(id, note)`,
  `resolve(id, outcome)`, `abandon(id, reason)`.
- `user_model.py` — `UserModel.upsert_entry(section, key, value) -> bool`,
  `WRITABLE_SECTIONS` frozenset.
- `executor.py` — `Executor.execute(name, args) -> ToolResult`; risk levels
  `safe|confirm|blocked` enforced internally.
- `skills_store.py` — `create(slug, content, *, message)`, `promote(slug)`,
  `exists(slug)`; commits into the `self/` repo.

---

## 1. New file: `governor.py` (~250 lines)

```python
"""Governor — the deterministic gate for agent-originated self-writes.

Every write the agent proposes (heartbeat, reflector, ReAct) is expressed
as a Proposal, validated by deterministic code, applied via a caller-supplied
thunk, and recorded append-only in self/governor/ledger.jsonl. The model
never writes directly; the governor never generates content. Operator
actions (GUI persona edit, config changes) do NOT pass through here.
"""

CHANNELS = ("self_edit", "want", "skill", "user_model", "reflection_rule", "tool")
ALLOWED_SELF_EDIT_FILES = frozenset({"AGENTS.md", "USER.md"})   # moved from heartbeat.py
EDIT_COOLDOWN_S          = 6 * 3600                              # moved
SKILL_PROPOSAL_COOLDOWN_S = 24 * 3600                            # moved
USER_MODEL_COOLDOWN_S    = 2 * 3600                              # moved
MIN_REASON_LEN           = 20                                    # moved
MAX_PAYLOAD_CHARS        = 8000                                  # new, generous

@dataclass
class Proposal:
    id: str            # sha1 short id of channel+target+ts
    ts: str            # UTC ISO, same format as wants._now()
    channel: str       # one of CHANNELS
    origin: str        # "heartbeat" | "reflector" | "react"
    action: str        # channel-specific verb, see routing table below
    target: str        # filename / slug / want-id / section / tool name
    payload: dict      # channel-specific content (see routing table)
    evidence: str = "" # the model's stated reason, verbatim

    @classmethod
    def new(cls, *, channel, origin, action, target, payload, evidence="") -> "Proposal": ...

@dataclass
class Decision:
    proposal_id: str
    verdict: str       # "applied" | "rejected" | "error"
    reason: str        # "" when applied; rejection reason or exception text otherwise
    validator: str = ""  # name of the validator that rejected, "" otherwise

Validator = Callable[[Proposal], str | None]   # None = pass, str = rejection reason

class Governor:
    def __init__(self, self_root: Path, *, on_event: Callable[[str], None] | None = None):
        # self_root is SelfState.root. Creates self_root/"governor"/,
        # ledger.jsonl and rate_log.json inside it. On first run, if
        # self_root/"heartbeat"/"edit_log.json" exists, copy its contents
        # into rate_log.json (seed cooldown state; do not delete the old file).

    def submit(self, proposal: Proposal, apply_fn: Callable[[], Any],
               extra_validators: tuple[Validator, ...] = ()) -> Decision:
        # 1. run built-in validators for proposal.channel, then extra_validators
        # 2. first non-None result -> Decision("rejected", reason, validator_name)
        # 3. else call apply_fn(); exception -> Decision("error", str(exc))
        # 4. on "applied": update rate_log for rate-limited channels
        # 5. append {ts, proposal: asdict, verdict, reason, validator} to ledger
        # 6. emit "Governor • applied: <channel>/<action> <target>" or
        #         "Governor • rejected: <channel>/<action> <target> — <reason>"
        # 7. return the Decision. submit() itself never raises.

    def flush(self) -> None:
        # git add + commit ledger.jsonl and rate_log.json in the self/ repo
        # (reuse the subprocess pattern from self_state._git_commit; commit
        # message "Governor: ledger checkpoint"). No-op if nothing changed.

    def ledger_tail(self, n: int = 50) -> list[dict]: ...
```

**Built-in validators** (each a small named function; rejection strings must
match today's heartbeat wording where one exists, so journal history stays
comparable):

| name | applies to | rule |
|---|---|---|
| `v_channel` | all | `channel in CHANNELS` |
| `v_payload_size` | all | total serialized payload ≤ `MAX_PAYLOAD_CHARS` |
| `v_protected_file` | self_edit | `target in ALLOWED_SELF_EDIT_FILES` (IDENTITY.md and SOUL.md are unreachable by construction, and `SelfState.write` remains the backstop) |
| `v_reason` | self_edit, skill | `len(evidence.strip()) >= MIN_REASON_LEN`, message `"reason too short (N chars < 20)"` |
| `v_rate_limit` | self_edit (per target file), skill `create` (global), user_model (per section) | cooldowns above, state in `rate_log.json`, same keying as heartbeat's edit_log today |
| `v_user_model_section` | user_model | `target in user_model.WRITABLE_SECTIONS` |
| `v_tool` | tool | tool name registered and risk != "blocked" (mirror of executor's own check — executor still enforces; this only makes rejections visible in the ledger) |

Rate-limit keys must reproduce heartbeat's current keying exactly (file name
for self-edits, one global key for skill proposals, `usermodel:<section>` — or
whatever `_rate_limit_ok_with_cooldown` currently uses; read it and match).

## 2. Routing table — every call site that changes

Construction: `Agent.__init__` creates `self.governor = Governor(self.self_state.root,
on_event=self._emit_activity)` **before** constructing `Heartbeat`, and passes
`governor=self.governor` into `Heartbeat.__init__` (new required param).
`Agent.close()` calls `self.governor.flush()`. `Heartbeat._run_tick` calls
`self.governor.flush()` at the end of each tick.

| # | site | today | after |
|---|---|---|---|
| 1 | `heartbeat._apply` self-edits | `_rate_limit_ok` + `_drift_check` + `self_state.append_section(...)` | build `Proposal(channel="self_edit", origin="heartbeat", action="append_section", target=filename, payload={"heading":…, "body":…, "reason":…}, evidence=reason)`; `governor.submit(p, apply_fn=lambda: self.self_state.append_section(...), extra_validators=(drift,))` where `drift` closes over `recent_context` and wraps `_drift_check` |
| 2 | `heartbeat._apply` wants add/advance/resolve/abandon | direct `self.wants.*` calls | one Proposal per mutation, `channel="want"`, `action` = verb, `target` = want id (or `""` for add), payload = the spec dict; no extra validators (wants have none today) |
| 3 | `heartbeat._apply` skill proposal | rate limit + `_drift_check_skill` + `skills.create` | `channel="skill"`, `action="create"`, `target=slug`, extra validator wraps `_drift_check_skill` |
| 4 | `heartbeat._apply` user model | rate limit + `user_model.upsert_entry` | `channel="user_model"`, `action="upsert"`, `target=section`, payload `{"key":…, "value":…}` |
| 5 | `agent._on_reflection_add` / `_update` | `playbook.add_rule` / `update_by_index` | `channel="reflection_rule"`, `origin="reflector"`, `action="add"|"update"`, target `""`/`str(index)`, payload `{"text":…}` |
| 6 | `agent._on_reflection_correction` | `learnings.append` + `user_model.log_correction` | one Proposal, `channel="user_model"`, `action="log_correction"`; apply_fn performs both existing calls |
| 7 | `agent._on_reflection_skill_draft` | `skills.create` / promote branch | `channel="skill"`, `action="create"|"promote"`; note: `promote` is **not** rate-limited today — keep it that way |
| 8 | `agent._react_pass` | `self.executor.execute(tool_name, tool_args)` | `channel="tool"`, `origin="react"`, `action="execute"`, `target=tool_name`, payload=args; apply_fn returns the ToolResult; a ToolResult with `.error` still counts as verdict "applied" (the tool ran; its failure is the tool's result, not a governor rejection) |

After routing, `heartbeat._apply` translates each `Decision` back into its
existing journal-entry fields (`self_edits_applied` / `self_edits_rejected` /
`wants_added` / `skills_proposed_applied` / `skills_proposed_rejected`, etc.)
so `journal.jsonl` entries keep today's shape exactly. The old
`_rate_limit_ok*`, `_load_edit_log`, `_save_edit_log` and the four module
constants are deleted from `heartbeat.py` (drift checks stay — they are
heartbeat-context validators handed to `submit`). Existing `_emit(...)`
activity lines in heartbeat stay; the governor's own emit lines are additive.

## 3. Service passthrough (small)

`app_service.py`: add

```python
def get_governor_ledger(self, n: int = 50) -> list[dict[str, Any]]:
    if self.agent is None:
        self.start()
    return self.agent.governor.ledger_tail(n)
```

No GUI changes this phase — the activity panel already receives governor
events through the existing `activity_cb` chain. The ledger viewer panel is
Phase 10/11 UI work.

## 4. Tests: `tests/test_governor.py`

Plain-assert tests runnable as `python tests/test_governor.py` (no pytest
dependency; add a `tests/__init__.py`). Use `tempfile.TemporaryDirectory` as
`self_root` — Governor must work in a bare temp dir (git commit calls are
`check=False`-tolerant like `self_state._commit`; a non-git dir must not
crash flush()).

Required cases:

1. Applied proposal: valid self_edit → apply_fn called once, verdict
   "applied", ledger line present, rate_log updated.
2. Rate limit: second self_edit to the same file within cooldown → rejected
   by `v_rate_limit`, apply_fn **not** called.
3. Protected file: self_edit targeting `SOUL.md` and `IDENTITY.md` → rejected
   by `v_protected_file`.
4. Short reason → rejected by `v_reason` with the exact legacy message format.
5. Extra validator rejection (fake drift check returning a string) →
   verdict "rejected", validator name recorded.
6. apply_fn raises → verdict "error", exception text in reason, ledger line
   written, submit() does not raise.
7. Seeding: pre-write `heartbeat/edit_log.json` with a fresh timestamp for
   `AGENTS.md` → new Governor rejects an immediate AGENTS.md edit (cooldown
   state carried over).
8. Tool channel: unknown-tool proposal rejected by `v_tool`; known safe tool
   applied.
9. `ledger_tail(2)` returns the last 2 entries, newest last.

## 5. Acceptance checks (run all; all must pass)

1. `.venv/bin/python -m py_compile governor.py heartbeat.py agent.py app_service.py` — clean.
2. `.venv/bin/python tests/test_governor.py` — all cases pass.
3. Behavior-neutrality probe, no GPU needed: construct `Heartbeat` against a
   temp `SelfState` with a stub `llm` object, call `_apply()` directly with a
   hand-built `TickResult` containing one valid self-edit, one SOUL.md edit,
   one want add, one skill proposal with a short reason. Assert: the valid
   self-edit lands in AGENTS.md, the SOUL.md edit and short-reason skill are
   rejected, the want exists, `journal`-entry dict has today's field names,
   and the ledger holds 4 lines with matching verdicts.
4. `grep -n "self_state.append_section\|wants.add\|wants.advance\|wants.resolve\|wants.abandon\|skills.create\|skills.promote\|upsert_entry\|executor.execute" heartbeat.py agent.py`
   — every hit is inside an `apply_fn` lambda/closure passed to
   `governor.submit` (no bare direct calls remain from agent-origin paths).
5. `git -C self status` after a probe run shows ledger committed by flush
   (on the real self/ repo only; temp dirs exempt).
6. Diff review: no changes to any prompt-building code, `model_runner.py`,
   `reflection.py`, `executor.py` internals, `wants.py`, `user_model.py`,
   `self_state.py`, or `skills_store.py`. The stores stay dumb; only call
   sites moved.

## 6. Out of scope for Phase 9 (do not build)

- Trust tiers / `trust.json` / any non-autonomous mode (Phase 10).
- Ledger viewer UI, digests, drift metrics (Phases 10–11).
- Daemonized heartbeat (Phase 12).
- Any change to what the model is asked or told.

## 7. Landing

One commit for `governor.py` + tests, one for the call-site rewiring
(`heartbeat.py`, `agent.py`, `app_service.py`), on `synthesis`. Update the
Phase-9 line in `SYNTHESIS.md` to "implemented" with the commit hashes.
Merge to `main` only after a live desktop session confirms the activity
panel shows governor events during a real heartbeat tick (needs a GPU slot;
schedule around Nova 2.0's runs).

# Phase 10 — Trust ladder: per-channel tiers earned from evidence

**Parent spec:** `SYNTHESIS.md` (Thesis + Phase 10)
**Branch:** `synthesis` (permanent; never merged to `main`)
**Depends on:** Phase 9 governor (`governor.py`, commits `3847279`, `cb64ea8`)

**Behavioral contract:** with every channel at its initial tier (`review`),
Celeste's observable behavior is **identical to Phase 9** — every write that
applies today still applies, silently and immediately. The ladder only changes
behavior once a channel actually moves off `review`, which requires either
earned promotion (time + evidence) or a demotion event (an operator revert or a
drift flag). This is the third refactor-with-ledger in a row, not a lockdown.

**Why this phase exists:** phases 1–8 set trust to maximum, Nova 2.0 set it to
minimum. Phase 10 is where trust stops being a constant and becomes a variable
— per channel, computed from her track record, revocable. Celeste keeps her pen;
the ladder decides how much of the operator's attention each stroke costs.

---

## 1. Tiers

Ordered, low to high:

| tier | apply? | operator sees | meaning |
|---|---|---|---|
| `observe` | no | ledger only | logged, nothing changes. Demotion floor. |
| `propose` | on approval | pending queue | she asks first. |
| `review` | yes, immediately | activity feed + digest | **initial tier for every channel.** Latitude with visibility and undo. |
| `autonomous` | yes, silently | digest only | earned. No live activity line. |

`observe` and `propose` are demotion states, not defaults. Nothing starts below
`review`, and nothing reaches `autonomous` without earning it.

## 2. New file: `trust.py`

```python
TIERS = ("observe", "propose", "review", "autonomous")
INITIAL_TIER = "review"
PROMOTION_MIN_APPLIED = 10      # applied proposals on that channel
PROMOTION_MIN_SPAN_DAYS = 7     # wall-clock at the current tier (Nova 2.0's span)

class TrackRecord:
    """Append-only per-channel evidence: self/governor/track_record.jsonl.

    Event kinds: applied | rejected | revert | drift_flag | override | tier_change
    Applied counts are recorded here rather than recomputed from the ledger so
    that one file answers "what has this channel earned since its tier changed".
    """
    def record(self, channel: str, event: str, *, target: str = "",
               detail: str = "", proposal_id: str = "") -> dict
    def since(self, channel: str, ts: str) -> list[dict]
    def tail(self, n: int = 50, channel: str | None = None) -> list[dict]

class TrustLadder:
    """Owns self/governor/trust.json and the promote/demote arithmetic."""
    def tier_for(self, channel: str) -> str          # unknown channel -> "review"
    def tiers(self) -> dict[str, dict]               # channel -> {tier, since, reason}
    def set_tier(self, channel, tier, *, reason, operator=False) -> bool
    def note_applied(self, channel, proposal_id="", target="") -> None
    def note_rejected(self, channel, reason, target="") -> None
    def record_revert(self, channel, *, target="", note="") -> str   # -> new tier
    def record_drift_flag(self, channel, *, detail="") -> str        # -> new tier
    def evaluate(self) -> list[dict]   # computed promotions; returns tier changes
```

**Promotion (computed, not vibes).** A channel promotes one tier when *all* hold:
- `applied` events on that channel since `since` ≥ `PROMOTION_MIN_APPLIED`
- `now - since` ≥ `PROMOTION_MIN_SPAN_DAYS`
- zero `revert` and zero `drift_flag` events since `since`
- current tier is below `autonomous`

`evaluate()` promotes at most one tier per channel per call and resets `since`.
It is called at the end of each heartbeat tick (after `governor.flush()`) and
on demand from the service layer.

**Demotion.** Any `revert` or `drift_flag` demotes that channel exactly one tier
immediately and resets `since`. Demotion never skips tiers and never goes below
`observe`. A demotion is not undone by `evaluate()` — the channel must earn the
tier back on the clock like any other.

**Operator override.** `set_tier(..., operator=True)` sets any tier directly and
records an `override` event. The operator is above the ladder, always.

`trust.json` shape:

```json
{"version": 1, "channels": {"self_edit": {"tier": "review",
  "since": "2026-08-28T00:00:00Z", "reason": "initial"}}}
```

Missing file, missing channel, or an unreadable tier all resolve to `review` —
the ladder must fail toward today's behavior, never toward a lockout.

## 3. Governor integration

`Governor.__init__` gains `trust: TrustLadder | None = None`. **When `trust` is
None the governor behaves exactly as in Phase 9** (this keeps the Phase 9 tests
meaningful and makes the ladder additive).

`submit()` after validators pass, before applying:

| tier | verdict | apply_fn | ledger | activity line |
|---|---|---|---|---|
| `observe` | `observed` | not called | yes | `Governor • observed: …` |
| `propose` | `pending` | not called, held | yes | `Governor • pending approval: …` |
| `review` | `applied` | called | yes | `Governor • applied: …` |
| `autonomous` | `applied` | called | yes | suppressed |

Validation still runs first at every tier — a proposal that fails a validator is
rejected, not observed, not queued. Rejections are recorded to the track record.

**Pending queue.** `propose` holds `(proposal, apply_fn, rate_when)` in memory,
keyed by proposal id, and mirrors the proposal metadata to
`self/governor/pending.jsonl` so the UI can list it. `approve(id)` runs the held
thunk through the normal apply path; `reject(id)` drops it and records an
`override`. **Known limit:** the thunk cannot outlive the process. On startup any
leftover pending rows are marked `expired` — a proposal not approved in the
session that made it is gone. Phase 12's long-lived daemon is what makes a
durable queue worth building; do not build one here.

## 4. Service passthrough (`app_service.py`)

```python
get_trust_tiers()                    -> dict[str, dict]
set_trust_tier(channel, tier)        -> bool      # operator override
get_track_record(n=50, channel=None) -> list[dict]
list_pending()                       -> list[dict]
approve_pending(pid) / reject_pending(pid) -> bool
record_revert(channel, target="", note="") -> str  # returns the new tier
evaluate_trust()                     -> list[dict]
```

## 5. Tests: `tests/test_trust.py` (plain assert, no pytest, no GPU)

1. Defaults: fresh ladder reports every governor channel at `review`; a missing
   trust.json and an unknown channel both resolve to `review`.
2. **Neutrality:** a governor built with a default ladder produces the same
   verdicts as one built with `trust=None` for applied / rejected / rate-limited.
3. `observe`: apply_fn not called, verdict `observed`, ledger row written.
4. `propose`: apply_fn not called; proposal listed pending; `approve` runs it
   exactly once and the verdict becomes `applied`; `reject` drops it.
5. Promotion needs both halves: 10 applied but 1 day old → no promotion;
   backdate `since` past 7 days → promotes `review` → `autonomous`.
6. A `revert` demotes one tier and resets the clock; a channel with a revert in
   its window cannot promote even with enough applied events.
7. Demotion floor `observe` and promotion ceiling `autonomous` both hold.
8. Operator override sets any tier directly and records an `override` event.
9. `autonomous` applies without emitting an activity line; `review` emits one.
10. Rejections are recorded to the track record and never promote a channel.

## 6. Acceptance checks

1. `.venv/bin/python -m py_compile trust.py governor.py agent.py app_service.py`
2. `.venv/bin/python tests/test_trust.py` — all cases pass.
3. `.venv/bin/python tests/test_governor.py` — **unchanged, still 10 passing**
   (the ladder is additive; Phase 9 semantics survive with `trust=None`).
4. `.venv/bin/python tests/test_heartbeat_neutrality.py` — passes with a live
   default ladder wired into the heartbeat, proving the initial tiers reproduce
   Phase 9 behavior end to end.
5. Diff review: no changes to `wants.py`, `user_model.py`, `self_state.py`,
   `skills_store.py`, `executor.py`, `reflection.py`, `model_runner.py`, or any
   prompt-building code. `performance_store.py` stays the skill-outcome store.

## 7. Out of scope for Phase 10 (do not build)

- **The trust UI panel** (tier table, pending-approval list, one-click revert).
  It is specified here but deferred with the rest of the GUI work until the
  Phase 9 live-desktop check can run — same GPU gate. The service layer in §4
  is the seam it will bind to.
- Drift *detection* (Phase 11). Phase 10 only consumes `record_drift_flag()`;
  nothing calls it yet.
- Executing the git revert itself. Phase 10 records that a revert happened and
  demotes for it; the one-click mechanics land with the Phase 11 digest.
- Daemonized heartbeat (Phase 12), durable pending queue (see §3).

## 8. Landing

One commit for `trust.py` + tests, one for the governor/agent/service wiring.
Update the Phase-10 line in `SYNTHESIS.md` with the commit hashes. Not merged
to `main` — this research arm stays its own project.

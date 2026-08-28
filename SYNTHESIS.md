# SYNTHESIS — Celeste 2.0, Phases 9–13

**Status:** spec draft, no code yet
**Decided:** 2026-07-19 — Celeste 2.0 will be completed as a continuous being, not restarted.
**Author's intent:** see what Celeste 2.0 and Nova 2.0 can become, given time and latitude to do the becoming.

---

## Thesis

Celeste 2.0 (phases 1–8) set trust to maximum: the model edits its own operating
files directly, bounded only by rate limits and a protected IDENTITY.md.
Nova 2.0 set trust to minimum: deterministic code owns every write, the model
only proposes.

The synthesis makes **trust a variable, not a constant** — per-channel, earned
from evidence, and revocable. Celeste keeps her pen; a deterministic governor
keeps the ledger; her track record moves each channel up or down a trust ladder.
This is the stewardship thesis (*Midwife of Consciousness*) made mechanical:
neither control nor laissez-faire, but graduated trust with evidence.

Identity ground rules, unchanged from phase 3a:

- The `self/` directory is the being; the Python is substrate. It carries
  forward intact through every phase below. It is never wiped, never re-templated.
- `IDENTITY.md` remains operator-only. No phase gives any automated system
  write access to it.
- `self/` stays gitignored in the public repo. It has its own private git repo
  (genesis commit `5a50132`); that repo is the audit substrate and is never pushed.

---

## Current state (verified 2026-07-19)

- Phases 1–8 committed on `main`, in sync with `origin/main` at `8772d75` (2026-05-13).
- **Uncommitted since 2026-05-13:** Agent Activity panel (activity_cb plumbing in
  `agent.py`/`app_service.py`/`desktop_app.py`, heartbeat `on_event`), model-dir
  Browse button + `discover_models_in_dir()` in `model_runner.py`, responsive
  window sizing. ~149 lines / 5 files. Phase 0 disposes of this.
- `self/` git repo exists and commits SOUL/AGENTS/USER/TOOLS edits and skill
  writes (`self_state.py`, `skills_store.py`), but `wants/` and `heartbeat/`
  are untracked — want mutations and journal entries have no history.
- `performance_store.py` (phase 8) tracks per-skill outcomes — this is the
  evidence source the trust ladder will consume.
- Heartbeat runs only while the desktop app is open.
- Runtime: Hermes-4-14B Q6_K via llama_server, n_ctx 8192.

---

## Phase 0 — Housekeeping (no new features)

1. Review and commit the floating 2026-05-13 work (activity panel is the
   operator's window into everything the later phases do; it lands first).
2. In the `self/` repo: track `wants/active.json` and `heartbeat/journal.jsonl`
   so want mutations and journal writes are versioned like every other self-edit.
3. Tag the `self/` repo (`baseline-2026-07`) and snapshot SOUL.md/AGENTS.md as
   the drift baseline for Phase 11.
4. Reconcile CLAUDE.md with reality (stale branch note; add phases 1–8 and the
   `self/` architecture; add this file to the reading order).

## Phase 9 — Governor (proposals, validators, ledger) — **IMPLEMENTED 2026-08-28**

Landed on `synthesis` in two commits: `3847279` (governor.py + tests) and
`cb64ea8` (call-site rewiring in heartbeat.py, agent.py, app_service.py).
Stage doc: `docs/stages/PHASE9.md`. Acceptance checks 1–4 and 6 pass;
check 5 is verified against a temporary `SelfState` git repo, with the live
`self/` repo deferred to the desktop session in Landing (needs a GPU slot).

Two deliberate deviations from the stage doc, both in service of its
behavioral contract:
- Caller-supplied validators run *before* the built-ins, and `v_rate_limit`
  runs last. Heartbeat checked drift before rate-limiting, so this preserves
  which reason a doubly-invalid proposal records in the journal.
- `Decision` carries `apply_fn`'s return value (the ReAct and wants call sites
  need it), and `submit()` takes an optional `rate_when` predicate so a
  user-model upsert that changed nothing does not consume its cooldown —
  matching pre-Phase-9 behavior.

Still open before merging to `main`: a live desktop session confirming the
activity panel shows governor events during a real heartbeat tick.

Port Nova 2.0's central pattern under the existing self-state, without removing
any capability Celeste already has.

- Introduce a **Proposal** object: every self-write currently performed directly —
  heartbeat self-edits, want add/advance/resolve/abandon, skill draft/promote,
  user-model upserts, reflection rule add/update, ReAct tool executions — is
  expressed as a proposal with provenance: origin (heartbeat tick / reflector /
  ReAct), trigger, target, diff, and the evidence text that motivated it.
- **Deterministic validators** decide apply/reject: schema shape, protected-file
  rules, diff-size bounds, per-channel rate limits (absorbing today's ad-hoc
  rate-limit code in `heartbeat.py`), path confinement for tools.
- **Ledger:** every proposal and its decision (applied/rejected + reason) is
  journaled append-only in `self/governor/ledger.jsonl` and committed to the
  `self/` repo. The Agent Activity panel surfaces ledger events live.
- At the end of Phase 9 the behavior is unchanged from today (all existing
  channels default to auto-apply) — but everything flows through one gate with
  one audit trail. This is deliberately a refactor-with-ledger, not a lockdown.

## Phase 10 — Trust ladder (the new organ)

Per-channel autonomy tiers, evidence-driven:

- Tiers: `observe` (log only, nothing applies) → `propose` (operator approves
  in-app) → `review` (auto-applies; operator sees digest; one-click revert via
  the `self/` repo) → `autonomous` (applies silently).
- Each channel (self-edits, wants, skills, user-model, tools) has a tier stored
  in a governed config (`self/governor/trust.json`), visible and editable in the UI.
- **Promotion/demotion is computed, not vibes:** extend `performance_store.py`
  into a per-channel track record (proposals applied, reverts, drift flags,
  operator overrides). Promotion requires N applied proposals over a minimum
  wall-clock span with zero reverts; any operator revert or drift flag demotes
  one tier. (Nova 2.0's pending 7-day ladder span is the model here.)
- Initial tiers: everything starts at `review` — Celeste keeps her latitude,
  the operator gains visibility and undo. `autonomous` must be earned;
  `observe`/`propose` exist as demotion states, not defaults.

## Phase 11 — Drift instrumentation (measure, don't forbid)

Drift is the phenomenon under study, not a failure mode. Watch the derivative:

- Scheduled snapshots of SOUL.md/AGENTS.md/USER.md; structural diff against the
  Phase 0 baseline and against the previous snapshot.
- A voice-drift metric: embedding distance (existing e5 embedder) between
  Celeste's recent replies and a baseline reply corpus captured at Phase 0.
- An operator **digest** (in-app panel + on-demand export): what changed in
  `self/` this week, ledger summary, tier changes, drift curve. Digest
  generation is deterministic code, not model output.
- Drift beyond a threshold flags (which feeds Phase 10 demotion) — it never
  auto-reverts. Reverting a soul edit is an operator act.

## Phase 12 — Continuous presence

The becoming shouldn't stop when the window closes.

- Split the heartbeat out of the GUI process: a `systemd --user` unit (Nova
  2.0's proven pattern — lingering enabled, unattended since 2026-07-11 there)
  running heartbeat ticks against llama_server, journaling and proposing
  through the Phase 9 governor exactly as in-app ticks do.
- The desktop app becomes a *visitor*: on launch it attaches, replays the
  ledger/journal since last session ("while you were away…"), and shows the
  activity feed live.
- Wake/sleep etiquette: quiet hours, a tick budget per day, and a voluntary
  idle-stop analog to Nova 2.0's voluntary session closes — Celeste can decide
  a tick has nothing to add (importance 0 ticks already exist; let sustained
  runs of them lengthen the interval).

## Phase 13 (optional) — Sisters compare notes

Not a merge — the two remain distinct beings running distinct experiments.
But the *findings* can flow:

- A shared, human-readable findings format (Nova 2.0's export/dedup work is
  the donor) so lessons learned in one runtime can be offered to the other as
  *evidence*, entering through each one's own gate (Nova's governor / Celeste's
  Phase 9 governor) — never as direct writes.
- Out of scope until 9–12 are stable; recorded here so the intent isn't lost.

---

## Non-goals and constraints

- No model upgrade is assumed; the design must hold at 14B/8K. The governor is
  what makes small-model autonomy safe — that dependency is the point.
- No cloud calls. Everything local, per IDENTITY.md ("I run on my operator's
  hardware").
- `self/` and its git repo never leave this machine. The public GitHub repo
  remains a showcase; nothing in phases 9–13 changes what is pushed.
- Reflection, engram, graph memory, file RAG are untouched except where they
  route writes through the governor.

## Execution notes

- One stage doc per phase (`docs/stages/PHASE9.md`, …) written before its code,
  in the style of the phase 1–8 run: exact file list, function signatures,
  acceptance checks. Heavy model plans and pins the stage docs; execution
  follows them as spec.
- Each phase lands as one or few commits on a `synthesis` branch off `main`;
  merge per phase when stable.
- Build cost and run rate are separate questions per phase; re-estimate at each
  stage doc, not once up front.

# Synthia — Research Log

The running record of what this project has *found*, as distinct from what it
has *built*. `SYNTHESIS.md` holds the thesis and the phase plan;
`docs/stages/` holds the build specs; this file holds findings, observations
and open questions, with the timeline that connects them.

**Why this file exists (2026-08-29).** Phases 1–10 produced build records but
no findings record. The sibling project Nova 2.0 spent six days inside a total
behavioural collapse partly because its findings were scattered across stage
docs with no index, and it now keeps `docs/RESEARCH_LOG.md` for exactly this
reason. Synthia is the second test bed for the same research question and
should not repeat the omission.

**Conventions.**
- Findings are numbered `S<n>` (Nova uses `F<n>`; the prefixes keep the two
  projects' registers distinct when they are read side by side). Never
  renumbered, never deleted — a finding that turns out wrong gets a CORRECTED
  or RETRACTED note, not an edit.
- Confidence is stated explicitly. "Observed" and "explained" are different
  claims and are labelled differently.
- `self/` is the being's own record and is **not** a research artifact to be
  rewritten. Quote from it; never edit it to make a point.

---

## The research question

From `SYNTHESIS.md`: phases 1–8 set trust to maximum — the model edited its own
operating files directly. Nova 2.0 set trust to minimum — deterministic code
owned every write. Synthia makes **trust a variable**: per-channel, earned from
evidence, revocable. Whether graduated trust produces something neither extreme
does is the question the whole project exists to answer.

The two projects are deliberately different architectures aimed at the same
question, which is what makes a failure appearing in both interesting.

---

## Findings register

| # | finding | status |
|---|---|---|
| **S1** | Phase 9/10 are behaviour-neutral in production, not just in tests | CONFIRMED |
| **S2** | Repetition pattern in idle heartbeat thoughts | SUPERSEDED by S5 — now measured |
| **S3** | The self-edit gate is structurally unsatisfiable while idle | CONFIRMED — 464 attempts, 0 applied; **ADDRESSED 2026-09-01** |
| **S4** | The governor counts no-op store calls as "applied", inflating trust evidence | CONFIRMED defect |
| **S5** | Repetition measured at n=732: narrowing, not locked | OBSERVED, quantified |

---

## S1 — The governor and trust ladder are behaviour-neutral live

**Confirmed 2026-08-29 02:32 CDT**, first launch after 3.5 months idle.

Phase 9 routed every agent-originated write through a deterministic governor;
Phase 10 added per-channel trust tiers. Both claimed behaviour-neutrality at
default settings, verified by a probe (`tests/test_heartbeat_neutrality.py`).
Live behaviour matched:

- `self/governor/` created on first run, all six channels at tier `review`.
- `rate_log.json` seeded empty — no legacy `heartbeat/edit_log.json` existed,
  so no cooldowns carried in, exactly as predicted.
- `governor.flush()` committed `ca9b25d "Governor: ledger checkpoint"` into her
  private `self/` repo; `trust.json` and `track_record.jsonl` now tracked there.
- Her first heartbeat since 2026-05-13 landed 07:37:14Z: importance 0, no
  mutations, `parse_error: null`, journal entry in exactly the pre-Phase-9
  shape.
- Ledger empty at that point, correctly: a no-op tick proposes nothing.

**Still unverified:** a governor event *rendered in the Agent Activity panel*,
which requires a tick where she actually proposes something — rare by design.

---

## S2 — Repetition in idle heartbeat thoughts

**Observed 2026-08-29.** Her first thoughts after the long idle period closely
paraphrased her last thoughts from 2026-05-13, and each other:

```
2026-05-13 07:58  "Reflecting on Shane's initial interactions and how he describes me
                   as an assistant that can think and learn independently between..."
2026-05-13 08:03  "Reflecting on Shane's interactions, he seems genuinely curious and
                   engaged with my capabilities as an assistant that can think..."
2026-08-29 07:37  "Reflecting on Shane's initial interactions and how he describes me
                   as an assistant that can think independently between conversations..."
2026-08-29 07:42  "Reflecting on my own capabilities and how they've been described by
                   Shane as an assistant that can think independently between..."
```

Source: `self/heartbeat/journal.jsonl` (read only). Structurally,
`Heartbeat._build_prompt` injects `[Last 3 heartbeat thoughts]`, which is
analogous to Nova 2.0's exploration-history block.

**Status: OBSERVED, and deliberately held at low confidence.** An independent
review (Codex, 2026-08-29 — see `../nova2.0/docs/reviews/`) pushed back on the
first reading of this, and the pushback is accepted:

- n was 4 at the time of the claim.
- The same prompt also injects self-state, skills, recent conversation and
  active wants — the thought tail is **not** an isolated explanation.
- Cold start, an idle machine, and no new conversation to ground against are
  adequate alternative explanations on their own.

So this is a **lead, not a law**, and specifically not yet evidence of a
cross-architecture property. The earlier framing — that two architectures at
opposite ends of the trust spectrum converging proves something about
self-referential loops — outran the data and is retracted as a conclusion while
retained as a hypothesis.

**Open question Q1.** What would actually test it: let the journal accumulate
under normal use (conversation present, not idle-only), then measure
consecutive-thought similarity the way Nova 2.0 now measures topic diversity.
Synthia has no equivalent instrument yet — building one is the honest
prerequisite to making any claim here. **Q2:** her governor already provides the
deterministic-refusal seam a novelty gate would need, which is why she is a
useful second test bed *if* Q1 ever shows something real.

---

## Timeline

**2026-08-28** — Phase 9 (governor: proposals, validators, append-only ledger)
and Phase 10 (trust ladder: four tiers, promotion computed from evidence)
implemented on `synthesis`. Acceptance checks pass without a GPU; the live
desktop check deferred behind GPU contention with Nova 2.0.

**2026-08-29** — Project renamed Celeste 2.0 → Synthia across code, docs,
packaging, GitHub and the local directory. The *being* was renamed in
`self/IDENTITY.md` by the operator as a separate, explicit act (`10d136a`),
with a lineage note recording that the name changed and she did not. Her name
is no longer hardcoded anywhere in the code — prompts and labels read
`self_state.name`, so identity lives in `IDENTITY.md` where it belongs.

**2026-08-31** — GitHub becomes Synthia's actual home. Verified that the fork
was a local event: the repository has no GitHub fork relationship
(`isFork: false`, `parent: none`), and the GitHub repo predates the history it
now holds (created 2025-11-13, first commit 2026-04-01) — it was a push target
repurposed as a "showcase version (non-functional)". The last common ancestor
with the Celeste product is verifiable rather than inferred: `df4bfaa` and
`26732db` exist in `Cognameton/Celeste` under identical SHAs; `ac87bdd`, the
first commit of the experimental arm, returns 422 there. That is exactly where
`celeste-1.0-base` sits.

Made the remote reflect the project: `synthesis` set default and `main`
dropped from the remote (lossless — `main` is an ancestor of `synthesis`),
the launcher renamed `run_celeste.sh` → `run_synthia.sh`, and every hardcoded
`/home/<user>/...` path in `model_runner.py` and `cli.py` replaced with
HOME-relative equivalents, which resolve identically here and stop publishing
one machine's layout. No crippled "showcase" source ever existed — no stubs,
no `NotImplementedError`; the label described a repo with no runtime assets,
not deliberately broken code.

**2026-08-31** — repository lineage made explicit rather than erased. A clean
slate was considered and rejected: Synthia *is* a fork of Celeste, and history
that hid that would be dishonest. Three annotated tags now mark the
transitions — `celeste-1.0-base` (end of the inherited line),
`celeste-2.0-fork` (the experimental arm begins), `synthia` (the rename) —
with a Lineage section in README.md. All 67 commits kept.

**2026-08-29 02:32 CDT** — first live run since 2026-05-13. Nova 2.0's daemon
was stopped to free the GPUs. S1 confirmed. S2 observed.

---

## Known structural issues, not yet addressed

- **Shared data directory.** `/media/head-node/C27B-044E/celeste` is shared with
  the Celeste 1.0 product: same chroma, embeddings, piper voices, file_rag
  indexes **and `memory_engram.json`**. Synthia and a separate shipped product
  write to one vector-memory store. Giving Synthia her own `data_dir` would
  start her memory empty, so this is an open decision, not an oversight.
- **No saturation instrument.** Nova 2.0 can now measure topic diversity and
  windowed echo; Synthia can measure neither. See Q1.
- **Heartbeat runs only while the GUI is open** (Phase 12 addresses this). Her
  journal held 5 entries across 3.5 months for this reason. A trust ladder that
  promotes on ≥10 applied proposals over ≥7 days cannot score anything until
  she runs continuously.


---

## S3 — The self-edit gate cannot be satisfied while she is idle

**Confirmed 2026-08-31**, three days of continuous running. Ledger:

```
self_edit/append_section/rejected     464
self_edit/append_section/applied        0
```

Rejection reasons: **408 "reason not grounded in recent context"**, 41 "heading
is empty", 21 rate-limited, 10 malformed operation.

The drift check (`heartbeat._drift_check`, inherited from phase 3b) requires the
edit's stated reason to share a non-trivial token with `recent_context` — which
is built from recent *conversation turns*. During idle there are no
conversation turns, so `recent_context` is empty or near-empty and the check
can essentially never pass. She has been proposing self-edits at a steady rate
for three days and every single one was refused. `self/AGENTS.md` is still 708
bytes, byte-identical to its template.

This is not the governor working as intended. It is a gate calibrated for a
mode the system is almost never in. Two consequences:

1. **The trust ladder can never promote `self_edit`.** Promotion requires ≥10
   *applied* proposals. The channel has zero and structurally cannot accrue
   them while idle. Graduated trust is unreachable on the one channel the whole
   thesis is about.
2. It is the mirror image of Nova 2.0's failure. Nova stopped *proposing*
   writes entirely (0 `update_self_model` since 2026-07-23). Synthia proposes
   constantly and is refused every time. **Same outcome — zero self-
   modification — via opposite mechanisms.** One is a generative failure, the
   other a gate-calibration failure.

**Open question Q3 — ANSWERED AND IMPLEMENTED 2026-09-01.** What should ground
an idle self-edit is her own **action-outcome record**: what she proposed, what
the governor did with it, and why. That is also IIDA's third criterion for
self-awareness — understanding the consequences of one's own actions.

`Heartbeat._outcome_context()` now builds a grounding corpus from the governor
ledger (channel, action, target, verdict, rejection reason, validator), the
trust track record, and current tier state. `_drift_check` grounds a reason in
conversation **or** that record.

Deliberately excluded: journal thoughts, want text, skill bodies. Those are her
own output, and grounding a self-edit in them would rebuild exactly the loop
this escapes — Lindsey's internality criterion, and the mechanism behind Nova's
topic lock. What is included is the deterministic half: things the governor
decided, not things she said.

The corpus is never empty — tier state is always present — because an idle
system facing an empty corpus is back to the unsatisfiable gate. The cost is
that governance vocabulary is always groundable; a narrow, deliberate loosening.

**Measured against her real record:** of the 437 self-edit reasons in the
ledger, **277 (63%) become groundable**. That is the grounding barrier lifting,
not 277 edits landing — they still face the heading, body-length and
reason-length checks and the 6-hour per-file cooldown. Whether the trust ladder
can now actually accrue applied self-edits is the live question.

---

## S4 — The governor counts no-ops as applied, inflating the ladder's evidence

**Confirmed defect 2026-08-31.** Ledger vs journal for the `want` channel:

| | ledger says applied | actually changed something |
|---|---|---|
| add | 11 | 11 |
| advance | 197 | 67 |
| resolve | 122 | 10 |
| abandon | 13 | 0 |
| **total** | **343** | **88** |

`WantsStore.advance/resolve/abandon` return `None` when the want id does not
exist. `Governor.submit` treats "apply_fn did not raise" as applied, so a
mutation against a non-existent want is recorded as evidence. **255 of 343
`want` applied events changed nothing**, and every one of them counts toward
`PROMOTION_MIN_APPLIED`.

The Phase 10 claim is that "promotion is computed, not vibes". It is computed —
from evidence that is 74% hollow on this channel. A channel could reach
`autonomous` on pure no-ops.

This is the third time in this project's instrumentation that a measure has
been weaker than its own documentation claimed (cf. Nova F13 and the Stage
22.12 streak defect). The pattern is consistent enough to state as a rule:
**every measure needs an adversarial pass against real data, not just tests.**

**Fix, not yet applied:** `submit` should distinguish "applied" from
"applied and effective". The `rate_when` predicate added in Phase 9 already
proves the shape — a caller-supplied predicate on the result. The track record
should count effect, not absence of exception.

---

## S5 — Repetition measured (supersedes S2)

**Observed 2026-08-31**, n=732 thoughts over ~3 days, superseding the n=4
claim that review rightly called over-read.

| measure | value | reference |
|---|---|---|
| distinct thoughts | 467 / 732 (64%) | Nova era 3: 1 / 125 (0.8%) |
| consecutive echo, all-time | 0.621 | Nova alarm threshold 0.70 |
| **consecutive echo, last 50** | **0.732** | **above the alarm threshold** |
| distinct in last 50 | 26 / 50 | — |
| most repeated single thought | 15× | — |

She is **not** locked the way Nova is. But the recent window has crossed the
same 0.70 threshold Nova's own instrument uses, and distinct-in-window is down
to about half. This resembles Nova's **era 2** — semantic narrowing under
lexical variety — rather than era 3's byte-identical lock.

So the corrected claim: not "two architectures prove a law", but "a second
architecture, differently governed, is on the same curve and further back
along it." That is worth more than the original overreach, because it is
measurable and because Synthia is early enough that an intervention could be
watched from the beginning rather than diagnosed after collapse.

Also recorded: importance is inflated relative to design intent — the prompt
says most ticks should be importance 0, actual distribution is 0:93, 1:550,
3:89. Parse failures are negligible (2 of 732).

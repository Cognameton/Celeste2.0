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
| **S2** | Repetition pattern in idle heartbeat thoughts | OBSERVED — weak, do not over-read |

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
`/home/head-node/...` path in `model_runner.py` and `cli.py` replaced with
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

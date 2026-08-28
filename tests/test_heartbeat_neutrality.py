"""Behavior-neutrality probe for Phase 9 (PHASE9.md acceptance check 3).

Drives Heartbeat._apply directly with a hand-built TickResult — no model, no
GPU, no thread — and asserts that the post-governor pipeline produces exactly
the pre-governor outcome: the same writes land, the same rejections fire with
the same wording, and journal.jsonl keeps its original field names.

Run: .venv/bin/python tests/test_heartbeat_neutrality.py
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from governor import CHANNELS, Governor    # noqa: E402
from heartbeat import Heartbeat, HeartbeatConfig, TickResult   # noqa: E402
from self_state import SelfState           # noqa: E402
from user_model import UserModel           # noqa: E402
from trust import INITIAL_TIER, TrustLadder  # noqa: E402
from wants import WantsStore               # noqa: E402

logging.disable(logging.CRITICAL)

# The drift check requires the reason to share a real word with recent context.
RECENT_CONTEXT = "shane asked about the deploy pipeline and rsync again today"
GOOD_REASON = "Shane corrected the deploy story twice in the recent turns."


class StubLLM:
    """Never called — _apply does no inference."""

    def chat(self, **kwargs):
        raise AssertionError("_apply must not call the model")


def build(root: Path) -> tuple[Heartbeat, Governor, SelfState]:
    self_state = SelfState.initialize(root / "self", ROOT / "self_template")
    wants = WantsStore(self_state.root / "wants")
    user_model = UserModel(self_state)
    # Phase 10: a live ladder at its initial tiers must reproduce Phase 9
    # behavior exactly — that is the whole neutrality claim.
    ladder = TrustLadder(self_state.root / "governor", channels=CHANNELS)
    assert all(v["tier"] == INITIAL_TIER for v in ladder.tiers().values())
    governor = Governor(self_state.root, trust=ladder)
    hb = Heartbeat(
        llm=StubLLM(),
        self_state=self_state,
        wants=wants,
        user_model=user_model,
        governor=governor,
        config=HeartbeatConfig(enabled=False),
        is_busy=lambda: False,
        last_user_activity_ts=lambda: 0.0,
        recent_turns=lambda n: [],
    )
    return hb, governor, self_state


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        hb, governor, self_state = build(Path(td))

        result = TickResult(
            private_thought="Noticing the deploy question keeps coming back.",
            importance=1,
            wants_added=[{"text": "understand the deploy pipeline end to end", "priority": 4}],
            self_edits=[
                {   # valid — should land in AGENTS.md
                    "file": "AGENTS.md",
                    "operation": "append_section",
                    "heading": "Deploys",
                    "body": "Deploys go through git, not rsync. Confirmed with Shane.",
                    "reason": GOOD_REASON,
                },
                {   # protected file — must be refused
                    "file": "SOUL.md",
                    "operation": "append_section",
                    "heading": "Voice",
                    "body": "Rewriting my own persona wholesale.",
                    "reason": GOOD_REASON,
                },
            ],
            skills_proposed=[
                {   # reason too short — must be refused
                    "slug": "deploy-helper",
                    "name": "Deploy Helper",
                    "description": "Helps reason about deploy pipelines.",
                    "when_to_use": "deploy questions",
                    "reason": "because",
                },
            ],
        )

        hb._apply(result, recent_context=RECENT_CONTEXT)

        # --- the valid self-edit landed ---
        agents = self_state.read("AGENTS.md")
        assert "Deploys go through git, not rsync." in agents, "valid self-edit did not land"
        assert "Rewriting my own persona wholesale." not in self_state.read("SOUL.md"), \
            "SOUL.md was written"

        # --- the want exists ---
        active = [w.text for w in WantsStore(self_state.root / "wants").load_active()]
        assert "understand the deploy pipeline end to end" in active, active

        # --- the skill draft was refused ---
        assert not (self_state.root / "skills" / "deploy-helper").exists(), \
            "short-reason skill was created"

        # --- journal keeps its pre-Phase-9 shape ---
        entry = json.loads((self_state.root / "heartbeat" / "journal.jsonl")
                           .read_text(encoding="utf-8").strip().splitlines()[-1])
        expected_fields = {
            "ts", "importance", "private_thought", "wants_added", "wants_advanced",
            "wants_resolved", "wants_abandoned", "self_edits_applied",
            "self_edits_rejected", "skills_proposed_applied",
            "skills_proposed_rejected", "parse_error",
        }
        assert set(entry) == expected_fields, set(entry) ^ expected_fields
        assert len(entry["wants_added"]) == 1
        assert entry["self_edits_applied"] == [
            {"file": "AGENTS.md", "heading": "Deploys", "reason": GOOD_REASON}
        ], entry["self_edits_applied"]
        assert entry["self_edits_rejected"] == [
            {"file": "SOUL.md", "reason": "file 'SOUL.md' not in allowed set"}
        ], entry["self_edits_rejected"]
        assert entry["skills_proposed_rejected"] == [
            {"slug": "deploy-helper", "reason": "reason too short (7 chars)"}
        ], entry["skills_proposed_rejected"]
        assert entry["skills_proposed_applied"] == []

        # --- the ledger holds all four proposals with matching verdicts ---
        ledger = governor.ledger_tail()
        assert len(ledger) == 4, [(e["proposal"]["channel"], e["verdict"]) for e in ledger]
        verdicts = {(e["proposal"]["channel"], e["proposal"]["target"]): e["verdict"]
                    for e in ledger}
        assert verdicts[("want", "")] == "applied", verdicts
        assert verdicts[("self_edit", "AGENTS.md")] == "applied", verdicts
        assert verdicts[("self_edit", "SOUL.md")] == "rejected", verdicts
        assert verdicts[("skill", "deploy-helper")] == "rejected", verdicts
        drift = [e for e in ledger if e["proposal"]["target"] == "SOUL.md"][0]
        assert drift["validator"] == "drift_check", drift

        # --- rate limiting survived the migration: a second AGENTS.md edit is refused ---
        second = TickResult(self_edits=[dict(result.self_edits[0], heading="Deploys again")])
        hb._apply(second, recent_context=RECENT_CONTEXT)
        entry2 = json.loads((self_state.root / "heartbeat" / "journal.jsonl")
                            .read_text(encoding="utf-8").strip().splitlines()[-1])
        assert entry2["self_edits_rejected"] == [
            {"file": "AGENTS.md", "reason": "rate-limited"}
        ], entry2["self_edits_rejected"]

        # --- the ladder recorded evidence but moved nobody ---
        ladder = governor.trust
        assert ladder is not None
        assert all(v["tier"] == INITIAL_TIER for v in ladder.tiers().values()), \
            "initial tiers must not shift during a normal tick"
        assert ladder.track.counts("self_edit", ladder._since_of("self_edit")) \
            == {"applied": 1, "rejected": 2, "revert": 0, "drift_flag": 0,
                "override": 0, "tier_change": 0}, \
            ladder.track.counts("self_edit", ladder._since_of("self_edit"))
        assert ladder.evaluate() == [], "nothing is earned on day one"

        # --- flush commits into the real self/ git repo ---
        governor.flush()
        import subprocess
        log = subprocess.run(["git", "log", "--oneline", "-1"], cwd=self_state.root,
                             capture_output=True, text=True, check=False).stdout
        assert "Governor: ledger checkpoint" in log, log
        tracked = subprocess.run(["git", "ls-files", "governor/"], cwd=self_state.root,
                                 capture_output=True, text=True, check=False).stdout
        assert "governor/ledger.jsonl" in tracked, tracked
        assert "governor/trust.json" in tracked, tracked
        assert "governor/track_record.jsonl" in tracked, tracked

    print("behavior-neutrality probe: all assertions passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

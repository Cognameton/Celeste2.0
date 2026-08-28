"""Plain-assert tests for the Phase 10 trust ladder.

Run: .venv/bin/python tests/test_trust.py    (no pytest, no GPU, no model)
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from governor import CHANNELS, Governor, Proposal   # noqa: E402
from trust import (  # noqa: E402
    INITIAL_TIER,
    PROMOTION_MIN_APPLIED,
    PROMOTION_MIN_SPAN_DAYS,
    TIERS,
    TrustLadder,
)

logging.disable(logging.CRITICAL)

GOOD_REASON = "The operator corrected the deploy path twice in recent turns."
_PASSED: list[str] = []


def case(fn):
    with tempfile.TemporaryDirectory() as td:
        fn(Path(td))
    _PASSED.append(fn.__name__)
    print(f"  ok  {fn.__name__}")


def _ladder(root: Path, **kw) -> TrustLadder:
    return TrustLadder(root / "governor", channels=CHANNELS, **kw)


def _gov(root: Path, ladder: TrustLadder | None, **kw) -> Governor:
    return Governor(root, trust=ladder, **kw)


def _edit(target="AGENTS.md"):
    return Proposal.new(channel="self_edit", origin="heartbeat",
                        action="append_section", target=target,
                        payload={"heading": "H", "body": "Deploys go through git."},
                        evidence=GOOD_REASON)


def _backdate(ladder: TrustLadder, channel: str, days: int) -> None:
    """Move a channel's `since` into the past without touching its tier."""
    data = json.loads(ladder.path.read_text())
    stamp = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    data["channels"][channel]["since"] = stamp
    ladder.path.write_text(json.dumps(data))


def _earn(ladder: TrustLadder, channel: str, n: int = PROMOTION_MIN_APPLIED) -> None:
    for _ in range(n):
        ladder.note_applied(channel)


# 1. Defaults
def test_defaults(root: Path):
    ladder = _ladder(root)
    tiers = ladder.tiers()
    assert set(tiers) == set(CHANNELS), tiers
    assert all(v["tier"] == INITIAL_TIER for v in tiers.values()), tiers
    assert ladder.tier_for("no_such_channel") == INITIAL_TIER
    ladder.path.unlink()
    assert ladder.tier_for("self_edit") == INITIAL_TIER, "missing trust.json must not lock out"
    ladder.path.write_text("{ not json")
    assert ladder.tier_for("self_edit") == INITIAL_TIER, "corrupt trust.json must not lock out"


# 2. Neutrality: a default ladder matches Phase 9's trust=None governor
def test_neutrality(root: Path):
    for name, ladder in (("none", None), ("default", _ladder(root))):
        sub = root / name
        sub.mkdir()
        g = _gov(sub, ladder)
        calls = []
        assert g.submit(_edit(), lambda: calls.append(1)).verdict == "applied", name
        assert g.submit(_edit(), lambda: calls.append(1)).verdict == "rejected", name
        assert g.submit(_edit(target="SOUL.md"), lambda: calls.append(1)).verdict == "rejected", name
        assert len(calls) == 1, (name, calls)


# 3. observe: nothing applies
def test_observe(root: Path):
    ladder = _ladder(root)
    ladder.set_tier("self_edit", "observe", reason="test", operator=True)
    g = _gov(root, ladder)
    calls = []
    d = g.submit(_edit(), lambda: calls.append(1))
    assert d.verdict == "observed", d
    assert not calls, "apply_fn ran at tier observe"
    assert g.ledger_tail()[-1]["verdict"] == "observed"
    # an observed proposal is not evidence of anything
    assert ladder.track.counts("self_edit", ladder._since_of("self_edit"))["applied"] == 0


# 4. propose: held, then approved or rejected
def test_propose(root: Path):
    ladder = _ladder(root)
    ladder.set_tier("self_edit", "propose", reason="test", operator=True)
    g = _gov(root, ladder)
    calls = []
    p = _edit()
    d = g.submit(p, lambda: calls.append(1))
    assert d.verdict == "pending", d
    assert not calls
    pending = g.list_pending()
    assert len(pending) == 1 and pending[0]["id"] == p.id, pending

    approved = g.approve(p.id)
    assert approved is not None and approved.verdict == "applied", approved
    assert len(calls) == 1, calls
    assert g.list_pending() == []
    assert g.approve(p.id) is None, "approving twice must not re-run the thunk"

    # reject drops the thunk
    ladder.set_tier("self_edit", "propose", reason="test", operator=True)
    p2 = _edit(target="USER.md")
    assert g.submit(p2, lambda: calls.append(2)).verdict == "pending"
    assert g.reject(p2.id) is True
    assert len(calls) == 1, "rejected proposal was applied"
    assert g.reject(p2.id) is False


# 5. Promotion needs evidence AND time
def test_promotion(root: Path):
    ladder = _ladder(root)
    _earn(ladder, "want")
    assert ladder.evaluate() == [], "promoted without serving the wall-clock span"
    assert ladder.tier_for("want") == "review"

    _backdate(ladder, "want", PROMOTION_MIN_SPAN_DAYS + 1)
    changes = ladder.evaluate()
    assert changes == [{"channel": "want", "from": "review", "to": "autonomous"}], changes
    assert ladder.tier_for("want") == "autonomous"

    # the clock resets, so it does not immediately promote again
    assert ladder.evaluate() == []


def test_promotion_needs_volume(root: Path):
    ladder = _ladder(root)
    _backdate(ladder, "want", PROMOTION_MIN_SPAN_DAYS + 1)
    _earn(ladder, "want", PROMOTION_MIN_APPLIED - 1)
    assert ladder.evaluate() == [], "promoted without enough applied proposals"
    ladder.note_applied("want")
    assert ladder.evaluate() != []


# 6. A revert demotes and poisons the window
def test_revert_demotes(root: Path):
    ladder = _ladder(root)
    new_tier = ladder.record_revert("self_edit", target="AGENTS.md", note="bad append")
    assert new_tier == "propose", new_tier
    assert ladder.tier_for("self_edit") == "propose"

    _earn(ladder, "self_edit")
    _backdate(ladder, "self_edit", PROMOTION_MIN_SPAN_DAYS + 1)
    ladder.record_revert("self_edit", note="another one")
    assert ladder.tier_for("self_edit") == "observe"
    _earn(ladder, "self_edit")
    _backdate(ladder, "self_edit", PROMOTION_MIN_SPAN_DAYS + 1)
    ladder.record_drift_flag("self_edit", detail="voice drift 0.4")
    assert ladder.tier_for("self_edit") == "observe", "demotion must not go below the floor"


def test_revert_blocks_promotion(root: Path):
    ladder = _ladder(root)
    ladder.set_tier("want", "review", reason="reset", operator=True)
    _earn(ladder, "want")
    ladder.track.record("want", "revert", detail="in-window revert")
    _backdate(ladder, "want", PROMOTION_MIN_SPAN_DAYS + 1)
    assert ladder.evaluate() == [], "a revert in the window must block promotion"


# 7. Ceiling
def test_ceiling(root: Path):
    ladder = _ladder(root)
    ladder.set_tier("tool", "autonomous", reason="test", operator=True)
    _earn(ladder, "tool")
    _backdate(ladder, "tool", PROMOTION_MIN_SPAN_DAYS + 1)
    assert ladder.evaluate() == []
    assert ladder.tier_for("tool") == TIERS[-1]


# 8. Operator override
def test_operator_override(root: Path):
    ladder = _ladder(root)
    assert ladder.set_tier("skill", "autonomous", reason="I trust her here", operator=True)
    assert ladder.tier_for("skill") == "autonomous"
    events = [e["event"] for e in ladder.track.tail(channel="skill")]
    assert "override" in events and "tier_change" in events, events
    assert ladder.set_tier("skill", "not-a-tier") is False
    assert ladder.tier_for("skill") == "autonomous"


# 9. autonomous is quiet, review is not
def test_emit_suppression(root: Path):
    ladder = _ladder(root)
    seen: list[str] = []
    g = _gov(root, ladder, on_event=seen.append)
    g.submit(_edit(), lambda: None)
    assert any("applied: self_edit" in m for m in seen), seen

    ladder.set_tier("want", "autonomous", reason="test", operator=True)
    seen.clear()
    p = Proposal.new(channel="want", origin="heartbeat", action="add",
                     target="", payload={"text": "x"})
    assert g.submit(p, lambda: "w").verdict == "applied"
    assert seen == [], seen
    assert g.ledger_tail()[-1]["verdict"] == "applied", "the ledger still sees it"


# 10. Rejections are evidence, but never promote
def test_rejections_recorded(root: Path):
    ladder = _ladder(root)
    g = _gov(root, ladder)
    for _ in range(PROMOTION_MIN_APPLIED + 2):
        g.submit(_edit(target="SOUL.md"), lambda: None)
    counts = ladder.track.counts("self_edit", ladder._since_of("self_edit"))
    assert counts["rejected"] == PROMOTION_MIN_APPLIED + 2, counts
    assert counts["applied"] == 0, counts
    _backdate(ladder, "self_edit", PROMOTION_MIN_SPAN_DAYS + 1)
    assert ladder.evaluate() == [], "rejections must not earn a promotion"


# extra: stale pending rows cannot be approved after a restart
def test_pending_expires_across_processes(root: Path):
    ladder = _ladder(root)
    ladder.set_tier("want", "propose", reason="test", operator=True)
    g1 = _gov(root, ladder)
    p = Proposal.new(channel="want", origin="heartbeat", action="add",
                     target="", payload={"text": "survive a restart"})
    assert g1.submit(p, lambda: "w").verdict == "pending"

    g2 = _gov(root, ladder)          # new process, new governor
    assert g2.list_pending() == [], "in-memory queue must not appear to survive"
    assert g2.approve(p.id) is None
    states = [r.get("state") for r in g2._read_pending_rows()]
    assert "expired" in states, states


def main() -> int:
    print("trust ladder tests")
    for fn in (test_defaults, test_neutrality, test_observe, test_propose,
               test_promotion, test_promotion_needs_volume, test_revert_demotes,
               test_revert_blocks_promotion, test_ceiling, test_operator_override,
               test_emit_suppression, test_rejections_recorded,
               test_pending_expires_across_processes):
        case(fn)
    print(f"\n{len(_PASSED)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

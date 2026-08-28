"""Plain-assert tests for governor.py. Run: .venv/bin/python tests/test_governor.py

No pytest dependency, no GPU, no model. Every case uses a bare temp dir as
self_root — the Governor must work outside a git repo without crashing.
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from governor import (  # noqa: E402
    Governor,
    Proposal,
    MIN_REASON_LEN,
    SKILL_RATE_KEY,
)

# Governor logs apply failures; test_apply_error triggers one on purpose.
logging.disable(logging.CRITICAL)

GOOD_REASON = "The operator corrected the deploy path twice in recent turns."
_PASSED: list[str] = []


def _gov(root: Path, **kw) -> Governor:
    return Governor(root, **kw)


def _self_edit(target="AGENTS.md", reason=GOOD_REASON, heading="Deploys", body="Use git, not rsync."):
    return Proposal.new(
        channel="self_edit", origin="heartbeat", action="append_section",
        target=target, payload={"heading": heading, "body": body, "reason": reason},
        evidence=reason,
    )


def case(fn):
    with tempfile.TemporaryDirectory() as td:
        fn(Path(td))
    _PASSED.append(fn.__name__)
    print(f"  ok  {fn.__name__}")


# 1. Applied proposal
def test_applied(root: Path):
    calls = []
    g = _gov(root)
    d = g.submit(_self_edit(), apply_fn=lambda: calls.append(1))
    assert d.verdict == "applied", d
    assert len(calls) == 1, calls
    tail = g.ledger_tail()
    assert len(tail) == 1 and tail[0]["verdict"] == "applied", tail
    assert tail[0]["proposal"]["channel"] == "self_edit"
    assert json.loads(g.rate_log_path.read_text())["AGENTS.md"], "rate_log not updated"


# 2. Rate limit blocks the second edit to the same file
def test_rate_limit(root: Path):
    calls = []
    g = _gov(root)
    assert g.submit(_self_edit(), lambda: calls.append(1)).verdict == "applied"
    d = g.submit(_self_edit(), lambda: calls.append(2))
    assert d.verdict == "rejected" and d.validator == "v_rate_limit", d
    assert d.reason == "rate-limited", d.reason
    assert len(calls) == 1, "apply_fn ran despite rate limit"


# 3. Protected files are unreachable
def test_protected_file(root: Path):
    g = _gov(root)
    for target in ("SOUL.md", "IDENTITY.md"):
        calls = []
        d = g.submit(_self_edit(target=target), lambda: calls.append(1))
        assert d.verdict == "rejected" and d.validator == "v_protected_file", d
        assert d.reason == f"file '{target}' not in allowed set", d.reason
        assert not calls


# 4. Short reason, legacy message format
def test_short_reason(root: Path):
    g = _gov(root)
    d = g.submit(_self_edit(reason="too short"), lambda: None)
    assert d.verdict == "rejected" and d.validator == "v_reason", d
    assert d.reason == f"reason too short (9 chars < {MIN_REASON_LEN})", d.reason
    # skill channel keeps its own legacy wording (no "< 20" suffix)
    p = Proposal.new(channel="skill", origin="heartbeat", action="create",
                     target="a-skill", payload={}, evidence="short")
    d2 = g.submit(p, lambda: None)
    assert d2.reason == "reason too short (5 chars)", d2.reason


# 5. Extra validator rejection
def test_extra_validator(root: Path):
    def fake_drift(proposal):
        return "reason not grounded in recent context"

    calls = []
    g = _gov(root)
    d = g.submit(_self_edit(), lambda: calls.append(1), extra_validators=(fake_drift,))
    assert d.verdict == "rejected", d
    assert d.validator == "fake_drift", d.validator
    assert d.reason == "reason not grounded in recent context"
    assert not calls
    assert g.ledger_tail()[-1]["validator"] == "fake_drift"


# 6. apply_fn raises -> verdict "error", submit() does not raise
def test_apply_error(root: Path):
    def boom():
        raise PermissionError("IDENTITY.md is immutable to the agent.")

    g = _gov(root)
    d = g.submit(_self_edit(), boom)
    assert d.verdict == "error", d
    assert "immutable" in d.reason, d.reason
    tail = g.ledger_tail()
    assert len(tail) == 1 and tail[0]["verdict"] == "error"
    # an errored apply must not consume the cooldown
    assert "AGENTS.md" not in json.loads(g.rate_log_path.read_text())


# 7. Cooldown state is seeded from heartbeat/edit_log.json
def test_seed_from_edit_log(root: Path):
    hb = root / "heartbeat"
    hb.mkdir(parents=True)
    fresh = (datetime.now(timezone.utc) - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    (hb / "edit_log.json").write_text(json.dumps({"AGENTS.md": fresh}))
    g = _gov(root)
    d = g.submit(_self_edit(), lambda: None)
    assert d.verdict == "rejected" and d.validator == "v_rate_limit", d
    assert (hb / "edit_log.json").exists(), "legacy edit_log must not be deleted"
    # a file with no seeded entry is still free
    stale = (datetime.now(timezone.utc) - timedelta(hours=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
    g._write_rate_log({"AGENTS.md": stale})
    assert g.submit(_self_edit(), lambda: None).verdict == "applied"


# 8. Tool channel
def test_tool_channel(root: Path):
    registry = {"read_file": "safe", "danger": "blocked"}
    g = _gov(root, tool_risk_lookup=lambda name: registry.get(name))

    def tool(name):
        return Proposal.new(channel="tool", origin="react", action="execute",
                            target=name, payload={"path": "/etc/hostname"})

    d = g.submit(tool("nope"), lambda: "ran")
    assert d.verdict == "rejected" and d.validator == "v_tool", d
    assert d.reason == "Unknown tool: 'nope'", d.reason

    d = g.submit(tool("danger"), lambda: "ran")
    assert d.verdict == "rejected" and d.reason == "Tool 'danger' is blocked", d

    d = g.submit(tool("read_file"), lambda: "contents")
    assert d.verdict == "applied" and d.result == "contents", d
    # tools are not rate-limited
    assert g.submit(tool("read_file"), lambda: "again").verdict == "applied"


# 9. ledger_tail ordering
def test_ledger_tail(root: Path):
    g = _gov(root)
    g.submit(_self_edit(target="SOUL.md"), lambda: None)
    g.submit(_self_edit(reason="nope"), lambda: None)
    g.submit(_self_edit(target="USER.md"), lambda: None)
    tail = g.ledger_tail(2)
    assert len(tail) == 2, tail
    assert tail[0]["validator"] == "v_reason", tail[0]
    assert tail[-1]["verdict"] == "applied", tail[-1]


# extra: channels that carry no rate limit, and flush() outside a git repo
def test_unrated_channels_and_flush(root: Path):
    g = _gov(root)
    for _ in range(3):
        p = Proposal.new(channel="want", origin="heartbeat", action="add",
                         target="", payload={"text": "learn the deploy story"})
        assert g.submit(p, lambda: "want").verdict == "applied"
    promote = Proposal.new(channel="skill", origin="reflector", action="promote",
                           target="a-skill", payload={}, evidence=GOOD_REASON)
    assert g.submit(promote, lambda: True).verdict == "applied"
    assert g.submit(promote, lambda: True).verdict == "applied", "promote must not be rate-limited"
    assert SKILL_RATE_KEY not in json.loads(g.rate_log_path.read_text())
    g.flush()   # bare temp dir is not a git repo — must not raise
    # user_model: unwritable section rejected, writable applied then rate-limited
    bad = Proposal.new(channel="user_model", origin="heartbeat", action="upsert",
                       target="Secrets", payload={"key": "k", "value": "v"})
    assert g.submit(bad, lambda: True).validator == "v_user_model_section"
    good = Proposal.new(channel="user_model", origin="heartbeat", action="upsert",
                        target="Expertise", payload={"key": "Python", "value": "deep"})
    assert g.submit(good, lambda: True).verdict == "applied"
    assert g.submit(good, lambda: True).validator == "v_rate_limit"
    # rate_when: an upsert that changed nothing must not consume the cooldown
    noop = Proposal.new(channel="user_model", origin="heartbeat", action="upsert",
                        target="Inferred Goals", payload={"key": "k", "value": "v"})
    d = g.submit(noop, lambda: False, rate_when=lambda changed: bool(changed))
    assert d.verdict == "applied" and d.result is False, d
    assert "__user_model_Inferred Goals__" not in json.loads(g.rate_log_path.read_text())
    d = g.submit(noop, lambda: True, rate_when=lambda changed: bool(changed))
    assert d.verdict == "applied"
    assert g.submit(noop, lambda: True).validator == "v_rate_limit"
    # log_correction is not rate-limited
    corr = Proposal.new(channel="user_model", origin="reflector", action="log_correction",
                        target="Corrections", payload={"content": "x"})
    assert g.submit(corr, lambda: True).verdict == "applied"
    assert g.submit(corr, lambda: True).verdict == "applied"


def main() -> int:
    print("governor tests")
    for fn in (test_applied, test_rate_limit, test_protected_file, test_short_reason,
               test_extra_validator, test_apply_error, test_seed_from_edit_log,
               test_tool_channel, test_ledger_tail, test_unrated_channels_and_flush):
        case(fn)
    print(f"\n{len(_PASSED)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

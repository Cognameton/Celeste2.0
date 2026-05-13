"""UserModel — structured section-level access to USER.md.

Wraps SelfState to provide upsert, append, and correction-log operations
on named sections. All writes are serialized with a threading.Lock so
the heartbeat thread and reflection thread can safely share one instance.
"""
from __future__ import annotations

import re
import threading
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from self_state import SelfState

FILE = "USER.md"

# Sections the heartbeat is allowed to target
WRITABLE_SECTIONS = frozenset({
    "Expertise",
    "Inferred Goals",
    "Working preferences",
    "Things I'm still figuring out about him",
})


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class UserModel:
    def __init__(self, self_state: "SelfState"):
        self.self_state = self_state
        self._lock = threading.Lock()

    # ---- read ----

    def read_section(self, section: str) -> str:
        return _extract_section(self.self_state.read(FILE), section)

    def full_text(self) -> str:
        return self.self_state.read(FILE).strip()

    # ---- write ----

    def upsert_entry(self, section: str, key: str, value: str) -> bool:
        """Add or update a `- key: value` bullet in a section. Returns True if changed."""
        with self._lock:
            content = self.self_state.read(FILE)
            updated = _upsert_entry(content, section, key, value)
            if updated == content:
                return False
            self.self_state.write(FILE, updated, message=f"User model: {section} / {key}")
            return True

    def append_to_section(self, section: str, text: str) -> None:
        """Append a line to the end of a named section."""
        with self._lock:
            content = self.self_state.read(FILE)
            updated = _append_to_section(content, section, text.strip())
            if updated != content:
                self.self_state.write(FILE, updated, message=f"User model: {section}")

    def log_correction(self, description: str) -> None:
        """Append a timestamped entry to the Corrections section."""
        self.append_to_section("Corrections", f"- {_now()}: {description}")


# ---- parsing helpers ----

def _section_bounds(content: str, section: str) -> tuple[int, int] | None:
    """Return (body_start, body_end) for a ## section, or None if not found."""
    m = re.search(
        r"^##\s+" + re.escape(section) + r"\s*$",
        content,
        re.MULTILINE | re.IGNORECASE,
    )
    if not m:
        return None
    body_start = m.end()
    next_h = re.search(r"^##\s+", content[body_start:], re.MULTILINE)
    body_end = body_start + next_h.start() if next_h else len(content)
    return body_start, body_end


def _extract_section(content: str, section: str) -> str:
    bounds = _section_bounds(content, section)
    if bounds is None:
        return ""
    return content[bounds[0]:bounds[1]].strip()


def _upsert_entry(content: str, section: str, key: str, value: str) -> str:
    bounds = _section_bounds(content, section)
    if bounds is None:
        return content.rstrip() + f"\n\n## {section}\n- {key}: {value}\n"
    body_start, body_end = bounds
    body = content[body_start:body_end]
    entry_re = re.compile(
        r"^(\s*-\s*" + re.escape(key) + r"\s*:).*$",
        re.MULTILINE | re.IGNORECASE,
    )
    new_body, n = entry_re.subn(f"- {key}: {value}", body)
    if n == 0:
        new_body = body.rstrip() + f"\n- {key}: {value}\n"
    return content[:body_start] + new_body + content[body_end:]


def _append_to_section(content: str, section: str, text: str) -> str:
    bounds = _section_bounds(content, section)
    if bounds is None:
        return content.rstrip() + f"\n\n## {section}\n{text}\n"
    body_start, body_end = bounds
    body = content[body_start:body_end]
    new_body = body.rstrip() + f"\n{text}\n"
    return content[:body_start] + new_body + content[body_end:]

"""ContextCompressor — summarizes old conversation turns to preserve long-session coherence.

When _recent_turns grows past a configurable threshold, the compressor condenses
the oldest turns into a structured Session Memory block. The block is kept as a
special {"role": "summary", ...} entry at the head of _recent_turns, so subsequent
prompts always have the full research thread without blowing the context window.

Compression is triggered at the start of each respond() call. The LLM call is
synchronous — it runs on the primary model for now; Phase 5 (secondary agent)
can offload it later.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any


_SUMMARY_SECTIONS = (
    "Active Thread",
    "Decisions Made",
    "Open Questions",
    "Sources Consulted",
)

_SUMMARY_PROMPT = """\
You are summarizing a conversation segment to preserve its key content as a compact \
Session Memory block. The conversation is between a user (Shane) and an AI research \
partner ({assistant_name}).

Produce a structured summary with exactly these four sections. Be specific and concrete \
— this will be the only record of these turns. Omit sections that have nothing to report \
(write "none" for them).

Format:
Active Thread: [what topic or task was in focus]
Decisions Made: [conclusions, choices, or directions agreed on]
Open Questions: [unresolved questions still worth tracking]
Sources Consulted: [any files, documents, or tools referenced]

Conversation to summarize:
{turns}

Output only the four-section block. No preamble, no commentary."""


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ContextCompressor:
    """
    Monitors _recent_turns and compresses when the list exceeds max_turns.

    After compression:
        _recent_turns = [{"role": "summary", ...}] + last keep_turns turns
    """

    def __init__(self, llm: Any, *, assistant_name: str = "Synthia",
                 max_turns: int = 16, keep_turns: int = 6,
                 max_summary_tokens: int = 500):
        self.llm = llm
        self.assistant_name = assistant_name or "Synthia"
        self.max_turns = max_turns
        self.keep_turns = keep_turns
        self.max_summary_tokens = max_summary_tokens

    def should_compress(self, turns: list[dict[str, Any]]) -> bool:
        non_summary = [t for t in turns if t.get("role") != "summary"]
        return len(non_summary) >= self.max_turns

    def compress(self, turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return a new turns list with old turns replaced by a summary entry."""
        non_summary = [t for t in turns if t.get("role") != "summary"]
        prior_summary = next((t for t in turns if t.get("role") == "summary"), None)

        to_compress = non_summary[: -self.keep_turns]
        to_keep = non_summary[-self.keep_turns :]

        if not to_compress:
            return turns

        summary_text = self._summarize(to_compress, prior_summary=prior_summary)

        summary_entry: dict[str, Any] = {
            "role": "summary",
            "content": summary_text,
            "compressed_at": _now_utc(),
            "turns_compressed": len(to_compress) + (
                prior_summary.get("turns_compressed", 0) if prior_summary else 0
            ),
        }
        logging.info(
            "ContextCompressor: compressed %d turns into session memory",
            len(to_compress),
        )
        return [summary_entry] + to_keep

    def _summarize(
        self, turns: list[dict[str, Any]], prior_summary: dict[str, Any] | None
    ) -> str:
        turns_text = self._format_turns_for_summary(turns)

        if prior_summary:
            prior_text = prior_summary.get("content", "").strip()
            turns_text = (
                f"[Prior session memory]\n{prior_text}\n\n"
                f"[New turns to incorporate]\n{turns_text}"
            )

        prompt = _SUMMARY_PROMPT.format(turns=turns_text, assistant_name=self.assistant_name)
        try:
            raw = self.llm.generate(
                prompt,
                max_new_tokens=self.max_summary_tokens,
                temperature=0.2,
            )
        except Exception:
            try:
                raw = self.llm.chat(
                    messages=[{"role": "user", "content": prompt}],
                    max_new_tokens=self.max_summary_tokens,
                    temperature=0.2,
                )
            except Exception:
                raw = ""

        result = (raw or "").strip()
        if not result:
            # Fallback: plain concatenation so we don't lose turns entirely
            lines = []
            for t in turns[-4:]:
                role = "User" if t.get("role") == "user" else self.assistant_name
                snippet = (t.get("content") or "")[:200].replace("\n", " ")
                lines.append(f"{role}: {snippet}")
            result = (
                "Active Thread: (compression failed — partial record)\n"
                + "\n".join(lines)
                + "\nDecisions Made: none\nOpen Questions: none\nSources Consulted: none"
            )

        return result

    @staticmethod
    def _format_turns_for_summary(turns: list[dict[str, Any]]) -> str:
        lines = []
        for t in turns:
            role = "User" if t.get("role") == "user" else self.assistant_name
            content = (t.get("content") or "").strip()
            if len(content) > 600:
                content = content[:600] + "…"
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

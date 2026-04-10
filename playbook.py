from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any

DEFAULT_RULEBOOK_THRESHOLD = 20
DEFAULT_MAX_RULES = 30

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "be", "to", "of", "and", "in",
    "it", "you", "i", "do", "not", "for", "on", "with", "that", "this",
    "at", "by", "or", "but", "as", "if", "so", "no", "up", "out",
}


class Playbook:
    """
    Structured behavioral rulebook with JSON persistence.
    Thread-safe: all mutations are lock-protected.

    Rules carry a source tag ("user" | "teacher"), a score, and usage
    metadata so the teacher can consolidate and prune over time.
    """

    def __init__(
        self,
        path: str,
        *,
        threshold: int = DEFAULT_RULEBOOK_THRESHOLD,
        max_rules: int = DEFAULT_MAX_RULES,
    ):
        self.path = path
        self.threshold = threshold
        self.max_rules = max_rules
        self._lock = threading.RLock()
        self._rules: list[dict[str, Any]] = []
        self._next_id: int = 1
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._load()

    # ---- Persistence --------------------------------------------------------

    def _load(self) -> None:
        if not os.path.exists(self.path):
            self._rules = []
            self._next_id = 1
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._rules = [
                r for r in data.get("rules", [])
                if isinstance(r, dict) and r.get("text")
            ]
            self._next_id = int(data.get("next_id", len(self._rules) + 1))
        except Exception as exc:
            logging.warning("Playbook: load failed (%s), starting empty", exc)
            self._rules = []
            self._next_id = 1

    def _save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(
                    {"rules": self._rules, "next_id": self._next_id},
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception as exc:
            logging.warning("Playbook: save failed: %s", exc)

    # ---- Queries ------------------------------------------------------------

    def get_rules(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._rules)

    def rule_count(self) -> int:
        with self._lock:
            return len(self._rules)

    def at_threshold(self) -> bool:
        with self._lock:
            return len(self._rules) >= self.threshold

    # ---- Mutations ----------------------------------------------------------

    def add_rule(self, text: str, *, source: str = "teacher") -> dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {}
        with self._lock:
            text_lower = text.lower()
            for rule in self._rules:
                if rule.get("text", "").strip().lower() == text_lower:
                    return rule  # exact duplicate — skip
            now = time.time()
            rule: dict[str, Any] = {
                "id": self._next_id,
                "text": text,
                "source": source,
                "score": 1.0,
                "created_at": now,
                "updated_at": now,
                "times_applied": 0,
            }
            self._rules.append(rule)
            self._next_id += 1
            if len(self._rules) > self.max_rules:
                self._prune_to_cap()
            self._save()
            logging.info("Playbook: rule added [%s]: %s", source, text[:80])
            return rule

    def update_by_index(self, one_based_index: int, text: str) -> bool:
        """Update rule at teacher-visible 1-based position."""
        text = (text or "").strip()
        if not text:
            return False
        with self._lock:
            idx = one_based_index - 1
            if 0 <= idx < len(self._rules):
                self._rules[idx]["text"] = text
                self._rules[idx]["updated_at"] = time.time()
                self._save()
                logging.info("Playbook: rule %d consolidated: %s", one_based_index, text[:80])
                return True
            return False

    def update_by_id(self, rule_id: int, text: str) -> bool:
        """Update rule by persistent id (for UI edits)."""
        text = (text or "").strip()
        if not text:
            return False
        with self._lock:
            for rule in self._rules:
                if rule.get("id") == rule_id:
                    rule["text"] = text
                    rule["updated_at"] = time.time()
                    self._save()
                    return True
            return False

    def delete_by_id(self, rule_id: int) -> bool:
        with self._lock:
            before = len(self._rules)
            self._rules = [r for r in self._rules if r.get("id") != rule_id]
            if len(self._rules) < before:
                self._save()
                return True
            return False

    def _prune_to_cap(self) -> None:
        """Drop lowest-scored teacher rules to stay under max_rules.
        User-authored rules are never pruned automatically."""
        user_rules = [r for r in self._rules if r.get("source") == "user"]
        teacher_rules = [r for r in self._rules if r.get("source") != "user"]
        teacher_rules.sort(key=lambda r: float(r.get("score", 1.0)), reverse=True)
        keep = max(0, self.max_rules - len(user_rules))
        self._rules = user_rules + teacher_rules[:keep]

    # ---- Formatting ---------------------------------------------------------

    def format_for_teacher(self) -> str:
        """Numbered list for the teacher's conflict-detection prompt."""
        with self._lock:
            if not self._rules:
                return "(no rules yet)"
            lines: list[str] = []
            for i, rule in enumerate(self._rules, 1):
                tag = "[user]" if rule.get("source") == "user" else "[teacher]"
                lines.append(f"{i}. {tag} {rule['text']}")
            return "\n".join(lines)

    def format_for_prompt(self, query: str = "", top_k: int = 5) -> str:
        """Select most relevant rules for injection into the main model prompt."""
        with self._lock:
            if not self._rules:
                return ""
            query_tokens = (
                set(re.findall(r"[a-z]+", query.lower())) - _STOPWORDS
                if query else set()
            )
            scored: list[tuple[float, str]] = []
            for rule in self._rules:
                text = rule.get("text", "")
                if not text:
                    continue
                rule_tokens = set(re.findall(r"[a-z]+", text.lower()))
                overlap = len(query_tokens & rule_tokens) if query_tokens else 0
                base = float(rule.get("score", 1.0))
                if rule.get("source") == "user":
                    base += 2.0  # user rules always surface
                scored.append((base + overlap * 0.5, text))
            scored.sort(reverse=True)
            selected = [text for _, text in scored[:top_k]]
            return "\n".join(f"- {t}" for t in selected)

    # ---- Legacy compatibility -----------------------------------------------

    def read(self) -> str:
        return self.format_for_prompt()

    def update(self, improvements_markdown: str) -> None:
        """Legacy: parse bullet points from old markdown and add as teacher rules."""
        for line in (improvements_markdown or "").splitlines():
            line = re.sub(r"^[-*\u2022]\s*", "", line).strip()
            if len(line) > 10:
                self.add_rule(line, source="teacher")

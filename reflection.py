from __future__ import annotations

import json
import logging
import os
import re
import threading
from typing import Any, Callable, Optional

from config_types import AgentConfig
from model_runner import LLMRunner


TEACHER_SYSTEM = (
    "You are a behavioral critique model. Your only job is to maintain a rulebook "
    "for an AI assistant by evaluating exchanges and managing rules. "
    "You output JSON only. You never interact with users directly."
)

TEACHER_PROMPT = """\
[Rulebook]
{rulebook}

[Exchange]
User: {user}
Assistant: {answer}

[Task]
Evaluate this exchange against the rulebook. Work through three stages:

1. Did the assistant violate any existing rule? Note which one.
2. Is there a new behavioral pattern worth capturing?
   - Can it fold into an existing rule by rewriting that rule more broadly?
     (preferred — keeps the rulebook lean and consolidated)
   - Or is it genuinely new with no overlap?
3. Output exactly ONE JSON object and nothing else:

No action needed:
{{"action": "none"}}

Add a new rule (one concise sentence, no conflicts with existing rules):
{{"action": "add", "rule": "rule text here"}}

Consolidate into an existing rule (use the rule number from the Rulebook above):
{{"action": "update", "index": N, "rule": "rewritten rule absorbing both patterns"}}

Rulebook is getting too large to manage effectively:
{{"action": "flag", "reason": "brief reason here"}}

Output only the JSON. No preamble, no explanation.\
"""


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    """Robustly extract a JSON object from model output."""
    text = (text or "").strip()
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    # Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Extract first { ... last } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidate = text[start : end + 1]
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)   # trailing commas
        candidate = re.sub(r"'([^']*)'", r'"\1"', candidate)   # single quotes
        try:
            return json.loads(candidate)
        except Exception:
            pass
    # Field-by-field regex fallback (salvage partial output)
    result: dict[str, Any] = {}
    m = re.search(r'"action"\s*:\s*"(\w+)"', text)
    if m:
        result["action"] = m.group(1)
    m = re.search(r'"rule"\s*:\s*"([^"]+)"', text)
    if m:
        result["rule"] = m.group(1)
    m = re.search(r'"index"\s*:\s*(\d+)', text)
    if m:
        result["index"] = int(m.group(1))
    m = re.search(r'"reason"\s*:\s*"([^"]+)"', text)
    if m:
        result["reason"] = m.group(1)
    return result if result else None


class Reflector:
    """
    Async behavioral reflector backed by a dedicated teacher LLM.

    The teacher runs in a background daemon thread after every chat turn,
    never touching the main model's context window.  If no teacher model
    is configured, the main LLM is used as a fallback.
    """

    def __init__(self, main_llm: LLMRunner, cfg: AgentConfig):
        self.main_llm = main_llm
        self.cfg = cfg
        self._lock = threading.Lock()
        self._teacher_llm: Optional[LLMRunner] = None
        self._teacher_loaded = False
        reflection_cfg = dict(getattr(cfg, "reflection", {}) or {})
        self._enabled = bool(reflection_cfg.get("enabled", False))
        self._teacher_model_path = str(reflection_cfg.get("model_path", "") or "").strip()
        self._reflection_cfg = reflection_cfg

    # ---- Teacher LLM --------------------------------------------------------

    def _get_teacher(self) -> LLMRunner:
        """Lazy-load the teacher on first reflection call."""
        with self._lock:
            if self._teacher_loaded:
                return self._teacher_llm or self.main_llm
            self._teacher_loaded = True
            if self._teacher_model_path and os.path.isfile(self._teacher_model_path):
                try:
                    teacher_cfg = self._build_teacher_cfg()
                    self._teacher_llm = LLMRunner(teacher_cfg)
                    logging.info(
                        "Reflector: teacher model loaded: %s",
                        os.path.basename(self._teacher_model_path),
                    )
                except Exception as exc:
                    logging.warning(
                        "Reflector: teacher load failed (%s) — falling back to main LLM", exc
                    )
            else:
                if self._teacher_model_path:
                    logging.warning(
                        "Reflector: teacher model path not found (%s) — falling back to main LLM",
                        self._teacher_model_path,
                    )
            return self._teacher_llm or self.main_llm

    def _build_teacher_cfg(self) -> AgentConfig:
        r = self._reflection_cfg
        return AgentConfig(
            backend=str(r.get("backend", "llama_cpp")),
            model_path=self._teacher_model_path,
            n_ctx=int(r.get("n_ctx", 2048)),
            n_gpu_layers=int(r.get("n_gpu_layers", 0)),
            n_threads=int(r.get("n_threads", 4)),
            n_batch=256,
            n_ubatch=256,
            max_new_tokens=int(r.get("max_new_tokens", 256)),
            llama_verbose=False,
            data_dir=self.cfg.data_dir,
            persist_dir=self.cfg.persist_dir,
            embedding_model=self.cfg.embedding_model,
        )

    def shutdown(self) -> None:
        with self._lock:
            if self._teacher_llm is not None and self._teacher_llm is not self.main_llm:
                try:
                    self._teacher_llm.shutdown()
                except Exception:
                    pass
                self._teacher_llm = None
            self._teacher_loaded = False

    # ---- Async reflection ---------------------------------------------------

    def reflect_async(
        self,
        user: str,
        answer: str,
        rulebook_text: str,
        *,
        on_add: Callable[[str], None] | None = None,
        on_update: Callable[[int, str], None] | None = None,
        on_flag: Callable[[str], None] | None = None,
    ) -> None:
        """Fire reflection in a background daemon thread. Non-blocking."""
        if not self._enabled:
            return
        threading.Thread(
            target=self._run,
            args=(user, answer, rulebook_text),
            kwargs={"on_add": on_add, "on_update": on_update, "on_flag": on_flag},
            name="celeste-reflection-worker",
            daemon=True,
        ).start()

    def _run(
        self,
        user: str,
        answer: str,
        rulebook_text: str,
        *,
        on_add: Callable[[str], None] | None = None,
        on_update: Callable[[int, str], None] | None = None,
        on_flag: Callable[[str], None] | None = None,
    ) -> None:
        try:
            llm = self._get_teacher()
            prompt = TEACHER_PROMPT.format(
                rulebook=rulebook_text or "(no rules yet)",
                user=(user or "").strip()[:600],
                answer=(answer or "").strip()[:1000],
            )
            raw = llm.generate(prompt, max_new_tokens=256).strip()
            result = _extract_json(raw)
            if not result:
                logging.warning("Reflector: unparseable output: %.200s", raw)
                return
            action = str(result.get("action", "none")).strip().lower()
            logging.info("Reflector: action=%s", action)
            if action == "add":
                rule_text = str(result.get("rule", "") or "").strip()
                if rule_text and on_add:
                    on_add(rule_text)
            elif action == "update":
                index = int(result.get("index", 0) or 0)
                rule_text = str(result.get("rule", "") or "").strip()
                if index > 0 and rule_text and on_update:
                    on_update(index, rule_text)
            elif action == "flag":
                reason = str(result.get("reason", "") or "").strip()
                if on_flag:
                    on_flag(reason or "Rulebook review recommended.")
            # "none" → do nothing
        except Exception:
            logging.exception("Reflector: background reflection failed")

    # ---- Legacy compatibility -----------------------------------------------

    def reflect(
        self, user: str, answer: str, prior_state: dict
    ) -> tuple[Optional[str], Optional[str], dict]:
        """Legacy synchronous stub — reflection is now fully async."""
        return None, None, prior_state

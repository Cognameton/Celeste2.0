# agent.py — Celeste agent (chat-style prompt, strict stops, perspective guard)
import os
import json
import re
from typing import Callable, Dict, Any, Tuple, Optional, List

from rich.console import Console

from config_types import AgentConfig
from model_runner import LLMRunner
from memory import MemoryPipeline
from file_rag import FileRAG
from reflection import Reflector
from playbook import Playbook
from tts import TTSManager

console = Console()


class Agent:
    def __init__(self, cfg: AgentConfig, status_cb: Callable[[str], None] | None = None):
        self.cfg = cfg
        self._status_cb = status_cb
        self._emit_status("Preparing Celeste workspace...")
        os.makedirs(self.cfg.data_dir, exist_ok=True)

        # Defaults for self-state
        self._default_state = {"identity": "Celeste", "focus": "helpful, precise, offline"}
        self._live_guidance: str = ""
        self._retry_phrases = (
            "i'm sorry, but i am an ai language model",
            "i'm sorry, but as an ai language model",
            "i'm sorry, but as an ai model",
            "i do not have the ability to provide explanations",
            "i cannot provide explanations",
            "as an ai language model, i cannot",
            "i am not able to",
            "i am unable to",
            "i'm unable to",
            "i cannot help",
            "i can't help",
            "i can't assist",
            "my capabilities are limited",
            "there are many resources available online",
            "i'm sorry, but i am an ai model",
            "i'm sorry, but i am just an ai",
            "i'm sorry, but i can't assist",
            "i'm sorry, but i can't help",
            "i'm just an ai language model",
            "as an ai, i cannot",
            "i'm an ai language model and cannot",
            "i'm a language model and cannot",
            "i do not have the capability",
            "i don't have the ability",
            "i lack the ability",
            "i do not have access to external",
            "there are many resources you can consult",
        )

        # Core subsystems
        self._emit_status(f"Loading language model: {os.path.basename(self.cfg.model_path)}")
        self.llm = LLMRunner(cfg, status_cb=status_cb)
        self._emit_status("Starting memory system...")
        self.mem = MemoryPipeline(cfg)
        self._emit_status("Starting file search...")
        self.file_rag = FileRAG(
            cfg,
            shared_embedder=getattr(self.mem, "embedder", None),
            shared_device=getattr(self.mem, "device", "cpu"),
        )
        self._emit_status("Starting reflection tools...")
        self.reflector = Reflector(self.llm, cfg)
        self._emit_status("Starting playbook and speech...")
        self.playbook = Playbook(os.path.join(cfg.data_dir, "playbook.md"))
        self.tts = TTSManager(cfg)

        # Self-state (identity/focus/etc.)
        self.self_state_path = os.path.join(self.cfg.data_dir, "self_state.json")
        if not os.path.exists(self.self_state_path):
            with open(self.self_state_path, "w", encoding="utf-8") as f:
                json.dump(self._default_state, f, indent=2)
        self._emit_status("Celeste startup complete.")

    def _emit_status(self, message: str) -> None:
        if self._status_cb:
            self._status_cb(message)

    def close(self) -> None:
        try:
            self.tts.shutdown()
        except Exception:
            pass
        try:
            self.llm.shutdown()
        except Exception:
            pass

    def set_file_rag_dirs(self, directories: List[str]) -> Dict[str, Any]:
        self.cfg.file_rag_dirs = directories
        return self.file_rag.rebuild(directories)

    def reindex_file_rag(self) -> Dict[str, Any]:
        return self.file_rag.rebuild(self.cfg.file_rag_dirs)

    def _reflection_enabled(self) -> bool:
        reflection_cfg = getattr(self.cfg, "reflection", {}) or {}
        if isinstance(reflection_cfg, dict):
            return bool(reflection_cfg.get("enabled", False))
        return False

    # ---------- Self-state ----------
    def _load_self_state(self) -> Dict[str, Any]:
        try:
            with open(self.self_state_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            raw = {}
        return self._sanitize_self_state(raw)

    def _save_self_state(self, state: Dict[str, Any]) -> None:
        clean = self._sanitize_self_state(state)
        with open(self.self_state_path, "w", encoding="utf-8") as f:
            json.dump(clean, f, indent=2)

    def _sanitize_self_state(self, state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Ensure identity/focus stay populated with simple strings."""
        clean: Dict[str, Any] = dict(self._default_state)
        if isinstance(state, dict):
            for key, value in state.items():
                if key in ("identity", "focus"):
                    if isinstance(value, str) and value.strip():
                        clean[key] = value.strip()
                else:
                    clean[key] = value
        return clean

    def _maybe_update_identity(self, user_msg: str) -> None:
        """
        Detect patterns like 'your name is ...' and persist the provided identity.
        Keeps existing identity when the extracted value is empty.
        """
        lowered = user_msg.lower()
        trigger = "your name is"
        if trigger not in lowered:
            return

        # Naive span capture after the trigger; accept up to sentence end.
        try:
            start = lowered.index(trigger) + len(trigger)
            proposed = user_msg[start:].strip(" .!?\n\t\"'")
            if proposed:
                state = self._load_self_state()
                state["identity"] = proposed
                self._save_self_state(state)
                self.mem.add(
                    f"Assistant identity updated to {proposed}",
                    kind="meta",
                    metadata={"via": "identity-update"},
                )
        except ValueError:
            pass

    # ---------- Prompt pieces ----------
    def build_system_prompt(self) -> str:
        self_state = self._load_self_state()
        playbook_text = self.playbook.read().strip()

        parts: List[str] = [
            (self.cfg.system_preamble or "").strip(),
            "[Behavior]\n"
            "- Reply concisely.\n"
            "- For code/IT tasks, use ordered steps.\n"
            "- Do not repeat headings. Output only the answer unless steps are requested.\n"
            "- Use second-person (“you/your”) for the user and first-person (“I/my”) for yourself.\n"
            "- Provide direct help even when the task is creative, exploratory, or strategic.\n"
            "- Offer outlines, examples, or drafts for writing requests instead of refusing.\n"
            "- Propose actionable plans or alternatives when direct execution is impossible.\n"
            "- Avoid apologizing for being an AI unless safety limits require refusal.\n"
            "- Only decline when the task is unsafe/illegal or truly impossible due to missing hardware; otherwise describe a workaround or best-effort assistance.\n"
            "- Do not quote or echo the raw memory notes verbatim; speak naturally.",
            "[Self-State]\n" + json.dumps(self_state, ensure_ascii=False, indent=2),
        ]
        if playbook_text:
            parts.append("[Playbook]\n" + playbook_text)
        if getattr(self, "_live_guidance", None):
            parts.append("[Live Playbook Reminder]\n" + self._live_guidance)
            self._live_guidance = ""

        return "\n".join([p for p in parts if p])

    def _count_tokens(self, text: str) -> int:
        try:
            if getattr(self.llm, "backend", "") == "llama_cpp":
                return len(self.llm.llm.tokenize(text.encode("utf-8"), add_bos=True))
            else:
                return len(self.llm.tok(text, return_tensors="pt").input_ids[0])
        except Exception:
            return max(1, len(text) // 4)

    def _truncate_for_budget(self, texts: List[str], token_budget: int, per_snippet_chars: int = 200) -> List[str]:
        """Keep short, high-signal snippets until token budget is hit."""
        acc = 0
        kept: List[str] = []
        for t in texts:
            t = (t or "").strip().replace("\n", " ")
            if len(t) > per_snippet_chars:
                t = t[:per_snippet_chars] + "…"
            est = self._count_tokens(t) + 2  # bullet/formatting cost
            if acc + est > token_budget:
                break
            kept.append(t)
            acc += est
        return kept

    # ---------- Perspective guard ----------
    def _enforce_perspective(self, user_msg: str, answer: str) -> str:
        """
        Fix common pronoun inversions so the reply sounds conversational:
        - If the user asks about themselves (“my …”), prefer “Your …”
        - If the user asks about the assistant (“your …” / “your name”), prefer “My …”
        - Special-case: “what is your name” => “My name is <identity>.”
        """
        u = user_msg.strip().lower()
        a = answer.strip()

        # Identity special-case
        identity_triggers = (
            "your name" in u
            or "what is your name" in u
            or u.startswith("who are you")
        )
        if identity_triggers:
            name = self._load_self_state().get("identity", self._default_state["identity"])
            return f"My name is {name}."

        # If the model volunteered its name without being asked, steer back to helping
        if not identity_triggers and a.lower().startswith("my name is"):
            name = self._load_self_state().get("identity", self._default_state["identity"])
            return f"I'm {name}, ready to help you with whatever you need."

        # If user refers to themselves (my/I), answer in second person
        if " my " in f" {u} " or u.startswith("my ") or " i " in f" {u} ":
            if a[:3].lower() == "my ":
                return "Your " + a[3:]
            if a[:2].lower() == "i ":
                return "You " + a[2:]
            # If the model already replied with a bare value like "royal blue.",
            # that's fine (concise). Otherwise, prefer “Your …”
            return a

        # If user asks about the assistant (your/you), answer in first person
        if " your " in f" {u} " or u.startswith("your ") or " about you" in u:
            if a[:5].lower() == "your ":
                return "My " + a[5:]
            return a

        return a

    def _needs_retry(self, answer: str) -> bool:
        low = (answer or "").strip().lower()
        if not low:
            return True
        for phrase in self._retry_phrases:
            if phrase in low:
                return True
        if low.startswith("i'm sorry") and (
            "cannot" in low or "can't" in low or "not able" in low or "unable" in low
        ):
            return True
        if "my capabilities are limited" in low:
            return True
        if "i am just an ai" in low or ("as an ai" in low and ("cannot" in low or "can't" in low)):
            return True
        if "i do not have the capability" in low or "i don't have the ability" in low:
            return True
        if "i lack the ability" in low:
            return True
        if "i cannot comply" in low or "i can't comply" in low:
            return True
        if "you can consult" in low and "resources" in low:
            return True
        return False

    def _build_retry_instruction(self, prior_answer: str, user_msg: str) -> str:
        cleaned = (prior_answer or "").strip().replace("\n", " ")
        if len(cleaned) > 300:
            cleaned = cleaned[:300] + "…"
        hints = [
            "Your previous reply was unsatisfactory. Provide a clear, detailed answer with actionable guidance.",
            "Do not apologize for being an AI or refuse the task unless it is unsafe.",
            "If physical hardware is required, state the limitation briefly and outline workarounds or simulations the user can run instead.",
        ]
        user_low = (user_msg or "").lower()
        if any(
            key in user_low
            for key in ("write", "essay", "story", "article", "draft", "paragraph", "blog")
        ):
            hints.append(
                "The user asked for writing assistance. Outline the piece, highlight key similarities, and draft helpful paragraphs."
            )
        hints.append(f"Prior reply (for critique): {cleaned}")
        return " ".join(hints)

    def _select_grounding_sources(
        self,
        file_context: dict[str, Any],
        *,
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add(record: dict[str, Any], source_kind: str) -> None:
            key = str(record.get("chunk_id") or f"{record.get('path', '')}:{record.get('chunk_index', -1)}")
            if not key or key in seen:
                return
            text = str(record.get("text", "") or "").strip()
            if not text:
                return
            seen.add(key)
            selected.append(
                {
                    "label": len(selected) + 1,
                    "source_kind": source_kind,
                    "path": record.get("path"),
                    "rel_path": record.get("rel_path") or os.path.basename(str(record.get("path", ""))),
                    "chunk_id": record.get("chunk_id"),
                    "chunk_index": int(record.get("chunk_index", 0)),
                    "doc_id": record.get("doc_id"),
                    "score": float(record.get("score", 0.0)),
                    "text": text,
                }
            )

        for record in file_context.get("snippet_records", []) or []:
            add(record, "file")
            if len(selected) >= limit:
                return selected
        for record in file_context.get("library_snippets", []) or []:
            add(record, "library")
            if len(selected) >= limit:
                return selected
        return selected

    def _format_grounding_notes(
        self,
        sources: list[dict[str, Any]],
        *,
        token_budget: int,
    ) -> tuple[str, list[dict[str, Any]]]:
        if not sources:
            return "", []
        consumed = 0
        kept: list[dict[str, Any]] = []
        lines: list[str] = []
        for source in sources:
            excerpt = source["text"].replace("\n", " ").strip()
            if len(excerpt) > 360:
                excerpt = excerpt[:360] + "…"
            line = f"[{source['label']}] {source['rel_path']} | chunk {source['chunk_index']} | {excerpt}"
            cost = self._count_tokens(line) + 2
            if kept and consumed + cost > token_budget:
                break
            consumed += cost
            kept.append(source)
            lines.append(line)
        if not kept:
            return "", []
        return "Grounding Sources:\n" + "\n".join(lines), kept

    def _citation_labels(self, answer: str, max_label: int) -> list[int]:
        labels: list[int] = []
        seen: set[int] = set()
        for match in re.finditer(r"\[(\d+)\]", answer or ""):
            label = int(match.group(1))
            if 1 <= label <= max_label and label not in seen:
                seen.add(label)
                labels.append(label)
        return labels

    def _sanitize_answer_citations(self, answer: str, max_label: int) -> str:
        if not answer:
            return ""
        return re.sub(
            r"\[(\d+)\]",
            lambda match: match.group(0) if 1 <= int(match.group(1)) <= max_label else "",
            answer,
        )

    def _append_source_list(self, answer: str, sources: list[dict[str, Any]]) -> str:
        cited = self._citation_labels(answer, max_label=len(sources))
        if not cited:
            return answer.strip()
        source_map = {source["label"]: source for source in sources}
        lines = [
            f"[{label}] {source_map[label]['rel_path']} (chunk {source_map[label]['chunk_index']})"
            for label in cited
            if label in source_map
        ]
        if not lines:
            return answer.strip()
        return answer.strip() + "\n\nSources:\n" + "\n".join(lines)

    def _handle_document_query(self, user_msg: str) -> Optional[str]:
        u_low = user_msg.strip().lower()
        total_docs = self.file_rag.document_count()
        if total_docs <= 0:
            return "I do not see any indexed documents right now."

        count_phrases = (
            "how many documents can you see",
            "how many files can you see",
            "how many documents do you see",
            "how many files do you see",
        )
        if any(phrase in u_low for phrase in count_phrases):
            return f"I can see {total_docs} indexed documents."

        only_phrases = (
            "are those the only documents",
            "are those the only files",
            "are these the only documents",
            "are these the only files",
        )
        if any(phrase in u_low for phrase in only_phrases):
            return (
                f"No. I can see {total_docs} indexed documents in total. "
                "Earlier I was only showing the files in the current retrieval context."
            )

        list_phrases = (
            "list the documents you see",
            "list documents you see",
            "list the files you see",
            "list files you see",
            "what documents can you see",
            "what files can you see",
        )
        if any(phrase in u_low for phrase in list_phrases):
            shown = self.file_rag.list_documents(limit=25)
            lines = "\n".join(f"- {name}" for name in shown)
            if total_docs > len(shown):
                return (
                    f"I can see {total_docs} indexed documents. Here are the first {len(shown)}:\n"
                    f"{lines}\n\n"
                    "I can list more if you want."
                )
            return f"I can see {total_docs} indexed documents:\n{lines}"

        topical_prefixes = ("what documents", "which documents", "what files", "which files")
        topical_cues = ("discuss", "mention", "cover", "about", "related to", "on ")
        if u_low.startswith(topical_prefixes) and any(cue in u_low for cue in topical_cues):
            matches = self.file_rag.search_document_titles(user_msg, top_k=8)
            if not matches:
                return "I do not see any indexed documents that clearly match that topic."
            lines = "\n".join(f"- {hit['rel_path']}" for hit in matches)
            return f"The best matching indexed documents I can see are:\n{lines}"

        return None

    # ---------- Main interaction ----------
    def respond(self, user: str) -> Tuple[str, Optional[str], Optional[str]]:
        u_strip = user.strip()
        u_low = u_strip.lower()

        # Command: remember this: <note>
        if u_low.startswith("remember this:") or u_low.startswith("remember:"):
            note = u_strip.split(":", 1)[1].strip() if ":" in u_strip else u_strip
            if note:
                self.mem.add(note, kind="note", metadata={"via": "remember-cmd"})
                return "Noted.", None, None
            else:
                return "What should I remember?", None, None

        document_answer = self._handle_document_query(u_strip)
        if document_answer is not None:
            return document_answer, None, None

        # Retrieve filtered memories (no assistant/meta blobs)
        raw_mems = self.mem.search(user, top_k=max(1, min(self.cfg.top_k, 4)), kinds=["note", "user"])
        mem_texts = [m["text"] for m in raw_mems]
        file_context = self.file_rag.get_context(
            user,
            top_k=max(1, min(getattr(self.cfg, "file_rag_top_k", 4), 4)),
        )
        file_hits = file_context.get("matches", [])
        file_list = [
            f"{hit.get('rel_path') or os.path.basename(hit.get('path', ''))}"
            for hit in file_hits
        ]
        opened_file = file_context.get("opened_file")
        library_hits = file_context.get("library_snippets", [])
        file_snippets = []
        if opened_file:
            file_snippets = [
                f"File={opened_file.get('rel_path') or os.path.basename(opened_file.get('path', ''))} | Excerpt={snippet.strip()}"
                for snippet in file_context.get("snippets", [])
                if snippet.strip()
            ]
        library_snippets = [
            f"File={hit.get('rel_path') or os.path.basename(hit.get('path', ''))} | Excerpt={hit.get('text', '').strip()}"
            for hit in library_hits
            if (hit.get("text") or "").strip()
        ]

        filename_query = any(
            phrase in u_low for phrase in ("can you see", "do you see", "is there a file", "do you have a file", "how about")
        ) and not any(token in u_low for token in ("documents", "document", "files", "list", "how many"))
        if filename_query and file_hits and not opened_file:
            top_hit = file_hits[0]
            rel_path = top_hit.get("rel_path") or os.path.basename(top_hit.get("path", ""))
            return f"Yes. I can see `{rel_path}`.", None, None

        # Update identity when the user explicitly assigns one
        self._maybe_update_identity(u_strip)

        # Prepare system + trimmed memory notes with a strict budget (~25% of context)
        system = self.build_system_prompt()
        mem_token_budget = max(96, int(self.cfg.n_ctx * 0.15))
        file_token_budget = max(128, int(self.cfg.n_ctx * 0.20))
        kept_snippets = self._truncate_for_budget(mem_texts, token_budget=mem_token_budget)
        kept_file_snippets = self._truncate_for_budget(file_snippets, token_budget=file_token_budget, per_snippet_chars=320)
        grounding_sources = self._select_grounding_sources(file_context, limit=6)
        grounding_note, grounding_sources = self._format_grounding_notes(
            grounding_sources,
            token_budget=max(128, int(self.cfg.n_ctx * 0.18)),
        )

        note_sections: List[str] = []
        if kept_snippets:
            note_sections.append("Notes:\n" + "\n".join(f"- {s}" for s in kept_snippets))
        if file_list:
            note_sections.append("Available Files:\n" + "\n".join(f"- {s}" for s in file_list))
        if kept_file_snippets:
            note_sections.append("File Context:\n" + "\n".join(f"- {s}" for s in kept_file_snippets))
        kept_library_snippets = self._truncate_for_budget(library_snippets, token_budget=file_token_budget, per_snippet_chars=320)
        if kept_library_snippets:
            note_sections.append("Library Context:\n" + "\n".join(f"- {s}" for s in kept_library_snippets))
        if grounding_note:
            note_sections.append(grounding_note)
        notes_block = "\n\n".join(note_sections)
        library_requested = any(
            token in u_low for token in ("library", "indexed", "documents", "document", "files", "file", "docs")
        )
        require_citations = bool(grounding_sources) and (
            library_requested
            or bool(opened_file)
            or bool(library_hits)
        )
        base_instruction_lines = [
            "Answer the user's message succinctly. Output ONLY the answer text. No quotes, no headings, no code fences.",
            "Reply in the same language as the user's message. If the user wrote in English, reply in English.",
        ]
        if library_requested:
            if kept_file_snippets or kept_library_snippets:
                base_instruction_lines.append(
                    "Use the retrieved File Context and Library Context when answering. Prefer that context over general background knowledge."
                )
            else:
                base_instruction_lines.append(
                    "The user asked about the indexed library. If the retrieved library context is insufficient, say that clearly instead of inventing details."
                )
        if require_citations:
            base_instruction_lines.extend(
                [
                    "Blend the retrieved grounding sources with your general knowledge when useful.",
                    "When you use a grounding source, cite it inline with its exact label like [1] or [2].",
                    "Do not invent citations or cite labels that were not provided.",
                    "General background statements that do not come from the grounding sources do not need citations.",
                ]
            )
        citation_tail_budget = 0
        if require_citations:
            citation_tail_budget = max(
                96,
                24 * len(grounding_sources) + 80,
            )
        answer_floor_budget = 256 if require_citations else 128

        # Chat-style prompt
        def make_prompt(notes: str, extra_instruction: str = "", prior_attempt: str = "") -> str:
            instruction_lines = list(base_instruction_lines)
            extra_instruction = (extra_instruction or "").strip()
            if extra_instruction:
                instruction_lines.append(extra_instruction)
            instruction = "\n".join(instruction_lines)

            prior_section = ""
            cleaned_prior = (prior_attempt or "").strip().replace("\n", " ")
            if cleaned_prior:
                if len(cleaned_prior) > 300:
                    cleaned_prior = cleaned_prior[:300] + "…"
                prior_section = f"Prior reply (for awareness): {cleaned_prior}\n\n"

            return f"""System:
{system}

{notes if notes else ""}

{prior_section}Instruction:
{instruction}

User: {user}
Assistant:"""

        def make_messages(notes: str, extra_instruction: str = "", prior_attempt: str = "") -> List[Dict[str, str]]:
            instruction_lines = list(base_instruction_lines)
            extra_instruction = (extra_instruction or "").strip()
            if extra_instruction:
                instruction_lines.append(extra_instruction)
            instruction = "\n".join(instruction_lines)

            system_sections: List[str] = [system]
            if notes:
                system_sections.append(notes)

            cleaned_prior = (prior_attempt or "").strip().replace("\n", " ")
            if cleaned_prior:
                if len(cleaned_prior) > 300:
                    cleaned_prior = cleaned_prior[:300] + "…"
                system_sections.append(f"Prior reply (for awareness): {cleaned_prior}")

            system_sections.append("Instruction:\n" + instruction)

            return [
                {"role": "system", "content": "\n\n".join([s for s in system_sections if s])},
                {"role": "user", "content": user},
            ]

        def prep_prompt(notes: str, extra_instruction: str = "", prior_attempt: str = "") -> Tuple[str, int]:
            prompt_text = make_prompt(notes, extra_instruction=extra_instruction, prior_attempt=prior_attempt)
            margin = 64
            prompt_tokens = self._count_tokens(prompt_text)
            if prompt_tokens > self.cfg.n_ctx - margin:
                prompt_text = make_prompt("", extra_instruction=extra_instruction, prior_attempt=prior_attempt)
                prompt_tokens = self._count_tokens(prompt_text)
            usable_budget = max(64, self.cfg.n_ctx - prompt_tokens - margin)
            generation_budget = min(
                usable_budget,
                max(answer_floor_budget, usable_budget - citation_tail_budget),
            )
            max_new = min(self.cfg.max_new_tokens, generation_budget)
            return prompt_text, max_new

        # Build prompt and ensure it fits the window
        prompt, max_new = prep_prompt(notes_block)
        use_chat_api = getattr(self.llm, "backend", "") == "llama_server" and hasattr(self.llm, "chat")

        # Strict stops to prevent prompt echo and code sprawl
        stop = [
            "\nUser:", "\nSystem:",    # end of assistant turn
            "\n#", "\n[",             # stop on headings/comments
            "```", '"""',             # stop on code fences / triple quotes
            "<|endoftext|>", "</s>",
        ]

        messages = make_messages(notes_block)
        answer = (
            self.llm.chat(
                messages,
                max_new_tokens=max_new,
                temperature=0.3,
                top_p=0.9,
                stop=stop,
                repeat_penalty=1.12,
                repeat_last_n=256,
            ).strip()
            if use_chat_api
            else self.llm.generate(
                prompt,
                max_new_tokens=max_new,
                temperature=0.3,
                top_p=0.9,
                stop=stop,
                repeat_penalty=1.12,   # may be ignored if unsupported; model_runner filters accordingly
                repeat_last_n=256,
            ).strip()
        )

        # Perspective correction (conversational voice)
        answer = self._enforce_perspective(user, answer)
        answer = self._sanitize_answer_citations(answer, max_label=len(grounding_sources))

        # Fallback: if answer is empty or generic refusal, retry once with stronger instruction
        if self._needs_retry(answer):
            extra_instruction = self._build_retry_instruction(answer, user)
            retry_prompt, retry_max_new = prep_prompt(notes_block, extra_instruction=extra_instruction, prior_attempt=answer)
            retry_messages = make_messages(notes_block, extra_instruction=extra_instruction, prior_attempt=answer)
            retry_answer = (
                self.llm.chat(
                    retry_messages,
                    max_new_tokens=retry_max_new,
                    temperature=0.2,
                    top_p=0.85,
                    stop=stop,
                    repeat_penalty=1.1,
                    repeat_last_n=256,
                ).strip()
                if use_chat_api
                else self.llm.generate(
                    retry_prompt,
                    max_new_tokens=retry_max_new,
                    temperature=0.2,
                    top_p=0.85,
                    stop=stop,
                    repeat_penalty=1.1,
                    repeat_last_n=256,
                ).strip()
            )
            answer = self._enforce_perspective(user, retry_answer)
            answer = self._sanitize_answer_citations(answer, max_label=len(grounding_sources))

        if require_citations and not self._citation_labels(answer, max_label=len(grounding_sources)):
            extra_instruction = (
                "Revise the answer so that claims drawn from the grounding sources cite the provided labels inline, "
                "for example [1] or [2]. If the sources are insufficient, say that clearly."
            )
            retry_prompt, retry_max_new = prep_prompt(notes_block, extra_instruction=extra_instruction, prior_attempt=answer)
            retry_messages = make_messages(notes_block, extra_instruction=extra_instruction, prior_attempt=answer)
            retry_answer = (
                self.llm.chat(
                    retry_messages,
                    max_new_tokens=retry_max_new,
                    temperature=0.2,
                    top_p=0.85,
                    stop=stop,
                    repeat_penalty=1.1,
                    repeat_last_n=256,
                ).strip()
                if use_chat_api
                else self.llm.generate(
                    retry_prompt,
                    max_new_tokens=retry_max_new,
                    temperature=0.2,
                    top_p=0.85,
                    stop=stop,
                    repeat_penalty=1.1,
                    repeat_last_n=256,
                ).strip()
            )
            answer = self._enforce_perspective(user, retry_answer)
            answer = self._sanitize_answer_citations(answer, max_label=len(grounding_sources))

        answer = self._append_source_list(answer, grounding_sources)

        # Persist logs
        self.mem.add(user, kind="user")
        self.mem.add(answer, kind="assistant")

        critique: Optional[str] = None
        improvements: Optional[str] = None
        new_state = self._load_self_state()
        if self._reflection_enabled():
            critique, improvements, new_state = self.reflector.reflect(user, answer, new_state)
            if critique:
                self.mem.add("CRITIQUE: " + critique, kind="meta")
            if improvements:
                self.playbook.update(improvements)
                self._live_guidance = improvements
            if new_state:
                self._save_self_state(new_state)

        return answer, critique, improvements

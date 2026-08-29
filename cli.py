#!/usr/bin/env python3
import os
import sys
import re
from datetime import datetime

# Ensure local imports work when launched from anywhere
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from app_config import load_config
from config_types import AgentConfig
from agent import Agent
from model_runner import discover_local_models

console = Console()

def show_banner():
    console.print(Panel.fit("Synthia — Offline Reflective Agent", title="Synthia", style="bold"))

def show_config(cfg: AgentConfig):
    tbl = Table.grid(padding=(0, 2))
    tbl.add_row("backend:", str(cfg.backend))
    tbl.add_row("model_path:", str(cfg.model_path))
    tbl.add_row("use_chroma:", str(cfg.use_chroma))
    tbl.add_row("embedding_model:", str(cfg.embedding_model))
    tbl.add_row("persist_dir:", str(cfg.persist_dir))
    tbl.add_row("data_dir:", str(cfg.data_dir))
    tbl.add_row("n_ctx/n_gpu_layers:", f"{cfg.n_ctx}/{cfg.n_gpu_layers}")
    tbl.add_row("main_gpu/tensor_split:", f"{cfg.main_gpu}/{cfg.tensor_split or '-'}")
    tbl.add_row("max_new_tokens:", str(cfg.max_new_tokens))
    console.print(Panel(tbl, title="Configuration"))

# ------------------------------ Command wiring ------------------------------ #
CMD_PREFIX = "!"
# Lives alongside the runtime data dir from config.yaml (override with
# SYNTHIA_PLAYBOOK_PATH). Derived rather than hardcoded so it follows the
# configured data_dir instead of pinning one machine's layout.
def _playbook_path(cfg) -> str:
    override = os.environ.get("SYNTHIA_PLAYBOOK_PATH")
    if override:
        return override
    base = getattr(cfg, "data_dir", "") or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "playbook.md")

def is_command(text: str) -> bool:
    return text.strip().startswith(CMD_PREFIX)

def _memory_add(memory, text: str, meta: dict | None = None, cfg: AgentConfig | None = None):
    """Try to save into the agent's memory; fallback to a notes file under persist_dir."""
    if memory is not None:
        # Try common memory APIs used in RAG agents
        if hasattr(memory, "add_message"):
            try:
                memory.add_message(role="system", content=text, metadata=meta or {})
                return True
            except Exception:
                pass
        if hasattr(memory, "add_text"):
            try:
                memory.add_text(text=text, metadata=meta or {})
                return True
            except Exception:
                pass
    # Fallback to a simple notes file
    persist_dir = getattr(cfg, "persist_dir", "/tmp")
    os.makedirs(persist_dir, exist_ok=True)
    note_path = os.path.join(persist_dir, "manual_notes.md")
    try:
        with open(note_path, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] {text}\n")
        return True
    except Exception:
        return False

def _playbook_add(text: str, cfg) -> str:
    path = _playbook_path(cfg)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n## Update (" + datetime.now().isoformat(timespec="seconds") + ")\n")
        f.write(text.strip() + "\n")
    return path

def handle_command(cmd_line: str, cfg: AgentConfig, agent: Agent) -> str:
    """Process in-chat commands. Returns a status string; does NOT send text to the model."""
    # Split the first token (verb) from the rest
    parts = cmd_line.strip().split(" ", 1)
    verb = parts[0].lower()
    payload = parts[1].strip() if len(parts) > 1 else ""

    memory = getattr(agent, "memory", None)

    if verb in ("!help", "!commands"):
        return (
            "Commands:\n"
            "  !remember TEXT          → save a note to memory\n"
            "  !note TEXT              → alias for !remember\n"
            "  !forget TEXT            → record a forget request in memory\n"
            "  !playbook TEXT          → append text to the playbook file\n"
            "  !identity user=Name assistant=Name  → update names in memory\n"
            "  !help                   → show this help\n"
        )

    if verb == "!remember" or verb == "!note":
        ok = _memory_add(memory, payload, meta={"type": "note", "source": "user_command"}, cfg=cfg)
        return "✅ saved to memory." if ok else "⚠️ failed to save to memory."

    if verb == "!forget":
        ok = _memory_add(memory, f"[FORGET REQUEST] {payload}", meta={"type": "forget", "source": "user_command"}, cfg=cfg)
        return "🧹 forget request recorded." if ok else "⚠️ failed to record forget request."

    if verb == "!playbook":
        try:
            path = _playbook_add(payload, cfg)
            return f"📒 playbook updated → {path}"
        except Exception as exc:
            return f"⚠️ failed to update playbook: {exc}"

    if verb == "!identity":
        # Accept forms like: user=Shane assistant=Synthia (order independent)
        u = re.search(r"user\s*=\s*([^\s,;]+)", payload, re.I)
        a = re.search(r"assistant\s*=\s*([^\s,;]+)", payload, re.I)
        msgs = []
        if u:
            name = u.group(1)
            ok = _memory_add(memory, f"Set user name: {name}", meta={"type": "identity", "source": "user_command"}, cfg=cfg)
            msgs.append("user=" + name + ("✅" if ok else "⚠️"))
        if a:
            name = a.group(1)
            ok = _memory_add(memory, f"Set assistant name: {name}", meta={"type": "identity", "source": "user_command"}, cfg=cfg)
            msgs.append("assistant=" + name + ("✅" if ok else "⚠️"))
        return "👤 identity updated: " + ", ".join(msgs) if msgs else "👤 provide values like: user=Shane assistant=Synthia"

    return "⚠️ unknown command. Try !help"

# ------------------------------------ Main ----------------------------------- #
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(here, "config.yaml")
    cfg = load_config(cfg_path)
    try:
        agent = Agent(cfg)
    except FileNotFoundError:
        console.print("[bold red]Model file not found.[/bold red]")
        console.print(f"Configured `model_path`: {cfg.model_path}")
        console.print(
            "Update /home/head-node/Dev/ai-lab/synthia/config.yaml "
            "or set `SYNTHIA_MODEL_PATH` before launching."
        )
        candidates = discover_local_models(limit=8)
        if candidates:
            console.print("\nLocal GGUF candidates:")
            for candidate in candidates:
                console.print(f"  - {candidate}")
        raise SystemExit(1)

    show_banner()
    show_config(cfg)

    console.print("[dim]Type !help for in-chat commands (memory/playbook).[/dim]")

    try:
        while True:
            try:
                user = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Bye.[/dim]")
                break

            if not user:
                continue
            if user in (":q", ":quit", ":exit"):
                console.print("[dim]Exiting.[/dim]")
                break
            if user == ":help":
                console.print(
                    "Commands: :help  :quit\n"
                    "Tips: start messages with 'remember this:' to persist notes.\n"
                    "Also see in-chat commands with !help"
                )
                continue

            # --- NEW: handle bang-commands locally (no model call) ---
            if is_command(user):
                status = handle_command(user, cfg, agent)
                console.print(status)
                continue

            # Get response from the agent
            answer, critique, improvements = agent.respond(user)

            # Always show the model's answer
            response_text = Text(answer or "")
            response_text.stylize("#4EDB4C")
            console.print(response_text)

            if getattr(agent.tts, "enabled", False):
                agent.tts.speak(answer)

            # Only show these if they have content (prevents blank panels)
            if critique and critique.strip():
                console.print(Panel(critique, title="Critique", style="dim"))
            if improvements and improvements.strip():
                console.print(Panel(improvements, title="Playbook Update", style="green"))
    finally:
        agent.close()

if __name__ == "__main__":
    main()

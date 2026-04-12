# Celeste — Claude Code Context

## Project Overview

Celeste is a local-first desktop AI assistant built with Python and PySide6. It runs fully offline using local GGUF models via llama.cpp and exposes a rich settings panel alongside the chat UI.

**Primary developer:** Shane  
**Primary dev machine:** Ubuntu 24.04 (Linux is the lead platform)  
**Secondary machine:** Windows 10 Pro (used for Windows installer packaging)  
**Working branch:** `windows/installer-troubleshooting` (branch off main for active dev; merge into `main` when stable)  
**GitHub:** The local repo is the authoritative source — GitHub is a push target, not the source of truth.

---

## Architecture

```
desktop_app.py      PySide6 GUI — chat panel + settings panel
app_service.py      CelesteService — thin service layer between GUI and Agent
agent.py            Agent — prompt assembly, grounding, reflection, graph memory
model_runner.py     LLMRunner — llama_cpp / llama_server / transformers backends
file_rag.py         FileRAG — hybrid lexical (TF-IDF) + semantic (embeddings) retrieval
memory.py           MemoryPipeline — Chroma vector DB for episodic memory (Engram)
graph_memory.py     SQLite graph of facts about the user and environment
graph_facts.py      Populates graph_memory at startup (runtime facts, RAG facts)
reflection.py       Reflector — async background reflection; updates playbook entries
playbook.py         Playbook — persistent behavioral rules loaded into system prompt
tts.py              TTSManager — pyttsx3 or Piper TTS
config_types.py     AgentConfig (Pydantic) — single source of truth for all settings
app_config.py       load_config / save_config (YAML)
app_paths.py        Platform-aware paths for packaged vs. dev mode
setup_wizard.py     First-run GUI wizard (shown when config.yaml is missing)
validate_environment.py  Pre-launch dependency checks
cli.py              Optional CLI interface
```

### Data flow for a chat message

1. `desktop_app.py` — user submits message → `ServiceWorker.send_message()` (background thread)
2. `app_service.py` — `CelesteService.chat(message, token_cb)` → `Agent.respond()`
3. `agent.py` — assembles prompt: system preamble + graph memory + engram recall + file RAG grounding + recent turns
4. `model_runner.py` — runs inference (streaming if `token_cb` provided)
5. Tokens stream back via `token_cb` signal → live preview widget in GUI
6. Final reply post-processed (perspective guard, retry on refusal phrases) → returned to GUI

---

## Key Systems

### File RAG (`file_rag.py`)
- Two index tiers: **TF-IDF** (fast lexical, always available) and **deep index** (semantic embeddings, built on demand)
- Deep index uses sentence-transformers; supports multi-GPU via `file_rag_multi_gpu: true`
- `files_per_directory()` returns per-directory file counts for UI display
- Deep index is lazy-loaded on first query; pre-warmed in a background thread at startup

### Graph Memory (`graph_memory.py`, `graph_facts.py`)
- SQLite-backed graph of facts (subject → predicate → object)
- Populated at startup by `record_runtime_graph_facts` and `record_file_rag_graph_facts` (runs in background thread to avoid blocking startup)
- Injected into prompt context as structured facts

### Engram Memory (`memory.py`)
- Chroma vector DB for episodic memory across sessions
- Auto-prune can be enabled; manual purge available via UI
- Retention controlled by `purge_engram_memory(seconds=N)`

### Reflection (`reflection.py`)
- Async background process: after each reply, reflects on the exchange
- Updates `playbook.py` entries — behavioral rules that persist across sessions

### Streaming Output
- `Agent.respond()` accepts optional `token_cb: Callable[[str], None]`
- `model_runner.py` `generate_stream()` / `chat_stream()` yield tokens via `stream=True` (llama_cpp)
- GUI: `token_received` Signal feeds live preview `QPlainTextEdit` in a `QFrame` (shown during generation, hidden after)

### Persona / System Preamble
- `AgentConfig.system_preamble` — position-zero instruction injected into every prompt
- Editable via "Edit Persona" button → `PersonaDialog` modal
- `CelesteService.set_persona()` updates in-memory config and saves to `config.yaml`

### Token Usage Indicator
- `Agent._last_prompt_tokens` set after `prep_prompt()`
- `CelesteService.get_token_usage()` returns `(used, n_ctx)`
- GUI: `QProgressBar` color-coded green/amber/red at bottom of chat panel

---

## Configuration (`config.yaml`)

`config.yaml` is gitignored — it is machine-specific. Use `config.example.yaml` as a template.

Key fields in `AgentConfig` (defined in `config_types.py`):
- `backend`: `"llama_cpp"` | `"llama_server"` | `"transformers"`
- `model_path`: path to GGUF file
- `n_ctx`: context window size (e.g. 4096)
- `n_gpu_layers`: layers offloaded to GPU (0 = CPU only)
- `embedding_model`: sentence-transformers model name or local path
- `file_rag_dirs`: list of document library directories
- `system_preamble`: persistent identity/persona instruction
- `file_rag_use_embeddings`: enables semantic deep index (requires `build_deep_index` from UI)
- `file_rag_multi_gpu`: use all available GPUs for embedding

Config is loaded by `app_config.load_config()` and saved by `app_config.save_config()`.

---

## Running in Dev Mode

```bash
# Linux
source .venv/bin/activate
python desktop_app.py

# Windows
.venv\Scripts\activate
python desktop_app.py
```

If `config.yaml` does not exist, the setup wizard runs automatically.

---

## Packaging

See `PACKAGING.md` for full details. Summary:

**Windows** (run on Windows machine):
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\packaging\windows\build_windows.ps1
```
Outputs: `dist\Celeste\` bundle + `dist\installer\Celeste-Setup-0.1.0.exe` (if Inno Setup present)

**Linux** (run on Ubuntu machine):
```bash
./packaging/linux/build_linux.sh
```
Outputs: `dist/Celeste/` bundle + `.tar.gz` + `.AppImage` (if `appimagetool` present)

Packaged config paths:
- Windows: `%LOCALAPPDATA%\Celeste\config.yaml`
- Linux: `~/.config/Celeste/config.yaml`

---

## Branch Strategy

- `main` — stable, always deployable
- `windows/installer-troubleshooting` — active Windows development branch
- Merge into `main` when a set of changes is stable; push `main` to GitHub

---

## Known Constraints / Notes

- `config.yaml` and model files are gitignored — never commit them
- Deep index build can take minutes for large libraries (895+ docs); triggered from UI → "Reindex" or build deep index button
- Graph facts writes (`record_runtime_graph_facts`, `record_file_rag_graph_facts`) run in a background thread at startup — do not block on them
- Anti-hallucination instructions are in `Agent.respond()` prompt: "Only reference documents that appear in the Grounding Sources section. Do not invent document titles."
- Anti-Markdown instructions are in the prompt: plain prose responses only
- `model_runner.py` `generate_stream()` uses `inspect.signature` to filter kwargs before calling llama_cpp to avoid parameter errors across llama_cpp versions

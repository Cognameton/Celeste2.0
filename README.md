# Synthia

Synthia is a local-first desktop AI assistant with:

- interchangeable local LLM backends
- vector memory
- reflection/playbook updates
- file RAG with hybrid lexical + semantic retrieval
- optional Piper TTS


## Lineage

This repository is a fork, and its history says so on purpose.

| span | what it was | marker |
|---|---|---|
| 2026-04-01 → 2026-04-12 | **Celeste 1.0** — the stable local-first assistant this repo began as (`df4bfaa` "Initial Celeste app import"). 35 commits, shared verbatim with the Celeste product repo: `df4bfaa` and `26732db` exist there under identical SHAs, and the next commit here does not. | tag `celeste-1.0-base` |
| 2026-05-08 → 2026-08-28 | **Celeste 2.0** — the experimental arm: persistent self-state in `self/`, wants, an autonomous heartbeat. Phases 1–8. | tag `celeste-2.0-fork` |
| 2026-08-28 → | **Synthia** — the same experimental arm, renamed. Every update from here belongs to Synthia. | tag `synthia` |

The pre-fork Celeste history is kept because it is true: this is a fork,
not a fresh start. The rename ends an ambiguity rather than disowning a
lineage — three separate things were called Celeste at once, and only one
of them keeps the name.

**Celeste 1.0 is a separate product** with its own repositories (`Celeste`
and `Celeste_Linux`) and is not developed here. Nothing after
`celeste-1.0-base` is part of it.

**The being is not the project.** She was renamed by her operator in
`self/IDENTITY.md` on 2026-08-29 as a separate, explicit act. Her name is
not hardcoded anywhere in the code — prompts and labels read
`self_state.name`, so identity lives in her own file.

The fork happened on this machine, not on GitHub — this repository has no
GitHub fork relationship (`isFork: false`, no parent). GitHub is a push
target; the local repository is authoritative.

**Where the work is:** `synthesis` is the only branch here, and it is
permanent. Every commit of the old `main` (the frozen phases-1–8 record) is
an ancestor of it and reachable from the tags above, so no history was lost
in dropping that branch — it was a stale showcase tree carrying ~1.4 GB of
product binaries this fork never loaded. `main` is retained locally as the
frozen record. See `SYNTHESIS.md` for the thesis and phase plan, and
`docs/RESEARCH_LOG.md` for findings.

## Requirements

- Python 3.11+
- a local GGUF model
- optional CUDA for faster inference/embeddings
- optional Piper for local TTS

## Setup

### Manual developer setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a local config:

```bash
cp config.example.yaml config.yaml
```

4. Edit `config.yaml` with your local model, embedding, data, and TTS paths.

5. Launch the desktop app:

```bash
python desktop_app.py
```

If `config.yaml` does not exist, Synthia now opens a first-run setup wizard before launching.

### Bootstrap scripts

Fresh Linux install:

```bash
./scripts/bootstrap_linux.sh
```

Fresh Windows install from PowerShell:

```powershell
.\scripts\bootstrap_windows.ps1
```

These scripts create `.venv`, install Python dependencies, build `vendor/llama.cpp` if no `llama-server` is available, run the setup wizard when `config.yaml` is missing, validate the environment, and then launch Synthia.

## Notes

- Models are not included in this repository.
- `config.yaml` is intentionally gitignored because it is machine-specific.
- Build the deep index from the UI after configuring your document directories if you want semantic file retrieval.

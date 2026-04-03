# Celeste

Celeste is a local-first desktop AI assistant with:

- interchangeable local LLM backends
- vector memory
- reflection/playbook updates
- file RAG with hybrid lexical + semantic retrieval
- optional Piper TTS

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

If `config.yaml` does not exist, Celeste now opens a first-run setup wizard before launching.

### Bootstrap scripts

Fresh Linux install:

```bash
./scripts/bootstrap_linux.sh
```

Fresh Windows install from PowerShell:

```powershell
.\scripts\bootstrap_windows.ps1
```

These scripts create `.venv`, install Python dependencies, build `vendor/llama.cpp` if no `llama-server` is available, run the setup wizard when `config.yaml` is missing, validate the environment, and then launch Celeste.

## Notes

- Models are not included in this repository.
- `config.yaml` is intentionally gitignored because it is machine-specific.
- Build the deep index from the UI after configuring your document directories if you want semantic file retrieval.

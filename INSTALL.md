# Install Celeste

## What You Need

- Linux with Python 3.11+
- a local GGUF model on your machine
- optional CUDA-capable GPU
- optional Piper install and voice files for TTS

Celeste does not ship with models. You point it at your own local files in `config.yaml`.

## Quick Bootstrap

Linux:

```bash
git clone https://github.com/Cognameton/Celeste.git
cd Celeste
./scripts/bootstrap_linux.sh
```

Windows PowerShell:

```powershell
git clone https://github.com/Cognameton/Celeste.git
cd Celeste
.\scripts\bootstrap_windows.ps1
```

If `config.yaml` does not exist, a first-run wizard opens so you can use Celeste default folders or browse to your own model, embedding, document, and data paths.

## Manual Setup

## 1. Clone The Repo

```bash
git clone https://github.com/Cognameton/Celeste.git
cd Celeste
```

## 2. Create A Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Create Your Local Config

```bash
cp config.example.yaml config.yaml
```

Edit `config.yaml` and set:

- `model_path`
- `embedding_model`
- `persist_dir`
- `data_dir`
- `file_rag_dirs`
- Piper paths if you want TTS
- `llama_server_executable` if your `llama-server` binary is not in `vendor/llama.cpp` or `PATH`

Important:
- `config.yaml` is machine-specific and is intentionally not tracked by git.
- Use absolute paths.

## 5. Verify Your Model Path

Make sure your configured model actually exists:

```bash
ls -lh /absolute/path/to/your/model.gguf
```

## 6. Launch Celeste

Desktop UI:

```bash
python desktop_app.py
```

If `config.yaml` is missing, the setup wizard opens automatically.

CLI mode:

```bash
python cli.py
```

## 7. Build The Deep Index

If you want semantic file retrieval:

1. Start the desktop app.
2. Add your document directory in the File RAG section.
3. Click `Build Deep Index`.

The first build can take a while on a large library.

## 8. First-Time Behavior To Expect

- Large GGUF models may take time to load on startup.
- Deep indexing may take a long time on a large corpus.
- Semantic retrieval requires the embedding model path to be valid.
- If TTS is enabled and Piper paths are wrong, speaking will fail even if chat works.

## Common Problems

Model file not found:

- Check `model_path` in `config.yaml`
- Verify external drives are mounted

Stuck on startup:

- Celeste may still be loading the model
- watch the status text in the UI

No semantic retrieval:

- verify `file_rag_use_embeddings: true`
- verify `embedding_model` points to a valid local embedding model
- rebuild the deep index after changing the embedding model

No documents found:

- verify `file_rag_dirs`
- reindex or rebuild the deep index

## Recommended Next Setup Step

After first launch, test these in the UI:

- ordinary chat
- a file-specific query
- a topic query using the indexed library
- a grounded answer with citations

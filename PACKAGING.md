# Packaging Synthia

Synthia now has two install paths:

- **Source/developer mode**: clone the repo, use `.venv`, and run `python desktop_app.py`.
- **Packaged/end-user mode**: build a native app bundle with PyInstaller and ship a Windows installer or Linux AppImage/tarball.

Your current local `config.yaml` workflow is preserved. When Synthia runs from source, it still uses the repo-local `config.yaml`. When it runs as a packaged executable, it stores config in a user-writable location:

- Windows: `%LOCALAPPDATA%\Synthia\config.yaml`
- Linux: `~/.config/Synthia/config.yaml` unless `XDG_CONFIG_HOME` is set

## Dependency Split

- `requirements.txt`: developer/source install path
- `requirements-runtime.txt`: packaged runtime dependencies, excluding `llama-cpp-python`
- `requirements-build.txt`: runtime deps + PyInstaller

Packaged builds should use `requirements-build.txt`, not `requirements.txt`.

## Windows Build

Prerequisites on the build machine:

- Python 3.11+
- a prebuilt `vendor\llama.cpp\build\bin\Release\llama-server.exe`
- optional Inno Setup (`ISCC.exe`) if you want a `.exe` installer

Build:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\packaging\windows\build_windows.ps1
```

Outputs:

- `dist\Synthia\` onedir app bundle
- `dist\installer\Synthia-Setup-0.1.0.exe` if Inno Setup is installed

## Linux Build

Prerequisites on the build machine:

- Python 3.11+
- a prebuilt `vendor/llama.cpp/build/bin/llama-server`
- optional `appimagetool` if you want an AppImage

Build:

```bash
./packaging/linux/build_linux.sh
```

Outputs:

- `dist/Synthia/` onedir app bundle
- `dist/packages/Synthia-linux-x86_64.tar.gz`
- `dist/packages/Synthia-x86_64.AppImage` if `appimagetool` is installed

## First-Run Setup

If no `config.yaml` exists in the packaged app's user config directory, Synthia opens the setup wizard and asks for:

- GGUF model path
- embedding model path
- `llama-server` executable path
- document library path
- data/vector DB paths
- optional Piper TTS paths

The wizard offers default Synthia folders but also lets the user browse to custom paths.

## Optional Bundled Runtime Assets

If you want the installer to ship with bundled runtime assets, place them in these repo-local folders before running the package build:

```text
models/
  default.gguf
embeddings/
  e5-small-v2/
    config.json
    modules.json
    sentence_bert_config.json
    model.safetensors
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    vocab.txt
    1_Pooling/
      config.json
voices/
  en_US-amy-medium/
    en_US-amy-medium.onnx
    en_US-amy-medium.onnx.json
piper/
  linux/
    ...
  windows/
    ...
```

The packaging scripts copy these folders into `dist/Synthia/` if present:

- `models/`
- `embeddings/`
- `voices/`
- `piper/linux/` for Linux packages
- `piper/windows/` for Windows packages

The setup wizard prefers bundled runtime paths from the installed app when present, while still keeping user data paths under the user profile. Users can still browse to different model, embedding, or TTS assets.

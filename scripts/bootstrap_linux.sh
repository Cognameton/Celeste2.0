#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

LLAMA_SERVER="$PROJECT_ROOT/vendor/llama.cpp/build/bin/llama-server"
if [ ! -x "$LLAMA_SERVER" ] && ! command -v llama-server >/dev/null 2>&1; then
  mkdir -p "$PROJECT_ROOT/vendor"
  if [ ! -d "$PROJECT_ROOT/vendor/llama.cpp/.git" ]; then
    git clone https://github.com/ggml-org/llama.cpp.git "$PROJECT_ROOT/vendor/llama.cpp"
  fi
  if cmake -S "$PROJECT_ROOT/vendor/llama.cpp" -B "$PROJECT_ROOT/vendor/llama.cpp/build" -DGGML_CUDA=ON -DLLAMA_CURL=OFF; then
    :
  else
    cmake -S "$PROJECT_ROOT/vendor/llama.cpp" -B "$PROJECT_ROOT/vendor/llama.cpp/build" -DLLAMA_CURL=OFF
  fi
  cmake --build "$PROJECT_ROOT/vendor/llama.cpp/build" --config Release -j "${SYNTHIA_BUILD_JOBS:-$(nproc)}"
fi

if [ ! -f "$PROJECT_ROOT/config.yaml" ]; then
  python "$PROJECT_ROOT/setup_wizard.py" --config "$PROJECT_ROOT/config.yaml"
fi

python "$PROJECT_ROOT/validate_environment.py" --config "$PROJECT_ROOT/config.yaml"
python "$PROJECT_ROOT/desktop_app.py"

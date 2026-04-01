#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
LLAMA_SERVER_PATTERN="/home/head-node/Dev/ai-lab/llama.cpp/build/bin/llama-server"
CLI_PATTERN="$ROOT_DIR/.venv/bin/python $ROOT_DIR/cli.py"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Missing virtualenv python at $VENV_PYTHON" >&2
  exit 1
fi

kill_pattern() {
  local pattern="$1"
  if pgrep -f "$pattern" >/dev/null 2>&1; then
    echo "Stopping stale process: $pattern"
    pkill -f "$pattern" || true
    sleep 1
  fi
}

kill_pattern "$LLAMA_SERVER_PATTERN"
kill_pattern "$CLI_PATTERN"

cd "$ROOT_DIR"
exec "$VENV_PYTHON" cli.py

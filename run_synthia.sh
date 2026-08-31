#!/usr/bin/env bash
# Launch the Synthia CLI, clearing any stale processes from a previous run.
# For the desktop app use: .venv/bin/python desktop_app.py
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$ROOT_DIR/.venv/bin/python"
CLI_PATTERN="$ROOT_DIR/.venv/bin/python $ROOT_DIR/cli.py"

# Any llama-server this project started. Override if you run several.
LLAMA_SERVER_PATTERN="${SYNTHIA_LLAMA_SERVER_PATTERN:-llama-server}"

if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Missing virtualenv python at $VENV_PYTHON" >&2
  echo "Run scripts/bootstrap_linux.sh first." >&2
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

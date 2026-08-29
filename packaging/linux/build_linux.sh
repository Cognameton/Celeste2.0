#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-build.txt

LLAMA_BIN_DIR="$PROJECT_ROOT/vendor/llama.cpp/build/bin"
LLAMA_SERVER="$LLAMA_BIN_DIR/llama-server"
if [ ! -x "$LLAMA_SERVER" ]; then
  echo "Missing $LLAMA_SERVER. Build llama.cpp first, then rerun this packager." >&2
  exit 1
fi

rm -rf "$PROJECT_ROOT/dist/Synthia" "$PROJECT_ROOT/build/Synthia"
python -m PyInstaller packaging/pyinstaller/synthia.spec --noconfirm --clean

mkdir -p "$PROJECT_ROOT/dist/Synthia/vendor/llama.cpp/build/bin"
cp -a "$LLAMA_BIN_DIR/." "$PROJECT_ROOT/dist/Synthia/vendor/llama.cpp/build/bin/"
chmod +x "$PROJECT_ROOT/dist/Synthia/vendor/llama.cpp/build/bin/llama-server"

if [ -d "$PROJECT_ROOT/models" ]; then
  cp -a "$PROJECT_ROOT/models" "$PROJECT_ROOT/dist/Synthia/models"
fi

if [ -d "$PROJECT_ROOT/embeddings" ]; then
  cp -a "$PROJECT_ROOT/embeddings" "$PROJECT_ROOT/dist/Synthia/embeddings"
fi

if [ -d "$PROJECT_ROOT/piper/linux" ]; then
  mkdir -p "$PROJECT_ROOT/dist/Synthia/piper"
  cp -a "$PROJECT_ROOT/piper/linux" "$PROJECT_ROOT/dist/Synthia/piper/linux"
fi

if [ -d "$PROJECT_ROOT/voices" ]; then
  cp -a "$PROJECT_ROOT/voices" "$PROJECT_ROOT/dist/Synthia/voices"
fi

mkdir -p "$PROJECT_ROOT/dist/packages"
tar -C "$PROJECT_ROOT/dist" -czf "$PROJECT_ROOT/dist/packages/Synthia-linux-x86_64.tar.gz" Synthia

if command -v appimagetool >/dev/null 2>&1; then
  APPDIR="$PROJECT_ROOT/dist/Synthia.AppDir"
  rm -rf "$APPDIR"
  mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/share/applications"
  cp -a "$PROJECT_ROOT/dist/Synthia/." "$APPDIR/usr/bin/"
  cat > "$APPDIR/AppRun" <<'SH'
#!/usr/bin/env bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$HERE/usr/bin/Synthia" "$@"
SH
  chmod +x "$APPDIR/AppRun"
  cat > "$APPDIR/Synthia.desktop" <<'DESKTOP'
[Desktop Entry]
Name=Synthia
Exec=Synthia
Icon=synthia_icon
Type=Application
Categories=Utility;
DESKTOP
  cp "$PROJECT_ROOT/assets/synthia_icon.png" "$APPDIR/synthia_icon.png"
  cp "$APPDIR/Synthia.desktop" "$APPDIR/usr/share/applications/Synthia.desktop"
  appimagetool "$APPDIR" "$PROJECT_ROOT/dist/packages/Synthia-x86_64.AppImage"
else
  echo "appimagetool not found. Generated dist/packages/Synthia-linux-x86_64.tar.gz, but skipped AppImage creation."
fi

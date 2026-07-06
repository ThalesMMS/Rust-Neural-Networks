#!/usr/bin/env bash
set -euo pipefail

# Build Rust NN Studio and package it as a macOS .app bundle.
#
# Usage:
#   scripts/build_macos_app.sh
#
# Output:
#   target/release/bundle/macos/Rust NN Studio.app

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAURI_BIN="$ROOT_DIR/ui/node_modules/.bin/tauri"

# Non-interactive shells on macOS often miss Homebrew's Node path.
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Error: macOS .app bundles can only be built on macOS." >&2
  exit 1
fi

if [[ ! -x "$TAURI_BIN" ]]; then
  echo "Error: Tauri CLI not found at $TAURI_BIN" >&2
  echo "Run: npm --prefix ui install" >&2
  exit 1
fi

if ! command -v node >/dev/null 2>&1; then
  echo "Error: node is not on PATH. Install Node.js or add it to PATH." >&2
  exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "Error: npm is not on PATH. Install Node.js or add it to PATH." >&2
  exit 1
fi

cd "$ROOT_DIR"

echo "Building Rust NN Studio .app bundle..."
(
  cd "$ROOT_DIR/src-tauri"
  "$TAURI_BIN" build --ci --bundles app --no-sign
)

APP_PATH="$(find "$ROOT_DIR/target/release/bundle/macos" -maxdepth 1 -type d -name "*.app" -print | sort | tail -n 1)"

if [[ -z "$APP_PATH" ]]; then
  echo "Error: build finished, but no .app bundle was found in target/release/bundle/macos." >&2
  exit 1
fi

echo
echo "App bundle created:"
echo "$APP_PATH"

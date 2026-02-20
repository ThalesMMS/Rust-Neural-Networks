#!/usr/bin/env bash
# Wrapper script to run visualize_vit_attention.py with correct Python environment
# Requires: numpy, matplotlib, pillow

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec /usr/bin/env python3 "$SCRIPT_DIR/visualize_vit_attention.py" "$@"

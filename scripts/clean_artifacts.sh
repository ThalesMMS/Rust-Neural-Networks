#!/usr/bin/env bash
set -euo pipefail

# Repository artifact cleanup script
#
# Deletes only standardized *generated* artifact directories as defined in
# docs/artifacts.md.
#
# Usage:
#   scripts/clean_artifacts.sh [--dry-run] [--all]
#
# Options:
#   --dry-run   Print what would be removed (default)
#   --all       Actually delete files/directories
#   -h, --help  Show help

DRY_RUN=1

usage() {
  sed -n '1,80p' "$0" | sed -n '1,80p' | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --all)
      DRY_RUN=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo
      usage >&2
      exit 2
      ;;
  esac
done

# Paths are relative to repo root.
TARGETS=(
  "logs"
  "runs"
  "artifacts"
  "benchmarks/results"
  "wasm/pkg"
  "wasm/target"
)

rm_path() {
  local p="$1"

  if [[ ! -e "$p" ]]; then
    return 0
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] rm -rf $p"
    return 0
  fi

  echo "rm -rf $p"
  rm -rf -- "$p"
}

# Ensure we are running from repo root (best effort).
if [[ ! -f "Cargo.toml" ]]; then
  echo "Error: run this script from the repository root (Cargo.toml not found)." >&2
  exit 1
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry-run mode (no files will be deleted). Use --all to delete." >&2
fi

for t in "${TARGETS[@]}"; do
  rm_path "$t"
done

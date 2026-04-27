#!/usr/bin/env bash
set -euo pipefail

# Build the manuscript PDF from main.tex.
# Usage:
#   ./build.sh          # build PDF
#   ./build.sh clean    # remove aux files

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MAIN_TEX="main.tex"
MAIN_BASENAME="${MAIN_TEX%.tex}"

clean_aux() {
  rm -f \
    "${MAIN_BASENAME}.aux" \
    "${MAIN_BASENAME}.bbl" \
    "${MAIN_BASENAME}.blg" \
    "${MAIN_BASENAME}.fdb_latexmk" \
    "${MAIN_BASENAME}.fls" \
    "${MAIN_BASENAME}.log" \
    "${MAIN_BASENAME}.out" \
    "${MAIN_BASENAME}.toc"
}

if [[ "${1:-}" == "clean" ]]; then
  clean_aux
  echo "Cleaned LaTeX auxiliary files in $SCRIPT_DIR"
  exit 0
fi

if [[ ! -f "$MAIN_TEX" ]]; then
  echo "Error: $MAIN_TEX not found in $SCRIPT_DIR" >&2
  exit 1
fi

tectonic $MAIN_TEX --outdir build

echo "Built ${MAIN_BASENAME}.pdf in $SCRIPT_DIR"

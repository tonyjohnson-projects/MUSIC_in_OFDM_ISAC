#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/artifacts/presentation/signal_model_math}"
DPI="${DPI:-220}"

TEX_SOURCE="$SCRIPT_DIR/signal_model_math.tex"
PDF_OUTPUT="$ROOT_DIR/signal_model_math_169.pdf"
PNG_STEM="$ROOT_DIR/signal_model_math_169_png"
PNG_OUTPUT="${PNG_STEM}.png"

mkdir -p "$BUILD_DIR"

if ! xelatex \
  -halt-on-error \
  -interaction=nonstopmode \
  -output-directory="$BUILD_DIR" \
  "$TEX_SOURCE" >"$BUILD_DIR/xelatex.stdout" 2>"$BUILD_DIR/xelatex.stderr"; then
  cat "$BUILD_DIR/xelatex.stdout"
  cat "$BUILD_DIR/xelatex.stderr" >&2
  exit 1
fi

cp "$BUILD_DIR/signal_model_math.pdf" "$PDF_OUTPUT"
pdftoppm -png -singlefile -r "$DPI" "$PDF_OUTPUT" "$PNG_STEM"

echo "Wrote $PDF_OUTPUT"
echo "Wrote $PNG_OUTPUT"

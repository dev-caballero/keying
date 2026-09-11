#!/usr/bin/env bash
# Downloads the segmentation model (isnet-general-use, MIT) to
# resources/models/ (gitignored).
set -euo pipefail

MODEL=isnet-general-use
URL="https://github.com/danielgatis/rembg/releases/download/v0.0.0/${MODEL}.onnx"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$REPO_ROOT/resources/models"
OUT="$OUT_DIR/${MODEL}.onnx"

if [[ -f "$OUT" ]]; then
  echo "model already present: $OUT"
  exit 0
fi

mkdir -p "$OUT_DIR"
echo "== downloading ${MODEL} ($URL) =="
curl -fL "$URL" -o "$OUT"
ls -lh "$OUT"
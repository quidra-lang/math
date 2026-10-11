#!/usr/bin/env bash
set -euo pipefail
QUIDRA="${1:?usage: half_math.sh /path/to/quidra}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export QUIDRA_CACHE_DIR="$TMP/run-cache"
mkdir "$TMP/packages"
ln -s "$ROOT" "$TMP/packages/math"
QUIDRA_PACKAGE_PATH="$TMP/packages" "$QUIDRA" run "$ROOT/tests/half_math.qui" > "$TMP/out"
python3 - "$TMP/out" <<'PY'
import pathlib,sys
assert pathlib.Path(sys.argv[1]).read_text().splitlines() == ['true'] * 20
PY

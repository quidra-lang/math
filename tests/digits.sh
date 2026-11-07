#!/usr/bin/env bash
set -euo pipefail

QUIDRA="${1:?usage: digits.sh /path/to/quidra}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export QUIDRA_CACHE_DIR="$TMP/run-cache"
export QUIDRA_PACKAGE_PATH="$(dirname "$ROOT")"

"$QUIDRA" "$ROOT/tests/digits.qui" >"$TMP/values.out"
if [[ "$(cat "$TMP/values.out")" != "Math integer digit tests passed" ]]; then
    echo "unexpected integer digit output:" >&2
    cat "$TMP/values.out" >&2
    exit 1
fi

# The generic integer contract must reject non-integer scalars and containers.
for value in 'true' 'real(12)' 'real16(12)' 'real16b(12)' 'real32(12)' 'real64(12)' \
             'com(12)' '"12"' '[1, 2]' 'tensor.ones<int32>([2])'; do
    for call in "math.digits(value)" "value.digits()"; do
        cat >"$TMP/noninteger.qui" <<QUI
import math
value := $value
print($call)
QUI
        if "$QUIDRA" check "$TMP/noninteger.qui" >"$TMP/noninteger.out" 2>&1; then
            echo "integer digits unexpectedly accepted $value through $call" >&2
            exit 1
        fi
    done
done

cat >"$TMP/no-import.qui" <<'QUI'
value := 12
print(value.digits())
QUI
if "$QUIDRA" check "$TMP/no-import.qui" >"$TMP/no-import.out" 2>&1; then
    echo "integer digits unexpectedly available without importing Math" >&2
    exit 1
fi

cat "$TMP/values.out"

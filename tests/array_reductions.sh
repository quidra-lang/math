#!/usr/bin/env bash
set -euo pipefail

QUIDRA="${1:?usage: array_reductions.sh /path/to/quidra}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export QUIDRA_CACHE_DIR="$TMP/run-cache"
export QUIDRA_TUNING_DIR="$TMP/tuning"
mkdir -p "$TMP/packages/math"
cp "$ROOT/"*.qui "$ROOT/project.toml" "$ROOT/quidra.package" "$TMP/packages/math/"
cp -R "$ROOT/compiler" "$ROOT/native" "$TMP/packages/math/"
export QUIDRA_PACKAGE_PATH="$TMP/packages"
cp "$ROOT/tests/array_reductions.qui" "$TMP/array_reductions.qui"

if ! "$QUIDRA" "$TMP/array_reductions.qui" >"$TMP/values.out" 2>"$TMP/values.err"; then
    cat "$TMP/values.err" >&2
    exit 1
fi
expected="$(printf 'true\n%.0s' {1..82})"
if [[ "$(cat "$TMP/values.out")" != "$expected" ]]; then
    echo "unexpected numeric-array reduction output:" >&2
    cat "$TMP/values.out" "$TMP/values.err" >&2
    exit 1
fi

# Ordinary array dot rejects unequal input lengths without partial results.
cat >"$TMP/dot-length.qui" <<'QUI'
import math
int[] left = [1, 2, 3]
int[] right = [4, 5]
print(math.dot(left, right))
QUI
dot_status=0
"$QUIDRA" "$TMP/dot-length.qui" >"$TMP/dot-length.out" 2>"$TMP/dot-length.err" || dot_status=$?
if [[ "$dot_status" -ne 101 || -s "$TMP/dot-length.out" ]] ||
   ! grep -Fq 'runtime error[MATH_SHAPE]' "$TMP/dot-length.err"; then
    echo "unexpected mismatched array dot outcome (status $dot_status):" >&2
    cat "$TMP/dot-length.out" "$TMP/dot-length.err" >&2
    exit 1
fi

expect_overflow() {
    local name="$1" type="$2" values="$3" code="$4" operation="${5:-sum}"
    cat >"$TMP/$name.qui" <<QUI
import math
$type[] values = $values
print(math.$operation(values))
QUI
    local status=0
    "$QUIDRA" "$TMP/$name.qui" >"$TMP/$name.out" 2>"$TMP/$name.err" || status=$?
    if [[ "$status" -ne 101 || -s "$TMP/$name.out" ]] ||
       ! grep -Fq "Quidra runtime error[$code]" "$TMP/$name.err"; then
        echo "unexpected $name outcome (status $status):" >&2
        cat "$TMP/$name.out" "$TMP/$name.err" >&2
        exit 1
    fi
}

# A later cancelling term cannot hide an earlier checked overflow.
expect_overflow signed-overflow int8 '[127, 1, -1]' INTEGER_OVERFLOW
expect_overflow natural-overflow nat8 '[255, 1]' INTEGER_OVERFLOW
expect_overflow real-overflow real32 '[3.0e38, 3.0e38, -3.0e38]' REAL_OVERFLOW
expect_overflow complex-overflow com32 '[3.0e38 + 0.0i, 3.0e38 + 0.0i]' REAL_OVERFLOW
expect_overflow mean-overflow real32 '[3.0e38, 3.0e38, -3.0e38]' REAL_OVERFLOW mean

for operation_type in 'mean real32' 'mean int8' 'mean com32' 'min_all real32' 'max_all real32'; do
    read -r operation element_type <<<"$operation_type"
    cat >"$TMP/empty-$operation.qui" <<QUI
import math
$element_type[] values = []
print(math.$operation(values))
QUI
    status=0
    "$QUIDRA" "$TMP/empty-$operation.qui" >"$TMP/empty.out" 2>"$TMP/empty.err" || status=$?
    if [[ "$status" -ne 101 || -s "$TMP/empty.out" ]] ||
       ! grep -Fq 'Quidra runtime error[MATH_SHAPE]' "$TMP/empty.err"; then
        echo "unexpected empty-array $operation outcome (status $status):" >&2
        cat "$TMP/empty.out" "$TMP/empty.err" >&2
        exit 1
    fi
done

for operation in sum mean min_all max_all; do
    cat >"$TMP/nonnumeric.qui" <<QUI
import math
bool[] values = [true, false]
print(math.$operation(values))
QUI
    if "$QUIDRA" check "$TMP/nonnumeric.qui" >"$TMP/nonnumeric.out" 2>&1; then
        echo "numeric-array $operation unexpectedly accepted bool[]" >&2
        exit 1
    fi
done

for operation in min_all max_all; do
    cat >"$TMP/complex.qui" <<QUI
import math
com32[] values = [1.0 + 2.0i]
print(math.$operation(values))
QUI
    if "$QUIDRA" check "$TMP/complex.qui" >"$TMP/complex.out" 2>&1; then
        echo "numeric-array $operation unexpectedly accepted unordered complex values" >&2
        exit 1
    fi
done

echo "Math numeric-array reduction tests passed"

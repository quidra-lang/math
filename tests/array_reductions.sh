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
    # Preserve the original numeric checks. On an LLVM verification failure,
    # print the producing function instead of suppressing the assertion or
    # silently replacing the exact-value comparison with a weaker one.
    if "$QUIDRA" llvm "$TMP/array_reductions.qui" >"$TMP/failed.ll" 2>"$TMP/llvm.err"; then
        echo "LLVM around failed string equality (if present):" >&2
        grep -n -B 45 -A 15 'str.equal.179' "$TMP/failed.ll" >&2 || true
        echo "LLVM definitions/references of the missing operand (if present):" >&2
        grep -n -E '%v178([^0-9]|$)' "$TMP/failed.ll" | tail -25 >&2 || true
    else
        cat "$TMP/llvm.err" >&2
    fi
    exit 1
fi
expected="$(printf 'true\n%.0s' {1..89})"
if [[ "$(cat "$TMP/values.out")" != "$expected" ]]; then
    echo "unexpected numeric-array reduction output:" >&2
    cat "$TMP/values.out" "$TMP/values.err" >&2
    exit 1
fi

# When a new array contract breaks LLVM emission, keep the concrete
# pre-optimization and post-optimization operands visible in CI logs.
# Do not change any numeric assertions or suppress compiler failures.
diagnose_failed_ir() {
    local source="$1" label="$2"
    if "$QUIDRA" ir "$source" >"$TMP/$label.ir" 2>"$TMP/$label.ir.err"; then
        echo "Typed IR around suspected missing operand in $label:" >&2
        grep -n -B 28 -A 14 -E '%178([^0-9]|$)|%179([^0-9]|$)' "$TMP/$label.ir" | tail -125 >&2 || true
    else
        cat "$TMP/$label.ir.err" >&2
    fi
    if "$QUIDRA" llvm "$source" >"$TMP/$label.ll" 2>"$TMP/$label.llvm.err"; then
        echo "LLVM around suspected missing SSA operand in $label:" >&2
        grep -n -B 55 -A 18 -E 'str\.equal\.179|%v178([^0-9]|$)' "$TMP/$label.ll" | tail -165 >&2 || true
    else
        cat "$TMP/$label.llvm.err" >&2
    fi
}

# Exercise the complete range -> array -> comparison/mask -> method pipeline.
if ! "$QUIDRA" "$ROOT/tests/array_pipeline.qui" >"$TMP/pipeline.out" 2>"$TMP/pipeline.err"; then
    echo "Math array pipeline failed:" >&2
    cat "$TMP/pipeline.err" >&2
    diagnose_failed_ir "$ROOT/tests/array_pipeline.qui" pipeline
    exit 1
fi
if [[ "$(cat "$TMP/pipeline.out")" != "true" ]]; then
    echo "unexpected numeric-array pipeline output:" >&2
    cat "$TMP/pipeline.out" "$TMP/pipeline.err" >&2
    exit 1
fi

# Array scalar dispatch must preserve real64 and exact-real precision.
if ! "$QUIDRA" "$ROOT/tests/array_precision.qui" >"$TMP/precision.out" 2>"$TMP/precision.err"; then
    echo "Math array precision failed:" >&2
    cat "$TMP/precision.err" >&2
    diagnose_failed_ir "$ROOT/tests/array_precision.qui" precision
    exit 1
fi
if [[ "$(cat "$TMP/precision.out")" != "$(printf 'true\n%.0s' {1..6})" ]]; then
    echo "array and scalar precision differ:" >&2
    cat "$TMP/precision.out" "$TMP/precision.err" >&2
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

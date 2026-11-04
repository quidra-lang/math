#!/usr/bin/env bash
set -euo pipefail

QUIDRA="${1:?usage: aot.sh /path/to/quidra}"
REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export QUIDRA_CACHE_DIR="$TMP/run-cache"

cat > "$TMP/math-aot.qui" <<'QUI'
import math

int | error run()
    real64 root := math.sqrt(real64(9.0))
    println(root == real64(3.0))

    tensor<real32> values := tensor.ones<real32>([2]) * real32(4)
    tensor<real32> roots := math.sqrt(values)
    println(roots[0].item() == real32(2))
    println(roots[1].item() == real32(2))
    return 0

int | error result := run()
match result
    int
        int ignored := result
    error problem
        println(problem)
QUI

OUTPUT="$TMP/math-aot"
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" build "$TMP/math-aot.qui" -o "$OUTPUT"
actual="$("$OUTPUT")"
expected="$(printf 'true\ntrue\ntrue')"
if [[ "$actual" != "$expected" ]]; then
    echo "unexpected Math AOT output:" >&2
    printf '%s\n' "$actual" >&2
    exit 1
fi

repl_output="$(
    QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" repl < "$TMP/math-aot.qui"
)"
true_lines="$(grep -Fxc "true" <<< "$repl_output" || true)"
if [[ "$true_lines" -lt 3 ]]; then
    echo "Math package-native REPL/JIT load did not execute expected math operations:" >&2
    printf '%s\n' "$repl_output" >&2
    exit 1
fi


# Tracked products through the package-native bridge in an AOT binary and in
# the REPL/JIT: the custom autograd callbacks must resolve in both. Values
# equal the portable composition, so the listed native events (logged under
# QUIDRA_MATH_TEST_NATIVE_TRACE) must also appear when the binary runs.
check_native_aot() {
    local name="$1"
    local count="$2"
    shift 2
    QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" build "$TMP/$name.qui" -o "$TMP/$name"
    local actual
    actual="$("$TMP/$name")"
    local expected
    expected="$(printf 'true\n%.0s' $(seq 1 "$count"))"
    if [[ "$actual" != "$expected" ]]; then
        echo "unexpected Math $name AOT output:" >&2
        printf '%s\n' "$actual" >&2
        exit 1
    fi
    local traced
    traced="$(QUIDRA_MATH_TEST_NATIVE_TRACE=1 "$TMP/$name" 2>&1 >/dev/null)"
    local event
    for event in "$@"; do
        if ! grep -Fxq "quidra-math native $event" <<< "$traced"; then
            echo "Math $name AOT binary did not run native '$event':" >&2
            printf '%s\n' "$traced" >&2
            exit 1
        fi
    done
    local repl_output
    repl_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" repl < "$TMP/$name.qui")"
    local repl_true
    repl_true="$(grep -Fxc "true" <<< "$repl_output" || true)"
    if [[ "$repl_true" -lt "$count" ]]; then
        echo "Math $name REPL/JIT run did not produce expected results:" >&2
        printf '%s\n' "$repl_output" >&2
        exit 1
    fi
}

cat > "$TMP/math-native-matmul-aot.qui" <<'QUI'
import math

tensor<real32> x := tensor.ones<real32>([3, 4]).track()
tensor<real32> w := (tensor.ones<real32>([2, 4]) * real32(0.5)).track()
tensor<real32> y := math.matmul(x, w.transpose(0, 1))
(y * y).gather([5], []).backward(&x, &w)
println(y.untrack()[2, 1].item() == real32(2))
println(w.grad[1, 0].item() == real32(4) and w.grad[0, 0].item() == real32(0))
println(x.grad[2, 3].item() == real32(2) and x.grad[0, 0].item() == real32(0))
QUI
check_native_aot math-native-matmul-aot 3 \
    "matmul cpu float32" "matmul-backward cpu float32"

# Tracked reductions (max_last, sum_last, mean, max_all/min_all) with products.
cat > "$TMP/math-native-reductions-aot.qui" <<'QUI'
import math

tensor<real32> x := tensor.ones<real32>([3, 4]).track()
tensor<real32> w := (tensor.ones<real32>([2, 4]) * real32(0.5)).track()
tensor<real32> logits := math.matmul(x, w.transpose(0, 1))
tensor<real32> shifted := logits - math.max_last(logits)
tensor<real32> loss := math.mean(math.sum_last(shifted * shifted) + math.sum_last(logits))
loss.backward(&x, &w)
println(loss.untrack().item() == real32(4))
println(w.grad[1, 3].item() == real32(1))
println(math.abs(x.grad[2, 0].item() - real32(1) / real32(3)) < real32(0.000001))
println(math.max_all(w.grad).item() == real32(1) and math.min_all(w.grad).item() == real32(1))
QUI
check_native_aot math-native-reductions-aot 4 \
    "max_last cpu float32" "sum_last cpu float32" "mean cpu float32" \
    "max_all cpu float32" "min_all cpu float32" "reduce-backward cpu float32" \
    "extrema-backward cpu float32"

# Views: tracked non-contiguous inputs (value-only copies with the view as the
# autograd parent) and results consumed through transposed views.
cat > "$TMP/math-native-views-aot.qui" <<'QUI'
import math

tensor<real32> x := tensor.ones<real32>([4, 3]).track()
tensor<real32> xv := x.transpose(0, 1)
tensor<real32> loss := math.sum(math.max_last(xv)) + math.sum(math.sum_last(x).transpose(0, 1))
loss.backward(&x)
println(loss.untrack().item() == real32(48))
println(x.grad[0, 0].item() == real32(7) and x.grad[1, 0].item() == real32(3) and x.grad[3, 2].item() == real32(3))
tensor<real32> z := tensor.ones<real32>([2, 3, 2]).track()
tensor<real32> product := math.matmul(z.transpose(0, 2), tensor.ones<real32>([2, 5]))
math.sum(product).backward(&z)
println(product.untrack()[1, 2, 4].item() == real32(2) and z.grad[1, 2, 0].item() == real32(5))
QUI
check_native_aot math-native-views-aot 3 \
    "max_last cpu float32" "sum_last cpu float32" "matmul cpu float32"

echo "math AOT and REPL/JIT integration: ok"

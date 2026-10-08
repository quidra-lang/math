#!/usr/bin/env bash
set -euo pipefail

QUIDRA="${1:?usage: integration.sh /path/to/quidra}"
REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/math.qui" <<'QUI'
import math

float pi_fast = math.pi
bigreal pi_exact = math.pi
float e_fast = math.e
bigreal e_exact = math.e
print(pi_fast > 3.0)
print(NL)
print(pi_exact > bigreal(3))
print(NL)
print(e_fast > 2.0)
print(NL)
print(e_exact > bigreal(2))
print(NL)
print(pi_exact > e_exact)
print(NL)
bigreal exact_sqrt = math.sqrt(bigreal(4))
bigreal exact_log = math.log(bigreal(1))
print(exact_sqrt.string() == "2.0")
print(NL)
print(math.sin(bigreal(0)).string() == "0.0")
print(NL)
print(math.cos(bigreal(0)).string() == "1.0")
print(NL)
print(math.tan(bigreal(0)).string() == "0.0")
print(NL)
print(exact_log.string() == "0.0")
print(NL)
print(math.exp(bigreal(0)).string() == "1.0")
print(NL)

tensor<float32> left = tensor.ones<float32>([2, 3])
tensor<float32> right = tensor.ones<float32>([3, 2])
tensor<float32> product = math.matmul(left, right)
print(product.shape()[0] == 2 and product.shape()[1] == 2)
print(NL)
print(product[1, 1].item() == float32(3))
print(NL)

tensor<int32> integer_left = tensor.ones<int32>([1, 2])
tensor<int32> integer_right = tensor.ones<int32>([2, 1])
tensor<int32> integer_product = math.matmul(integer_left, integer_right)
print(integer_product[0, 0].item() == int32(2))
print(NL)

tensor<int32> integer_vector = tensor.ones<int32>([2])
print(math.dot(integer_vector, integer_vector) == int32(2))
print(NL)

tensor<float32> vector = tensor.ones<float32>([3])
print(math.dot(vector, vector) == float32(3))
print(NL)

print(math.abs(float(-5.0)) == float(5.0))
print(NL)
float sqrt64 = math.sqrt(float(9.0))
float32 sqrt32 = math.sqrt(float32(9.0))
print(sqrt64 == float(3.0))
print(NL)
print(sqrt32 == float32(3.0))
print(NL)
print(math.min(float(2.0), float(3.0)) == float(2.0))
print(NL)
print(math.max(float(2.0), float(3.0)) == float(3.0))
print(NL)
print(math.is_finite(float(1.0)))
print(NL)
int truncated = math.trunc(float(-1.75))
int rounded_positive = math.round(float(1.5))
int rounded_negative = math.round(float(-1.5))
int floored = math.floor(float(-1.25))
int ceiled = math.ceil(float(-1.25))
print(truncated == int(-1))
print(NL)
print(rounded_positive == int(2))
print(NL)
print(rounded_negative == int(-2))
print(NL)
print(floored == int(-2))
print(NL)
print(ceiled == int(-1))
print(NL)
print(math.pow(float(2.0), float(3.0)) == float(8.0))
print(NL)

tensor<float32> unary_values = tensor.zeros<float32>([4])
unary_values[0] = float32(2)
unary_values[1] = float32(4)
unary_values[2] = float32(1)
unary_values[3] = float32(0)
print(math.abs(unary_values)[0].item() == float32(2))
print(NL)
print(math.sqrt(unary_values)[1].item() == float32(2))
print(NL)
print(math.log(unary_values)[2].item() == float32(0))
print(NL)
print(math.exp(unary_values)[3].item() == float32(1))
print(NL)

tensor<float32> abs_source = tensor.zeros<float32>([1])
abs_source[0] = float32(-2)
tensor<float32> abs_tracked = abs_source.track()
math.sum(math.abs(abs_tracked)).backward(&abs_tracked)
print(abs_tracked.grad[0].item() == float32(-1))
print(NL)

tensor<float32> sqrt_source = tensor.zeros<float32>([1])
sqrt_source[0] = float32(4)
tensor<float32> sqrt_tracked = sqrt_source.track()
math.sum(math.sqrt(sqrt_tracked)).backward(&sqrt_tracked)
print(sqrt_tracked.grad[0].item() == float32(0.25))
print(NL)

tensor<float32> log_source = tensor.zeros<float32>([1])
log_source[0] = float32(2)
tensor<float32> log_tracked = log_source.track()
math.sum(math.log(log_tracked)).backward(&log_tracked)
print(log_tracked.grad[0].item() == float32(0.5))
print(NL)

tensor<float32> exp_source = tensor.zeros<float32>([1]).track()
math.sum(math.exp(exp_source)).backward(&exp_source)
print(exp_source.grad[0].item() == float32(1))
print(NL)

tensor<float32> higher_unary = tensor.ones<float32>([]).track()
math.log(higher_unary).backward(&higher_unary, track = true)
tensor<float32> higher_unary_first = higher_unary.grad
print(higher_unary_first.is_tracked())
print(NL)
print(higher_unary_first.untrack().item() == float32(1))
print(NL)
higher_unary.clear_grad()
higher_unary_first.backward(&higher_unary)
print(higher_unary.grad.untrack().item() == float32(-1))
print(NL)

tensor<float32> higher_exp = tensor.zeros<float32>([]).track()
math.exp(higher_exp).backward(&higher_exp, track = true)
tensor<float32> higher_exp_first = higher_exp.grad
print(higher_exp_first.is_tracked())
print(NL)
print(higher_exp_first.untrack().item() == float32(1))
print(NL)
higher_exp.clear_grad()
higher_exp_first.backward(&higher_exp)
print(higher_exp.grad.untrack().item() == float32(1))
print(NL)

tensor<float32> higher_sqrt = (tensor.ones<float32>([]) * float32(4)).track()
math.sqrt(higher_sqrt).backward(&higher_sqrt, track = true)
tensor<float32> higher_sqrt_first = higher_sqrt.grad
print(higher_sqrt_first.is_tracked())
print(NL)
print(higher_sqrt_first.untrack().item() == float32(0.25))
print(NL)
higher_sqrt.clear_grad()
higher_sqrt_first.backward(&higher_sqrt)
print(higher_sqrt.grad.untrack().item() == float32(-0.03125))
print(NL)

tensor<float32> higher_abs = (tensor.ones<float32>([]) * float32(-2)).track()
math.abs(higher_abs).backward(&higher_abs, track = true)
tensor<float32> higher_abs_first = higher_abs.grad
print(higher_abs_first.is_tracked())
print(NL)
print(higher_abs_first.untrack().item() == float32(-1))
print(NL)
higher_abs.clear_grad()
higher_abs_first.backward(&higher_abs)
print(higher_abs.grad.untrack().item() == float32(0))
print(NL)

tensor<float32> reduction_values = tensor.ones<float32>([2, 3])
print(math.sum(reduction_values).item() == float32(6))
print(NL)
print(math.mean(reduction_values).item() == float32(1))
print(NL)
print(math.sum_last(reduction_values)[0, 0].item() == float32(3))
print(NL)

tensor<int32> integer_sum_values = tensor.ones<int32>([2, 3])
print(math.sum(integer_sum_values).item() == int32(6))
print(NL)

tensor<float32> empty_sum_values = tensor.zeros<float32>([0])
print(math.sum(empty_sum_values).item() == float32(0))
print(NL)

tensor<float32> empty_tracked = tensor.zeros<float32>([0]).track()
tensor<float32> empty_total = math.sum(empty_tracked)
empty_total.backward(&empty_tracked, track = true)
print(empty_tracked.grad.is_tracked())
print(NL)
print(empty_tracked.grad.shape()[0] == 0)
print(NL)

tensor<float32> tracked_whole = tensor.ones<float32>([2, 3]).track()
(math.sum(tracked_whole) * math.sum(tracked_whole)).backward(
    &tracked_whole,
    track = true
)
tensor<float32> tracked_whole_first = tracked_whole.grad
print(tracked_whole_first.is_tracked())
print(NL)
print(tracked_whole_first.untrack()[1, 2].item() == float32(12))
print(NL)
tracked_whole.clear_grad()
math.mean(tracked_whole_first).backward(&tracked_whole)
print(tracked_whole.grad.untrack()[0, 0].item() == float32(2))
print(NL)

tensor<int32> integer_reduction = tensor.ones<int32>([2, 2])
print(math.max_last(integer_reduction)[1, 1].item() == int32(1))
print(NL)
print(math.min_last(integer_reduction)[0, 0].item() == int32(1))
print(NL)

tensor<float32> tie_values = tensor.zeros<float32>([1, 3])
tie_values[0, 0] = float32(2)
tie_values[0, 1] = float32(2)
tie_values[0, 2] = float32(1)
tensor<float32> tie_tracked = tie_values.track()
math.mean(math.max_last(tie_tracked)).backward(&tie_tracked)
print(tie_tracked.grad[0, 0].item() == float32(1))
print(NL)
print(tie_tracked.grad[0, 1].item() == float32(0))
print(NL)

tensor<float32> sum_tracked = tensor.ones<float32>([1, 3]).track()
math.mean(math.sum_last(sum_tracked)).backward(&sum_tracked)
tensor<float32> sum_last_grad = sum_tracked.grad.untrack()
print(sum_last_grad[0, 2].item() == float32(1))
print(NL)

tensor<float32> higher_extrema_values = tensor.zeros<float32>([2])
higher_extrema_values[0] = float32(2)
higher_extrema_values[1] = float32(1)
tensor<float32> higher_extrema = higher_extrema_values.track()
tensor<float32> higher_max = math.max_last(higher_extrema)
math.mean((higher_max * higher_max)).backward(&higher_extrema, track = true)
tensor<float32> higher_max_first = higher_extrema.grad
print(higher_max_first.is_tracked())
print(NL)
print(higher_max_first.untrack()[0].item() == float32(4))
print(NL)
print(higher_max_first.untrack()[1].item() == float32(0))
print(NL)
math.mean(higher_max_first).backward(&higher_extrema)
tensor<float32> higher_max_accumulated = higher_extrema.grad.untrack()
print(higher_max_accumulated[0].item() == float32(5))
print(NL)
print(higher_max_accumulated[1].item() == float32(0))
print(NL)

tensor<float32> scalar_product = math.matmul(vector, vector)
print(len(scalar_product.shape()) == 0)
print(NL)
print(scalar_product.item() == float32(3))
print(NL)

tensor<float32> tracked_left = tensor.ones<float32>([2, 3]).track()
tensor<float32> tracked_right = tensor.ones<float32>([3, 2]).track()
tensor<float32> tracked_product = math.matmul(tracked_left, tracked_right)
math.mean(tracked_product).backward(&tracked_left, &tracked_right)
print(tracked_left.grad[0, 0].item() == float32(0.5))
print(NL)
print(tracked_right.grad[0, 0].item() == float32(0.5))
print(NL)

tensor<float32> fc_input = tensor.ones<float32>([1, 3]).track()
tensor<float32> fc_weight = tensor.ones<float32>([2, 3]).track()
tensor<float32> fc_product = math.matmul(
    fc_input,
    fc_weight.transpose(0, 1)
)
math.mean(fc_product).backward(&fc_input, &fc_weight)
print(fc_input.grad[0, 0].item() == float32(1))
print(NL)
print(fc_weight.grad[0, 0].item() == float32(0.5))
print(NL)

tensor<float32> batched_left = tensor.ones<float32>([2, 2, 3]).track()
tensor<float32> batched_right = tensor.ones<float32>([3, 4]).track()
tensor<float32> batched = math.matmul(batched_left, batched_right)
math.mean(batched).backward(&batched_left, &batched_right)
print(batched.shape()[0] == 2 and batched.shape()[1] == 2 and batched.shape()[2] == 4)
print(NL)
print(batched_left.grad[1, 1, 2].item() == float32(0.25))
print(NL)
print(batched_right.grad[2, 3].item() == float32(0.25))
print(NL)

tensor<float32> higher_left = tensor.ones<float32>([1, 2]).track()
tensor<float32> higher_right = tensor.ones<float32>([2, 1]).track()
tensor<float32> higher = math.matmul(higher_left, higher_right)
math.mean((higher * higher)).backward(&higher_left, &higher_right, track = true)
print(higher_left.grad.is_tracked())
print(NL)
higher_right.clear_grad()
math.mean(higher_left.grad).backward(&higher_right)
print(higher_right.grad[0, 0].item() == float32(4))
print(NL)


QUI

expected="$(printf 'true\n%.0s' {1..79})"
set +e
actual="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/math.qui" 2>&1)"
math_status=$?
set -e
if [[ "$math_status" -ne 0 ]]; then
    echo "Math integration program failed with status $math_status:" >&2
    printf '%s\n' "$actual" >&2
    exit 1
fi
if [[ "$actual" != "$expected" ]]; then
    echo "unexpected Math integration output:" >&2
    printf '%s\n' "$actual" >&2
    exit 1
fi

cat > "$TMP/round-range.qui" <<'QUI'
import math
int value = math.trunc(float(1.0e30))
print(value)
print(NL)
QUI
set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/round-range.qui" >"$TMP/round-range.out" 2>&1
round_range_status=$?
set -e
if [[ "$round_range_status" -ne 101 ]]; then
    echo "out-of-range Math rounding unexpectedly succeeded" >&2
    cat "$TMP/round-range.out" >&2
    exit 1
fi

cat > "$TMP/integer-transcendental.qui" <<'QUI'
import math
int value = 1
auto result = math.sin(value)
print(result)
print(NL)
QUI
set +e
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/integer-transcendental.qui" \
    >"$TMP/integer-transcendental.out" 2>&1
integer_transcendental_status=$?
set -e
if [[ "$integer_transcendental_status" -eq 0 ]]; then
    echo "integer Math transcendental unexpectedly type-checked" >&2
    cat "$TMP/integer-transcendental.out" >&2
    exit 1
fi

cat > "$TMP/whole-extrema.qui" <<'QUI'
import math

tensor<int32> integer_values = tensor.zeros<int32>([3])
integer_values[0] = int32(5)
integer_values[1] = int32(-2)
integer_values[2] = int32(4)
print(math.max_all(integer_values).item() == int32(5))
print(NL)
print(math.min_all(integer_values).item() == int32(-2))
print(NL)

tensor<float32> tie_values = tensor.zeros<float32>([3])
tie_values[0] = float32(2)
tie_values[1] = float32(2)
tie_values[2] = float32(1)
tensor<float32> tracked = tie_values.track()
tensor<float32> maximum = math.max_all(tracked)
print(maximum.untrack().item() == float32(2))
print(NL)
maximum.backward(&tracked, track = true)
tensor<float32> first = tracked.grad
print(first.is_tracked())
print(NL)
print(first.untrack()[0].item() == float32(1))
print(NL)
print(first.untrack()[1].item() == float32(0))
print(NL)

tensor<float32> minimum_values = tensor.zeros<float32>([3])
minimum_values[0] = float32(1)
minimum_values[1] = float32(-3)
minimum_values[2] = float32(-3)
tensor<float32> tracked_min = minimum_values.track()
math.min_all(tracked_min).backward(&tracked_min)
print(tracked_min.grad[1].item() == float32(1))
print(NL)
print(tracked_min.grad[2].item() == float32(0))
print(NL)
QUI

set +e
whole_extrema_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/whole-extrema.qui" 2>&1)"
whole_extrema_status=$?
set -e
whole_extrema_expected="$(printf 'true\n%.0s' {1..8})"
if [[ "$whole_extrema_status" -ne 0 ]]; then
    echo "whole-tensor extrema program failed with status $whole_extrema_status:" >&2
    printf '%s\n' "$whole_extrema_output" >&2
    exit 1
fi
if [[ "$whole_extrema_output" != "$whole_extrema_expected" ]]; then
    echo "unexpected whole-tensor extrema output:" >&2
    printf '%s\n' "$whole_extrema_output" >&2
    exit 1
fi

# Whole-tensor extrema must resolve to their own compiler-extension operation
# identities. Scalar math.max/math.min are separate APIs and must never be
# mistaken for tensor reductions by the generic tensor-region optimizer.
set +e
extrema_ir="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" ir "$TMP/whole-extrema.qui" 2>&1)"
extrema_ir_status=$?
set -e
if [[ "$extrema_ir_status" -ne 0 ]]; then
    echo "whole-tensor extrema typed-IR generation failed with status $extrema_ir_status:" >&2
    printf '%s\n' "$extrema_ir" >&2
    exit 1
fi
for marker in "math.graph:max_all" "math.graph:min_all"; do
    if ! grep -Fq "$marker" <<< "$extrema_ir"; then
        echo "Math extrema compiler operation marker missing: $marker" >&2
        printf '%s\n' "$extrema_ir" >&2
        exit 1
    fi
done

for operation in max min; do
    cat > "$TMP/empty-extrema.qui" <<QUI
import math
tensor<float32> empty = tensor.zeros<float32>([0])
tensor<float32> value = math.${operation}_all(empty)
print(value.item())
print(NL)
QUI
    set +e
    QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/empty-extrema.qui" \
        >"$TMP/empty-extrema.out" 2>&1
    empty_extrema_status=$?
    set -e
    if [[ "$empty_extrema_status" -ne 101 ]]; then
        echo "empty math.${operation}_all unexpectedly succeeded" >&2
        cat "$TMP/empty-extrema.out" >&2
        exit 1
    fi
done

# Compiler extensions are package-owned semantics carried through Core's
# domain-neutral typed-IR pipeline. Importing Math must register its
# descriptor without teaching Core what matmul or linear algebra means.
set +e
ir_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" ir "$TMP/math.qui" 2>&1)"
ir_status=$?
set -e
if [[ "$ir_status" -ne 0 ]]; then
    echo "Math typed-IR generation failed with status $ir_status:" >&2
    printf '%s\n' "$ir_output" >&2
    exit 1
fi
if ! grep -Fq "compiler-extension math.graph" <<< "$ir_output"; then
    echo "Math compiler extension was not registered in typed IR" >&2
    printf '%s\n' "$ir_output" >&2
    exit 1
fi

# Exercise the real package-import alias path all the way through Core's
# domain-neutral optimizer: descriptor discovery -> alias-resolved package call
# -> operation ID -> tensor region. Executable numerical backend selection stays
# in Math's implementation; descriptor-only optimization tables are forbidden.
cat > "$TMP/compiler-alias.qui" <<'QUI'
import numerical = math

tensor<float32> left = tensor.ones<float32>([1, 2])
tensor<float32> right = tensor.ones<float32>([2, 1])
tensor<float32> product = numerical.matmul(left, right)
print(product[0, 0].item())
print(NL)
QUI
set +e
alias_ir="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" ir "$TMP/compiler-alias.qui" 2>&1)"
alias_status=$?
set -e
if [[ "$alias_status" -ne 0 ]]; then
    echo "Math alias typed-IR generation failed with status $alias_status:" >&2
    printf '%s\n' "$alias_ir" >&2
    exit 1
fi
for marker in \
    "compiler-extension math.graph" \
    "operations=math.graph:matmul" \
    "tables=math.graph:"
do
    if ! grep -Fq "$marker" <<< "$alias_ir"; then
        echo "Math compiler extension E2E marker missing: $marker" >&2
        printf '%s\n' "$alias_ir" >&2
        exit 1
    fi
done

# Runs one Quidra program that prints a "true" line per check.
expect_true_lines() {
    local label="$1"
    local program="$2"
    local count="$3"
    local output
    local status
    set +e
    output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$program" 2>&1)"
    status=$?
    set -e
    local expected
    expected="$(printf 'true\n%.0s' $(seq 1 "$count"))"
    if [[ "$status" -ne 0 || "$output" != "$expected" ]]; then
        echo "unexpected $label output (status $status):" >&2
        printf '%s\n' "$output" >&2
        exit 1
    fi
}

# Math-native matrix products on the CPU: finite-difference gradients (both or
# one side tracked, transposed views), backward(track = true) Hessian-vector
# products with cross terms and third order, rank rules, bit-exact agreement
# with the portable "round each product, add in order" semantics, and
# untracked non-contiguous operands.
cat > "$TMP/native-matmul.qui" <<'QUI'
import math

// Deterministic, non-symmetric test data.
tensor<float> pattern64(int rows, int columns, int seed)
    tensor<float> value = tensor.zeros<float>([rows, columns])
    for i in range(rows)
        for j in range(columns)
            int h = (i * 131 + j * 71 + seed * 17) % 97
            value[i, j] = float(h) / 48.0 - 1.0
    return value

tensor<float32> pattern32(int rows, int columns, int seed)
    tensor<float32> value = tensor.zeros<float32>([rows, columns])
    for i in range(rows)
        for j in range(columns)
            int h = (i * 131 + j * 71 + seed * 17) % 97
            value[i, j] = float32(h) / float32(48) - float32(1)
    return value

float max_abs64(tensor<float> value)
    return math.max_all(math.abs(value)).item()

float loss64(tensor<float> a, tensor<float> b, tensor<float> w)
    return math.sum(math.matmul(a, b) * w).item()

// Central differences of loss64 with respect to every element of `a` (or `b`).
tensor<float> numeric_gradient(tensor<float> a, tensor<float> b, tensor<float> w, bool left)
    tensor<float> target = b
    if left
        target = a
    int rows = target.shape()[0]
    int columns = target.shape()[1]
    tensor<float> result = tensor.zeros<float>([rows, columns])
    float eps = 0.000001
    for i in range(rows)
        for j in range(columns)
            tensor<float> step = tensor.zeros<float>([rows, columns])
            step[i, j] = eps
            float plus = 0.0
            float minus = 0.0
            if left
                plus = loss64(a + step, b, w)
                minus = loss64(a - step, b, w)
            else
                plus = loss64(a, b + step, w)
                minus = loss64(a, b - step, w)
            result[i, j] = (plus - minus) / (2.0 * eps)
    return result

// 1-3: first-order gradients (both tracked) match finite differences.
tensor<float> a64 = pattern64(3, 4, 1)
tensor<float> b64 = pattern64(4, 5, 2)
tensor<float> w64 = pattern64(3, 5, 3)
tensor<float> ta = a64.track()
tensor<float> tb = b64.track()
math.sum(math.matmul(ta, tb) * w64).backward(&ta, &tb)
print(max_abs64(ta.grad - numeric_gradient(a64, b64, w64, true)) < 0.000001)
print(NL)
print(max_abs64(tb.grad - numeric_gradient(a64, b64, w64, false)) < 0.000001)
print(NL)
print(math.matmul(ta, tb).is_tracked())
print(NL)

// 4-5: one side tracked.
tensor<float> only_a = a64.track()
math.sum(math.matmul(only_a, b64) * w64).backward(&only_a)
print(max_abs64(only_a.grad - numeric_gradient(a64, b64, w64, true)) < 0.000001)
print(NL)
tensor<float> only_b = b64.track()
math.sum(math.matmul(a64, only_b) * w64).backward(&only_b)
print(max_abs64(only_b.grad - numeric_gradient(a64, b64, w64, false)) < 0.000001)
print(NL)

// 6-7: FC/Conv pattern -- tracked right operand is a transposed view.
tensor<float> weight64 = pattern64(5, 4, 4)
tensor<float> tw = weight64.track()
tensor<float> tx = a64.track()
math.sum(math.matmul(tx, tw.transpose(0, 1)) * w64).backward(&tx, &tw)
tensor<float> weight_numeric = numeric_gradient(a64, weight64.transpose(0, 1).contiguous(), w64, false)
print(max_abs64(tw.grad - weight_numeric.transpose(0, 1)) < 0.000001)
print(NL)
print(max_abs64(tx.grad - numeric_gradient(a64, weight64.transpose(0, 1).contiguous(), w64, true)) < 0.000001)
print(NL)

// 8: tracked transposed left operand.
tensor<float> left_storage = pattern64(4, 3, 5).track()
math.sum(math.matmul(left_storage.transpose(0, 1), b64) * w64).backward(&left_storage)
tensor<float> left_numeric = numeric_gradient(left_storage.untrack().transpose(0, 1).contiguous(), b64, w64, true)
print(max_abs64(left_storage.grad - left_numeric.transpose(0, 1)) < 0.000001)
print(NL)

// 9: a tracked rank-3 view (not a rank-2 transpose) is read natively through
// a value-only contiguous copy while the view stays the autograd parent, so
// its gradient flows back through the view.
tensor<float> cube = tensor.ones<float>([2, 3, 4]).track()
tensor<float> cube_view = cube.transpose(0, 2)
tensor<float> cube_right = pattern64(2, 3, 6)
math.sum(math.matmul(cube_view, cube_right)).backward(&cube)
print(cube.grad[1, 2, 3].item() == cube_right[1, 0].item() + cube_right[1, 1].item() + cube_right[1, 2].item())
print(NL)

// 10-11: backward(track = true) second derivatives (Hessian-vector product,
// including the cross terms) match finite differences of the first gradient.
tensor<float> va = pattern64(3, 4, 7)
tensor<float> vb = pattern64(4, 5, 8)
tensor<float> ha = a64.track()
tensor<float> hb = b64.track()
tensor<float> hp = math.matmul(ha, hb)
math.sum(hp * hp * w64).backward(&ha, &hb, track = true)
tensor<float> ga = ha.grad
tensor<float> gb = hb.grad
ha.clear_grad()
hb.clear_grad()
(math.sum(ga * va) + math.sum(gb * vb)).backward(&ha, &hb)
float eps = 0.000001
tensor<float> pa = (a64 + va * eps).track()
tensor<float> pb = (b64 + vb * eps).track()
tensor<float> pp = math.matmul(pa, pb)
math.sum(pp * pp * w64).backward(&pa, &pb)
tensor<float> ma = (a64 - va * eps).track()
tensor<float> mb = (b64 - vb * eps).track()
tensor<float> mp = math.matmul(ma, mb)
math.sum(mp * mp * w64).backward(&ma, &mb)
tensor<float> hessian_a = (pa.grad - ma.grad) / (2.0 * eps)
tensor<float> hessian_b = (pb.grad - mb.grad) / (2.0 * eps)
print(max_abs64(ha.grad.untrack() - hessian_a) < 0.00001)
print(NL)
print(max_abs64(hb.grad.untrack() - hessian_b) < 0.00001)
print(NL)

// 12: third order stays available (tracked second gradient).
tensor<float> cubic = tensor.ones<float>([1, 1]).track()
tensor<float> scale = tensor.ones<float>([1, 1]) * 3.0
tensor<float> cubic_value = math.matmul(math.matmul(cubic, cubic), math.matmul(cubic, scale))
math.sum(cubic_value).backward(&cubic, track = true)
tensor<float> first_cubic = cubic.grad
cubic.clear_grad()
math.sum(first_cubic).backward(&cubic, track = true)
tensor<float> second_cubic = cubic.grad
cubic.clear_grad()
math.sum(second_cubic).backward(&cubic)
print(first_cubic.untrack()[0, 0].item() == 9.0 and second_cubic.untrack()[0, 0].item() == 18.0 and cubic.grad[0, 0].item() == 18.0)
print(NL)

// 13-16: rank rules.
tensor<float32> vector3 = tensor.ones<float32>([3])
print(len(math.matmul(vector3, vector3).shape()) == 0)
print(NL)
tensor<float32> matrix23 = pattern32(2, 3, 1)
tensor<float32> mv = math.matmul(matrix23, vector3)
print(len(mv.shape()) == 1 and mv.shape()[0] == 2)
print(NL)
tensor<float32> batched = math.matmul(tensor.ones<float32>([2, 5, 3]), pattern32(3, 4, 2))
print(len(batched.shape()) == 3 and batched.shape()[0] == 2 and batched.shape()[1] == 5 and batched.shape()[2] == 4)
print(NL)
tensor<float32> batched_vector = math.matmul(tensor.ones<float32>([2, 5, 3]), vector3)
print(len(batched_vector.shape()) == 2 and batched_vector.shape()[1] == 5)
print(NL)

// 17: CPU products equal the portable "round each product, add k in order"
// semantics bit for bit.
tensor<float32> exact_left = pattern32(7, 19, 3)
tensor<float32> exact_right = pattern32(19, 11, 4)
tensor<float32> exact_product = math.matmul(exact_left, exact_right)
bool exact_ok = true
for i in range(7)
    for j in range(11)
        float32 total = exact_left[i, 0].item() * exact_right[0, j].item()
        for k in range(1, 19)
            float32 product = exact_left[i, k].item() * exact_right[k, j].item()
            total = total + product
        if exact_product[i, j].item() != total
            exact_ok = false
print(exact_ok)
print(NL)

// 18-19: untracked non-contiguous operands are read in place / materialized.
tensor<float32> transposed_input = pattern32(3, 7, 5).transpose(0, 1)
tensor<float32> via_view = math.matmul(transposed_input, pattern32(3, 2, 6))
tensor<float32> via_copy = math.matmul(transposed_input.contiguous(), pattern32(3, 2, 6))
print(math.max_all(math.abs(via_view - via_copy)).item() == float32(0))
print(NL)
print(math.dot(pattern32(1, 4, 1).reshape([4]), pattern32(4, 1, 2).reshape([4])) == math.matmul(pattern32(1, 4, 1), pattern32(4, 1, 2))[0, 0].item())
print(NL)
QUI
expect_true_lines "Math native matmul" "$TMP/native-matmul.qui" 19

# Whole-tensor sum/mean keep the portable pairwise tree; mean backward is
# upstream / count as one node.
cat > "$TMP/native-sum-mean.qui" <<'QUI'
import math

// Deterministic, non-symmetric test data.
tensor<float> pattern64(int rows, int columns, int seed)
    tensor<float> value = tensor.zeros<float>([rows, columns])
    for i in range(rows)
        for j in range(columns)
            int h = (i * 131 + j * 71 + seed * 17) % 97
            value[i, j] = float(h) / 48.0 - 1.0
    return value

tensor<float32> pattern32(int rows, int columns, int seed)
    tensor<float32> value = tensor.zeros<float32>([rows, columns])
    for i in range(rows)
        for j in range(columns)
            int h = (i * 131 + j * 71 + seed * 17) % 97
            value[i, j] = float32(h) / float32(48) - float32(1)
    return value

float max_abs64(tensor<float> value)
    return math.max_all(math.abs(value)).item()

// 1-3: whole-tensor sum/mean keep the portable pairwise tree.
tensor<float32> pairwise = tensor.zeros<float32>([4])
pairwise[0] = float32(100000000)
pairwise[1] = float32(1)
pairwise[2] = float32(-100000000)
pairwise[3] = float32(1)
print(math.sum(pairwise).item() == float32(0))
print(NL)
print(math.mean(pairwise).item() == float32(0))
print(NL)
tensor<float32> odd = tensor.ones<float32>([7]) * float32(0.1)
print(math.sum(odd).item() == ((float32(0.1) + float32(0.1)) + (float32(0.1) + float32(0.1))) + ((float32(0.1) + float32(0.1)) + float32(0.1)))
print(NL)

// 4: mean backward is upstream / N as one node.
tensor<float> mean_input = pattern64(3, 5, 9).track()
math.mean(mean_input).backward(&mean_input)
print(max_abs64(mean_input.grad - tensor.ones<float>([3, 5]) / 15.0) == 0.0)
print(NL)
QUI
expect_true_lines "Math native sum/mean" "$TMP/native-sum-mean.qui" 4

# Last-axis reductions keep the input shape; extrema are first-occurrence with
# strict comparison (ties and NaN), and gradients go only to the winner.
cat > "$TMP/native-last-axis.qui" <<'QUI'
import math

// Deterministic, non-symmetric test data.
tensor<float> pattern64(int rows, int columns, int seed)
    tensor<float> value = tensor.zeros<float>([rows, columns])
    for i in range(rows)
        for j in range(columns)
            int h = (i * 131 + j * 71 + seed * 17) % 97
            value[i, j] = float(h) / 48.0 - 1.0
    return value

tensor<float32> pattern32(int rows, int columns, int seed)
    tensor<float32> value = tensor.zeros<float32>([rows, columns])
    for i in range(rows)
        for j in range(columns)
            int h = (i * 131 + j * 71 + seed * 17) % 97
            value[i, j] = float32(h) / float32(48) - float32(1)
    return value

// 1-2: sum_last keeps the input shape and adds each row left to right.
tensor<float32> row_values = pattern32(3, 4, 2)
tensor<float32> row_sums = math.sum_last(row_values)
print(len(row_sums.shape()) == 2 and row_sums.shape()[0] == 3 and row_sums.shape()[1] == 4)
print(NL)
float32 row1 = ((row_values[1, 0].item() + row_values[1, 1].item()) + row_values[1, 2].item()) + row_values[1, 3].item()
print(row_sums[1, 0].item() == row1 and row_sums[1, 3].item() == row1)
print(NL)

// 3-6: first-occurrence ties with strict comparison, NaN handling.
tensor<float32> ties = tensor.zeros<float32>([2, 4])
ties[0, 0] = float32(1)
ties[0, 1] = float32(5)
ties[0, 2] = float32(5)
ties[0, 3] = float32(-2)
ties[1, 0] = float32(-3)
ties[1, 1] = float32(-3)
ties[1, 2] = float32(4)
ties[1, 3] = float32(-3)
tensor<float32> tied = ties.track()
math.sum(math.max_last(tied) * pattern32(2, 4, 1)).backward(&tied)
tensor<float32> upstream = pattern32(2, 4, 1)
float32 row0_upstream = ((upstream[0, 0].item() + upstream[0, 1].item()) + upstream[0, 2].item()) + upstream[0, 3].item()
print(tied.grad[0, 1].item() == row0_upstream and tied.grad[0, 2].item() == float32(0) and tied.grad[0, 0].item() == float32(0))
print(NL)
tensor<float32> tied_min = ties.track()
math.sum(math.min_last(tied_min)).backward(&tied_min)
print(tied_min.grad[1, 0].item() == float32(4) and tied_min.grad[1, 1].item() == float32(0) and tied_min.grad[1, 3].item() == float32(0))
print(NL)
tensor<float32> nan_values = tensor.zeros<float32>([2, 3])
float32 zero32 = float32(0)
nan_values[0, 0] = zero32 / zero32
nan_values[0, 1] = float32(7)
nan_values[1, 0] = float32(2)
nan_values[1, 1] = zero32 / zero32
nan_values[1, 2] = float32(9)
tensor<float32> nan_max = math.max_last(nan_values)
print(not math.is_finite(nan_max[0, 2].item()) and nan_max[1, 0].item() == float32(9))
print(NL)
tensor<float32> whole = pattern32(4, 6, 3)
tensor<float32> whole_tracked = whole.track()
math.min_all(whole_tracked).backward(&whole_tracked)
print(math.sum(whole_tracked.grad).item() == float32(1) and whole_tracked.grad[0, 0].item() + float32(1) == float32(1) + whole_tracked.grad[0, 0].item())
print(NL)

// 7: float64 last-axis sums add left to right.
tensor<float> doubles = pattern64(5, 8, 2)
float double_row = doubles[2, 0].item()
for k in range(1, 8)
    double_row = double_row + doubles[2, k].item()
print(math.sum_last(doubles)[2, 5].item() == double_row)
print(NL)
QUI
expect_true_lines "Math native last-axis reductions" "$TMP/native-last-axis.qui" 7

# Reduction derivatives are Math custom nodes too: backward(track = true)
# through sum_last, max_last, and min_all stays differentiable.
cat > "$TMP/native-reduction-higher-order.qui" <<'QUI'
import math

// Deterministic, non-symmetric test data.
tensor<float> pattern64(int rows, int columns, int seed)
    tensor<float> value = tensor.zeros<float>([rows, columns])
    for i in range(rows)
        for j in range(columns)
            int h = (i * 131 + j * 71 + seed * 17) % 97
            value[i, j] = float(h) / 48.0 - 1.0
    return value

// 1-2: higher-order through sum_last and max_last stays package-owned.
tensor<float> hx = pattern64(2, 3, 4).track()
tensor<float> hs = math.sum_last(hx)
math.sum(hs * hs).backward(&hx, track = true)
tensor<float> hx_first = hx.grad
hx.clear_grad()
math.sum(hx_first).backward(&hx)
print(hx_first.is_tracked() and hx.grad[0, 0].item() == 18.0)
print(NL)
tensor<float> mx = pattern64(2, 3, 4).track()
tensor<float> mm = math.max_last(mx)
math.sum(mm * mm).backward(&mx, track = true)
tensor<float> mx_first = mx.grad
mx.clear_grad()
math.sum(mx_first).backward(&mx)
print(mx_first.is_tracked() and math.sum(mx.grad).item() == 12.0)
print(NL)

// 3: whole-tensor extrema: second derivative of min_all(x)^2 through the
// scatter/select adjoint pair.
tensor<float> ex = pattern64(3, 4, 5).track()
tensor<float> smallest = math.min_all(ex)
(smallest * smallest).backward(&ex, track = true)
tensor<float> ex_first = ex.grad
ex.clear_grad()
math.sum(ex_first).backward(&ex)
print(ex_first.is_tracked() and math.sum(ex.grad).item() == 2.0 and math.max_all(ex.grad).item() == 2.0)
print(NL)
QUI
expect_true_lines "Math native reduction higher-order" "$TMP/native-reduction-higher-order.qui" 3

# Non-contiguous views: a tracked view is read through a value-only contiguous
# copy while the view stays the autograd parent, and results consumed through
# views send their gradients back correctly. Everything equals the contiguous
# reference exactly.
cat > "$TMP/native-views.qui" <<'QUI'
import math

tensor<float32> pattern32(int rows, int columns, int seed)
    tensor<float32> value = tensor.zeros<float32>([rows, columns])
    for i in range(rows)
        for j in range(columns)
            int h = (i * 131 + j * 71 + seed * 17) % 97
            value[i, j] = float32(h) / float32(48) - float32(1)
    return value

tensor<float32> pattern3(int a, int b, int c, int seed)
    return pattern32(a * b, c, seed).reshape([a, b, c])

tensor<float> pattern64(int rows, int columns, int seed)
    tensor<float> value = tensor.zeros<float>([rows, columns])
    for i in range(rows)
        for j in range(columns)
            int h = (i * 131 + j * 71 + seed * 17) % 97
            value[i, j] = float(h) / 48.0 - 1.0
    return value

bool same32(tensor<float32> left, tensor<float32> right)
    return math.max_all(math.abs(left - right)).item() == float32(0)

bool same64(tensor<float> left, tensor<float> right)
    return math.max_all(math.abs(left - right)).item() == 0.0

// 1-2: a tracked rank-3 transposed left operand (neither contiguous nor a
// transposed matrix) keeps its graph; values and gradients equal the
// contiguous reference.
tensor<float32> x_data = pattern3(3, 4, 2, 1)
tensor<float32> w_data = pattern32(3, 5, 2)
tensor<float32> up = pattern3(2, 4, 5, 3)
tensor<float32> x = x_data.track()
tensor<float32> w = w_data.track()
tensor<float32> y = math.matmul(x.transpose(0, 2), w)
math.sum(y * up).backward(&x, &w)
tensor<float32> xr = x_data.transpose(0, 2).contiguous().track()
tensor<float32> wr = w_data.track()
tensor<float32> yr = math.matmul(xr, wr)
math.sum(yr * up).backward(&xr, &wr)
print(y.is_tracked() and same32(y.untrack(), yr.untrack()) and same32(w.grad, wr.grad))
print(NL)
print(same32(x.grad.transpose(0, 2), xr.grad))
print(NL)

// 3: backward(track = true) through the non-contiguous operand: the
// Hessian-vector product equals the contiguous reference.
tensor<float> hx_data = pattern64(6, 4, 4).reshape([2, 3, 4])
tensor<float> hw_data = pattern64(2, 3, 5)
tensor<float> hx = hx_data.track()
tensor<float> hw = hw_data.track()
tensor<float> hy = math.matmul(hx.transpose(0, 2), hw)
math.sum(hy * hy).backward(&hx, &hw, track = true)
tensor<float> hw_first = hw.grad
hx.clear_grad()
math.sum(hw_first * hw_first).backward(&hx)
tensor<float> rx = hx_data.transpose(0, 2).contiguous().track()
tensor<float> rw = hw_data.track()
tensor<float> ry = math.matmul(rx, rw)
math.sum(ry * ry).backward(&rx, &rw, track = true)
tensor<float> rw_first = rw.grad
rx.clear_grad()
math.sum(rw_first * rw_first).backward(&rx)
print(hw_first.is_tracked() and same64(hw_first.untrack(), rw_first.untrack()) and same64(hx.grad.transpose(0, 2), rx.grad))
print(NL)

// 4-5: every reduction of a tracked transposed input equals the contiguous
// reference, values and gradients.
tensor<float32> r_data = pattern32(6, 5, 4)
tensor<float32> weights = pattern32(5, 6, 5)
tensor<float32> r = r_data.track()
tensor<float32> rv = r.transpose(0, 1)
tensor<float32> rc = r_data.transpose(0, 1).contiguous().track()
print(same32(math.sum_last(rv).untrack(), math.sum_last(rc).untrack()) and same32(math.max_last(rv).untrack(), math.max_last(rc).untrack()) and same32(math.min_last(rv).untrack(), math.min_last(rc).untrack()) and math.sum(rv).untrack().item() == math.sum(rc).untrack().item() and math.mean(rv).untrack().item() == math.mean(rc).untrack().item() and math.max_all(rv).untrack().item() == math.max_all(rc).untrack().item() and math.min_all(rv).untrack().item() == math.min_all(rc).untrack().item())
print(NL)
tensor<float32> view_loss = math.sum(math.sum_last(rv) * weights) + math.sum(math.max_last(rv) * weights) - math.sum(math.min_last(rv) * weights) + math.mean(rv) + math.sum(rv) + math.min_all(rv) * float32(3) + math.max_all(rv)
view_loss.backward(&r)
tensor<float32> copy_loss = math.sum(math.sum_last(rc) * weights) + math.sum(math.max_last(rc) * weights) - math.sum(math.min_last(rc) * weights) + math.mean(rc) + math.sum(rc) + math.min_all(rc) * float32(3) + math.max_all(rc)
copy_loss.backward(&rc)
print(same32(r.grad.transpose(0, 1), rc.grad))
print(NL)

// 6: results consumed through transposed views send the right gradients.
tensor<float32> q_data = pattern32(4, 7, 6)
tensor<float32> across = pattern32(7, 4, 7)
tensor<float32> square = pattern32(4, 4, 8)
tensor<float32> q = q_data.track()
tensor<float32> view_sum = math.sum(math.max_last(q).transpose(0, 1) * across) + math.sum(math.sum_last(q).transpose(0, 1) * across) + math.sum(math.matmul(q, q.transpose(0, 1)).transpose(0, 1) * square)
view_sum.backward(&q)
tensor<float32> qr = q_data.track()
tensor<float32> plain_sum = math.sum(math.max_last(qr) * across.transpose(0, 1)) + math.sum(math.sum_last(qr) * across.transpose(0, 1)) + math.sum(math.matmul(qr, qr.transpose(0, 1)) * square.transpose(0, 1))
plain_sum.backward(&qr)
print(same32(q.grad, qr.grad))
print(NL)
QUI
expect_true_lines "Math native views" "$TMP/native-views.qui" 6

# Whole-tensor sums keep the portable odd-carry definition (`x + x * 0`), so an
# infinity in a carried position yields NaN on every path.
cat > "$TMP/native-sum-carry.qui" <<'QUI'
import math

bool is_nan(float32 value)
    return value != value

// 1: odd-carry definition: [1, 1, inf] -> NaN (inf carried), [1, inf] and
// [inf, 1, 1, 1, 1] -> inf, on contiguous and view inputs alike.
float32 infinity = float32(1) / float32(0)
tensor<float32> carried = tensor.ones<float32>([1, 3])
carried[0, 2] = infinity
tensor<float32> paired = tensor.ones<float32>([1, 2])
paired[0, 1] = infinity
tensor<float32> leading = tensor.ones<float32>([1, 5])
leading[0, 0] = infinity
print(is_nan(math.sum(carried).item()) and is_nan(math.mean(carried).item()) and is_nan(math.sum(carried.track().transpose(0, 1)).untrack().item()) and math.sum(paired).item() == infinity and math.sum(leading).item() == infinity and math.mean(leading.track().transpose(0, 1)).untrack().item() == infinity)
print(NL)
QUI
expect_true_lines "Math native sum carry" "$TMP/native-sum-carry.qui" 1

# Shapes the native bridges do not claim keep the portable behaviour. Until
# Core stops discarding bare error() statements, the portable composition runs
# past its diagnostic, so both outcomes are accepted: the Math diagnostic, or
# exactly what the develop composition produces.
expect_diagnostic_or_develop() {
    local label="$1"
    local program="$2"
    local diagnostic="$3"
    local develop="$4"
    local output
    local status
    set +e
    output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$program" 2>&1)"
    status=$?
    set -e
    if [[ "$status" -ne 0 ]] && grep -Fq "$diagnostic" <<< "$output"; then
        return
    fi
    if [[ -n "$develop" && "$status" -eq 0 && "$output" == "$develop" ]]; then
        return
    fi
    if [[ -z "$develop" && "$status" -ne 0 ]]; then
        return
    fi
    echo "unexpected $label outcome (status $status):" >&2
    printf '%s\n' "$output" >&2
    exit 1
}

write_shape_case() {
    local name="$1"
    local body="$2"
    printf 'import math\n%s\n' "$body" > "$TMP/$name.qui"
}

write_shape_case inner-mismatch 'tensor<float32> c = math.matmul(tensor.ones<float32>([2, 3]), tensor.ones<float32>([4, 5]))
print("{c.shape()[0]} {c.shape()[1]} {c.gather([0], []).item()}{NL}")'
expect_diagnostic_or_develop "matmul inner mismatch" "$TMP/inner-mismatch.qui" \
    "math.matmul inner dimensions do not match" "2 5 3.0"
write_shape_case inner-mismatch-tracked 'tensor<float32> a = tensor.ones<float32>([2, 3]).track()
tensor<float32> b = tensor.ones<float32>([4, 5]).track()
math.sum(math.matmul(a, b)).backward(&a, &b)
print("{a.grad.gather([0], []).item()} {b.grad.gather([0], []).item()}{NL}")'
expect_diagnostic_or_develop "tracked matmul inner mismatch" "$TMP/inner-mismatch-tracked.qui" \
    "math.matmul inner dimensions do not match" "5.0 2.0"
# Zero extents never produce a native result: they stop with an error.
write_shape_case zero-rows 'tensor<float32> c = math.matmul(tensor.ones<float32>([0, 3]), tensor.ones<float32>([3, 2]))
print("{c.shape()[0]}{NL}")'
expect_diagnostic_or_develop "matmul zero rows" "$TMP/zero-rows.qui" \
    "math.matmul requires positive tensor extents" ""
write_shape_case zero-inner 'tensor<float32> c = math.matmul(tensor.ones<float32>([2, 0]), tensor.ones<float32>([0, 2]))
print("{c.shape()[0]}{NL}")'
expect_diagnostic_or_develop "matmul zero inner" "$TMP/zero-inner.qui" \
    "math.matmul inner dimensions do not match" ""
write_shape_case zero-columns 'tensor<float32> c = math.matmul(tensor.ones<float32>([2, 3]), tensor.ones<float32>([3, 0]))
print("{c.shape()[0]}{NL}")'
expect_diagnostic_or_develop "matmul zero columns" "$TMP/zero-columns.qui" \
    "math.matmul requires a positive output width" ""
write_shape_case empty-max 'tensor<float32> m = math.max_all(tensor.ones<float32>([0, 3]))
print("after{NL}")'
expect_diagnostic_or_develop "max_all of an empty tensor" "$TMP/empty-max.qui" \
    "math.max requires at least one element" ""

# Empty rows and zero-width rows keep their shapes; empty tracked sums stay
# differentiable.
cat > "$TMP/native-empty.qui" <<'QUI'
import math

tensor<float32> rows_empty = tensor.ones<float32>([0, 3])
tensor<float32> s = math.sum_last(rows_empty)
tensor<float32> m = math.max_last(rows_empty)
tensor<float32> n = math.min_last(rows_empty)
print(s.shape()[0] == 0 and s.shape()[1] == 3 and m.shape()[1] == 3 and n.shape()[1] == 3)
print(NL)
tensor<float32> tracked_empty = tensor.ones<float32>([0, 3]).track()
print(math.sum_last(tracked_empty).is_tracked() and math.max_last(tracked_empty).is_tracked())
print(NL)
tensor<float32> width_empty = tensor.ones<float32>([3, 0])
tensor<float32> ws = math.sum_last(width_empty)
tensor<float32> wm = math.max_last(width_empty)
print(ws.shape()[0] == 3 and ws.shape()[1] == 0 and wm.shape()[0] == 3 and wm.shape()[1] == 0)
print(NL)
tensor<float32> total = math.sum(tracked_empty)
total.backward(&tracked_empty)
print(total.untrack().item() == float32(0) and tracked_empty.grad.shape()[0] == 0 and tracked_empty.grad.shape()[1] == 3)
print(NL)
QUI
expect_true_lines "Math native empty reductions" "$TMP/native-empty.qui" 4

# Signed zeros: every native product equals the portable fold p0 + p1 + ... bit
# for bit, including the sign of exact-zero outputs (products that are all -0
# give -0, any +0 product gives +0), for tracked and untracked operands,
# depth blocks, transposed views, vectors, and float64.
cat > "$TMP/native-signed-zero.qui" <<'QUI'
import math

bool same_bits32(float32 x, float32 y)
    if x != x
        return y != y
    if x == float32(0)
        return y == float32(0) and (float32(1) / x < float32(0)) == (float32(1) / y < float32(0))
    return x == y

bool same_bits64(float x, float y)
    if x != x
        return y != y
    if x == 0.0
        return y == 0.0 and (1.0 / x < 0.0) == (1.0 / y < 0.0)
    return x == y

bool negative_zero32(float32 x)
    return x == float32(0) and float32(1) / x < float32(0)

// Row i: all negative (i % 3 == 0), all positive (1), or alternating (2).
// Column j: all +0 (j % 4 == 0), all -0 (1), mixed signed zeros (2), or
// small positive integers (3), so every sum is exact in any order.
tensor<float32> signed_left32(int rows, int inner)
    tensor<float32> value = tensor.zeros<float32>([rows, inner])
    for i in range(rows)
        for k in range(inner)
            float32 magnitude = float32(1 + (i + k) % 5)
            if i % 3 == 0 or (i % 3 == 2 and k % 2 == 0)
                magnitude = -magnitude
            value[i, k] = magnitude
    return value

tensor<float32> signed_right32(int inner, int columns)
    tensor<float32> value = tensor.zeros<float32>([inner, columns])
    float32 negative_zero = float32(0) * float32(-1)
    for k in range(inner)
        for j in range(columns)
            if j % 4 == 1 or (j % 4 == 2 and k % 3 == 0)
                value[k, j] = negative_zero
            if j % 4 == 3
                value[k, j] = float32(1 + (k + j) % 3)
    return value

bool fold_matches32(tensor<float32> left, tensor<float32> right, tensor<float32> product)
    int rows = left.shape()[0]
    int inner = left.shape()[1]
    int columns = right.shape()[1]
    for i in range(rows)
        for j in range(columns)
            float32 total = left[i, 0].item() * right[0, j].item()
            for k in range(1, inner)
                float32 term = left[i, k].item() * right[k, j].item()
                total = total + term
            if not same_bits32(product[i, j].item(), total)
                return false
    return negative_zero32(product[0, 0].item())

tensor<float> widen(tensor<float32> value)
    int rows = value.shape()[0]
    int columns = value.shape()[1]
    tensor<float> result = tensor.zeros<float>([rows, columns])
    for i in range(rows)
        for j in range(columns)
            result[i, j] = float(value[i, j].item())
    return result

bool fold_matches64(tensor<float> left, tensor<float> right, tensor<float> product)
    int rows = left.shape()[0]
    int inner = left.shape()[1]
    int columns = right.shape()[1]
    for i in range(rows)
        for j in range(columns)
            float total = left[i, 0].item() * right[0, j].item()
            for k in range(1, inner)
                float term = left[i, k].item() * right[k, j].item()
                total = total + term
            if not same_bits64(product[i, j].item(), total)
                return false
    return 1.0 / product[0, 0].item() < 0.0

// 1-2: untracked and tracked float32, short inner axis.
tensor<float32> l3 = signed_left32(6, 3)
tensor<float32> r3 = signed_right32(3, 8)
print(fold_matches32(l3, r3, math.matmul(l3, r3)))
print(NL)
print(fold_matches32(l3, r3, math.matmul(l3.track(), r3).untrack()))
print(NL)

// 3-4: an inner axis longer than one depth block, tracked right operand read
// through its transposed storage.
tensor<float32> l300 = signed_left32(6, 300)
tensor<float32> r300 = signed_right32(300, 8)
print(fold_matches32(l300, r300, math.matmul(l300, r300.track()).untrack()))
print(NL)
tensor<float32> r300_storage = r300.transpose(0, 1).contiguous().track()
print(fold_matches32(l300, r300, math.matmul(l300, r300_storage.transpose(0, 1)).untrack()))
print(NL)

// 5: float64, tracked and untracked.
tensor<float> l64 = widen(l300)
tensor<float> r64 = widen(r300)
print(fold_matches64(l64, r64, math.matmul(l64, r64)) and fold_matches64(l64, r64, math.matmul(l64.track(), r64.track()).untrack()))
print(NL)

// 6: matrix x vector and dot keep -0 for all -0 products.
tensor<float32> negative_zeros = r3.transpose(0, 1).contiguous().gather([3, 4, 5], [3])
tensor<float32> mv = math.matmul(l3, negative_zeros)
tensor<float32> negative_row = l3.gather([0, 1, 2], [3])
tensor<float32> plus_zeros = tensor.zeros<float32>([3])
print(negative_zero32(mv[1].item()) and not negative_zero32(mv[0].item()) and not negative_zero32(mv[2].item()) and negative_zero32(math.dot(negative_row, plus_zeros)) and negative_zero32(math.matmul(l3.track(), plus_zeros).untrack()[0].item()))
print(NL)
QUI
expect_true_lines "Math native signed zeros" "$TMP/native-signed-zero.qui" 6

# Native dispatch. The native results equal the portable composition bit for
# bit, so values cannot show which path served a call (a silent fallback would
# pass every check above). QUIDRA_MATH_TEST_NATIVE_TRACE makes the native
# bridges log each forward/backward they serve on stderr; every listed event
# must appear, and an event suffixed "=N" must appear exactly N times.
expect_native_events() {
    local label="$1"
    local program="$2"
    local count="$3"
    shift 3
    local status
    set +e
    QUIDRA_MATH_TEST_NATIVE_TRACE=1 QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" \
        "$QUIDRA" "$program" >"$TMP/trace.out" 2>"$TMP/trace.err"
    status=$?
    set -e
    local expected
    expected="$(printf 'true\n%.0s' $(seq 1 "$count"))"
    if [[ "$status" -ne 0 || "$(cat "$TMP/trace.out")" != "$expected" ]]; then
        echo "unexpected $label output (status $status):" >&2
        cat "$TMP/trace.out" "$TMP/trace.err" >&2
        exit 1
    fi
    local event
    local found
    for event in "$@"; do
        found="$(grep -Fxc "quidra-math native ${event%=*}" "$TMP/trace.err" || true)"
        if [[ "$event" == *=* && "$found" -ne "${event##*=}" ]] ||
           [[ "$event" != *=* && "$found" -eq 0 ]]; then
            local wanted="at least once"
            [[ "$event" == *=* ]] && wanted="${event##*=} time(s)"
            echo "$label: native event '${event%=*}' seen $found time(s), expected $wanted" >&2
            cat "$TMP/trace.err" >&2
            exit 1
        fi
    done
}

cat > "$TMP/native-dispatch.qui" <<'QUI'
import math

tensor<float32> pattern32(int rows, int columns, int seed)
    tensor<float32> value = tensor.zeros<float32>([rows, columns])
    for i in range(rows)
        for j in range(columns)
            value[i, j] = float32((i * 131 + j * 71 + seed * 17) % 97) / float32(48) - float32(1)
    return value

tensor<float> pattern64(int rows, int columns, int seed)
    tensor<float> value = tensor.zeros<float>([rows, columns])
    for i in range(rows)
        for j in range(columns)
            value[i, j] = float((i * 131 + j * 71 + seed * 17) % 97) / 48.0 - 1.0
    return value

// 1: float32 first order through every native forward and backward.
tensor<float32> x = pattern32(6, 5, 1).track()
tensor<float32> w = pattern32(4, 5, 2).track()
tensor<float32> y = math.matmul(x, w.transpose(0, 1))
(math.sum(math.max_last(y)) + math.sum(math.min_last(y)) + math.max_all(y) - math.min_all(y) + math.mean(math.sum_last(y))).backward(&x, &w)
print(x.grad.shape()[0] == 6 and w.grad.shape()[0] == 4)
print(NL)

// 2: float64 second order: tracked backward of every node family, then the
// backward of the nodes it attached.
tensor<float> a = pattern64(3, 4, 3).track()
tensor<float> b = pattern64(4, 2, 4).track()
tensor<float> p = math.matmul(a, b)
(math.sum(p * p) + math.max_all(p) + math.mean(math.sum_last(p))).backward(&a, &b, track = true)
tensor<float> ga = a.grad
a.clear_grad()
b.clear_grad()
math.sum(ga * ga).backward(&a, &b)
print(a.grad.shape()[0] == 3 and b.grad.shape()[1] == 2)
print(NL)
QUI
expect_native_events "Math native CPU dispatch" "$TMP/native-dispatch.qui" 2 \
    "matmul cpu float32" "sum cpu float32" "mean cpu float32" \
    "sum_last cpu float32" "max_last cpu float32" "min_last cpu float32" \
    "max_all cpu float32" "min_all cpu float32" "matmul-backward cpu float32" \
    "reduce-backward cpu float32" "extrema-backward cpu float32" \
    "matmul cpu float64" "sum cpu float64" "mean cpu float64" \
    "sum_last cpu float64" "max_all cpu float64" \
    "matmul-backward-tracked cpu float64" "reduce-backward-tracked cpu float64" \
    "extrema-backward-tracked cpu float64" "matmul-backward cpu float64" \
    "reduce-backward cpu float64" "extrema-backward cpu float64"

# Tracked non-contiguous views are served natively too: each operation below
# sees only a view (a rank-2 transpose or a rank-3 permutation), so each
# forward event must appear exactly once.
cat > "$TMP/native-dispatch-views.qui" <<'QUI'
import math

tensor<float32> base = tensor.zeros<float32>([5, 6])
for i in range(5)
    for j in range(6)
        base[i, j] = float32((i * 7 + j * 3) % 11) - float32(5)
tensor<float32> tracked = base.track()
tensor<float32> view = tracked.transpose(0, 1)
tensor<float32> cube = tensor.ones<float32>([2, 3, 4]).track()
tensor<float32> right = tensor.ones<float32>([2, 3])
tensor<float32> loss = math.sum(view) + math.mean(view) + math.sum_last(view).gather([7], []) + math.max_last(view).gather([3], []) + math.min_last(view).gather([11], []) + math.max_all(view) + math.min_all(view) + math.matmul(cube.transpose(0, 2), right).gather([5], [])
loss.backward(&tracked, &cube)
print(tracked.grad.shape()[0] == 5 and cube.grad[0, 0, 0].item() == float32(0))
print(NL)
QUI
expect_native_events "Math native CPU view dispatch" "$TMP/native-dispatch-views.qui" 1 \
    "sum cpu float32=1" "mean cpu float32=1" "sum_last cpu float32=1" \
    "max_last cpu float32=1" "min_last cpu float32=1" "max_all cpu float32=1" \
    "min_all cpu float32=1" "matmul cpu float32=1" "matmul-backward cpu float32=1" \
    "reduce-backward cpu float32=3" "extrema-backward cpu float32=4"

# Partially initialized CPU storage never reaches a native kernel: Core's CPU
# accessor rejects it, the bridge falls back, and the portable composition
# raises the language's UNINITIALIZED failure, for contiguous inputs and views.
# (GPU storage has no initialization query in the native ABI yet; see README.)
uninitialized_case=0
for operation in 'math.sum(t)' 'math.mean(t)' 'math.sum_last(t)' 'math.max_last(t)' \
    'math.min_last(t)' 'math.max_all(t)' 'math.min_all(t)' 'math.sum(t.transpose(0, 1))' \
    'math.max_last(t.transpose(0, 1))' 'math.matmul(t, tensor.ones<float32>([3, 2]))' \
    'math.matmul(tensor.ones<float32>([4, 2]), t)' 'math.matmul(t.transpose(0, 1), tensor.ones<float32>([2, 2]))'; do
    uninitialized_case=$((uninitialized_case + 1))
    cat > "$TMP/uninitialized-$uninitialized_case.qui" <<QUI
import math

tensor<float32> t = tensor<float32>([2, 3])
t[0, 0] = float32(1)
t[0, 1] = float32(2)
t[0, 2] = float32(3)
t[1, 0] = float32(4)
t[1, 1] = float32(5)
tensor<float32> result = $operation
print("unexpected success{NL}")
QUI
    set +e
    uninitialized_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/uninitialized-$uninitialized_case.qui" 2>&1)"
    uninitialized_status=$?
    set -e
    if [[ "$uninitialized_status" -ne 101 ]] ||
       ! grep -Fq "runtime error[UNINITIALIZED]" <<< "$uninitialized_output"; then
        echo "$operation on partially initialized CPU storage did not fail as UNINITIALIZED (status $uninitialized_status):" >&2
        printf '%s\n' "$uninitialized_output" >&2
        exit 1
    fi
done

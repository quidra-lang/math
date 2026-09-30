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

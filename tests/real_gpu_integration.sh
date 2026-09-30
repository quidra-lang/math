#!/usr/bin/env bash
set -euo pipefail

QUIDRA="${1:-}"
if [[ -z "$QUIDRA" ]]; then
    echo "usage: $0 /path/to/quidra" >&2
    exit 2
fi

REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT")"
GPU_INDEX="${QUIDRA_REAL_GPU_INDEX:-0}"
REQUIRE_REAL="${QUIDRA_REQUIRE_REAL_GPU:-0}"
REQUIRE_BACKEND="${QUIDRA_REQUIRE_GPU_BACKEND:-}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

set +e
gpu_info="$("$QUIDRA" gpu 2>&1)"
gpu_status=$?
set -e

skip_or_fail() {
    local reason="$1"
    if [[ "$REQUIRE_REAL" == "1" ]]; then
        echo "real Math GPU integration required but unavailable: $reason" >&2
        printf '%s\n' "$gpu_info" >&2
        exit 1
    fi
    echo "math real GPU integration: skipped ($reason)"
    exit 0
}

if [[ $gpu_status -ne 0 ]]; then skip_or_fail "quidra gpu failed"; fi
if grep -Fq "backend: TEST" <<<"$gpu_info"; then skip_or_fail "fake GPU backend is active"; fi
if ! grep -Fq "GPU $GPU_INDEX" <<<"$gpu_info"; then skip_or_fail "gpu($GPU_INDEX) is not present"; fi

gpu_block="$(awk -v target="GPU $GPU_INDEX" '
    $0 == target { found = 1; print; next }
    found && /^GPU [0-9]+$/ { exit }
    found { print }
' <<<"$gpu_info")"
if [[ -n "$REQUIRE_BACKEND" ]] && ! grep -Fq "backend: $REQUIRE_BACKEND" <<<"$gpu_block"; then
    skip_or_fail "gpu($GPU_INDEX) is not backend $REQUIRE_BACKEND"
fi

cat > "$TMP/math-real-gpu.qui" <<QUI
import math

tensor<float32> left = tensor.zeros<float32>([2, 3])
left[0, 0] = float32(1)
left[0, 1] = float32(2)
left[0, 2] = float32(3)
left[1, 0] = float32(4)
left[1, 1] = float32(5)
left[1, 2] = float32(6)

tensor<float32> right = tensor.zeros<float32>([3, 2])
right[0, 0] = float32(7)
right[0, 1] = float32(8)
right[1, 0] = float32(9)
right[1, 1] = float32(10)
right[2, 0] = float32(11)
right[2, 1] = float32(12)

tensor<float32> gpu_product = math.matmul(
    left.gpu($GPU_INDEX),
    right.gpu($GPU_INDEX)
).cpu()
print(gpu_product[0, 0].item() == float32(58))
print(NL)
print(gpu_product[0, 1].item() == float32(64))
print(NL)
print(gpu_product[1, 0].item() == float32(139))
print(NL)
print(gpu_product[1, 1].item() == float32(154))
print(NL)

tensor<float32> vector = tensor.zeros<float32>([3])
vector[0] = float32(1)
vector[1] = float32(2)
vector[2] = float32(3)
tensor<float32> gpu_vector_product = math.matmul(
    left.gpu($GPU_INDEX),
    vector.gpu($GPU_INDEX)
).cpu()
print(gpu_vector_product[0].item() == float32(14))
print(NL)
print(gpu_vector_product[1].item() == float32(32))
print(NL)

tensor<float32> tracked_left = tensor.ones<float32>([1, 2], gpu = $GPU_INDEX).track()
tensor<float32> tracked_right = tensor.ones<float32>([2, 1], gpu = $GPU_INDEX).track()
tensor<float32> tracked_product = math.matmul(tracked_left, tracked_right)
math.mean(tracked_product).backward(&tracked_left, &tracked_right)
print(tracked_left.grad.cpu()[0, 0].item() == float32(1))
print(NL)
print(tracked_right.grad.cpu()[0, 0].item() == float32(1))
print(NL)

tensor<float32> reduction_source = tensor.zeros<float32>([1, 3])
reduction_source[0, 0] = float32(5)
reduction_source[0, 1] = float32(-2)
reduction_source[0, 2] = float32(4)
tensor<float32> reduction_gpu = reduction_source.gpu($GPU_INDEX)
tensor<float32> reduced_sum = math.sum_last(reduction_gpu).cpu()
tensor<float32> reduced_max = math.max_last(reduction_gpu).cpu()
tensor<float32> reduced_min = math.min_last(reduction_gpu).cpu()
print(reduced_sum[0, 1].item() == float32(7))
print(NL)
print(reduced_max[0, 2].item() == float32(5))
print(NL)
print(reduced_min[0, 0].item() == float32(-2))
print(NL)

tensor<float32> tracked_reduction = reduction_source.gpu($GPU_INDEX).track()
math.mean(math.max_last(tracked_reduction)).backward(&tracked_reduction)
tensor<float32> reduction_gradient = tracked_reduction.grad.cpu()
print(reduction_gradient[0, 0].item() == float32(1))
print(NL)
print(reduction_gradient[0, 1].item() == float32(0))
print(NL)
print(reduction_gradient[0, 2].item() == float32(0))
print(NL)

tensor<float32> whole_maximum = math.max_all(reduction_gpu).cpu()
tensor<float32> whole_minimum = math.min_all(reduction_gpu).cpu()
print(whole_maximum.item() == float32(5))
print(NL)
print(whole_minimum.item() == float32(-2))
print(NL)

tracked_reduction.clear_grad()
math.max_all(tracked_reduction).backward(&tracked_reduction)
tensor<float32> whole_gradient = tracked_reduction.grad.cpu()
print(whole_gradient[0, 0].item() == float32(1))
print(NL)
print(whole_gradient[0, 1].item() == float32(0))
print(NL)

tensor<float32> unary_source = tensor.zeros<float32>([2])
unary_source[0] = float32(4)
unary_source[1] = float32(1)
tensor<float32> unary_gpu = unary_source.gpu($GPU_INDEX)
tensor<float32> unary_sqrt = math.sqrt(unary_gpu).cpu()
print(unary_sqrt[0].item() == float32(2) and unary_sqrt[1].item() == float32(1))
print(NL)
tensor<float32> unary_log = math.log(unary_gpu).cpu()
print(unary_log[1].item() == float32(0))
print(NL)
tensor<float32> unary_exp_input = tensor.zeros<float32>([1], gpu = $GPU_INDEX)
tensor<float32> unary_exp = math.exp(unary_exp_input).cpu()
print(unary_exp[0].item() == float32(1))
print(NL)

tensor<float32> abs_source = tensor.zeros<float32>([2])
abs_source[0] = float32(-2)
abs_source[1] = float32(3)
tensor<float32> unary_abs = math.abs(abs_source.gpu($GPU_INDEX)).cpu()
print(unary_abs[0].item() == float32(2) and unary_abs[1].item() == float32(3))
print(NL)

tensor<float32> exp_tracked = tensor.ones<float32>([2], gpu = $GPU_INDEX).track()
tensor<float32> exp_tracked_output = math.exp(exp_tracked)
math.mean(exp_tracked_output).backward(&exp_tracked)
tensor<float32> exp_gradient = exp_tracked.grad.cpu()
print(
    exp_gradient[0].item() > float32(1.359) and
    exp_gradient[0].item() < float32(1.360) and
    exp_gradient[1].item() > float32(1.359) and
    exp_gradient[1].item() < float32(1.360)
)
print(NL)

tensor<float32> abs_tracked = abs_source.gpu($GPU_INDEX).track()
math.mean(math.abs(abs_tracked)).backward(&abs_tracked)
tensor<float32> abs_gradient = abs_tracked.grad.cpu()
print(abs_gradient[0].item() == float32(-0.5) and abs_gradient[1].item() == float32(0.5))
print(NL)

tensor<float32> log_tracked = tensor.ones<float32>([2], gpu = $GPU_INDEX).track()
math.mean(math.log(log_tracked)).backward(&log_tracked)
tensor<float32> log_gradient = log_tracked.grad.cpu()
print(log_gradient[0].item() == float32(0.5) and log_gradient[1].item() == float32(0.5))
print(NL)

tensor<float32> sqrt_tracked = tensor.ones<float32>([2], gpu = $GPU_INDEX) * float32(4)
sqrt_tracked = sqrt_tracked.track()
math.mean(math.sqrt(sqrt_tracked)).backward(&sqrt_tracked)
tensor<float32> sqrt_gradient = sqrt_tracked.grad.cpu()
print(sqrt_gradient[0].item() == float32(0.125) and sqrt_gradient[1].item() == float32(0.125))
print(NL)

tensor<float32> noncontiguous = tensor.ones<float32>([2, 2], gpu = $GPU_INDEX).track()
tensor<float32> transposed = noncontiguous.transpose(0, 1)
tensor<float32> transposed_exp = math.exp(transposed)
math.mean(transposed_exp).backward(&noncontiguous)
tensor<float32> transposed_value = transposed_exp.untrack().cpu()
tensor<float32> transposed_gradient = noncontiguous.grad.cpu()
print(transposed_value[1, 1].item() > float32(2.718) and transposed_value[1, 1].item() < float32(2.719))
print(NL)
print(transposed_gradient[0, 0].item() > float32(0.679) and transposed_gradient[0, 0].item() < float32(0.680))
print(NL)
QUI

expected="$(printf 'true\n%.0s' {1..28})"
actual="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" run "$TMP/math-real-gpu.qui")"
if [[ "$actual" != "$expected" ]]; then
    echo "unexpected Math real-GPU output:" >&2
    printf '%s\n' "$actual" >&2
    exit 1
fi

echo "math real GPU integration: ok on gpu($GPU_INDEX)"

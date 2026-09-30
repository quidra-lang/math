#!/usr/bin/env bash
set -euo pipefail

QUIDRA="${1:?usage: device.sh /path/to/quidra}"
REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

export QUIDRA_TEST_FAKE_GPU_COUNT=1

cat > "$TMP/math-device.qui" <<'QUI'
import math

tensor<float32> left = tensor.ones<float32>([2, 3], gpu = 0)
tensor<float32> right = tensor.ones<float32>([3, 2], gpu = 0)
tensor<float32> product = math.matmul(left, right).cpu()
print(product[0, 0].item() == float32(3))
print(NL)
print(product[1, 1].item() == float32(3))
print(NL)

tensor<float32> weight = tensor.ones<float32>([2, 3], gpu = 0)
tensor<float32> transposed_product = math.matmul(left, weight.transpose(0, 1))
tensor<float32> device_preserved = transposed_product + tensor.zeros<float32>([2, 2], gpu = 0)
print(device_preserved.cpu()[0, 0].item() == float32(3))
print(NL)

tensor<int32> integer_left = tensor.ones<int32>([2, 2], gpu = 0)
tensor<int32> integer_right = tensor.ones<int32>([2, 2], gpu = 0)
tensor<int32> integer_device_product = math.matmul(integer_left, integer_right)
tensor<int32> integer_device_preserved = integer_device_product + tensor.zeros<int32>([2, 2], gpu = 0)
print(integer_device_preserved.cpu()[1, 1].item() == int32(2))
print(NL)
tensor<int32> integer_product = integer_device_product.cpu()
print(integer_product[1, 1].item() == int32(2))
print(NL)

tensor<float32> vector = tensor.ones<float32>([3], gpu = 0)
print(math.dot(vector, vector) == float32(3))
print(NL)

tensor<float32> tracked_left = tensor.ones<float32>([1, 2], gpu = 0).track()
tensor<float32> tracked_right = tensor.ones<float32>([2, 1], gpu = 0).track()
tensor<float32> tracked_product = math.matmul(tracked_left, tracked_right)
math.mean(tracked_product).backward(&tracked_left, &tracked_right)
tensor<float32> left_grad = tracked_left.grad.cpu()
tensor<float32> right_grad = tracked_right.grad.cpu()
print(left_grad[0, 0].item() == float32(1))
print(NL)
print(right_grad[0, 0].item() == float32(1))
print(NL)

tensor<float32> reduction_values = tensor.ones<float32>([1, 3], gpu = 0)
print(math.sum(reduction_values).cpu().item() == float32(3))
print(NL)
print(math.mean(reduction_values).cpu().item() == float32(1))
print(NL)
tensor<float32> reduced_sum = math.sum_last(reduction_values).cpu()
print(reduced_sum[0, 2].item() == float32(3))
print(NL)
tensor<float32> reduced_max = math.max_last(reduction_values).cpu()
tensor<float32> reduced_min = math.min_last(reduction_values).cpu()
print(reduced_max[0, 1].item() == float32(1))
print(NL)
print(reduced_min[0, 1].item() == float32(1))
print(NL)

tensor<int32> integer_reduction_values = tensor.zeros<int32>([2, 3], gpu = 0)
integer_reduction_values[0, 0] = int32(5)
integer_reduction_values[0, 1] = int32(-2)
integer_reduction_values[0, 2] = int32(4)
integer_reduction_values[1, 0] = int32(9)
integer_reduction_values[1, 1] = int32(3)
integer_reduction_values[1, 2] = int32(7)
tensor<int32> integer_max = math.max_last(integer_reduction_values).cpu()
tensor<int32> integer_min = math.min_last(integer_reduction_values).cpu()
print(integer_max[0, 2].item() == int32(5))
print(NL)
print(integer_max[1, 1].item() == int32(9))
print(NL)
print(integer_min[0, 0].item() == int32(-2))
print(NL)
print(integer_min[1, 2].item() == int32(3))
print(NL)

tensor<float32> tracked_reduction = tensor.ones<float32>([1, 3], gpu = 0).track()
math.mean(math.max_last(tracked_reduction)).backward(&tracked_reduction)
tensor<float32> reduction_grad = tracked_reduction.grad.cpu()
print(reduction_grad[0, 0].item() == float32(1))
print(NL)
print(reduction_grad[0, 1].item() == float32(0))
print(NL)
print(reduction_grad[0, 2].item() == float32(0))
print(NL)

tensor<float32> unary_source = tensor.ones<float32>([2], gpu = 0) * float32(4)
tensor<float32> unary_tracked = unary_source.track()
tensor<float32> unary_root = math.sqrt(unary_tracked)
print(unary_root.untrack().cpu()[0].item() == float32(2))
print(NL)
math.mean(unary_root).backward(&unary_tracked)
tensor<float32> unary_gradient = unary_tracked.grad.cpu()
print(unary_gradient[0].item() == float32(0.125))
print(NL)
print(unary_gradient[1].item() == float32(0.125))
print(NL)

tensor<float32> unary_matrix = tensor.ones<float32>([2, 2], gpu = 0).track()
tensor<float32> unary_view = unary_matrix.transpose(0, 1)
tensor<float32> unary_exp = math.exp(unary_view)
math.mean(unary_exp).backward(&unary_matrix)
tensor<float32> unary_exp_value = unary_exp.untrack().cpu()
tensor<float32> unary_exp_gradient = unary_matrix.grad.cpu()
print(unary_exp_value[1, 1].item() > float32(2.718) and unary_exp_value[1, 1].item() < float32(2.719))
print(NL)
print(unary_exp_gradient[0, 0].item() > float32(0.679) and unary_exp_gradient[0, 0].item() < float32(0.680))
print(NL)


QUI

expected="$(printf 'true\n%.0s' {1..25})"
actual="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/math-device.qui")"
if [[ "$actual" != "$expected" ]]; then
    echo "unexpected Math fake-GPU output:" >&2
    printf '%s\n' "$actual" >&2
    exit 1
fi


cat > "$TMP/whole-extrema-device.qui" <<'QUI'
import math

tensor<float32> values = tensor.zeros<float32>([3], gpu = 0)
values[0] = float32(5)
values[1] = float32(-2)
values[2] = float32(4)
print(math.max_all(values).cpu().item() == float32(5))
print(NL)
print(math.min_all(values).cpu().item() == float32(-2))
print(NL)

tensor<float32> tracked = values.track()
math.max_all(tracked).backward(&tracked)
tensor<float32> gradient = tracked.grad.cpu()
print(gradient[0].item() == float32(1))
print(NL)
print(gradient[1].item() == float32(0))
print(NL)
print(gradient[2].item() == float32(0))
print(NL)
QUI
whole_device_output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/whole-extrema-device.qui")"
whole_device_expected="$(printf 'true\n%.0s' {1..5})"
if [[ "$whole_device_output" != "$whole_device_expected" ]]; then
    echo "unexpected whole-tensor extrema fake-GPU output:" >&2
    printf '%s\n' "$whole_device_output" >&2
    exit 1
fi

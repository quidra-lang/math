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
export QUIDRA_CACHE_DIR="$TMP/run-cache"
# The suite pins the default Metal path; the MPS opt-in is enabled explicitly
# for the one program that tests it, never inherited from the caller.
unset QUIDRA_MATH_METAL_MPS

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

tensor<real32> left = tensor.zeros<real32>([2, 3])
left[0, 0] = real32(1)
left[0, 1] = real32(2)
left[0, 2] = real32(3)
left[1, 0] = real32(4)
left[1, 1] = real32(5)
left[1, 2] = real32(6)

tensor<real32> right = tensor.zeros<real32>([3, 2])
right[0, 0] = real32(7)
right[0, 1] = real32(8)
right[1, 0] = real32(9)
right[1, 1] = real32(10)
right[2, 0] = real32(11)
right[2, 1] = real32(12)

tensor<real32> gpu_product = math.matmul(
    left.gpu($GPU_INDEX),
    right.gpu($GPU_INDEX)
).cpu()
print(gpu_product[0, 0].item() == real32(58))
print(NL)
print(gpu_product[0, 1].item() == real32(64))
print(NL)
print(gpu_product[1, 0].item() == real32(139))
print(NL)
print(gpu_product[1, 1].item() == real32(154))
print(NL)

tensor<real32> vector = tensor.zeros<real32>([3])
vector[0] = real32(1)
vector[1] = real32(2)
vector[2] = real32(3)
tensor<real32> gpu_vector_product = math.matmul(
    left.gpu($GPU_INDEX),
    vector.gpu($GPU_INDEX)
).cpu()
print(gpu_vector_product[0].item() == real32(14))
print(NL)
print(gpu_vector_product[1].item() == real32(32))
print(NL)

tensor<real32> tracked_left = tensor.ones<real32>([1, 2], gpu = $GPU_INDEX).track()
tensor<real32> tracked_right = tensor.ones<real32>([2, 1], gpu = $GPU_INDEX).track()
tensor<real32> tracked_product = math.matmul(tracked_left, tracked_right)
math.mean(tracked_product).backward(&tracked_left, &tracked_right)
print(tracked_left.grad.cpu()[0, 0].item() == real32(1))
print(NL)
print(tracked_right.grad.cpu()[0, 0].item() == real32(1))
print(NL)

tensor<real32> reduction_source = tensor.zeros<real32>([1, 3])
reduction_source[0, 0] = real32(5)
reduction_source[0, 1] = real32(-2)
reduction_source[0, 2] = real32(4)
tensor<real32> reduction_gpu = reduction_source.gpu($GPU_INDEX)
tensor<real32> reduced_sum = math.sum_last(reduction_gpu).cpu()
tensor<real32> reduced_max = math.max_last(reduction_gpu).cpu()
tensor<real32> reduced_min = math.min_last(reduction_gpu).cpu()
print(reduced_sum[0, 1].item() == real32(7))
print(NL)
print(reduced_max[0, 2].item() == real32(5))
print(NL)
print(reduced_min[0, 0].item() == real32(-2))
print(NL)

tensor<real32> tracked_reduction = reduction_source.gpu($GPU_INDEX).track()
math.mean(math.max_last(tracked_reduction)).backward(&tracked_reduction)
tensor<real32> reduction_gradient = tracked_reduction.grad.cpu()
print(reduction_gradient[0, 0].item() == real32(1))
print(NL)
print(reduction_gradient[0, 1].item() == real32(0))
print(NL)
print(reduction_gradient[0, 2].item() == real32(0))
print(NL)

tensor<real32> whole_maximum = math.max_all(reduction_gpu).cpu()
tensor<real32> whole_minimum = math.min_all(reduction_gpu).cpu()
print(whole_maximum.item() == real32(5))
print(NL)
print(whole_minimum.item() == real32(-2))
print(NL)

tracked_reduction.clear_grad()
math.max_all(tracked_reduction).backward(&tracked_reduction)
tensor<real32> whole_gradient = tracked_reduction.grad.cpu()
print(whole_gradient[0, 0].item() == real32(1))
print(NL)
print(whole_gradient[0, 1].item() == real32(0))
print(NL)

tensor<real32> unary_source = tensor.zeros<real32>([2])
unary_source[0] = real32(4)
unary_source[1] = real32(1)
tensor<real32> unary_gpu = unary_source.gpu($GPU_INDEX)
tensor<real32> unary_sqrt = math.sqrt(unary_gpu).cpu()
print(unary_sqrt[0].item() == real32(2) and unary_sqrt[1].item() == real32(1))
print(NL)
tensor<real32> unary_log = math.log(unary_gpu).cpu()
print(unary_log[1].item() == real32(0))
print(NL)
tensor<real32> unary_exp_input = tensor.zeros<real32>([1], gpu = $GPU_INDEX)
tensor<real32> unary_exp = math.exp(unary_exp_input).cpu()
print(unary_exp[0].item() == real32(1))
print(NL)

tensor<real32> abs_source = tensor.zeros<real32>([2])
abs_source[0] = real32(-2)
abs_source[1] = real32(3)
tensor<real32> unary_abs = math.abs(abs_source.gpu($GPU_INDEX)).cpu()
print(unary_abs[0].item() == real32(2) and unary_abs[1].item() == real32(3))
print(NL)

tensor<real32> exp_tracked = tensor.ones<real32>([2], gpu = $GPU_INDEX).track()
tensor<real32> exp_tracked_output = math.exp(exp_tracked)
math.mean(exp_tracked_output).backward(&exp_tracked)
tensor<real32> exp_gradient = exp_tracked.grad.cpu()
print(
    exp_gradient[0].item() > real32(1.359) and
    exp_gradient[0].item() < real32(1.360) and
    exp_gradient[1].item() > real32(1.359) and
    exp_gradient[1].item() < real32(1.360)
)
print(NL)

tensor<real32> abs_tracked = abs_source.gpu($GPU_INDEX).track()
math.mean(math.abs(abs_tracked)).backward(&abs_tracked)
tensor<real32> abs_gradient = abs_tracked.grad.cpu()
print(abs_gradient[0].item() == real32(-0.5) and abs_gradient[1].item() == real32(0.5))
print(NL)

tensor<real32> log_tracked = tensor.ones<real32>([2], gpu = $GPU_INDEX).track()
math.mean(math.log(log_tracked)).backward(&log_tracked)
tensor<real32> log_gradient = log_tracked.grad.cpu()
print(log_gradient[0].item() == real32(0.5) and log_gradient[1].item() == real32(0.5))
print(NL)

tensor<real32> sqrt_tracked = tensor.ones<real32>([2], gpu = $GPU_INDEX) * real32(4)
sqrt_tracked = sqrt_tracked.track()
math.mean(math.sqrt(sqrt_tracked)).backward(&sqrt_tracked)
tensor<real32> sqrt_gradient = sqrt_tracked.grad.cpu()
print(sqrt_gradient[0].item() == real32(0.125) and sqrt_gradient[1].item() == real32(0.125))
print(NL)

tensor<real32> noncontiguous = tensor.ones<real32>([2, 2], gpu = $GPU_INDEX).track()
tensor<real32> transposed = noncontiguous.transpose(0, 1)
tensor<real32> transposed_exp = math.exp(transposed)
math.mean(transposed_exp).backward(&noncontiguous)
tensor<real32> transposed_value = transposed_exp.untrack().cpu()
tensor<real32> transposed_gradient = noncontiguous.grad.cpu()
print(transposed_value[1, 1].item() > real32(2.718) and transposed_value[1, 1].item() < real32(2.719))
print(NL)
print(transposed_gradient[0, 0].item() > real32(0.679) and transposed_gradient[0, 0].item() < real32(0.680))
print(NL)
QUI

expected="$(printf 'true\n%.0s' {1..28})"
actual="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" run "$TMP/math-real-gpu.qui")"
if [[ "$actual" != "$expected" ]]; then
    echo "unexpected Math real-GPU output:" >&2
    printf '%s\n' "$actual" >&2
    exit 1
fi


# Runs one Quidra program that prints a "true" line per check.
expect_true_lines() {
    local label="$1"
    local program="$2"
    local count="$3"
    local output
    local status
    set +e
    output="$(QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" run "$program" 2>&1)"
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

# Native products on any real backend: every Metal kernel family (per-output,
# tiled, split-inner) and CUDA's library path agree with the CPU, tracked
# FC-pattern gradients agree, and the deterministic policy is repeatable. The
# exact Metal numerics (bitwise kernels, the split-inner error bound, and the
# MPS opt-in) are pinned in the Metal-only section below.
cat > "$TMP/native-matmul-real-gpu.qui" <<QUI
import math

extern void execution_policy(int32 policy) = "qcore_execution_policy_set"

tensor<real32> pattern32(int rows, int columns, int seed)
    tensor<real32> value = tensor.zeros<real32>([nat(rows), nat(columns)])
    for i in range(rows)
        for j in range(columns)
            int h = (i * 131 + j * 71 + seed * 17) % 97
            value[i, j] = real32(h) / real32(48) - real32(1)
    return value

real32 max_difference(tensor<real32> left, tensor<real32> right)
    return math.max_all(math.abs(left - right)).item()

bool same32(tensor<real32> left, tensor<real32> right)
    return max_difference(left, right) == real32(0)

// Products agree with CPU within real32 accumulation tolerance on every
// backend: per-output (small), tiled, split-inner (few outputs, long inner),
// and a large product (Math's tiled kernel on Metal by default).
bool product_close(int m, int k, int n, int device)
    tensor<real32> a = pattern32(m, k, 1)
    tensor<real32> w = pattern32(n, k, 2)
    tensor<real32> expected = math.matmul(a, w.transpose(0, 1))
    tensor<real32> actual = math.matmul(a.gpu(nat(device)), w.gpu(nat(device)).transpose(0, 1))
    return actual.device() == device and real64(max_difference(actual.cpu(), expected)) < real64(0.00001) * real64(k)

print(product_close(45, 16, 2, $GPU_INDEX))
print(NL)
print(product_close(100, 40, 90, $GPU_INDEX))
print(NL)
print(product_close(9, 3000, 4, $GPU_INDEX))
print(NL)
print(product_close(256, 128, 192, $GPU_INDEX))
print(NL)

// Tracked FC-pattern gradients on Metal agree with CPU.
tensor<real32> a = pattern32(70, 33, 3)
tensor<real32> w = pattern32(65, 33, 4)
tensor<real32> g = pattern32(70, 65, 5)
tensor<real32> ga = a.gpu($GPU_INDEX).track()
tensor<real32> gw = w.gpu($GPU_INDEX).track()
math.sum(math.matmul(ga, gw.transpose(0, 1)) * g.gpu($GPU_INDEX)).backward(&ga, &gw)
tensor<real32> ca = a.track()
tensor<real32> cw = w.track()
math.sum(math.matmul(ca, cw.transpose(0, 1)) * g).backward(&ca, &cw)
print(max_difference(ga.grad.cpu(), ca.grad) < real32(0.0001) and max_difference(gw.grad.cpu(), cw.grad) < real32(0.0001))
print(NL)

// Deterministic policy: Math's own fixed-order kernels, bitwise repeatable.
execution_policy(int32(1))
tensor<real32> left = pattern32(256, 128, 10).gpu($GPU_INDEX)
tensor<real32> right = pattern32(128, 192, 11).gpu($GPU_INDEX)
tensor<real32> first = math.matmul(left, right).cpu()
tensor<real32> second = math.matmul(left, right).cpu()
execution_policy(int32(0))
print(same32(first, second) and real64(max_difference(first, math.matmul(pattern32(256, 128, 10), pattern32(128, 192, 11)))) < real64(0.000001) * real64(128.0))
print(NL)
QUI
expect_true_lines "Math native real-GPU matmul" "$TMP/native-matmul-real-gpu.qui" 6

# Whole-tensor sum/mean on the GPU use the same pairwise tree as the CPU.
cat > "$TMP/native-sum-mean-real-gpu.qui" <<QUI
import math

extern void execution_policy(int32 policy) = "qcore_execution_policy_set"

tensor<real32> pattern32(int rows, int columns, int seed)
    tensor<real32> value = tensor.zeros<real32>([nat(rows), nat(columns)])
    for i in range(rows)
        for j in range(columns)
            int h = (i * 131 + j * 71 + seed * 17) % 97
            value[i, j] = real32(h) / real32(48) - real32(1)
    return value

real32 max_difference(tensor<real32> left, tensor<real32> right)
    return math.max_all(math.abs(left - right)).item()

bool same32(tensor<real32> left, tensor<real32> right)
    return max_difference(left, right) == real32(0)

// Whole-tensor sum and mean run the same pairwise tree as the CPU: results are
// bit-identical, and mean backward equals the CPU exactly.
tensor<real32> big = pattern32(400, 257, 6)
tensor<real32> big_gpu = big.gpu($GPU_INDEX)
print(math.sum(big_gpu).cpu().item() == math.sum(big).item())
print(NL)
print(math.mean(big_gpu).cpu().item() == math.mean(big).item())
print(NL)
tensor<real32> r = pattern32(20, 48, 8)
tensor<real32> mg = r.gpu($GPU_INDEX).track()
math.mean(mg).backward(&mg)
tensor<real32> mc = r.track()
math.mean(mc).backward(&mc)
print(same32(mg.grad.cpu(), mc.grad))
print(NL)
QUI
expect_true_lines "Math native real-GPU sum/mean" "$TMP/native-sum-mean-real-gpu.qui" 3

# Last-axis sums and first-occurrence extrema on the GPU are bit-identical to
# the CPU, including their device-side gradients.
cat > "$TMP/native-last-axis-real-gpu.qui" <<QUI
import math

extern void execution_policy(int32 policy) = "qcore_execution_policy_set"

tensor<real32> pattern32(int rows, int columns, int seed)
    tensor<real32> value = tensor.zeros<real32>([nat(rows), nat(columns)])
    for i in range(rows)
        for j in range(columns)
            int h = (i * 131 + j * 71 + seed * 17) % 97
            value[i, j] = real32(h) / real32(48) - real32(1)
    return value

real32 max_difference(tensor<real32> left, tensor<real32> right)
    return math.max_all(math.abs(left - right)).item()

bool same32(tensor<real32> left, tensor<real32> right)
    return max_difference(left, right) == real32(0)

// Last-axis sums add each row left to right and extrema are first-occurrence
// with strict comparison on both backends: results and gradients are exact.
tensor<real32> big = pattern32(400, 257, 6)
tensor<real32> big_gpu = big.gpu($GPU_INDEX)
print(same32(math.sum_last(big_gpu).cpu(), math.sum_last(big)))
print(NL)
print(same32(math.max_last(big_gpu).cpu(), math.max_last(big)) and same32(math.min_last(big_gpu).cpu(), math.min_last(big)))
print(NL)
print(math.max_all(big_gpu).cpu().item() == math.max_all(big).item() and math.min_all(big_gpu).cpu().item() == math.min_all(big).item())
print(NL)

// Reduction gradients on Metal equal CPU exactly.
tensor<real32> r = pattern32(20, 48, 8)
tensor<real32> up = pattern32(20, 48, 9)
tensor<real32> rg = r.gpu($GPU_INDEX).track()
math.sum(math.sum_last(rg) * up.gpu($GPU_INDEX)).backward(&rg)
tensor<real32> rc = r.track()
math.sum(math.sum_last(rc) * up).backward(&rc)
print(same32(rg.grad.cpu(), rc.grad))
print(NL)
tensor<real32> xg = r.gpu($GPU_INDEX).track()
math.sum(math.max_last(xg) * up.gpu($GPU_INDEX)).backward(&xg)
tensor<real32> xc = r.track()
math.sum(math.max_last(xc) * up).backward(&xc)
print(same32(xg.grad.cpu(), xc.grad))
print(NL)
QUI
expect_true_lines "Math native real-GPU last-axis reductions" "$TMP/native-last-axis-real-gpu.qui" 5

# Wide rows (threadgroup extrema) with ties and NaNs match the sequential scan.
cat > "$TMP/native-wide-extrema-real-gpu.qui" <<QUI
import math

extern void execution_policy(int32 policy) = "qcore_execution_policy_set"

tensor<real32> pattern32(int rows, int columns, int seed)
    tensor<real32> value = tensor.zeros<real32>([nat(rows), nat(columns)])
    for i in range(rows)
        for j in range(columns)
            int h = (i * 131 + j * 71 + seed * 17) % 97
            value[i, j] = real32(h) / real32(48) - real32(1)
    return value

real32 max_difference(tensor<real32> left, tensor<real32> right)
    return math.max_all(math.abs(left - right)).item()

bool same32(tensor<real32> left, tensor<real32> right)
    return max_difference(left, right) == real32(0)

// Wide rows with ties and NaNs (group extrema kernel) match the sequential scan.
tensor<real32> wide = pattern32(3, 600, 7)
real32 zero = real32(0)
wide[0, 0] = zero / zero
wide[1, 10] = zero / zero
wide[1, 300] = real32(5)
wide[1, 450] = real32(5)
wide[2, 599] = real32(-7)
wide[2, 20] = real32(-7)
tensor<real32> wide_gpu = wide.gpu($GPU_INDEX)
tensor<real32> wide_max = math.max_last(wide_gpu).cpu()
tensor<real32> wide_min = math.min_last(wide_gpu).cpu()
print(not math.is_finite(wide_max[0, 5].item()) and wide_max[1, 0].item() == real32(5) and wide_min[2, 3].item() == real32(-7))
print(NL)
tensor<real32> wide_tracked = wide_gpu.track()
math.sum(math.max_last(wide_tracked)).backward(&wide_tracked)
tensor<real32> wide_grad = wide_tracked.grad.cpu()
print(wide_grad[1, 300].item() == real32(600) and wide_grad[1, 450].item() == real32(0) and wide_grad[0, 0].item() == real32(600))
print(NL)
QUI
expect_true_lines "Math native real-GPU wide extrema" "$TMP/native-wide-extrema-real-gpu.qui" 2

# Views and one-sided tracking on the real GPU: transposed results feeding the
# loss (non-contiguous upstream device gradients), tracked transposed inputs
# read through a value-only device copy (no per-element host readback), and
# one side tracked products.
cat > "$TMP/native-views-real-gpu.qui" <<QUI
import math

tensor<real32> pattern32(int rows, int columns, int seed)
    tensor<real32> value = tensor.zeros<real32>([nat(rows), nat(columns)])
    for i in range(rows)
        for j in range(columns)
            int h = (i * 131 + j * 71 + seed * 17) % 97
            value[i, j] = real32(h) / real32(48) - real32(1)
    return value

real32 max_difference(tensor<real32> left, tensor<real32> right)
    return math.max_all(math.abs(left - right)).item()

bool same32(tensor<real32> left, tensor<real32> right)
    return max_difference(left, right) == real32(0)

bool close32(tensor<real32> left, tensor<real32> right)
    return max_difference(left, right) < real32(0.00001)

// 1: one side tracked: only that side receives a gradient, equal to the CPU.
tensor<real32> a_data = pattern32(40, 24, 1)
tensor<real32> w_data = pattern32(36, 24, 2)
tensor<real32> up = pattern32(40, 36, 3)
tensor<real32> left_only = a_data.gpu($GPU_INDEX).track()
math.sum(math.matmul(left_only, w_data.gpu($GPU_INDEX).transpose(0, 1)) * up.gpu($GPU_INDEX)).backward(&left_only)
tensor<real32> right_only = w_data.gpu($GPU_INDEX).track()
math.sum(math.matmul(a_data.gpu($GPU_INDEX), right_only.transpose(0, 1)) * up.gpu($GPU_INDEX)).backward(&right_only)
tensor<real32> host_left = a_data.track()
tensor<real32> host_right = w_data.track()
math.sum(math.matmul(host_left, host_right.transpose(0, 1)) * up).backward(&host_left, &host_right)
print(close32(left_only.grad.cpu(), host_left.grad) and close32(right_only.grad.cpu(), host_right.grad))
print(NL)

// 2-3: transposed matmul / sum_last / max_last / min_last results feed the
// loss on the device.
tensor<real32> p_data = pattern32(5, 6, 4)
tensor<real32> q_data = pattern32(6, 7, 5)
tensor<real32> across = pattern32(7, 5, 6)
tensor<real32> gp = p_data.gpu($GPU_INDEX).track()
tensor<real32> gq = q_data.gpu($GPU_INDEX).track()
math.sum(math.matmul(gp, gq).transpose(0, 1) * across.gpu($GPU_INDEX)).backward(&gp, &gq)
tensor<real32> cp = p_data.track()
tensor<real32> cq = q_data.track()
math.sum(math.matmul(cp, cq).transpose(0, 1) * across).backward(&cp, &cq)
print(close32(gp.grad.cpu(), cp.grad) and close32(gq.grad.cpu(), cq.grad))
print(NL)
tensor<real32> x_data = pattern32(5, 6, 7)
tensor<real32> weights = pattern32(6, 5, 8)
tensor<real32> gx = x_data.gpu($GPU_INDEX).track()
math.sum((math.sum_last(gx) + math.max_last(gx) - math.min_last(gx)).transpose(0, 1) * weights.gpu($GPU_INDEX)).backward(&gx)
tensor<real32> cx = x_data.track()
math.sum((math.sum_last(cx) + math.max_last(cx) - math.min_last(cx)).transpose(0, 1) * weights).backward(&cx)
print(same32(gx.grad.cpu(), cx.grad))
print(NL)

// 4: tracked transposed inputs to every reduction: exact values and
// gradients, computed on the device.
tensor<real32> r_data = pattern32(48, 200, 9)
tensor<real32> gr = r_data.gpu($GPU_INDEX).track()
tensor<real32> gv = gr.transpose(0, 1)
tensor<real32> device_loss = math.sum(math.max_last(gv)) - math.sum(math.min_last(gv)) + math.sum(math.sum_last(gv)) + math.mean(gv) + math.max_all(gv) + math.min_all(gv)
device_loss.backward(&gr)
tensor<real32> cr = r_data.track()
tensor<real32> cv = cr.transpose(0, 1)
tensor<real32> host_loss = math.sum(math.max_last(cv)) - math.sum(math.min_last(cv)) + math.sum(math.sum_last(cv)) + math.mean(cv) + math.max_all(cv) + math.min_all(cv)
host_loss.backward(&cr)
print(gr.grad.device() == $GPU_INDEX and same32(gr.grad.cpu(), cr.grad))
print(NL)
QUI
expect_true_lines "Math native real-GPU views" "$TMP/native-views-real-gpu.qui" 4

# Very wide whole-tensor extrema split the search across threadgroups (span
# winners, then a merge) and still equal the sequential strict scan.
cat > "$TMP/native-wide-all-real-gpu.qui" <<QUI
import math

// 1: very wide whole-tensor extrema (span winners, then a merge) with ties,
// NaNs and signed zeros match the sequential scan, gradients included.
tensor<real32> wide = tensor.zeros<real32>([1, 70001])
for i in range(70001)
    wide[0, i] = real32((i * 131) % 9973) / real32(4096) - real32(1)
real32 zero = real32(0)
wide[0, 17] = zero / zero
wide[0, 5000] = real32(9)
wide[0, 60000] = real32(9)
wide[0, 30000] = real32(-9)
wide[0, 40000] = real32(-9)
tensor<real32> wide_gpu = wide.gpu($GPU_INDEX).track()
(math.max_all(wide_gpu) - math.min_all(wide_gpu)).backward(&wide_gpu)
tensor<real32> wide_grad = wide_gpu.grad.cpu()
print(math.max_all(wide.gpu($GPU_INDEX)).cpu().item() == real32(9) and wide_grad[0, 5000].item() == real32(1) and wide_grad[0, 60000].item() == real32(0) and wide_grad[0, 30000].item() == real32(-1) and wide_grad[0, 40000].item() == real32(0))
print(NL)
QUI
expect_true_lines "Math native real-GPU wide whole-tensor extrema" "$TMP/native-wide-all-real-gpu.qui" 1

# Whole-tensor sums keep the portable odd-carry definition on the device.
cat > "$TMP/native-sum-carry-real-gpu.qui" <<QUI
import math

bool is_nan(real32 value)
    return value != value

// 1: odd-carry definition on the device (bit-identical to the CPU tree).
real32 infinity = real32(1) / real32(0)
tensor<real32> carried = tensor.ones<real32>([1, 3])
carried[0, 2] = infinity
tensor<real32> long_carried = tensor.ones<real32>([1, 4097])
long_carried[0, 4096] = infinity
tensor<real32> paired = tensor.ones<real32>([1, 4096])
paired[0, 4095] = infinity
print(is_nan(math.sum(carried.gpu($GPU_INDEX)).cpu().item()) and is_nan(math.mean(long_carried.gpu($GPU_INDEX)).cpu().item()) and is_nan(math.sum(long_carried).item()) and math.sum(paired.gpu($GPU_INDEX)).cpu().item() == infinity)
print(NL)
QUI
expect_true_lines "Math native real-GPU sum carry" "$TMP/native-sum-carry-real-gpu.qui" 1

# Metal-only checks: Math serves Metal real32 through its own kernels (CUDA
# and HIP keep their existing paths, so these checks do not apply there).
if grep -Fq "backend: Metal" <<<"$gpu_block"; then

# Native dispatch on Metal. Values equal the CPU, so values alone cannot show
# which path ran; QUIDRA_MATH_TEST_NATIVE_TRACE logs every native forward,
# backward, and GEMM kernel choice. Every listed event must appear, and an
# event suffixed "=N" exactly N times.
expect_native_events() {
    local label="$1"
    local program="$2"
    local count="$3"
    shift 3
    local status
    set +e
    QUIDRA_MATH_TEST_NATIVE_TRACE=1 QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" \
        "$QUIDRA" run "$program" >"$TMP/trace.out" 2>"$TMP/trace.err"
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

# Signed zeros through each Math GEMM kernel (per-output, tiled, split-inner)
# under the deterministic policy: outputs equal the portable fold p0 + p1 + ...
# bit for bit, including the sign of exact-zero outputs. Every sum is a small
# integer, so the split-inner order is exact as well.
cat > "$TMP/native-signed-zero-real-gpu.qui" <<QUI
import math

extern void execution_policy(int32 policy) = "qcore_execution_policy_set"

bool same_bits32(real32 x, real32 y)
    if x != x
        return y != y
    if x == real32(0)
        return y == real32(0) and (real32(1) / x < real32(0)) == (real32(1) / y < real32(0))
    return x == y

tensor<real32> signed_left32(int rows, int inner)
    tensor<real32> value = tensor.zeros<real32>([nat(rows), nat(inner)])
    for i in range(rows)
        for k in range(inner)
            real32 magnitude = real32(1 + (i + k) % 5)
            if i % 3 == 0 or (i % 3 == 2 and k % 2 == 0)
                magnitude = -magnitude
            value[i, k] = magnitude
    return value

tensor<real32> signed_right32(int inner, int columns)
    tensor<real32> value = tensor.zeros<real32>([nat(inner), nat(columns)])
    real32 negative_zero = real32(0) * real32(-1)
    for k in range(inner)
        for j in range(columns)
            if j % 4 == 1 or (j % 4 == 2 and k % 3 == 0)
                value[k, j] = negative_zero
            if j % 4 == 3
                value[k, j] = real32(1 + (k + j) % 3)
    return value

bool fold_matches32(int rows, int inner, int columns, bool tracked)
    tensor<real32> left = signed_left32(rows, inner)
    tensor<real32> right = signed_right32(inner, columns)
    tensor<real32> device_left = left.gpu($GPU_INDEX)
    if tracked
        device_left = device_left.track()
    tensor<real32> device_product = math.matmul(device_left, right.gpu($GPU_INDEX))
    if tracked
        device_product = device_product.untrack()
    tensor<real32> product = device_product.cpu()
    for i in range(rows)
        for j in range(columns)
            real32 total = left[i, 0].item() * right[0, j].item()
            for k in range(1, inner)
                real32 term = left[i, k].item() * right[k, j].item()
                total = total + term
            if not same_bits32(product[i, j].item(), total)
                return false
    return real32(1) / product[0, 0].item() < real32(0)

execution_policy(int32(1))
// 1: per-output kernel, 2: tiled kernel, 3: split-inner kernel.
print(fold_matches32(6, 3, 8, true))
print(NL)
print(fold_matches32(40, 20, 40, false))
print(NL)
print(fold_matches32(6, 300, 8, true))
print(NL)
execution_policy(int32(0))
QUI
expect_native_events "Math native real-GPU signed zeros" "$TMP/native-signed-zero-real-gpu.qui" 3 \
    "gemm-simple metal real32=1" "gemm-tiled metal real32=1" \
    "gemm-split-inner metal real32=1" "matmul metal real32=3"

# Default Metal GEMM numerics. Math's own kernels serve every shape in the
# default (fast) policy and no product reaches MPS: the per-output and tiled
# kernels add the inner axis in ascending order like the CPU and must equal it
# bit for bit, large products included.
#
# Reordered sums (the split-inner kernel here, MPS when opted in below) cannot
# equal the CPU bit for bit. Each is held, per element, to the a-priori error
# bound of its summation order and the CPU's (Higham, Accuracy and Stability of
# Numerical Algorithms, 2nd ed., sec. 3.1):
#     |metal - cpu| <= (gamma(h_cpu) + gamma(h_metal)) * (|A| |B|)[i, j]
#     gamma(n) = n u / (1 - n u),  u = 2^-24 (real32 unit roundoff)
# where h is the longest chain of roundings one product goes through in that
# order. The bound holds for every input whose products and partial sums stay
# clear of real32 underflow and overflow. It grows with the inner length k and
# scales with the summed magnitudes rather than with the result, so outputs
# that cancel to near zero need no separate absolute term. A fixed rtol
# measured on one kind of input is not a bound: on same-sign inputs the CPU's
# ascending fold drifts by a roughly fixed share of k u.
#
# The CPU rounds every product and adds the k products in ascending order from
# -0: h_cpu = k. The split-inner kernel (few outputs, long inner axis: weight
# gradients) adds 256 fixed strided partial sums of ceil(k / 256) products each
# and then a fixed 8-level tree: h = ceil(k / 256) + 8. It is deterministic and
# must also repeat bit for bit.
#
# The split-inner cases cover both kinds of input. Mixed-sign pattern32 inputs
# (items 5 and 6) stay far inside the bound. Same-sign inputs (items 7 and 8: a
# constant forward and the weight gradient of a mean loss) are where the CPU's
# fold drifts furthest, by a roughly fixed share of the bound at every k; there
# Metal is the closer one to the exact sum, and a fixed rtol fitted to
# mixed-sign inputs fails.
cat > "$TMP/native-gemm-numerics-real-gpu.qui" <<QUI
import math

tensor<real32> pattern32(int rows, int columns, int seed)
    tensor<real32> value = tensor.zeros<real32>([nat(rows), nat(columns)])
    for i in range(rows)
        for j in range(columns)
            int h = (i * 131 + j * 71 + seed * 17) % 97
            value[i, j] = real32(h) / real32(48) - real32(1)
    return value

real32 max_difference(tensor<real32> left, tensor<real32> right)
    return math.max_all(math.abs(left - right)).item()

bool same32(tensor<real32> left, tensor<real32> right)
    return max_difference(left, right) == real32(0)

bool within_scaled(tensor<real32> actual, tensor<real32> expected, tensor<real32> scale, real32 rtol)
    return math.max_all(math.abs(actual - expected) - scale * rtol).item() <= real32(0)

// gamma(n) = n u / (1 - n u), u = 2^-24: n roundings in a row.
real32 gamma(int n)
    real32 nu = real32(n) / real32(16777216)
    return nu / (real32(1) - nu)

// A-priori bound of the CPU's ascending fold (h = k) against the split-inner
// order (h = ceil(k / 256) + 8), relative to (|A| |B|)[i, j].
real32 split_rtol(int k)
    return gamma(k) + gamma((k + 255) / 256 + 8)

// a[m, k] @ w[n, k]^T on the device equals the CPU bit for bit.
bool product_same(int m, int k, int n)
    tensor<real32> a = pattern32(m, k, 1)
    tensor<real32> w = pattern32(n, k, 2)
    tensor<real32> expected = math.matmul(a, w.transpose(0, 1))
    tensor<real32> actual = math.matmul(a.gpu($GPU_INDEX), w.gpu($GPU_INDEX).transpose(0, 1))
    return actual.device() == $GPU_INDEX and same32(actual.cpu(), expected)

// 1: per-output kernel, 2: tiled kernel, 3: large product (MPS-sized: rows and
// columns >= 64, inner >= 32, m*n*k >= 2^21) on the tiled kernel.
print(product_same(45, 16, 2))
print(NL)
print(product_same(100, 40, 90))
print(NL)
print(product_same(256, 128, 192))
print(NL)

// 4: tracked FC 128x128 -> 128 (every product MPS-sized): forward, dA = G W
// and dW = G^T A on the tiled kernel equal the CPU bit for bit.
tensor<real32> fa = pattern32(128, 128, 6)
tensor<real32> fw = pattern32(128, 128, 7)
tensor<real32> fg = pattern32(128, 128, 8)
tensor<real32> ga = fa.gpu($GPU_INDEX).track()
tensor<real32> gw = fw.gpu($GPU_INDEX).track()
tensor<real32> gy = math.matmul(ga, gw.transpose(0, 1))
math.sum(gy * fg.gpu($GPU_INDEX)).backward(&ga, &gw)
tensor<real32> ca = fa.track()
tensor<real32> cw = fw.track()
tensor<real32> cy = math.matmul(ca, cw.transpose(0, 1))
math.sum(cy * fg).backward(&ca, &cw)
print(same32(gy.untrack().cpu(), cy.untrack()) and same32(ga.grad.cpu(), ca.grad) and same32(gw.grad.cpu(), cw.grad))
print(NL)

// 5: split-inner forward within split_rtol(3000), and bitwise repeatable.
tensor<real32> sa = pattern32(9, 3000, 1)
tensor<real32> sw = pattern32(4, 3000, 2)
tensor<real32> split_cpu = math.matmul(sa, sw.transpose(0, 1))
tensor<real32> split_scale = math.matmul(math.abs(sa), math.abs(sw).transpose(0, 1))
tensor<real32> split_first = math.matmul(sa.gpu($GPU_INDEX), sw.gpu($GPU_INDEX).transpose(0, 1)).cpu()
tensor<real32> split_second = math.matmul(sa.gpu($GPU_INDEX), sw.gpu($GPU_INDEX).transpose(0, 1)).cpu()
print(within_scaled(split_first, split_cpu, split_scale, split_rtol(3000)) and same32(split_first, split_second))
print(NL)

// 6: split-inner FC weight gradient dW = G^T A over a batch of 1024 rows.
tensor<real32> ba = pattern32(1024, 16, 3)
tensor<real32> bw = pattern32(2, 16, 4)
tensor<real32> bg = pattern32(1024, 2, 5)
tensor<real32> device_weight = bw.gpu($GPU_INDEX).track()
math.sum(math.matmul(ba.gpu($GPU_INDEX), device_weight.transpose(0, 1)) * bg.gpu($GPU_INDEX)).backward(&device_weight)
tensor<real32> host_weight = bw.track()
math.sum(math.matmul(ba, host_weight.transpose(0, 1)) * bg).backward(&host_weight)
tensor<real32> weight_scale = math.matmul(math.abs(bg).transpose(0, 1), math.abs(ba))
print(within_scaled(device_weight.grad.cpu(), host_weight.grad, weight_scale, split_rtol(1024)))
print(NL)

// Same-sign inputs, where the CPU's ascending fold drifts furthest from the
// split-inner order (a roughly fixed share of split_rtol(k) at every k).
// 7: constant forward 9x3000 @ 3000x4 (a = 0.1, w = 1).
tensor<real32> ka = tensor.ones<real32>([9, 3000]) * real32(0.1)
tensor<real32> kw = tensor.ones<real32>([4, 3000])
tensor<real32> constant_cpu = math.matmul(ka, kw.transpose(0, 1))
tensor<real32> constant_metal = math.matmul(ka.gpu($GPU_INDEX), kw.gpu($GPU_INDEX).transpose(0, 1)).cpu()
print(within_scaled(constant_metal, constant_cpu, math.matmul(math.abs(ka), math.abs(kw).transpose(0, 1)), split_rtol(3000)))
print(NL)

// 8: FC weight gradient of a mean loss over a batch of 4096 rows
// (A = 0.1, G = 1 / 8192).
tensor<real32> ma = tensor.ones<real32>([4096, 16]) * real32(0.1)
tensor<real32> mw = tensor.ones<real32>([2, 16])
tensor<real32> mean_device = mw.gpu($GPU_INDEX).track()
math.mean(math.matmul(ma.gpu($GPU_INDEX), mean_device.transpose(0, 1))).backward(&mean_device)
tensor<real32> mean_host = mw.track()
math.mean(math.matmul(ma, mean_host.transpose(0, 1))).backward(&mean_host)
tensor<real32> mean_g = tensor.ones<real32>([4096, 2]) * (real32(1) / real32(8192))
print(within_scaled(mean_device.grad.cpu(), mean_host.grad, math.matmul(math.abs(mean_g).transpose(0, 1), math.abs(ma)), split_rtol(4096)))
print(NL)
QUI
expect_native_events "Math native real-GPU GEMM numerics" "$TMP/native-gemm-numerics-real-gpu.qui" 8 \
    "gemm-simple metal real32=3" "gemm-tiled metal real32=5" \
    "gemm-split-inner metal real32=5" "gemm-mps metal real32=0"

# Apple's MPSMatrixMultiplication is an explicit opt-in (QUIDRA_MATH_METAL_MPS=1)
# and then serves only large products in the fast policy. Its accumulation
# order and multiply-add use are Apple's and unknown, so it is held to the
# a-priori bound above for an arbitrary order (any real32 evaluation of a
# k-term dot product, fused multiply-adds included, has h <= k):
#     mps_rtol(k) = 2 gamma(k)
# never to bitwise equality. A fixed rtol is not a bound here either: the error
# grows with k, which the near-constant case 3 below exercises. In the tracked
# FC (case 2), dA has outputs that cancel to near zero and differ from the CPU
# by many times their own size, so a result-relative tolerance cannot hold.
cat > "$TMP/native-gemm-mps-real-gpu.qui" <<QUI
import math

extern void execution_policy(int32 policy) = "qcore_execution_policy_set"

tensor<real32> pattern32(int rows, int columns, int seed)
    tensor<real32> value = tensor.zeros<real32>([nat(rows), nat(columns)])
    for i in range(rows)
        for j in range(columns)
            int h = (i * 131 + j * 71 + seed * 17) % 97
            value[i, j] = real32(h) / real32(48) - real32(1)
    return value

real32 max_difference(tensor<real32> left, tensor<real32> right)
    return math.max_all(math.abs(left - right)).item()

bool same32(tensor<real32> left, tensor<real32> right)
    return max_difference(left, right) == real32(0)

bool within_scaled(tensor<real32> actual, tensor<real32> expected, tensor<real32> scale, real32 rtol)
    return math.max_all(math.abs(actual - expected) - scale * rtol).item() <= real32(0)

// gamma(n) = n u / (1 - n u), u = 2^-24: n roundings in a row.
real32 gamma(int n)
    real32 nu = real32(n) / real32(16777216)
    return nu / (real32(1) - nu)

// A-priori bound of the CPU's ascending fold against any order of k terms.
real32 mps_rtol(int k)
    return real32(2) * gamma(k)

tensor<real32> a = pattern32(256, 128, 1)
tensor<real32> w = pattern32(192, 128, 2)
tensor<real32> expected = math.matmul(a, w.transpose(0, 1))
tensor<real32> scale = math.matmul(math.abs(a), math.abs(w).transpose(0, 1))

// 1: fast policy, large product: MPS, within mps_rtol(128).
print(within_scaled(math.matmul(a.gpu($GPU_INDEX), w.gpu($GPU_INDEX).transpose(0, 1)).cpu(), expected, scale, mps_rtol(128)))
print(NL)

// 2: tracked FC 128x128 -> 128: forward, dA and dW through MPS, within
// mps_rtol(128) (every product has an inner length of 128).
tensor<real32> fa = pattern32(128, 128, 6)
tensor<real32> fw = pattern32(128, 128, 7)
tensor<real32> fg = pattern32(128, 128, 8)
tensor<real32> ga = fa.gpu($GPU_INDEX).track()
tensor<real32> gw = fw.gpu($GPU_INDEX).track()
tensor<real32> gy = math.matmul(ga, gw.transpose(0, 1))
math.sum(gy * fg.gpu($GPU_INDEX)).backward(&ga, &gw)
tensor<real32> ca = fa.track()
tensor<real32> cw = fw.track()
tensor<real32> cy = math.matmul(ca, cw.transpose(0, 1))
math.sum(cy * fg).backward(&ca, &cw)
bool forward_ok = within_scaled(gy.untrack().cpu(), cy.untrack(), math.matmul(math.abs(fa), math.abs(fw).transpose(0, 1)), mps_rtol(128))
bool left_ok = within_scaled(ga.grad.cpu(), ca.grad, math.matmul(math.abs(fg), math.abs(fw)), mps_rtol(128))
bool right_ok = within_scaled(gw.grad.cpu(), cw.grad, math.matmul(math.abs(fg).transpose(0, 1), math.abs(fa)), mps_rtol(128))
print(forward_ok and left_ok and right_ok)
print(NL)

// 3: near-constant same-sign product with a long inner axis, within
// mps_rtol(8192).
tensor<real32> na = pattern32(64, 8192, 1) * real32(0.001) + real32(0.1)
tensor<real32> nw = pattern32(64, 8192, 2) * real32(0.001) + real32(1)
tensor<real32> near_cpu = math.matmul(na, nw.transpose(0, 1))
tensor<real32> near_mps = math.matmul(na.gpu($GPU_INDEX), nw.gpu($GPU_INDEX).transpose(0, 1)).cpu()
print(within_scaled(near_mps, near_cpu, math.matmul(math.abs(na), math.abs(nw).transpose(0, 1)), mps_rtol(8192)))
print(NL)

// 4: small products stay on Math's kernels even when opted in: bitwise CPU.
tensor<real32> sa = pattern32(45, 16, 3)
tensor<real32> sw = pattern32(2, 16, 4)
print(same32(math.matmul(sa.gpu($GPU_INDEX), sw.gpu($GPU_INDEX).transpose(0, 1)).cpu(), math.matmul(sa, sw.transpose(0, 1))))
print(NL)

// 5: the deterministic policy never uses MPS, opted in or not: the large
// product runs on the tiled kernel and equals the CPU bit for bit.
execution_policy(int32(1))
tensor<real32> deterministic = math.matmul(a.gpu($GPU_INDEX), w.gpu($GPU_INDEX).transpose(0, 1)).cpu()
execution_policy(int32(0))
print(same32(deterministic, expected))
print(NL)
QUI
mps_events=("gemm-simple metal real32=1")
# Apple GPUs always support MPS; another Metal device may decline it, and the
# opted-in products then fall back to the tiled kernel.
if [[ "$(uname -m)" == "arm64" ]]; then
    mps_events+=("gemm-mps metal real32=5" "gemm-tiled metal real32=1")
fi
QUIDRA_MATH_METAL_MPS=1 expect_native_events "Math native real-GPU MPS opt-in" \
    "$TMP/native-gemm-mps-real-gpu.qui" 5 "${mps_events[@]}"

cat > "$TMP/native-dispatch-real-gpu.qui" <<QUI
import math

tensor<real32> pattern32(int rows, int columns, int seed)
    tensor<real32> value = tensor.zeros<real32>([nat(rows), nat(columns)])
    for i in range(rows)
        for j in range(columns)
            value[i, j] = real32((i * 131 + j * 71 + seed * 17) % 97) / real32(48) - real32(1)
    return value

// 1: every native forward and backward on the device.
tensor<real32> x = pattern32(6, 5, 1).gpu($GPU_INDEX).track()
tensor<real32> w = pattern32(4, 5, 2).gpu($GPU_INDEX).track()
tensor<real32> y = math.matmul(x, w.transpose(0, 1))
(math.sum(math.max_last(y)) + math.sum(math.min_last(y)) + math.max_all(y) - math.min_all(y) + math.mean(math.sum_last(y))).backward(&x, &w)
print(x.grad.device() == $GPU_INDEX and w.grad.device() == $GPU_INDEX)
print(NL)

// 2: a large product in the default (fast) policy stays on Math's own tiled
// kernel; Apple's MPS is never used unless the process opts in.
tensor<real32> big = pattern32(256, 128, 3).gpu($GPU_INDEX)
tensor<real32> big_right = pattern32(128, 192, 4).gpu($GPU_INDEX)
print(math.matmul(big, big_right).shape()[1] == 192)
print(NL)
QUI
expect_native_events "Math native real-GPU dispatch" "$TMP/native-dispatch-real-gpu.qui" 2 \
    "matmul metal real32" "sum metal real32" "mean metal real32" \
    "sum_last metal real32" "max_last metal real32" "min_last metal real32" \
    "max_all metal real32" "min_all metal real32" "matmul-backward metal real32" \
    "reduce-backward metal real32" "extrema-backward metal real32" \
    "gemm-tiled metal real32=1" "gemm-mps metal real32=0"

# Tracked non-contiguous views on Metal are served natively (no per-element
# host selection): each operation sees only a view, so each forward event
# appears exactly once.
cat > "$TMP/native-dispatch-views-real-gpu.qui" <<QUI
import math

tensor<real32> base = tensor.zeros<real32>([5, 6])
for i in range(5)
    for j in range(6)
        base[i, j] = real32((i * 7 + j * 3) % 11) - real32(5)
tensor<real32> tracked = base.gpu($GPU_INDEX).track()
tensor<real32> view = tracked.transpose(0, 1)
tensor<real32> cube = tensor.ones<real32>([2, 3, 4], gpu = $GPU_INDEX).track()
tensor<real32> right = tensor.ones<real32>([2, 3], gpu = $GPU_INDEX)
tensor<real32> loss = math.sum(view) + math.mean(view) + math.sum_last(view).gather([7], []) + math.max_last(view).gather([3], []) + math.min_last(view).gather([11], []) + math.max_all(view) + math.min_all(view) + math.matmul(cube.transpose(0, 2), right).gather([5], [])
loss.backward(&tracked, &cube)
print(tracked.grad.device() == $GPU_INDEX and cube.grad.cpu()[0, 0, 0].item() == real32(0))
print(NL)
QUI
expect_native_events "Math native real-GPU view dispatch" "$TMP/native-dispatch-views-real-gpu.qui" 1 \
    "sum metal real32=1" "mean metal real32=1" "sum_last metal real32=1" \
    "max_last metal real32=1" "min_last metal real32=1" "max_all metal real32=1" \
    "min_all metal real32=1" "matmul metal real32=1" "matmul-backward metal real32=1" \
    "reduce-backward metal real32=3" "extrema-backward metal real32=4"

fi

echo "math real GPU integration: ok on gpu($GPU_INDEX)"

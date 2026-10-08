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

# Math-native products on the fake GPU run the host reference kernels behind
# the backend-neutral bridge: results and gradients stay on the device and
# equal the CPU bit for bit (float32 and float64).
cat > "$TMP/native-matmul-device.qui" <<'QUI'
import math

tensor<float32> pattern32(int rows, int columns, int seed)
    tensor<float32> value = tensor.zeros<float32>([rows, columns])
    for i in range(rows)
        for j in range(columns)
            int h = (i * 131 + j * 71 + seed * 17) % 97
            value[i, j] = float32(h) / float32(48) - float32(1)
    return value

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

// 1-2: forward products on the fake GPU equal CPU exactly and stay on device.
tensor<float32> a = pattern32(6, 5, 1)
tensor<float32> w = pattern32(3, 5, 2)
tensor<float32> device_product = math.matmul(a.gpu(0), w.gpu(0).transpose(0, 1))
print(device_product.device() == 0)
print(NL)
print(same32(device_product.cpu(), math.matmul(a, w.transpose(0, 1))))
print(NL)

// 3-4: tracked FC pattern gradients equal CPU exactly.
tensor<float32> xa = a.gpu(0).track()
tensor<float32> xw = w.gpu(0).track()
math.sum(math.matmul(xa, xw.transpose(0, 1)) * pattern32(6, 3, 3).gpu(0)).backward(&xa, &xw)
tensor<float32> ca = a.track()
tensor<float32> cw = w.track()
math.sum(math.matmul(ca, cw.transpose(0, 1)) * pattern32(6, 3, 3)).backward(&ca, &cw)
print(same32(xa.grad.cpu(), ca.grad))
print(NL)
print(same32(xw.grad.cpu(), cw.grad))
print(NL)

// 5: float64 products and gradients on the fake GPU.
tensor<float> da = pattern64(4, 3, 1).gpu(0).track()
tensor<float> db = pattern64(3, 2, 2).gpu(0).track()
math.mean(math.matmul(da, db)).backward(&da, &db)
tensor<float> ha = pattern64(4, 3, 1).track()
tensor<float> hb = pattern64(3, 2, 2).track()
math.mean(math.matmul(ha, hb)).backward(&ha, &hb)
print(same64(da.grad.cpu(), ha.grad) and same64(db.grad.cpu(), hb.grad))
print(NL)
QUI
expect_true_lines "Math native fake-GPU matmul" "$TMP/native-matmul-device.qui" 5

# Native whole-tensor sum/mean on the fake GPU equal the CPU exactly.
cat > "$TMP/native-sum-mean-device.qui" <<'QUI'
import math

tensor<float32> pattern32(int rows, int columns, int seed)
    tensor<float32> value = tensor.zeros<float32>([rows, columns])
    for i in range(rows)
        for j in range(columns)
            int h = (i * 131 + j * 71 + seed * 17) % 97
            value[i, j] = float32(h) / float32(48) - float32(1)
    return value

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

// 1: whole-tensor sum and mean equal CPU exactly.
tensor<float32> r = pattern32(5, 7, 4)
tensor<float32> rg = r.gpu(0)
print(math.sum(rg).cpu().item() == math.sum(r).item() and math.mean(rg).cpu().item() == math.mean(r).item())
print(NL)

// 2: mean backward on the device equals CPU exactly (upstream / count).
tensor<float32> mean_device = rg.track()
math.mean(mean_device).backward(&mean_device)
tensor<float32> mean_host = r.track()
math.mean(mean_host).backward(&mean_host)
print(mean_device.grad.device() == 0 and same32(mean_device.grad.cpu(), mean_host.grad))
print(NL)
QUI
expect_true_lines "Math native fake-GPU sum/mean" "$TMP/native-sum-mean-device.qui" 2

# Native last-axis and whole-tensor extrema on the fake GPU equal the CPU
# exactly, including tie gradients.
cat > "$TMP/native-last-axis-device.qui" <<'QUI'
import math

tensor<float32> pattern32(int rows, int columns, int seed)
    tensor<float32> value = tensor.zeros<float32>([rows, columns])
    for i in range(rows)
        for j in range(columns)
            int h = (i * 131 + j * 71 + seed * 17) % 97
            value[i, j] = float32(h) / float32(48) - float32(1)
    return value

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

// 1-4: last-axis and whole-tensor extrema equal CPU exactly, keep shapes, and
// stay on device.
tensor<float32> r = pattern32(5, 7, 4)
tensor<float32> rg = r.gpu(0)
tensor<float32> rs = math.sum_last(rg)
print(rs.device() == 0 and rs.shape()[1] == 7 and same32(rs.cpu(), math.sum_last(r)))
print(NL)
print(same32(math.max_last(rg).cpu(), math.max_last(r)) and same32(math.min_last(rg).cpu(), math.min_last(r)))
print(NL)
print(math.max_all(rg).cpu().item() == math.max_all(r).item() and math.min_all(rg).cpu().item() == math.min_all(r).item())
print(NL)

// 5-7: reduction gradients on device equal CPU exactly (ties included).
tensor<float32> ties = tensor.zeros<float32>([2, 3])
ties[0, 1] = float32(4)
ties[0, 2] = float32(4)
ties[1, 0] = float32(-1)
ties[1, 1] = float32(-1)
tensor<float32> tg = ties.gpu(0).track()
math.sum(math.max_last(tg) * pattern32(2, 3, 5).gpu(0)).backward(&tg)
tensor<float32> tc = ties.track()
math.sum(math.max_last(tc) * pattern32(2, 3, 5)).backward(&tc)
print(same32(tg.grad.cpu(), tc.grad) and tc.grad[0, 2].item() == float32(0))
print(NL)
tensor<float32> sg = rg.track()
math.sum(math.sum_last(sg) * pattern32(5, 7, 6).gpu(0)).backward(&sg)
tensor<float32> sc = r.track()
math.sum(math.sum_last(sc) * pattern32(5, 7, 6)).backward(&sc)
print(same32(sg.grad.cpu(), sc.grad))
print(NL)
tensor<float32> mg = rg.track()
math.min_all(mg).backward(&mg)
tensor<float32> mc = r.track()
math.min_all(mc).backward(&mc)
print(same32(mg.grad.cpu(), mc.grad) and math.sum(mc.grad).item() == float32(1))
print(NL)
QUI
expect_true_lines "Math native fake-GPU last-axis reductions" "$TMP/native-last-axis-device.qui" 6

# Views on the fake GPU: results consumed through transposed views (the
# upstream device gradient arrives non-contiguous) and tracked non-contiguous
# inputs read through a value-only device copy. Everything equals the CPU
# exactly and stays on device.
cat > "$TMP/native-views-device.qui" <<'QUI'
import math

tensor<float32> pattern32(int rows, int columns, int seed)
    tensor<float32> value = tensor.zeros<float32>([rows, columns])
    for i in range(rows)
        for j in range(columns)
            int h = (i * 131 + j * 71 + seed * 17) % 97
            value[i, j] = float32(h) / float32(48) - float32(1)
    return value

bool same32(tensor<float32> left, tensor<float32> right)
    return math.max_all(math.abs(left - right)).item() == float32(0)

// 1-4: transposed matmul / sum_last / max_last / min_last results feed the
// loss on the device; gradients equal the CPU.
tensor<float32> a_data = pattern32(2, 3, 1)
tensor<float32> b_data = pattern32(3, 4, 2)
tensor<float32> across = pattern32(4, 2, 3)
tensor<float32> ga = a_data.gpu(0).track()
tensor<float32> gb = b_data.gpu(0).track()
math.sum(math.matmul(ga, gb).transpose(0, 1) * across.gpu(0)).backward(&ga, &gb)
tensor<float32> ca = a_data.track()
tensor<float32> cb = b_data.track()
math.sum(math.matmul(ca, cb).transpose(0, 1) * across).backward(&ca, &cb)
print(ga.grad.device() == 0 and same32(ga.grad.cpu(), ca.grad) and same32(gb.grad.cpu(), cb.grad))
print(NL)
tensor<float32> x_data = pattern32(3, 4, 4)
tensor<float32> weights = pattern32(4, 3, 5)
tensor<float32> gs = x_data.gpu(0).track()
math.sum(math.sum_last(gs).transpose(0, 1) * weights.gpu(0)).backward(&gs)
tensor<float32> cs = x_data.track()
math.sum(math.sum_last(cs).transpose(0, 1) * weights).backward(&cs)
print(same32(gs.grad.cpu(), cs.grad))
print(NL)
tensor<float32> gm = x_data.gpu(0).track()
math.sum(math.max_last(gm).transpose(0, 1) * weights.gpu(0)).backward(&gm)
tensor<float32> cm = x_data.track()
math.sum(math.max_last(cm).transpose(0, 1) * weights).backward(&cm)
print(same32(gm.grad.cpu(), cm.grad))
print(NL)
tensor<float32> gn = x_data.gpu(0).track()
math.sum(math.min_last(gn).transpose(0, 1) * weights.gpu(0)).backward(&gn)
tensor<float32> cn = x_data.track()
math.sum(math.min_last(cn).transpose(0, 1) * weights).backward(&cn)
print(same32(gn.grad.cpu(), cn.grad))
print(NL)

// 5-6: tracked transposed inputs to matmul and to every reduction.
tensor<float32> r_data = pattern32(6, 5, 6)
tensor<float32> rw = pattern32(5, 6, 7)
tensor<float32> right = pattern32(6, 2, 8)
tensor<float32> gr = r_data.gpu(0).track()
tensor<float32> gv = gr.transpose(0, 1)
tensor<float32> device_loss = math.sum(math.sum_last(gv) * rw.gpu(0)) + math.sum(math.max_last(gv) * rw.gpu(0)) - math.sum(math.min_last(gv) * rw.gpu(0)) + math.mean(gv) + math.max_all(gv) + math.min_all(gv) * float32(3) + math.sum(math.matmul(gv, right.gpu(0)))
device_loss.backward(&gr)
tensor<float32> cr = r_data.track()
tensor<float32> cv = cr.transpose(0, 1)
tensor<float32> host_loss = math.sum(math.sum_last(cv) * rw) + math.sum(math.max_last(cv) * rw) - math.sum(math.min_last(cv) * rw) + math.mean(cv) + math.max_all(cv) + math.min_all(cv) * float32(3) + math.sum(math.matmul(cv, right))
host_loss.backward(&cr)
print(device_loss.untrack().cpu().item() == host_loss.untrack().item())
print(NL)
print(gr.grad.device() == 0 and same32(gr.grad.cpu(), cr.grad))
print(NL)
QUI
expect_true_lines "Math native fake-GPU views" "$TMP/native-views-device.qui" 6

# Whole-tensor sums keep the portable odd-carry definition on the device.
cat > "$TMP/native-sum-carry-device.qui" <<'QUI'
import math

bool is_nan(float32 value)
    return value != value

// 1: odd-carry definition on the device.
float32 infinity = float32(1) / float32(0)
tensor<float32> carried = tensor.ones<float32>([1, 3])
carried[0, 2] = infinity
tensor<float32> paired = tensor.ones<float32>([1, 2])
paired[0, 1] = infinity
print(is_nan(math.sum(carried.gpu(0)).cpu().item()) and is_nan(math.mean(carried.gpu(0)).cpu().item()) and math.sum(paired.gpu(0)).cpu().item() == infinity)
print(NL)
QUI
expect_true_lines "Math native fake-GPU sum carry" "$TMP/native-sum-carry-device.qui" 1

# Shapes the native bridges do not claim keep the portable behaviour on the
# device too (see integration.sh for why both outcomes are accepted).
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

write_shape_case device-inner-mismatch 'tensor<float32> a = tensor.ones<float32>([2, 3], gpu = 0).track()
tensor<float32> b = tensor.ones<float32>([4, 5], gpu = 0).track()
tensor<float32> c = math.matmul(a, b)
math.sum(c).backward(&a, &b)
print("{c.shape()[0]} {c.shape()[1]} {a.grad.cpu().gather([0], []).item()} {b.grad.cpu().gather([0], []).item()}{NL}")'
expect_diagnostic_or_develop "fake-GPU matmul inner mismatch" "$TMP/device-inner-mismatch.qui" \
    "math.matmul inner dimensions do not match" "2 5 5.0 2.0"
write_shape_case device-zero-rows 'tensor<float32> c = math.matmul(tensor.ones<float32>([0, 3], gpu = 0), tensor.ones<float32>([3, 2], gpu = 0))
print("{c.shape()[0]}{NL}")'
expect_diagnostic_or_develop "fake-GPU matmul zero rows" "$TMP/device-zero-rows.qui" \
    "math.matmul requires positive tensor extents" ""
write_shape_case device-zero-inner 'tensor<float32> c = math.matmul(tensor.ones<float32>([2, 0], gpu = 0), tensor.ones<float32>([0, 2], gpu = 0))
print("{c.shape()[0]}{NL}")'
expect_diagnostic_or_develop "fake-GPU matmul zero inner" "$TMP/device-zero-inner.qui" \
    "math.matmul inner dimensions do not match" ""

cat > "$TMP/native-empty-device.qui" <<'QUI'
import math

tensor<float32> rows_empty = tensor.ones<float32>([0, 3], gpu = 0)
tensor<float32> s = math.sum_last(rows_empty)
tensor<float32> m = math.max_last(rows_empty)
print(s.shape()[1] == 3 and m.shape()[0] == 0 and m.device() == 0)
print(NL)
tensor<float32> width_empty = tensor.ones<float32>([3, 0], gpu = 0)
tensor<float32> ws = math.sum_last(width_empty)
tensor<float32> wn = math.min_last(width_empty)
print(ws.shape()[0] == 3 and ws.shape()[1] == 0 and wn.shape()[1] == 0)
print(NL)
tensor<float32> tracked_empty = tensor.ones<float32>([0, 3], gpu = 0).track()
tensor<float32> total = math.sum(tracked_empty)
total.backward(&tracked_empty)
print(total.untrack().cpu().item() == float32(0) and tracked_empty.grad.shape()[1] == 3)
print(NL)
QUI
expect_true_lines "Math native fake-GPU empty reductions" "$TMP/native-empty-device.qui" 3

# Signed zeros on the fake GPU: products equal the portable fold p0 + p1 + ...
# bit for bit, including the sign of exact-zero outputs.
cat > "$TMP/native-signed-zero-device.qui" <<'QUI'
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

// Row i: all negative (i % 3 == 0), all positive (1), or alternating (2).
// Column j: all +0, all -0, mixed signed zeros, or small positive integers.
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

bool fold_matches32(tensor<float32> left, tensor<float32> right, tensor<float32> device_product)
    tensor<float32> product = device_product.cpu()
    int inner = left.shape()[1]
    for i in range(left.shape()[0])
        for j in range(right.shape()[1])
            float32 total = left[i, 0].item() * right[0, j].item()
            for k in range(1, inner)
                float32 term = left[i, k].item() * right[k, j].item()
                total = total + term
            if not same_bits32(product[i, j].item(), total)
                return false
    return device_product.device() == 0 and float32(1) / product[0, 0].item() < float32(0)

tensor<float> widen(tensor<float32> value)
    tensor<float> result = tensor.zeros<float>(value.shape())
    for i in range(value.shape()[0])
        for j in range(value.shape()[1])
            result[i, j] = float(value[i, j].item())
    return result

bool fold_matches64(tensor<float> left, tensor<float> right, tensor<float> device_product)
    tensor<float> product = device_product.cpu()
    int inner = left.shape()[1]
    for i in range(left.shape()[0])
        for j in range(right.shape()[1])
            float total = left[i, 0].item() * right[0, j].item()
            for k in range(1, inner)
                float term = left[i, k].item() * right[k, j].item()
                total = total + term
            if not same_bits64(product[i, j].item(), total)
                return false
    return 1.0 / product[0, 0].item() < 0.0

// 1-2: untracked and tracked float32, short and depth-blocked inner axes.
tensor<float32> l3 = signed_left32(6, 3)
tensor<float32> r3 = signed_right32(3, 8)
print(fold_matches32(l3, r3, math.matmul(l3.gpu(0), r3.gpu(0))) and fold_matches32(l3, r3, math.matmul(l3.gpu(0).track(), r3.gpu(0)).untrack()))
print(NL)
tensor<float32> l300 = signed_left32(6, 300)
tensor<float32> r300 = signed_right32(300, 8)
tensor<float32> r300_storage = r300.transpose(0, 1).contiguous().gpu(0).track()
print(fold_matches32(l300, r300, math.matmul(l300.gpu(0), r300_storage.transpose(0, 1)).untrack()))
print(NL)

// 3: float64.
tensor<float> l64 = widen(l300)
tensor<float> r64 = widen(r300)
print(fold_matches64(l64, r64, math.matmul(l64.gpu(0).track(), r64.gpu(0)).untrack()))
print(NL)
QUI
expect_true_lines "Math native fake-GPU signed zeros" "$TMP/native-signed-zero-device.qui" 3

# Native dispatch on the fake GPU. Values equal the portable composition, so
# QUIDRA_MATH_TEST_NATIVE_TRACE shows which path served each call: every
# listed event must appear, and an event suffixed "=N" exactly N times.
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

cat > "$TMP/native-dispatch-device.qui" <<'QUI'
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
tensor<float32> x = pattern32(6, 5, 1).gpu(0).track()
tensor<float32> w = pattern32(4, 5, 2).gpu(0).track()
tensor<float32> y = math.matmul(x, w.transpose(0, 1))
(math.sum(math.max_last(y)) + math.sum(math.min_last(y)) + math.max_all(y) - math.min_all(y) + math.mean(math.sum_last(y))).backward(&x, &w)
print(x.grad.device() == 0 and w.grad.device() == 0)
print(NL)

// 2: float64 on the device (Core's backward(track = true) is CPU-only, so
// the tracked callbacks are covered by the CPU suite).
tensor<float> a = pattern64(3, 4, 3).gpu(0).track()
tensor<float> b = pattern64(4, 2, 4).gpu(0).track()
tensor<float> p = math.matmul(a, b)
(math.sum(p * p) + math.max_all(p) - math.min_last(p).gather([1], []) + math.mean(math.sum_last(p))).backward(&a, &b)
print(a.grad.device() == 0 and b.grad.shape()[1] == 2)
print(NL)
QUI
expect_native_events "Math native fake-GPU dispatch" "$TMP/native-dispatch-device.qui" 2 \
    "matmul test float32" "sum test float32" "mean test float32" \
    "sum_last test float32" "max_last test float32" "min_last test float32" \
    "max_all test float32" "min_all test float32" "matmul-backward test float32" \
    "reduce-backward test float32" "extrema-backward test float32" \
    "matmul test float64" "sum test float64" "mean test float64" \
    "sum_last test float64" "max_all test float64" "min_last test float64" \
    "matmul-backward test float64" "reduce-backward test float64" \
    "extrema-backward test float64"

# Tracked non-contiguous views on the fake GPU are served natively (no
# per-element host selection): each operation sees only a view, so each
# forward event appears exactly once, including a wide tracked max_last.
cat > "$TMP/native-dispatch-views-device.qui" <<'QUI'
import math

tensor<float32> base = tensor.zeros<float32>([5, 6])
for i in range(5)
    for j in range(6)
        base[i, j] = float32((i * 7 + j * 3) % 11) - float32(5)
tensor<float32> tracked = base.gpu(0).track()
tensor<float32> view = tracked.transpose(0, 1)
tensor<float32> cube = tensor.ones<float32>([2, 3, 4], gpu = 0).track()
tensor<float32> right = tensor.ones<float32>([2, 3], gpu = 0)
tensor<float32> wide = tensor.ones<float32>([4096, 8], gpu = 0).track()
tensor<float32> loss = math.sum(view) + math.mean(view) + math.sum_last(view).gather([7], []) + math.max_last(view).gather([3], []) + math.min_last(view).gather([11], []) + math.max_all(view) + math.min_all(view) + math.matmul(cube.transpose(0, 2), right).gather([5], []) + math.max_last(wide.transpose(0, 1)).gather([1], [])
loss.backward(&tracked, &cube, &wide)
print(tracked.grad.device() == 0 and cube.grad.cpu()[0, 0, 0].item() == float32(0) and wide.grad.cpu()[0, 0].item() == float32(1))
print(NL)
QUI
expect_native_events "Math native fake-GPU view dispatch" "$TMP/native-dispatch-views-device.qui" 1 \
    "sum test float32=1" "mean test float32=1" "sum_last test float32=1" \
    "max_last test float32=2" "min_last test float32=1" "max_all test float32=1" \
    "min_all test float32=1" "matmul test float32=1" "matmul-backward test float32=1" \
    "reduce-backward test float32=3" "extrema-backward test float32=5"

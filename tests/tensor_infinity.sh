#!/usr/bin/env bash
# Preserve the documented pairwise reduction order under the NaN-only policy.
set -euo pipefail

QUIDRA="${1:?usage: tensor_infinity.sh /path/to/quidra [cpu|gpu]}"
MODE="${2:-cpu}"
REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export QUIDRA_CACHE_DIR="$TMP/run-cache"
case "$MODE" in
    cpu) placement="" ;;
    gpu) placement=".gpu(0)"; export QUIDRA_TEST_FAKE_GPU_COUNT=1 ;;
    *) echo "unknown device mode: $MODE" >&2; exit 2 ;;
esac

# Paired infinity remains a valid value. Finite odd carries do not introduce
# NaN, including when the input is a tracked transposed view.
cat > "$TMP/paired.qui" <<QUI
import math
real32 infinity = real32(1) / real32(0)
tensor<real32> paired_host = tensor.ones<real32>([1, 2])
paired_host[0, 1] = infinity
tensor<real32> leading_host = tensor.ones<real32>([1, 5])
leading_host[0, 0] = infinity
tensor<real32> paired = paired_host${placement}
tensor<real32> leading = leading_host${placement}
tensor<real32> zero_input = tensor.zeros<real32>([1, 2])${placement}
tensor<real32> logs = math.log(zero_input).cpu()
print(logs[0, 0].item() == -infinity and logs[0, 1].item() == -infinity)
print(NL)
print(math.sum(paired).cpu().item() == infinity and math.mean(paired).cpu().item() == infinity)
print(NL)
print(math.sum(leading).cpu().item() == infinity and math.mean(leading).cpu().item() == infinity)
print(NL)
print(math.sum(paired.track().transpose(0, 1)).untrack().cpu().item() == infinity and math.mean(paired.track().transpose(0, 1)).untrack().cpu().item() == infinity)
print(NL)
print(math.sum(leading.track().transpose(0, 1)).untrack().cpu().item() == infinity and math.mean(leading.track().transpose(0, 1)).untrack().cpu().item() == infinity)
print(NL)
QUI
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/paired.qui"
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/paired.qui" > "$TMP/paired.out"
printf 'true\ntrue\ntrue\ntrue\ntrue\n' > "$TMP/expected.out"
if ! cmp -s "$TMP/expected.out" "$TMP/paired.out"; then
    echo "paired/leading infinity changed on $MODE:" >&2
    cat "$TMP/paired.out" >&2
    exit 1
fi

# The portable odd-carry formula is x + x * 0. Carrying infinity therefore
# produces NaN, which must fail at the reduction, NOT at infinity construction.
# Separate processes ensure the first failure cannot hide another path.
for operation in sum mean; do
    for layout in contiguous tracked-view; do
        view=""
        [[ "$layout" != tracked-view ]] || view=".track().transpose(0, 1)"
        cat > "$TMP/carried.qui" <<QUI
import math
real32 infinity = real32(1) / real32(0)
tensor<real32> host = tensor.ones<real32>([1, 3])
host[0, 2] = infinity
tensor<real32> carried = host${placement}${view}
print("before")
tensor<real32> result = math.${operation}(carried)
print("after")
QUI
        QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/carried.qui"
        set +e
        QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/carried.qui" \
            > "$TMP/carried.out" 2> "$TMP/carried.err"
        status=$?
        set -e
        if [[ "$status" -ne 101 || "$(cat "$TMP/carried.out")" != before ]] ||
           ! grep -Fq 'Quidra runtime error[NONFINITE_RESULT]' "$TMP/carried.err" ||
           ! grep -Fq "package operation math.${operation} produced a NaN real32 value" "$TMP/carried.err"; then
            echo "unexpected $MODE $operation $layout odd-carry outcome (status $status):" >&2
            cat "$TMP/carried.out" "$TMP/carried.err" >&2
            exit 1
        fi
    done
done
printf 'tensor infinity contracts (%s): ok\n' "$MODE"

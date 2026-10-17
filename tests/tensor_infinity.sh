#!/usr/bin/env bash
# Infinity is valid under the NaN-only policy; odd trailing carries retain it.
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

# A trailing infinity is a valid value in a pairwise reduction, including
# when its path crosses an odd-numbered level or a transposed tracked view.
for operation in sum mean; do
    for layout in contiguous tracked-view; do
        view=""
        untrack=""
        if [[ "$layout" == tracked-view ]]; then
            view=".track().transpose(0, 1)"
            untrack=".untrack()"
        fi
        cat > "$TMP/carried.qui" <<QUI
import math
real32 infinity = real32(1) / real32(0)
tensor<real32> host = tensor.ones<real32>([1, 3])
host[0, 2] = infinity
tensor<real32> carried = host${placement}${view}
tensor<real32> result = math.${operation}(carried)
print(result${untrack}.cpu().item() == infinity)
print(NL)
QUI
        QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/carried.qui"
        QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/carried.qui" > "$TMP/carried.out"
        if [[ "$(cat "$TMP/carried.out")" != true ]]; then
            echo "unexpected $MODE $operation $layout infinity carry:" >&2
            cat "$TMP/carried.out" >&2
            exit 1
        fi
    done
    # Mixing opposite infinities produces NaN, which must fail on use,
    # *after* the preceding user-visible print, not on construction.
    cat > "$TMP/opposed.qui" <<QUI
import math
real32 infinity = real32(1) / real32(0)
tensor<real32> host = tensor.ones<real32>([1, 2])
host[0, 0] = infinity
host[0, 1] = -infinity
tensor<real32> opposed = host${placement}
print("before")
print(NL)
tensor<real32> result = math.${operation}(opposed)
print("after")
QUI
    QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" check "$TMP/opposed.qui"
    set +e
    QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" "$TMP/opposed.qui" \
        > "$TMP/opposed.out" 2> "$TMP/opposed.err"
    status=$?
    set -e
    if [[ "$status" -ne 101 || "$(cat "$TMP/opposed.out")" != before ]] ||
       ! grep -Fq 'Quidra runtime error[NONFINITE_RESULT]' "$TMP/opposed.err" ||
       ! grep -Fq "package operation math.${operation} produced a NaN real32 value" "$TMP/opposed.err"; then
        echo "unexpected $MODE $operation opposite-infinity NaN outcome (status $status):" >&2
        cat "$TMP/opposed.out" "$TMP/opposed.err" >&2
        exit 1
    fi
done
printf 'tensor infinity contracts (%s): ok\n' "$MODE"

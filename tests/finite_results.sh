#!/usr/bin/env bash
set -euo pipefail
QUIDRA="${1:?usage: finite_results.sh /path/to/quidra [gpu-index]}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
export QUIDRA_CACHE_DIR="$TMP/run-cache"
mkdir "$TMP/packages"
ln -s "$ROOT" "$TMP/packages/math"
PLACEMENT=""
CODE="NONFINITE_RESULT"
if [[ -n "${2:-}" ]]; then
    PLACEMENT=".gpu($2)"
    CODE="TENSOR"
    if [[ "${QUIDRA_TEST_FAKE_GPU_COUNT:-0}" -gt 0 ]]; then CODE="NONFINITE_RESULT"; fi
fi
for OP in exp sum matmul; do
    case "$OP" in
        exp) VALUE="tensor.ones<real32>([2]) * real32(100)"; EXPR="math.exp(value)" ;;
        sum) VALUE="tensor.ones<real32>([2]) * real32(3.0e38)"; EXPR="math.sum(value)" ;;
        matmul) VALUE="tensor.ones<real32>([1, 2]) * real32(3.0e38)"; EXPR="math.matmul(value, tensor.ones<real32>([2, 1])$PLACEMENT)" ;;
    esac
    cat > "$TMP/$OP.qui" <<QUI
import math
tensor<real32> value = ($VALUE)$PLACEMENT
tensor<real32> result = $EXPR
print(result.cpu().shape()[0])
print(NL)
QUI
    set +e
    QUIDRA_PACKAGE_PATH="$TMP/packages" "$QUIDRA" run "$TMP/$OP.qui" > "$TMP/$OP.log" 2>&1
    RC=$?
    set -e
    [[ "$RC" == 101 ]]
    grep -F "Quidra runtime error[$CODE]" "$TMP/$OP.log"
    grep -F "package operation math.$OP produced a non-finite real32 value" "$TMP/$OP.log"
done

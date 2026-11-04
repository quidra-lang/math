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
if [[ -n "${2:-}" ]]; then
    PLACEMENT=".gpu($2)"
fi
# Finite overflow may produce signed infinity; only NaN is rejected.
for OP in exp sum matmul; do
    case "$OP" in
        exp) VALUE="tensor.ones<real32>([2]) * real32(100)"; EXPR="math.exp(value)" ;;
        sum) VALUE="tensor.ones<real32>([2]) * real32(3.0e38)"; EXPR="math.sum(value)" ;;
        matmul) VALUE="tensor.ones<real32>([1, 2]) * real32(3.0e38)"; EXPR="math.matmul(value, tensor.ones<real32>([2, 1])$PLACEMENT)" ;;
    esac
    cat > "$TMP/$OP.qui" <<QUI
import math
tensor<real32> value := ($VALUE)$PLACEMENT
tensor<real32> result := $EXPR
println(math.max_all(result.cpu()).item() == real32(inf))
QUI
    if ! QUIDRA_PACKAGE_PATH="$TMP/packages" "$QUIDRA" run "$TMP/$OP.qui" > "$TMP/$OP.log" 2>&1; then
        echo "math.$OP unexpectedly rejected overflow to infinity:" >&2
        cat "$TMP/$OP.log" >&2
        exit 1
    fi
    if [[ "$(cat "$TMP/$OP.log")" != true ]]; then
        echo "math.$OP did not preserve positive infinity:" >&2
        cat "$TMP/$OP.log" >&2
        exit 1
    fi
done

# These inputs are valid infinities; the package operation creates NaN.
for OP in sum matmul; do
    case "$OP" in
        sum) EXPR="math.sum(value)"; SECOND="-inf" ;;
        matmul) EXPR="math.matmul(value, tensor.zeros<real32>([2, 1])$PLACEMENT)"; SECOND="inf" ;;
    esac
    cat > "$TMP/$OP-nan.qui" <<QUI
import math
tensor<real32> host := tensor.ones<real32>([1, 2])
host[0, 0] = real32(inf)
host[0, 1] = real32($SECOND)
value := host$PLACEMENT
println("before")
result := $EXPR
println("after {result.cpu().item()}")
QUI
    set +e
    QUIDRA_PACKAGE_PATH="$TMP/packages" "$QUIDRA" run "$TMP/$OP-nan.qui" > "$TMP/$OP-nan.out" 2> "$TMP/$OP-nan.err"
    RC=$?
    set -e
    if [[ "$RC" != 101 || "$(cat "$TMP/$OP-nan.out")" != before ]] ||
       ! grep -Fq 'Quidra runtime error[' "$TMP/$OP-nan.err" ||
       ! grep -Fq "package operation math.$OP produced a NaN real32 value" "$TMP/$OP-nan.err"; then
        echo "unexpected math.$OP NaN outcome (status=$RC):" >&2
        cat "$TMP/$OP-nan.out" "$TMP/$OP-nan.err" >&2
        exit 1
    fi
done
echo 'Math overflow and NaN result tests passed'

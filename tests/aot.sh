#!/usr/bin/env bash
set -euo pipefail

QUIDRA="${1:?usage: aot.sh /path/to/quidra}"
REPOSITORY_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACKAGE_ROOT="$(dirname "$REPOSITORY_ROOT")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

cat > "$TMP/math-aot.qui" <<'QUI'
import math

int | error run()
    float root = math.sqrt(float(9.0))
    print(root == float(3.0))
    print(NL)

    tensor<float32> values = tensor.ones<float32>([2]) * float32(4)
    tensor<float32> roots = math.sqrt(values)
    print(roots[0].item() == float32(2))
    print(NL)
    print(roots[1].item() == float32(2))
    print(NL)
    return 0

auto | error result = run()
match result
    int
        int ignored = result
    error problem
        print(problem)
        print(NL)
QUI

OUTPUT="$TMP/math-aot"
QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" build "$TMP/math-aot.qui" -o "$OUTPUT"
actual="$("$OUTPUT")"
expected="$(printf 'true\ntrue\ntrue')"
if [[ "$actual" != "$expected" ]]; then
    echo "unexpected Math AOT output:" >&2
    printf '%s\n' "$actual" >&2
    exit 1
fi

repl_output="$(
    QUIDRA_PACKAGE_PATH="$PACKAGE_ROOT" "$QUIDRA" repl < "$TMP/math-aot.qui"
)"
true_lines="$(grep -Fxc "true" <<< "$repl_output" || true)"
if [[ "$true_lines" -lt 3 ]]; then
    echo "Math package-native REPL/JIT load did not execute expected math operations:" >&2
    printf '%s\n' "$repl_output" >&2
    exit 1
fi

echo "math AOT and REPL/JIT integration: ok"

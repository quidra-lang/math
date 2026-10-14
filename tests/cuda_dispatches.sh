#!/usr/bin/env bash
set -euo pipefail

CORE="${1:?usage: cuda_dispatches.sh /path/to/quidra-source}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
COMPILER="${CXX:-c++}"
LINK_FLAGS=(-pthread)
case "$(uname -s)" in
    Darwin) LINK_FLAGS+=(-Wl,-dead_strip) ;;
    Linux) LINK_FLAGS+=(-Wl,--gc-sections -ldl) ;;
    *) echo "CUDA dispatch mock supports Darwin and Linux" >&2; exit 1 ;;
esac
"$COMPILER" -std=c++20 -O0 -ffunction-sections -fdata-sections \
    -I"$CORE/include" "$ROOT/tests/cuda_dispatches.cpp" \
    "${LINK_FLAGS[@]}" -o "$TMP/cuda-dispatches"
"$TMP/cuda-dispatches"

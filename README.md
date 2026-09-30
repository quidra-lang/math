# Quidra Math

Generic numerical computing for Quidra.

**Core provides mechanisms. Math owns generic numerical semantics.**

Math contains reusable numerical operations that are not required to make
the Quidra language, tensor storage, autograd graph, or device substrate exist.
The first boundary moved here is matrix multiplication.

## API

```quidra
import math

tensor<float32> a = tensor.ones<float32>([2, 3])
tensor<float32> b = tensor.ones<float32>([3, 4])
tensor<float32> c = math.matmul(a, b)

float32 d = math.dot(
    tensor.ones<float32>([3]),
    tensor.ones<float32>([3])
)


```

`matmul` supports a left tensor with rank >= 1 and a right vector or matrix.
A right vector removes the contracted axis; a right matrix replaces it with
the matrix output width. Two vectors therefore produce a rank-0 tensor.

`math.sum(value)`, `math.mean(value)`, `math.max_all(value)`, and `math.min_all(value)` own whole-tensor numerical reduction semantics. They are composed from Core tensor shape/gather/arithmetic/device/autograd mechanisms rather than Core reduction kernels. Integer `sum` is untracked; floating reductions preserve the original differentiable path. Whole-tensor `max_all`/`min_all` use strict comparison on an untracked selector and gather the first winning element from the original tensor, so ties are first-occurrence and autograd remains package-composed. Empty floating `sum` keeps differentiability through a Math-owned zero-Jacobian custom operation; empty `mean`, `max_all`, and `min_all` are domain errors.

The former Core tensor methods `sum_last()`, `max_last()`, and
`min_last()` are package-owned here as `math.sum_last(value)`,
`math.max_last(value)`, and `math.min_last(value)`. Their portable
implementation uses only Core shape/indexing/gather/elementwise/autograd
mechanisms; Core no longer needs named last-axis reduction semantics.
`max_last` and `min_last` choose the winning index from an explicitly
untracked selector and gather from the original tensor, preserving
first-occurrence tie breaking and the differentiable path. A future general
axis-aware reduction API can supersede these convenience names without moving
the numerical semantics back into Core. Core exposes no public numerical
reduction methods; whole-tensor and axis-shaped reduction semantics live here.

`math.abs`, `math.sqrt`, `math.log`, and `math.exp` own tensor unary
semantics as package functions; the former Core tensor methods do not exist.
Their floating-tensor implementation uses Core's opaque device/native and
custom-autograd mechanisms while Math owns the operation kernels and derivative
formulas. CPU, fake-GPU test execution, and package-owned CUDA/Metal paths preserve
the input placement; non-contiguous inputs are materialized through Core's
generic gather mechanism without detaching tracked tensors. Backend placement
therefore stays mechanism in Core while mathematical semantics remain in Math.

The untracked CPU `float32` path uses the package-owned C++ implementation in
`native/math_native.cpp`. Untracked CUDA `float32` matmul attempts a
package-owned cuBLAS fast path loaded dynamically from the installed CUDA
toolkit. If cuBLAS is unavailable, and for tracked tensors or other placements,
Math falls back to a portable composition of Core tensor/autograd/device
primitives so gradients, higher-order gradients, and explicit placement remain
Core substrate concerns without making matrix multiplication a Core-owned
computational primitive.

Further ROCm unary backends, decompositions, FFTs, and compiler-native
numerical optimizations belong here rather than in Core.

`compiler/graph.toml` is the package-owned compiler descriptor. It assigns stable
semantic operation IDs and traits to Math operations without teaching Core
what matrix multiplication or linear algebra means.

## Ownership boundary

Quidra Core owns tensor representation, shape/stride/storage, CPU/GPU placement,
transfers, autograd graph mechanics, primitive tensor arithmetic, and generic
extension hooks. Math owns public numerical reductions and linear-algebra
semantics plus their autograd/native/backend/compiler implementations.
Domain packages such as NN, Vision, Video, and DNN may depend on Math but keep
their own domain-specific operations.

## Development

Development uses the permanent `develop` branch. After Math's first release, `main` is the latest released source; before then it contains bootstrap history only.

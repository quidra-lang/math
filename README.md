# Quidra Math

Generic numerical computing for Quidra.

**Core provides mechanisms. Math owns generic numerical semantics.**

Math contains reusable numerical operations that are not required to make
the Quidra language, tensor storage, autograd graph, or device substrate exist.
The first boundary moved here is matrix multiplication.

## API

```quidra
import math

tensor<real32> a = tensor.ones<real32>([2, 3])
tensor<real32> b = tensor.ones<real32>([3, 4])
tensor<real32> c = math.matmul(a, b)

real32 d = math.dot(
    tensor.ones<real32>([3]),
    tensor.ones<real32>([3])
)


```

`matmul` supports a left tensor with rank >= 1 and a right vector or matrix.
A right vector removes the contracted axis; a right matrix replaces it with
the matrix output width. Two vectors therefore produce a rank-0 tensor.

`math.sum(value)`, `math.mean(value)`, `math.max_all(value)`, and `math.min_all(value)` also accept numeric arrays. Array `sum` adds left to right from a zero of the element type and returns that type, including for an empty array. It supports every integer, real, and complex type, preserving the scalar addition's checked overflow and rounding. Array `mean` returns an exact `real` for integer arrays, accumulating exactly before division; real and complex arrays keep their element type and divide their left-to-right sum by the length converted to that type. Array `min_all`/`max_all` accept ordered numeric elements (integers and reals), return the element type, and keep the first occurrence of ties. Empty array `mean`, `min_all`, and `max_all` report `MATH_SHAPE`, as their tensor counterparts do. Array comprehensions work directly:

```quidra
int[] values = [-2, 1, 2, 3]
int squares = math.sum([for auto x = values[] if x > 0 then x * x])
real average = math.mean(values)
```

The tensor forms own whole-tensor numerical reduction semantics. Integer `sum` is untracked; floating reductions preserve the original differentiable path. Whole-tensor `sum`/`mean` add elements with a level-by-level pairwise tree (adjacent pairs; on native CPU and Metal an odd trailing element preserves signed infinity, without producing NaN from an identity carry; other devices currently use the portable tensor composition), and `mean` divides by the element count accumulated in the tensor's dtype from powers of two. Whole-tensor `max_all`/`min_all` use strict comparison, so ties are first-occurrence and a NaN never displaces an earlier winner. Empty floating `sum` keeps differentiability through a Math-owned zero-Jacobian custom operation; empty `mean`, `max_all`, and `min_all` are domain errors.

The former Core tensor methods `sum_last()`, `max_last()`, and
`min_last()` are package-owned here as `math.sum_last(value)`,
`math.max_last(value)`, and `math.min_last(value)`. They keep the input shape:
each row's sum (added left to right) or first-occurrence extremum is broadcast
across the last axis, so `weights / math.sum_last(weights)` normalizes rows.
A future general axis-aware reduction API can supersede these convenience
names without moving the numerical semantics back into Core. Core exposes no
public numerical reduction methods; whole-tensor and axis-shaped reduction
semantics live here.

For floating tensors on the CPU, the fake test GPU, and Metal (`real32`),
these reductions run as Math-native kernels with one Math-owned
custom-autograd node each. `mean` backward is `upstream / count`; `sum_last`
is its own adjoint; `max_last`/`min_last`/`max_all`/`min_all` record each
row's winning offset on the device, and backward sends the row's upstream sum
only to that winner. The derivative operations are themselves Math custom
nodes, so `backward(track = true)` keeps higher-order gradients. On these
placements no path reads values back to the host per element. A
non-contiguous view, tracked or not, is read through a value-only contiguous
copy on its own device while the view itself stays the autograd parent, so
nothing is detached. Integer tensors and placements without a Math-native
kernel (CUDA/HIP) keep the portable composition of Core
shape/gather/elementwise/autograd mechanisms, which defines the same
semantics; its floating `max_last`/`min_last`/`max_all`/`min_all` still select
the winner element by element on the host.

`math.abs`, `math.sqrt`, `math.log`, and `math.exp` own tensor unary
semantics as package functions; the former Core tensor methods do not exist.
Their floating-tensor implementation uses Core's opaque device/native and
custom-autograd mechanisms while Math owns the operation kernels and derivative
formulas. CPU, fake-GPU test execution, and package-owned CUDA/Metal paths preserve
the input placement; non-contiguous inputs are materialized through Core's
generic gather mechanism without detaching tracked tensors. Backend placement
therefore stays mechanism in Core while mathematical semantics remain in Math.
On the CPU a `sqrt` or `log` domain failure stops the program at the call, and
so does Math's CUDA kernel, which still reads its check back at once. On Metal
and the fake test GPU the check stays device-resident, as the language
requires for GPU tensor checks: the call returns at once, and the same message
is reported with the queuing statement at the device's next synchronization
point. On Metal that is a host read such as `.cpu()` or `.item()`,
`gpu.sync`, or program exit. Core's fake test GPU does not yet consume
deferred checks at host reads, only at `gpu.sync` and program exit, so there a
program can read a failed result and keep running until one of those.

For ordinary arrays, `math.dot(a, b)` / `a.dot(b)` requires equal lengths
and exactly matching numeric element types, returns that element type, and
returns its zero for two empty arrays. It visits indices from left to right,
computes each product with the scalar multiplication rules, then adds that
rounded product to the running total with the scalar addition rules. The
accumulator starts at the element type's zero. It does not reorder the sum,
widen its accumulator, or contract multiplication and addition into FMA.
`real16` and `real16b` scalar operations compute in `real32` and round each
stored intermediate back to the element type. Exact `int`, `nat`, `real`
and `com` retain their existing exact scalar semantics. Checked integer
overflow and NaN-producing operations stop at the first failing multiplication
or addition; real overflow to signed infinity is allowed, following Core's
scalar contract. A later finite term cannot undo an earlier overflow to infinity.
Array `log` accepts zero (including fixed-width negative zero) and returns
negative infinity, while negative inputs retain their catchable `MATH_DOMAIN`
error. Exact `real` displays infinity as `∞` or `-∞`; fixed-width reals use
`inf` or `-inf`. Array mappings and selections preserve these values without
modifying their inputs.
Complex dot is bilinear (it does not conjugate either operand).
This array contract is independent of the tensor matrix-product contract below.

Floating `matmul` (and `dot`) on the CPU, the fake test GPU, and Metal
(`real32`) runs through one backend-neutral native bridge for tracked and
untracked tensors alike. The forward product attaches a single Math-owned
custom-autograd node whose first backward computes `dA = G B^T` and
`dB = A^T G` with the same backend's native GEMM; `backward(track = true)`
attaches product nodes of the same family to those gradients, so higher-order
gradients remain available without a portable gather graph. A rank-2
transposed view such as `weight.transpose(0, 1)` is read in place through its
transposed storage while the original tensor stays the differentiable input;
other non-contiguous layouts are read through a value-only contiguous copy in
the same way. The CPU GEMM rounds every product before adding it and
accumulates the inner axis in order from the first product (its accumulators
start at `-0`, the exact identity of addition), matching the portable
composition bit for bit, signed zeros included. By default Metal serves every
shape, in both execution policies, with Math-owned kernels (per-output, tiled
threadgroup-memory, and split-inner reductions for long inner axes with few
outputs) compiled without multiply-add contraction and seeded the same way.
The per-output and tiled kernels add the inner axis in ascending order and
reproduce the CPU results bit for bit, large products included. The
split-inner kernel (weight gradients over long batches) adds 256 fixed strided
partial sums of `ceil(k / 256)` products each and then a fixed 8-level tree.
It is bitwise repeatable, but it does not add in the CPU's order, so it agrees
with the CPU only within the a-priori error bound of the two orders, which
grows with the inner length `k`:
`|metal - cpu| <= (gamma(k) + gamma(ceil(k / 256) + 8)) * (|A| |B|)[i, j]`
per element (away from real32 underflow and overflow), with
`gamma(n) = n u / (1 - n u)` and `u = 2^-24`; about `1.8e-4` at `k = 3000`.
Mixed-sign inputs stay far inside the bound. On same-sign inputs, such as a
constant product or the weight gradient of a mean loss over a large batch, the
CPU's ascending fold drifts by a roughly fixed share of `k u`, so the
difference grows with `k` and no fixed tolerance holds; there Metal is the
closer one to the exact sum. `tests/real_gpu_integration.sh` checks the bound
on both kinds of input. Apple's MPS is an explicit opt-in (see below).
Untracked CUDA `real32` matmul keeps its package-owned cuBLAS fast path
loaded dynamically from the installed CUDA toolkit. Other
placements and tracked CUDA tensors fall back to a portable composition of
Core tensor/autograd/device primitives, so matrix multiplication never
becomes a Core-owned computational primitive.

Partially initialized storage never reaches a native kernel. Core's CPU
accessors and device handle accessors refuse views containing uninitialized
elements. Math treats a refused borrow as unavailable instead of reading its
storage; a portable Core read reports the language's `UNINITIALIZED` failure.

Math's CPU kernels, which also serve the fake test GPU, split their
independent outputs (row blocks of a product, rows of a last-axis reduction,
elements of a unary operation) across Core's CPU threads with
`qcore_parallel_for`, never the terms of one sum; the whole-tensor pairwise
sum stays on one thread. `QUIDRA_CPU_THREADS` limits the threads, and results
are bitwise the same for every thread count.

Further ROCm unary backends, decompositions, FFTs, and compiler-native
numerical optimizations belong here rather than in Core.

`compiler/graph.toml` is the package-owned compiler descriptor. It assigns stable
semantic operation IDs and traits to Math operations without teaching Core
what matrix multiplication or linear algebra means.

### Opt-in: Apple MPS for large Metal products

Apple's `MPSMatrixMultiplication` adds in its own order and fuses
multiply-adds, so its results differ from the CPU and from Math's kernels in
the last bits, and an exact-zero output may carry `+0` where the portable
definition gives `-0`. Math never makes a vendor library with different
numerics its default, so a process opts in with an environment variable:

```sh
QUIDRA_MATH_METAL_MPS=1 quidra run train.qui
```

Only the value `1` enables it; the variable is read once, at the process's
first Metal product. Opted in, MPS serves `real32` products whose rows and
columns are at least 64, inner axis at least 32 and `m * n * k` at least
`2^21`, with 16-byte-aligned operands, and only under the fast execution
policy. The deterministic policy and every other shape keep Math's kernels.
Because MPS's order is Apple's, its results are held only to the a-priori
bound for an arbitrary order of the `k` products,
`|metal - cpu| <= 2 gamma(k) * (|A| |B|)[i, j]` per element. The error grows
with `k`, most on near-constant inputs with a long inner axis, so no fixed
tolerance such as `1e-6` holds for every inner length. A result-relative
tolerance would not hold either, because outputs that cancel to near zero can
differ by many times their own size.

The opt-in is a process environment variable rather than a Quidra-level API
because it is a deployment decision about the numerics of a whole run, not a
property of one call: the same source program keeps Math's reproducible
results unless whoever runs it asks otherwise, and a library cannot switch it
on behind its caller. It follows Math's existing `QUIDRA_MATH_*` native
configuration, needs no API, ABI, or compiler descriptor change, and behaves
the same under `quidra run`, the REPL, and AOT binaries. Core has no
per-package option mechanism.

## Errors

Math preserves every operation-specific error message, with a stable
category code for programmatic handling. Core errors propagated from Math
retain their originating codes.

| Code | Meaning | Typical message |
| --- | --- | --- |
| `MATH_DOMAIN` | Argument outside a mathematical function's domain | `math.sqrt requires a nonnegative value` |
| `MATH_RANGE` | Rounded result outside the permitted integer range | `math.round requires a finite value in int range` |
| `MATH_SHAPE` | Incompatible rank, shape or reduction extent | `math.matmul inner dimensions do not match` |
| `MATH_SIZE` | Tensor or temporary exceeds a supported size | `math.matmul output size overflows int` |
| `MATH_DTYPE` | Element type is unsupported for an operation | `math tensor unary is not implemented for this device dtype` |
| `MATH_DEVICE` | Backend does not support the operation | `math tensor unary is not implemented for this device backend` |
| `MATH_NATIVE` | Native kernel execution failed | `math tensor unary native execution failed` |
| `MATH_INTERNAL` | Internal invariant or gradient attachment failed | `math.sum failed to attach empty-tensor autograd` |

## Ownership boundary

Quidra Core owns tensor representation, shape/stride/storage, CPU/GPU placement,
transfers, autograd graph mechanics, primitive tensor arithmetic, and generic
extension hooks. Math owns public numerical reductions and linear-algebra
semantics plus their autograd/native/backend/compiler implementations.
Domain packages such as NN, Vision, Video, and DNN may depend on Math but keep
their own domain-specific operations.

## Development

Development uses the permanent `develop` branch. After Math's first release, `main` is the latest released source; before then it contains bootstrap history only.

The native matrix and reduction results equal the portable composition by
design, so values alone cannot show which path served a call. For Math's own
test suites, setting `QUIDRA_MATH_TEST_NATIVE_TRACE` makes those native
bridges write one `quidra-math native <event> <backend> <dtype>` line to
stderr for every forward, backward, and Metal GEMM kernel choice they serve.
It is a test diagnostic, not API; unset, nothing is printed.

## Exact and floating results

Math registers the version 2 exact provider. Atoms print as `π` and `e`;
Core prints radical layout, while Math extracts rational square factors and
canonicalizes identities such as `sin(0)` and `log(exp(x))`. Transcendental
comparisons and conversions use outward integer dyadic intervals, never
`double` approximations as proofs. A precision, series, argument-range or
domain proof that cannot be certified remains unknown and follows Core's
`EXACT_UNPROVEN` contract. The provider does not guess a sign at a pole.

Scalar `real16` and `real16b` functions widen exactly to `real32` and narrow
once after the operation. Half tensor unary functions use differentiable
floating casts, preserving the source gradient destination. The existing
`real32` and `real64` arithmetic paths retain their operation order.

Native unary, matrix-product and sum kernels reject NaN outputs. CPU
failures report `NONFINITE_RESULT`; device checks are consumed at the next
synchronization with Core's code and the same Math message and origin.
Extrema only select existing NaN-free inputs and declare that preservation.
Signed infinities remain valid unless an operation combines them into NaN.
A zero-width last axis returns its original empty shape before width guards.

The compiler descriptor uses version 2 and the current numeric type names.
Unary Metal programs compile in safe math with contraction disabled;
`QUIDRA_METAL_FAST_MATH=1` remains an explicit comparison switch.

Additional suites are `tests/provider_v2.sh`, `tests/finite_results.sh`
(optional explicit GPU index), and `tests/half_math.sh`.
`tests/exact_intervals.cpp` checks integer/interval invariants and independent
rational brackets when linked with Core's native-extension runtime.

## Extension methods

Elementwise array `abs`, `sqrt`, `log`, and `exp` recurse through nested
arrays, including unequal row lengths and empty rows. Each leaf uses the
corresponding scalar specialization, so exact `real` remains exact and
floating results keep the scalar function's rounding and domain rules.
These functions preserve the array's shape and create independent values.
Fixed inner extents are retained during recursive mapping.

Importing Math makes its public mathematical functions usable as statically
resolved extension methods. Both forms call the same implementation:

```quidra
import math
real64[] values = [1.0, 4.0, 9.0]
real64[] roots = values.sqrt()      // math.sqrt(values)
int[] a = [1, 2, 3]
int result = a.sum()                // math.sum(a)
int product = a.dot(a)             // math.dot(a, a)
```

Scalar, array and tensor overloads share the existing precision, error,
shape, CPU/GPU and autograd contracts. Methods never mutate their input
unless a library author explicitly declares a writable extension receiver.

Ranges, masks, array operators, and methods compose directly:

```quidra
import math
int[] values = [1, -2, 3, -4, 5]
int positive_square_sum = (values[values > 0] ^ 2).sum() // 35
int range_square_sum = (:9 ^ 2).sum()                    // 285
```

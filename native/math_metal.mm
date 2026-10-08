#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <quidra/native_extension.h>

#include <dlfcn.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <tuple>

namespace {

struct MetalPrograms {
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> forward_plain = nil;
    id<MTLComputePipelineState> forward_checked = nil;
    id<MTLComputePipelineState> backward = nil;
    id<MTLComputePipelineState> second_backward = nil;
};

std::mutex programs_mutex;
std::map<std::uintptr_t, MetalPrograms> programs_by_device;

NSString* math_metal_source() {
    static NSString* source = [[NSString alloc] initWithUTF8String:R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void unary_forward_plain(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    constant int& operation [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    const float value = input[index];
    output[index] = operation == 1 ? fabs(value) : exp(value);
}

kernel void unary_forward_checked(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device atomic_uint* status [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    constant int& operation [[buffer(4)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    const float value = input[index];
    bool valid = true;
    float result = 0.0f;
    if (operation == 2) {
        valid = !(value < 0.0f);
        result = sqrt(value);
    } else {
        valid = value > 0.0f;
        result = valid ? log(value) : 0.0f;
    }
    if (!valid)
        atomic_store_explicit(status, 1u, memory_order_relaxed);
    output[index] = result;
}

kernel void unary_backward(
    device const float* input [[buffer(0)]],
    device const float* output [[buffer(1)]],
    device const float* gradient_output [[buffer(2)]],
    device float* gradient_input [[buffer(3)]],
    constant uint& count [[buffer(4)]],
    constant int& operation [[buffer(5)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    const float x = input[index];
    float derivative = 0.0f;
    if (operation == 1)
        derivative = x < 0.0f ? -1.0f : (x > 0.0f ? 1.0f : 0.0f);
    else if (operation == 2)
        derivative = 1.0f / (2.0f * output[index]);
    else if (operation == 3)
        derivative = 1.0f / x;
    else
        derivative = output[index];
    gradient_input[index] = gradient_output[index] * derivative;
}

kernel void unary_second_backward(
    device const float* input [[buffer(0)]],
    device const float* first_gradient [[buffer(1)]],
    device const float* gradient_output [[buffer(2)]],
    device float* gradient_input [[buffer(3)]],
    device float* gradient_first [[buffer(4)]],
    constant uint& count [[buffer(5)]],
    constant int& operation [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= count) return;
    const float x = input[index];
    float first = 0.0f;
    float second = 0.0f;
    if (operation == 1) {
        first = x < 0.0f ? -1.0f : (x > 0.0f ? 1.0f : 0.0f);
        second = 0.0f;
    } else if (operation == 2) {
        const float root = sqrt(x);
        first = 1.0f / (2.0f * root);
        second = -0.25f / (x * root);
    } else if (operation == 3) {
        first = 1.0f / x;
        second = -1.0f / (x * x);
    } else {
        const float e = exp(x);
        first = e;
        second = e;
    }
    const float upstream = gradient_output[index];
    gradient_input[index] =
        upstream * first_gradient[index] * second;
    gradient_first[index] = upstream * first;
}
)MSL"];
    return source;
}

id<MTLComputePipelineState> make_pipeline(
    id<MTLDevice> device,
    id<MTLLibrary> library,
    NSString* name) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function) return nil;
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    [function release];
    return pipeline;
}

MetalPrograms* programs_for(id<MTLDevice> device) {
    if (!device) return nullptr;
    const auto key = reinterpret_cast<std::uintptr_t>(
        (__bridge void*)device);
    std::lock_guard<std::mutex> lock(programs_mutex);
    auto [it, inserted] = programs_by_device.try_emplace(key);
    auto& programs = it->second;
    if (inserted) {
        NSError* error = nil;
        programs.library =
            [device newLibraryWithSource:math_metal_source()
                                 options:nil
                                   error:&error];
        if (programs.library) {
            programs.forward_plain =
                make_pipeline(device, programs.library, @"unary_forward_plain");
            programs.forward_checked =
                make_pipeline(device, programs.library, @"unary_forward_checked");
            programs.backward =
                make_pipeline(device, programs.library, @"unary_backward");
            programs.second_backward =
                make_pipeline(device, programs.library, @"unary_second_backward");
        }
    }
    if (!programs.library || !programs.forward_plain ||
        !programs.forward_checked || !programs.backward ||
        !programs.second_backward)
        return nullptr;
    return &programs;
}

id<MTLCommandQueue> command_queue(const void* tensor) {
    const auto device = qcore_tensor_device(tensor);
    if (device < 0) return nil;
    const auto handle = qcore_device_queue_handle(device);
    if (handle == 0) return nil;
    return (__bridge id<MTLCommandQueue>)(
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(handle)));
}

// Core keeps one queue per Metal device for the life of the process; the
// borrowed handle is remembered per device so each Math call does not repeat
// Core's device enumeration.
std::mutex linear_queue_mutex;
std::map<long long, std::uint64_t> linear_queue_handles;

id<MTLCommandQueue> linear_command_queue(const void* tensor) {
    const auto device = qcore_tensor_device(tensor);
    if (device < 0) return nil;
    std::uint64_t handle = 0;
    {
        std::lock_guard<std::mutex> lock(linear_queue_mutex);
        if (const auto found = linear_queue_handles.find(device);
            found != linear_queue_handles.end())
            handle = found->second;
    }
    if (handle == 0) {
        handle = qcore_device_queue_handle(device);
        if (handle == 0) return nil;
        std::lock_guard<std::mutex> lock(linear_queue_mutex);
        linear_queue_handles[device] = handle;
    }
    return (__bridge id<MTLCommandQueue>)(
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(handle)));
}

id<MTLBuffer> const_buffer(const void* tensor) {
    const auto handle = qcore_tensor_device_handle_const(tensor);
    if (handle == 0) return nil;
    return (__bridge id<MTLBuffer>)(
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(handle)));
}

id<MTLBuffer> mutable_buffer(void* tensor) {
    const auto handle = qcore_tensor_device_handle(tensor);
    if (handle == 0) return nil;
    return (__bridge id<MTLBuffer>)(
        reinterpret_cast<void*>(static_cast<std::uintptr_t>(handle)));
}

bool metal_tensor(const void* tensor) {
    return tensor &&
        qcore_tensor_backend(tensor) == QCORE_BACKEND_METAL &&
        qcore_tensor_is_contiguous(tensor);
}

bool same_device(const void* left, const void* right) {
    return qcore_tensor_device(left) == qcore_tensor_device(right);
}

bool count_u32(const void* tensor, std::uint32_t& count) {
    const auto elements = qcore_tensor_element_count(tensor);
    if (elements > std::numeric_limits<std::uint32_t>::max())
        return false;
    count = static_cast<std::uint32_t>(elements);
    return true;
}

NSUInteger thread_count(id<MTLComputePipelineState> pipeline) {
    NSUInteger width = pipeline.threadExecutionWidth;
    if (width == 0) width = 1;
    const NSUInteger maximum = pipeline.maxTotalThreadsPerThreadgroup;
    if (maximum != 0 && width > maximum) width = maximum;
    return width;
}

void bind_const(
    id<MTLComputeCommandEncoder> encoder,
    const void* tensor,
    NSUInteger index) {
    [encoder setBuffer:const_buffer(tensor)
                offset:static_cast<NSUInteger>(
                    qcore_tensor_device_offset_bytes(tensor))
               atIndex:index];
}

void bind_mutable(
    id<MTLComputeCommandEncoder> encoder,
    void* tensor,
    NSUInteger index) {
    [encoder setBuffer:mutable_buffer(tensor)
                offset:static_cast<NSUInteger>(
                    qcore_tensor_device_offset_bytes(tensor))
               atIndex:index];
}

bool encode_dispatch(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    std::uint32_t count) {
    if (!encoder || !pipeline) return false;
    [encoder setComputePipelineState:pipeline];
    if (count != 0) {
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(thread_count(pipeline), 1, 1)];
    }
    return true;
}

} // namespace

extern "C" int math_native_metal_tensor_unary_forward(
    const void* input,
    void* output,
    std::int32_t operation) {
    @autoreleasepool {
        if (!metal_tensor(input) || !metal_tensor(output) ||
            !same_device(input, output) ||
            qcore_tensor_dtype(input) != QCORE_DTYPE_FLOAT32 ||
            qcore_tensor_dtype(output) != QCORE_DTYPE_FLOAT32)
            return qcore_tensor_dtype(input) == QCORE_DTYPE_FLOAT64 ? 7 : 1;

        std::uint32_t count = 0;
        if (!count_u32(input, count) ||
            qcore_tensor_element_count(output) != count)
            return 2;
        if (count == 0) return 0;

        id<MTLCommandQueue> queue = command_queue(input);
        if (!queue || !const_buffer(input) || !mutable_buffer(output))
            return 3;
        auto* programs = programs_for([queue device]);
        if (!programs) return 5;

        const bool checked = operation == 2 || operation == 3;
        if (!checked && operation != 1 && operation != 4)
            return 2;

        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!command || !encoder) return 5;

        if (checked) {
            id<MTLBuffer> status =
                [[queue device] newBufferWithLength:sizeof(std::uint32_t)
                                            options:MTLResourceStorageModeShared];
            if (!status) return 5;
            *static_cast<std::uint32_t*>([status contents]) = 0;
            bind_const(encoder, input, 0);
            bind_mutable(encoder, output, 1);
            [encoder setBuffer:status offset:0 atIndex:2];
            [encoder setBytes:&count length:sizeof(count) atIndex:3];
            [encoder setBytes:&operation length:sizeof(operation) atIndex:4];
            if (!encode_dispatch(
                    encoder, programs->forward_checked, count)) {
                [status release];
                return 5;
            }
            [encoder endEncoding];
            [command commit];
            [command waitUntilCompleted];
            const bool failed =
                *static_cast<const std::uint32_t*>([status contents]) != 0;
            const bool command_failed =
                [command status] == MTLCommandBufferStatusError;
            [status release];
            if (command_failed) return 5;
            return failed ? 4 : 0;
        }

        bind_const(encoder, input, 0);
        bind_mutable(encoder, output, 1);
        [encoder setBytes:&count length:sizeof(count) atIndex:2];
        [encoder setBytes:&operation length:sizeof(operation) atIndex:3];
        if (!encode_dispatch(encoder, programs->forward_plain, count))
            return 5;
        [encoder endEncoding];
        [command commit];
        return 0;
    }
}

extern "C" int math_native_metal_tensor_unary_backward(
    const void* input,
    const void* output,
    const void* gradient_output,
    void* gradient_input,
    std::int32_t operation) {
    @autoreleasepool {
        if (!metal_tensor(input) || !metal_tensor(output) ||
            !metal_tensor(gradient_output) || !metal_tensor(gradient_input) ||
            !same_device(input, output) ||
            !same_device(input, gradient_output) ||
            !same_device(input, gradient_input))
            return 1;
        if (qcore_tensor_dtype(input) != QCORE_DTYPE_FLOAT32 ||
            qcore_tensor_dtype(output) != QCORE_DTYPE_FLOAT32 ||
            qcore_tensor_dtype(gradient_output) != QCORE_DTYPE_FLOAT32 ||
            qcore_tensor_dtype(gradient_input) != QCORE_DTYPE_FLOAT32)
            return 7;

        std::uint32_t count = 0;
        if (!count_u32(input, count) ||
            qcore_tensor_element_count(output) != count ||
            qcore_tensor_element_count(gradient_output) != count ||
            qcore_tensor_element_count(gradient_input) != count)
            return 2;
        if (count == 0) return 0;

        id<MTLCommandQueue> queue = command_queue(input);
        if (!queue || !const_buffer(input) || !const_buffer(output) ||
            !const_buffer(gradient_output) || !mutable_buffer(gradient_input))
            return 3;
        auto* programs = programs_for([queue device]);
        if (!programs) return 5;

        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!command || !encoder) return 5;
        bind_const(encoder, input, 0);
        bind_const(encoder, output, 1);
        bind_const(encoder, gradient_output, 2);
        bind_mutable(encoder, gradient_input, 3);
        [encoder setBytes:&count length:sizeof(count) atIndex:4];
        [encoder setBytes:&operation length:sizeof(operation) atIndex:5];
        if (!encode_dispatch(encoder, programs->backward, count))
            return 5;
        [encoder endEncoding];
        [command commit];
        return 0;
    }
}

extern "C" int math_native_metal_tensor_unary_second_backward(
    const void* input,
    const void* first_gradient,
    const void* gradient_output,
    void* gradient_input,
    void* gradient_first,
    std::int32_t operation) {
    @autoreleasepool {
        if (!metal_tensor(input) || !metal_tensor(first_gradient) ||
            !metal_tensor(gradient_output) || !metal_tensor(gradient_input) ||
            !metal_tensor(gradient_first) ||
            !same_device(input, first_gradient) ||
            !same_device(input, gradient_output) ||
            !same_device(input, gradient_input) ||
            !same_device(input, gradient_first))
            return 1;
        if (qcore_tensor_dtype(input) != QCORE_DTYPE_FLOAT32 ||
            qcore_tensor_dtype(first_gradient) != QCORE_DTYPE_FLOAT32 ||
            qcore_tensor_dtype(gradient_output) != QCORE_DTYPE_FLOAT32 ||
            qcore_tensor_dtype(gradient_input) != QCORE_DTYPE_FLOAT32 ||
            qcore_tensor_dtype(gradient_first) != QCORE_DTYPE_FLOAT32)
            return 7;

        std::uint32_t count = 0;
        if (!count_u32(input, count) ||
            qcore_tensor_element_count(first_gradient) != count ||
            qcore_tensor_element_count(gradient_output) != count ||
            qcore_tensor_element_count(gradient_input) != count ||
            qcore_tensor_element_count(gradient_first) != count)
            return 2;
        if (count == 0) return 0;

        id<MTLCommandQueue> queue = command_queue(input);
        if (!queue || !const_buffer(input) || !const_buffer(first_gradient) ||
            !const_buffer(gradient_output) || !mutable_buffer(gradient_input) ||
            !mutable_buffer(gradient_first))
            return 3;
        auto* programs = programs_for([queue device]);
        if (!programs) return 5;

        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder =
            [command computeCommandEncoder];
        if (!command || !encoder) return 5;
        bind_const(encoder, input, 0);
        bind_const(encoder, first_gradient, 1);
        bind_const(encoder, gradient_output, 2);
        bind_mutable(encoder, gradient_input, 3);
        bind_mutable(encoder, gradient_first, 4);
        [encoder setBytes:&count length:sizeof(count) atIndex:5];
        [encoder setBytes:&operation length:sizeof(operation) atIndex:6];
        if (!encode_dispatch(encoder, programs->second_backward, count))
            return 5;
        [encoder endEncoding];
        [command commit];
        return 0;
    }
}

// ===========================================================================
// Math-owned dense matrix products and reductions on Metal.
//
// Core lends the device queue and tensor buffers; Math owns the kernels and
// their numerical semantics. The library is compiled without fast-math so
// floating-point comparisons keep IEEE NaN/signed-zero behavior and divisions
// round correctly, like the CPU implementation.
// ===========================================================================

// Same POD layout as MathGemmTask in math_native.cpp.
struct MathGemmTask {
    const void* left;
    const void* right;
    void* output;
    std::uint64_t rows;
    std::uint64_t inner;
    std::uint64_t columns;
    std::uint64_t left_row_stride;
    std::uint64_t left_inner_stride;
    std::uint64_t right_inner_stride;
    std::uint64_t right_column_stride;
};

// Opt-in test trace defined in math_native.cpp (QUIDRA_MATH_TEST_NATIVE_TRACE).
extern "C" void math_native_test_trace(const char* event, const void* tensor);

namespace {

struct LinearPrograms {
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> gemm_simple = nil;
    id<MTLComputePipelineState> gemm_tiled = nil;
    id<MTLComputePipelineState> gemm_reduce = nil;
    id<MTLComputePipelineState> row_reduce_broadcast = nil;
    id<MTLComputePipelineState> row_sums = nil;
    id<MTLComputePipelineState> row_broadcast = nil;
    id<MTLComputePipelineState> pairwise_block = nil;
    id<MTLComputePipelineState> row_extrema = nil;
    id<MTLComputePipelineState> row_extrema_group = nil;
    id<MTLComputePipelineState> extrema_partials = nil;
    id<MTLComputePipelineState> extrema_merge = nil;
    id<MTLComputePipelineState> row_select = nil;
    id<MTLComputePipelineState> row_scatter = nil;
};

std::mutex linear_programs_mutex;
std::map<std::uintptr_t, LinearPrograms> linear_programs_by_device;

// Threadgroup geometry shared by the MSL source and the host dispatcher.
constexpr std::uint32_t gemm_tile = 64;
constexpr std::uint32_t group_threads = 256;
constexpr std::uint32_t pairwise_block_elements = 2048;
// Rows at least this wide (with few rows) split their extremum search across
// groups of extrema_span_elements elements, at most extrema_max_spans per row.
constexpr std::uint64_t extrema_span_threshold = 16384;
constexpr std::uint64_t extrema_span_elements = 4096;
constexpr std::uint64_t extrema_max_spans = 1024;

NSString* math_linear_metal_source() {
    static NSString* source = [[NSString alloc] initWithUTF8String:R"MSL(
#include <metal_stdlib>
using namespace metal;

struct GemmParams {
    uint rows;
    uint inner;
    uint columns;
    uint left_row_stride;
    uint left_inner_stride;
    uint right_inner_stride;
    uint right_column_stride;
    uint reserved;
};

// Every accumulator starts at -0.0f, the exact identity of IEEE addition, so
// an output equals p0 + p1 + ... as the portable composition and the CPU GEMM
// define it, signed zeros included.

// One thread per output element; k accumulates in ascending order.
kernel void gemm_simple(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant GemmParams& p [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= p.rows * p.columns) return;
    const uint row = index / p.columns;
    const uint column = index - row * p.columns;
    device const float* a = left + row * p.left_row_stride;
    device const float* b = right + column * p.right_column_stride;
    float total = -0.0f;
    for (uint k = 0; k < p.inner; ++k)
        total += a[k * p.left_inner_stride] * b[k * p.right_inner_stride];
    output[index] = total;
}

// 64x64 output tile per 256-thread group, 4x4 outputs per thread, 16-deep
// k tiles staged in threadgroup memory. Each output still accumulates k in
// ascending order, so results are deterministic.
constant uint TILE = 64;
constant uint TILE_K = 16;

kernel void gemm_tiled(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant GemmParams& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint local [[thread_index_in_threadgroup]]) {
    threadgroup float left_tile[TILE_K][TILE + 1];
    threadgroup float right_tile[TILE_K][TILE + 1];
    const uint row0 = group.y * TILE;
    const uint column0 = group.x * TILE;
    const uint tx = local % 16;
    const uint ty = local / 16;
    float acc[4][4];
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 4; ++j)
            acc[i][j] = -0.0f;
    const bool left_k_contiguous = p.left_inner_stride == 1;
    const bool right_n_contiguous = p.right_column_stride == 1;
    for (uint k0 = 0; k0 < p.inner; k0 += TILE_K) {
        for (uint e = local; e < TILE * TILE_K; e += 256) {
            uint m;
            uint kk;
            if (left_k_contiguous) {
                kk = e % TILE_K;
                m = e / TILE_K;
            } else {
                m = e % TILE;
                kk = e / TILE;
            }
            const uint row = row0 + m;
            const uint k = k0 + kk;
            left_tile[kk][m] = (row < p.rows && k < p.inner)
                ? left[row * p.left_row_stride + k * p.left_inner_stride]
                : 0.0f;
        }
        for (uint e = local; e < TILE * TILE_K; e += 256) {
            uint n;
            uint kk;
            if (right_n_contiguous) {
                n = e % TILE;
                kk = e / TILE;
            } else {
                kk = e % TILE_K;
                n = e / TILE_K;
            }
            const uint column = column0 + n;
            const uint k = k0 + kk;
            right_tile[kk][n] = (column < p.columns && k < p.inner)
                ? right[k * p.right_inner_stride + column * p.right_column_stride]
                : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint depth = min(TILE_K, p.inner - k0);
        for (uint kk = 0; kk < depth; ++kk) {
            float a[4];
            float b[4];
            for (uint i = 0; i < 4; ++i) a[i] = left_tile[kk][ty + 16 * i];
            for (uint j = 0; j < 4; ++j) b[j] = right_tile[kk][tx + 16 * j];
            for (uint i = 0; i < 4; ++i)
                for (uint j = 0; j < 4; ++j)
                    acc[i][j] += a[i] * b[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 4; ++i) {
        const uint row = row0 + ty + 16 * i;
        if (row >= p.rows) continue;
        for (uint j = 0; j < 4; ++j) {
            const uint column = column0 + tx + 16 * j;
            if (column < p.columns)
                output[row * p.columns + column] = acc[i][j];
        }
    }
}

// One 256-thread group per output element for long inner dimensions with few
// outputs (weight gradients). Fixed strided partial sums plus a fixed tree:
// deterministic without atomics. This reorders the sum, so it agrees with the
// ascending CPU fold only within the a-priori error bound of the two orders,
// which grows with the inner length (tests/real_gpu_integration.sh checks
// it).
kernel void gemm_reduce(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant GemmParams& p [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    uint local [[thread_index_in_threadgroup]]) {
    threadgroup float partial[256];
    const uint row = group / p.columns;
    const uint column = group - row * p.columns;
    device const float* a = left + row * p.left_row_stride;
    device const float* b = right + column * p.right_column_stride;
    float total = -0.0f;
    for (uint k = local; k < p.inner; k += 256)
        total += a[k * p.left_inner_stride] * b[k * p.right_inner_stride];
    partial[local] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local < stride) partial[local] += partial[local + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (local == 0) output[group] = partial[0];
}

struct RowParams {
    uint rows;
    uint input_width;
    uint output_width;
    uint divide;
    float divisor;
    uint count;
    uint reserved0;
    uint reserved1;
};

// out[r, j] = sum_k in[r, k] (left to right, starting from the first element),
// optionally divided. One thread per output element.
kernel void row_reduce_broadcast(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant RowParams& p [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= p.rows * p.output_width) return;
    const uint row = index / p.output_width;
    device const float* source = input + row * p.input_width;
    float total = source[0];
    for (uint k = 1; k < p.input_width; ++k) total += source[k];
    if (p.divide != 0) total = total / p.divisor;
    output[index] = total;
}

kernel void row_sums(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant RowParams& p [[buffer(2)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    device const float* source = input + row * p.input_width;
    float total = source[0];
    for (uint k = 1; k < p.input_width; ++k) total += source[k];
    if (p.divide != 0) total = total / p.divisor;
    output[row] = total;
}

kernel void row_broadcast(
    device const float* sums [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant RowParams& p [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= p.rows * p.output_width) return;
    output[index] = sums[index / p.output_width];
}

// The portable composition carries an odd trailing element of a level with
// more than one element as `x + x * 0` (identity for finite values, NaN for
// infinities); the kernels keep that definition.
inline float pairwise_carry(float value) {
    return value + value * 0.0f;
}

// Level-by-level pairwise tree over an aligned 2048-element block: adjacent
// pairs are added and an odd trailing element is carried upward. Blocks and
// per-thread chunks are aligned to powers of two, so a local odd tail is the
// global tail of its level; it is carried when elements precede it or its
// local level still has more than one element. Repeating the pass over block
// results reproduces the whole-tensor tree exactly.
kernel void pairwise_block(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant RowParams& p [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],
    uint local [[thread_index_in_threadgroup]]) {
    threadgroup float values[256];
    const uint count = p.count;
    const uint block_base = group * 2048;
    const uint base = block_base + local * 8;
    float v[8];
    for (uint i = 0; i < 8; ++i) v[i] = 0.0f;
    uint width = base < count ? min(8u, count - base) : 0u;
    for (uint i = 0; i < width; ++i) v[i] = input[base + i];
    if (width > 0) {
        for (uint level = 0; level < 3; ++level) {
            if ((width & 1u) != 0 && (width > 1 || base > 0))
                v[width - 1] = pairwise_carry(v[width - 1]);
            const uint pairs = (width + 1) / 2;
            for (uint i = 0; i < pairs; ++i)
                v[i] = (2 * i + 1 < width) ? v[2 * i] + v[2 * i + 1] : v[2 * i];
            width = pairs;
        }
    }
    values[local] = v[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride < 256; stride <<= 1) {
        if ((local & (2 * stride - 1)) == 0) {
            const uint partner = local + stride;
            if (block_base + partner * 8 < count)
                values[local] = values[local] + values[partner];
            else if (block_base + local * 8 < count && block_base + local > 0)
                values[local] = pairwise_carry(values[local]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (local == 0) {
        float total = values[0];
        if (p.divide != 0) total = total / p.divisor;
        output[group] = total;
    }
}

struct SelectParams {
    uint rows;
    uint input_width;
    uint output_width;
    uint maximum;
};

// Sequential strict scan per row: first occurrence wins, a NaN first element
// wins its row, later NaNs never win.
kernel void row_extrema(
    device const float* input [[buffer(0)]],
    device int* winners [[buffer(1)]],
    constant SelectParams& p [[buffer(2)]],
    uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    device const float* x = input + row * p.input_width;
    float best = x[0];
    uint best_index = 0;
    for (uint k = 1; k < p.input_width; ++k) {
        const float candidate = x[k];
        const bool better =
            p.maximum != 0 ? candidate > best : candidate < best;
        if (better) {
            best = candidate;
            best_index = k;
        }
    }
    winners[row] = int(best_index);
}

// Parallel first-occurrence search. Every thread scans its elements in
// ascending order and skips NaNs; partial winners merge by strict comparison
// with ties resolved to the smaller index. Together with "a NaN first element
// wins its row" this equals the sequential strict scan.
inline bool extrema_prefers(
    float other, int other_index, float mine, int mine_index, bool maximum) {
    if (other_index < 0) return false;
    if (mine_index < 0) return true;
    const bool other_better = maximum ? other > mine : other < mine;
    const bool mine_better = maximum ? mine > other : mine < other;
    return other_better || (!mine_better && other_index < mine_index);
}

inline void extrema_scan(
    device const float* x, uint begin, uint end, uint step, bool maximum,
    thread float& best, thread int& index) {
    for (uint k = begin; k < end; k += step) {
        const float candidate = x[k];
        if (isnan(candidate)) continue;
        if (index < 0 || (maximum ? candidate > best : candidate < best)) {
            best = candidate;
            index = int(k);
        }
    }
}

inline void extrema_group_merge(
    threadgroup float* values, threadgroup int* indices, uint local,
    bool maximum) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (local < stride &&
            extrema_prefers(values[local + stride], indices[local + stride],
                            values[local], indices[local], maximum)) {
            values[local] = values[local + stride];
            indices[local] = indices[local + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Wide rows: one 256-thread group per row, coalesced strided scans.
kernel void row_extrema_group(
    device const float* input [[buffer(0)]],
    device int* winners [[buffer(1)]],
    constant SelectParams& p [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint local [[thread_index_in_threadgroup]]) {
    threadgroup float best_values[256];
    threadgroup int best_indices[256];
    device const float* x = input + row * p.input_width;
    const bool maximum = p.maximum != 0;
    float best = 0.0f;
    int index = -1;
    extrema_scan(x, local, p.input_width, 256, maximum, best, index);
    best_values[local] = best;
    best_indices[local] = index;
    extrema_group_merge(best_values, best_indices, local, maximum);
    if (local == 0)
        winners[row] = (isnan(x[0]) || best_indices[0] < 0) ? 0 : best_indices[0];
}

struct SpanParams {
    uint rows;
    uint width;
    uint spans;
    uint span;
    uint maximum;
    uint reserved0;
    uint reserved1;
    uint reserved2;
};

// Very wide rows (whole-tensor max_all/min_all): first pass, one group per
// contiguous span of a row writes that span's partial winner.
kernel void extrema_partials(
    device const float* input [[buffer(0)]],
    device float* partial_values [[buffer(1)]],
    device int* partial_indices [[buffer(2)]],
    constant SpanParams& p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint local [[thread_index_in_threadgroup]]) {
    threadgroup float best_values[256];
    threadgroup int best_indices[256];
    const uint row = group.y;
    device const float* x = input + row * p.width;
    const bool maximum = p.maximum != 0;
    const uint begin = min(group.x * p.span, p.width);
    const uint end = min(begin + p.span, p.width);
    float best = 0.0f;
    int index = -1;
    extrema_scan(x, begin + local, end, 256, maximum, best, index);
    best_values[local] = best;
    best_indices[local] = index;
    extrema_group_merge(best_values, best_indices, local, maximum);
    if (local == 0) {
        partial_values[row * p.spans + group.x] = best_values[0];
        partial_indices[row * p.spans + group.x] = best_indices[0];
    }
}

// Second pass: merge a row's span winners and apply the NaN-first rule.
kernel void extrema_merge(
    device const float* input [[buffer(0)]],
    device const float* partial_values [[buffer(1)]],
    device const int* partial_indices [[buffer(2)]],
    device int* winners [[buffer(3)]],
    constant SpanParams& p [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint local [[thread_index_in_threadgroup]]) {
    threadgroup float best_values[256];
    threadgroup int best_indices[256];
    const bool maximum = p.maximum != 0;
    float best = 0.0f;
    int index = -1;
    for (uint span = local; span < p.spans; span += 256) {
        const float candidate = partial_values[row * p.spans + span];
        const int candidate_index = partial_indices[row * p.spans + span];
        if (extrema_prefers(candidate, candidate_index, best, index, maximum)) {
            best = candidate;
            index = candidate_index;
        }
    }
    best_values[local] = best;
    best_indices[local] = index;
    extrema_group_merge(best_values, best_indices, local, maximum);
    if (local == 0) {
        const bool nan_first = isnan(input[row * p.width]);
        winners[row] = (nan_first || best_indices[0] < 0) ? 0 : best_indices[0];
    }
}

kernel void row_select(
    device const float* input [[buffer(0)]],
    device const int* winners [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant SelectParams& p [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= p.rows * p.output_width) return;
    const uint row = index / p.output_width;
    uint winner = uint(max(winners[row], 0));
    if (winner >= p.input_width) winner = 0;
    output[index] = input[row * p.input_width + winner];
}

kernel void row_scatter(
    device const float* input [[buffer(0)]],
    device const int* winners [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant SelectParams& p [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index >= p.rows * p.output_width) return;
    const uint row = index / p.output_width;
    const uint column = index - row * p.output_width;
    float total = 0.0f;
    if (int(column) == winners[row]) {
        device const float* source = input + row * p.input_width;
        for (uint k = 0; k < p.input_width; ++k) total += source[k];
    }
    output[index] = total;
}
)MSL"];
    return source;
}

struct GemmParams {
    std::uint32_t rows;
    std::uint32_t inner;
    std::uint32_t columns;
    std::uint32_t left_row_stride;
    std::uint32_t left_inner_stride;
    std::uint32_t right_inner_stride;
    std::uint32_t right_column_stride;
    std::uint32_t reserved;
};

struct RowParams {
    std::uint32_t rows;
    std::uint32_t input_width;
    std::uint32_t output_width;
    std::uint32_t divide;
    float divisor;
    std::uint32_t count;
    std::uint32_t reserved0;
    std::uint32_t reserved1;
};

struct SelectParams {
    std::uint32_t rows;
    std::uint32_t input_width;
    std::uint32_t output_width;
    std::uint32_t maximum;
};

struct SpanParams {
    std::uint32_t rows;
    std::uint32_t width;
    std::uint32_t spans;
    std::uint32_t span;
    std::uint32_t maximum;
    std::uint32_t reserved0;
    std::uint32_t reserved1;
    std::uint32_t reserved2;
};

LinearPrograms* linear_programs_for(id<MTLDevice> device) {
    if (!device) return nullptr;
    const auto key = reinterpret_cast<std::uintptr_t>((__bridge void*)device);
    std::lock_guard<std::mutex> lock(linear_programs_mutex);
    auto [it, inserted] = linear_programs_by_device.try_emplace(key);
    auto& programs = it->second;
    if (inserted) {
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        // Products are rounded before they are added (no fused multiply-add
        // contraction), like the CPU GEMM, so the Math kernels reproduce the
        // CPU results bit for bit. Compilers without the pragma get the plain
        // source, which is equally deterministic but may contract.
        NSString* strict = [@"#pragma METAL fp contract(off)\n"
            stringByAppendingString:math_linear_metal_source()];
        NSError* error = nil;
        programs.library = [device newLibraryWithSource:strict
                                                options:options
                                                  error:&error];
        if (!programs.library) {
            error = nil;
            programs.library =
                [device newLibraryWithSource:math_linear_metal_source()
                                     options:options
                                       error:&error];
        }
        [options release];
        if (programs.library) {
            programs.gemm_simple =
                make_pipeline(device, programs.library, @"gemm_simple");
            programs.gemm_tiled =
                make_pipeline(device, programs.library, @"gemm_tiled");
            programs.gemm_reduce =
                make_pipeline(device, programs.library, @"gemm_reduce");
            programs.row_reduce_broadcast =
                make_pipeline(device, programs.library, @"row_reduce_broadcast");
            programs.row_sums =
                make_pipeline(device, programs.library, @"row_sums");
            programs.row_broadcast =
                make_pipeline(device, programs.library, @"row_broadcast");
            programs.pairwise_block =
                make_pipeline(device, programs.library, @"pairwise_block");
            programs.row_extrema =
                make_pipeline(device, programs.library, @"row_extrema");
            programs.row_extrema_group =
                make_pipeline(device, programs.library, @"row_extrema_group");
            programs.extrema_partials =
                make_pipeline(device, programs.library, @"extrema_partials");
            programs.extrema_merge =
                make_pipeline(device, programs.library, @"extrema_merge");
            programs.row_select =
                make_pipeline(device, programs.library, @"row_select");
            programs.row_scatter =
                make_pipeline(device, programs.library, @"row_scatter");
        }
    }
    if (!programs.library || !programs.gemm_simple || !programs.gemm_tiled ||
        !programs.gemm_reduce || !programs.row_reduce_broadcast ||
        !programs.row_sums || !programs.row_broadcast ||
        !programs.pairwise_block || !programs.row_extrema ||
        !programs.row_extrema_group || !programs.extrema_partials ||
        !programs.extrema_merge || !programs.row_select ||
        !programs.row_scatter)
        return nullptr;
    // Group kernels are written for exactly 256 threads per threadgroup.
    if (programs.gemm_tiled.maxTotalThreadsPerThreadgroup < group_threads ||
        programs.gemm_reduce.maxTotalThreadsPerThreadgroup < group_threads ||
        programs.pairwise_block.maxTotalThreadsPerThreadgroup < group_threads ||
        programs.row_extrema_group.maxTotalThreadsPerThreadgroup < group_threads ||
        programs.extrema_partials.maxTotalThreadsPerThreadgroup < group_threads ||
        programs.extrema_merge.maxTotalThreadsPerThreadgroup < group_threads)
        return nullptr;
    return &programs;
}

bool float_metal_tensor(const void* tensor) {
    return metal_tensor(tensor) &&
           qcore_tensor_dtype(tensor) == QCORE_DTYPE_FLOAT32;
}

// Core's Metal synchronization only waits on command buffers that Core itself
// committed, so a host readback issued right after a package-committed
// command could observe stale memory. Until Core exposes a substrate hook for
// package command buffers, Math completes its own commands before returning.
constexpr bool wait_for_package_commands = true;

int finish_command(id<MTLCommandBuffer> command) {
    [command commit];
    if (wait_for_package_commands) {
        [command waitUntilCompleted];
        if ([command status] == MTLCommandBufferStatusError) return 5;
    }
    return 0;
}

void dispatch_threads(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    std::uint64_t count) {
    [encoder setComputePipelineState:pipeline];
    if (count == 0) return;
    [encoder dispatchThreads:MTLSizeMake(static_cast<NSUInteger>(count), 1, 1)
      threadsPerThreadgroup:MTLSizeMake(thread_count(pipeline), 1, 1)];
}

void dispatch_groups(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    std::uint64_t groups_x,
    std::uint64_t groups_y) {
    [encoder setComputePipelineState:pipeline];
    if (groups_x == 0 || groups_y == 0) return;
    [encoder dispatchThreadgroups:MTLSizeMake(
                 static_cast<NSUInteger>(groups_x),
                 static_cast<NSUInteger>(groups_y), 1)
            threadsPerThreadgroup:MTLSizeMake(group_threads, 1, 1)];
}

bool u32(std::uint64_t value) {
    return value <= std::numeric_limits<std::uint32_t>::max();
}

// Largest element offset touched by a strided matrix read, plus one.
bool strided_extent(
    std::uint64_t rows, std::uint64_t row_stride,
    std::uint64_t columns, std::uint64_t column_stride,
    std::uint64_t& extent) {
    if (rows == 0 || columns == 0) return false;
    const auto last_row = rows - 1;
    const auto last_column = columns - 1;
    if (row_stride != 0 &&
        last_row > std::numeric_limits<std::uint64_t>::max() / row_stride)
        return false;
    if (column_stride != 0 &&
        last_column > std::numeric_limits<std::uint64_t>::max() / column_stride)
        return false;
    const auto a = last_row * row_stride;
    const auto b = last_column * column_stride;
    if (a > std::numeric_limits<std::uint64_t>::max() - b - 1) return false;
    extent = a + b + 1;
    return true;
}

bool gemm_task_valid(const MathGemmTask& task, long long device) {
    if (!float_metal_tensor(task.left) || !float_metal_tensor(task.right) ||
        !float_metal_tensor(task.output) ||
        qcore_tensor_device(task.left) != device ||
        qcore_tensor_device(task.right) != device ||
        qcore_tensor_device(task.output) != device)
        return false;
    if (task.rows == 0 || task.inner == 0 || task.columns == 0 ||
        !u32(task.rows) || !u32(task.inner) || !u32(task.columns) ||
        !u32(task.left_row_stride) || !u32(task.left_inner_stride) ||
        !u32(task.right_inner_stride) || !u32(task.right_column_stride))
        return false;
    if (task.rows > std::numeric_limits<std::uint32_t>::max() / task.columns ||
        qcore_tensor_element_count(task.output) != task.rows * task.columns)
        return false;
    std::uint64_t left_extent = 0;
    std::uint64_t right_extent = 0;
    if (!strided_extent(task.rows, task.left_row_stride,
                        task.inner, task.left_inner_stride, left_extent) ||
        !strided_extent(task.inner, task.right_inner_stride,
                        task.columns, task.right_column_stride, right_extent))
        return false;
    return u32(left_extent) && u32(right_extent) &&
           left_extent <= qcore_tensor_element_count(task.left) &&
           right_extent <= qcore_tensor_element_count(task.right);
}

void encode_gemm(
    id<MTLComputeCommandEncoder> encoder,
    const LinearPrograms& programs,
    const MathGemmTask& task) {
    GemmParams params{
        static_cast<std::uint32_t>(task.rows),
        static_cast<std::uint32_t>(task.inner),
        static_cast<std::uint32_t>(task.columns),
        static_cast<std::uint32_t>(task.left_row_stride),
        static_cast<std::uint32_t>(task.left_inner_stride),
        static_cast<std::uint32_t>(task.right_inner_stride),
        static_cast<std::uint32_t>(task.right_column_stride),
        0};
    bind_const(encoder, task.left, 0);
    bind_const(encoder, task.right, 1);
    bind_mutable(encoder, task.output, 2);
    [encoder setBytes:&params length:sizeof(params) atIndex:3];

    const std::uint64_t outputs = task.rows * task.columns;
    const std::uint64_t tiles =
        ((task.rows + gemm_tile - 1) / gemm_tile) *
        ((task.columns + gemm_tile - 1) / gemm_tile);
    if (task.inner >= 256 && outputs <= 65536 &&
        (outputs <= 4096 || tiles < 16)) {
        // Few outputs, long reductions (weight gradients): one group each.
        dispatch_groups(encoder, programs.gemm_reduce, outputs, 1);
        math_native_test_trace("gemm-split-inner", task.output);
    } else if (task.rows >= 32 && task.columns >= 32 && task.inner >= 8) {
        dispatch_groups(
            encoder, programs.gemm_tiled,
            (task.columns + gemm_tile - 1) / gemm_tile,
            (task.rows + gemm_tile - 1) / gemm_tile);
        math_native_test_trace("gemm-tiled", task.output);
    } else {
        dispatch_threads(encoder, programs.gemm_simple, outputs);
        math_native_test_trace("gemm-simple", task.output);
    }
}

bool metal_rows_valid(
    const void* input, const void* output, std::uint64_t rows,
    std::uint64_t input_width, std::uint64_t output_width) {
    return float_metal_tensor(input) && float_metal_tensor(output) &&
           same_device(input, output) &&
           rows != 0 && input_width != 0 && output_width != 0 &&
           u32(rows) && u32(input_width) && u32(output_width) &&
           rows <= std::numeric_limits<std::uint32_t>::max() / input_width &&
           rows <= std::numeric_limits<std::uint32_t>::max() / output_width &&
           qcore_tensor_element_count(input) == rows * input_width &&
           qcore_tensor_element_count(output) == rows * output_width;
}

bool metal_winners_valid(
    const void* winners, const void* reference, std::uint64_t rows) {
    return winners &&
           qcore_tensor_backend(winners) == QCORE_BACKEND_METAL &&
           qcore_tensor_is_contiguous(winners) &&
           qcore_tensor_dtype(winners) == QCORE_DTYPE_INT32 &&
           same_device(winners, reference) &&
           qcore_tensor_element_count(winners) == rows;
}

struct Encoding {
    id<MTLCommandBuffer> command = nil;
    id<MTLComputeCommandEncoder> encoder = nil;
    LinearPrograms* programs = nullptr;
    id<MTLDevice> device = nil;
};

bool begin_encoding(const void* tensor, Encoding& encoding) {
    id<MTLCommandQueue> queue = linear_command_queue(tensor);
    if (!queue) return false;
    encoding.device = [queue device];
    encoding.programs = linear_programs_for(encoding.device);
    if (!encoding.programs) return false;
    encoding.command = [queue commandBuffer];
    if (!encoding.command) return false;
    encoding.encoder = [encoding.command computeCommandEncoder];
    return encoding.encoder != nil;
}

// Apple's MPSMatrixMultiplication is an explicit opt-in, never the default.
// Its accumulation order and multiply-add use are Apple's, so its results
// differ from the CPU and from the Math kernels above in the last bits (and an
// exact-zero output may lose its sign). A vendor library with different
// numerics is never Math's default, so MPS serves a product only when the
// process asks for it: QUIDRA_MATH_METAL_MPS=1 in the environment (read once,
// at the first Metal product), and then only for large products under the fast
// execution policy. Unset or any other value, every shape in either policy
// runs on the Math kernels, whose accumulation order is fixed. After opting
// in, the framework is loaded at runtime and its classes are resolved by name,
// so the package-native build gains no link dependency and the JIT keeps
// working when the host process has not linked it.
bool mps_opted_in() {
    static const bool enabled = [] {
        const char* value = std::getenv("QUIDRA_MATH_METAL_MPS");
        return value != nullptr && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

struct MpsApi {
    bool ready = false;
    Class descriptor = nil;
    Class matrix = nil;
    Class multiplication = nil;
    BOOL (*supports)(id<MTLDevice>) = nullptr;
};

MpsApi load_mps() {
    MpsApi api;
    void* handle = dlopen(
        "/System/Library/Frameworks/MetalPerformanceShaders.framework/"
        "MetalPerformanceShaders",
        RTLD_LAZY | RTLD_LOCAL);
    if (!handle) return api;
    api.descriptor = NSClassFromString(@"MPSMatrixDescriptor");
    api.matrix = NSClassFromString(@"MPSMatrix");
    api.multiplication = NSClassFromString(@"MPSMatrixMultiplication");
    api.supports = reinterpret_cast<BOOL (*)(id<MTLDevice>)>(
        dlsym(handle, "MPSSupportsMTLDevice"));
    api.ready = api.descriptor && api.matrix && api.multiplication &&
                api.supports;
    return api;
}

MpsApi& mps() {
    static MpsApi api = load_mps();
    return api;
}

using MpsKernelKey = std::tuple<
    std::uintptr_t, bool, bool, std::uint64_t, std::uint64_t, std::uint64_t>;
std::mutex mps_mutex;
std::map<MpsKernelKey, id> mps_kernels;

// Returns a cached MPSMatrixMultiplication for this problem shape with an
// extra reference owned by the caller (release it after encoding), so a cache
// eviction on another thread never frees a kernel that is still being encoded.
id mps_kernel(
    id<MTLDevice> device, bool transpose_left, bool transpose_right,
    std::uint64_t rows, std::uint64_t columns, std::uint64_t inner) {
    auto& api = mps();
    const MpsKernelKey key{
        reinterpret_cast<std::uintptr_t>((__bridge void*)device),
        transpose_left, transpose_right, rows, columns, inner};
    std::lock_guard<std::mutex> lock(mps_mutex);
    if (const auto found = mps_kernels.find(key); found != mps_kernels.end())
        return [found->second retain];
    if (mps_kernels.size() >= 256) {
        for (auto& entry : mps_kernels) [entry.second release];
        mps_kernels.clear();
    }
    MPSMatrixMultiplication* kernel =
        [(MPSMatrixMultiplication*)[api.multiplication alloc]
            initWithDevice:device
             transposeLeft:transpose_left
            transposeRight:transpose_right
                resultRows:static_cast<NSUInteger>(rows)
             resultColumns:static_cast<NSUInteger>(columns)
           interiorColumns:static_cast<NSUInteger>(inner)
                     alpha:1.0
                      beta:0.0];
    if (!kernel) return nil;
    mps_kernels.emplace(key, kernel);
    return [kernel retain];
}

// Describes one strided operand as an MPS matrix: row-major storage (unit
// column stride) or transposed storage (unit row stride).
struct MpsOperand {
    bool transposed;
    std::uint64_t stored_rows;
    std::uint64_t stored_columns;
    std::uint64_t row_bytes;
};

bool mps_operand(
    std::uint64_t rows, std::uint64_t columns,
    std::uint64_t row_stride, std::uint64_t column_stride,
    MpsOperand& operand) {
    if (column_stride == 1 && row_stride >= columns) {
        operand = {false, rows, columns, row_stride * sizeof(float)};
        return true;
    }
    if (row_stride == 1 && column_stride >= rows) {
        operand = {true, columns, rows, column_stride * sizeof(float)};
        return true;
    }
    return false;
}

bool mps_eligible(const MathGemmTask& task) {
    if (!mps_opted_in()) return false;
    if (qcore_execution_policy_get() != QCORE_EXECUTION_FAST) return false;
    if (task.rows < 64 || task.columns < 64 || task.inner < 32 ||
        task.rows * task.columns * task.inner < (1ull << 21))
        return false;
    const auto aligned = [](const void* tensor) {
        return qcore_tensor_device_offset_bytes(tensor) % 16 == 0;
    };
    return aligned(task.left) && aligned(task.right) && aligned(task.output);
}

id mps_matrix(
    id buffer, const void* tensor, const MpsOperand& operand) {
    auto& api = mps();
    MPSMatrixDescriptor* descriptor =
        [(id)api.descriptor
            matrixDescriptorWithRows:static_cast<NSUInteger>(operand.stored_rows)
                             columns:static_cast<NSUInteger>(operand.stored_columns)
                            rowBytes:static_cast<NSUInteger>(operand.row_bytes)
                            dataType:MPSDataTypeFloat32];
    if (!descriptor) return nil;
    return [(MPSMatrix*)[api.matrix alloc]
        initWithBuffer:buffer
                offset:static_cast<NSUInteger>(
                    qcore_tensor_device_offset_bytes(tensor))
            descriptor:descriptor];
}

// Encodes the product with MPS when eligible; false leaves the task to the
// Math kernels (nothing has been encoded).
bool encode_mps(
    id<MTLCommandBuffer> command, id<MTLDevice> device,
    const MathGemmTask& task) {
    auto& api = mps();
    if (!mps_eligible(task) || !api.ready || !api.supports(device))
        return false;
    MpsOperand left{};
    MpsOperand right{};
    if (!mps_operand(task.rows, task.inner, task.left_row_stride,
                     task.left_inner_stride, left) ||
        !mps_operand(task.inner, task.columns, task.right_inner_stride,
                     task.right_column_stride, right))
        return false;
    const MpsOperand result{false, task.rows, task.columns,
                            task.columns * sizeof(float)};
    id kernel = mps_kernel(
        device, left.transposed, right.transposed,
        task.rows, task.columns, task.inner);
    if (!kernel) return false;
    id left_matrix = mps_matrix(const_buffer(task.left), task.left, left);
    id right_matrix = mps_matrix(const_buffer(task.right), task.right, right);
    id result_matrix =
        mps_matrix(mutable_buffer(task.output), task.output, result);
    const bool ok = left_matrix && right_matrix && result_matrix;
    if (ok)
        [(MPSMatrixMultiplication*)kernel encodeToCommandBuffer:command
                                                     leftMatrix:left_matrix
                                                    rightMatrix:right_matrix
                                                   resultMatrix:result_matrix];
    [left_matrix release];
    [right_matrix release];
    [result_matrix release];
    [kernel release];
    if (ok) math_native_test_trace("gemm-mps", task.output);
    return ok;
}

} // namespace

extern "C" int math_native_metal_gemm(
    const MathGemmTask* tasks,
    std::uint64_t task_count) {
    @autoreleasepool {
        if (!tasks || task_count == 0 || !tasks[0].output) return 1;
        const auto device = qcore_tensor_device(tasks[0].output);
        for (std::uint64_t index = 0; index < task_count; ++index) {
            if (!gemm_task_valid(tasks[index], device)) return 2;
            if (!const_buffer(tasks[index].left) ||
                !const_buffer(tasks[index].right) ||
                !mutable_buffer(tasks[index].output))
                return 3;
        }
        Encoding encoding;
        if (!begin_encoding(tasks[0].output, encoding)) return 5;
        for (std::uint64_t index = 0; index < task_count; ++index) {
            if (mps_eligible(tasks[index])) {
                if (encoding.encoder) {
                    [encoding.encoder endEncoding];
                    encoding.encoder = nil;
                }
                if (encode_mps(encoding.command, encoding.device, tasks[index]))
                    continue;
            }
            if (!encoding.encoder) {
                encoding.encoder = [encoding.command computeCommandEncoder];
                if (!encoding.encoder) return 5;
            }
            encode_gemm(encoding.encoder, *encoding.programs, tasks[index]);
        }
        if (encoding.encoder) [encoding.encoder endEncoding];
        return finish_command(encoding.command);
    }
}

extern "C" int math_native_metal_row_reduce(
    const void* input,
    void* output,
    std::uint64_t rows,
    std::uint64_t input_width,
    std::uint64_t output_width,
    double divisor,
    int pairwise) {
    @autoreleasepool {
        if (!metal_rows_valid(input, output, rows, input_width, output_width))
            return 2;
        if (!const_buffer(input) || !mutable_buffer(output)) return 3;
        Encoding encoding;
        if (!begin_encoding(input, encoding)) return 5;
        auto* programs = encoding.programs;
        id<MTLComputeCommandEncoder> encoder = encoding.encoder;

        RowParams params{};
        params.rows = static_cast<std::uint32_t>(rows);
        params.input_width = static_cast<std::uint32_t>(input_width);
        params.output_width = static_cast<std::uint32_t>(output_width);
        params.divide = divisor != 1.0 ? 1u : 0u;
        params.divisor = static_cast<float>(divisor);

        if (pairwise && rows == 1 && output_width == 1 && input_width > 1) {
            // Whole-tensor tree: repeat block passes until one value remains.
            std::uint64_t count = input_width;
            id<MTLBuffer> source = nil;
            bool first = true;
            while (true) {
                const std::uint64_t groups =
                    (count + pairwise_block_elements - 1) /
                    pairwise_block_elements;
                RowParams pass = params;
                pass.count = static_cast<std::uint32_t>(count);
                pass.divide = groups == 1 ? params.divide : 0u;
                if (first) bind_const(encoder, input, 0);
                else [encoder setBuffer:source offset:0 atIndex:0];
                id<MTLBuffer> target = nil;
                if (groups == 1) {
                    bind_mutable(encoder, output, 1);
                } else {
                    target = [encoding.device
                        newBufferWithLength:static_cast<NSUInteger>(
                                                groups * sizeof(float))
                                    options:MTLResourceStorageModePrivate];
                    if (!target) {
                        [encoder endEncoding];
                        if (source) [source release];
                        return 5;
                    }
                    [encoder setBuffer:target offset:0 atIndex:1];
                }
                [encoder setBytes:&pass length:sizeof(pass) atIndex:2];
                dispatch_groups(encoder, programs->pairwise_block, groups, 1);
                // The command buffer retains every bound buffer until it
                // completes, so the scratch reference can be dropped here.
                if (source) [source release];
                source = target;
                first = false;
                if (groups == 1) break;
                count = groups;
            }
        } else if (input_width == 1 || output_width == 1 ||
                   input_width * output_width <= 4096) {
            bind_const(encoder, input, 0);
            bind_mutable(encoder, output, 1);
            [encoder setBytes:&params length:sizeof(params) atIndex:2];
            dispatch_threads(
                encoder, programs->row_reduce_broadcast, rows * output_width);
        } else {
            id<MTLBuffer> sums = [encoding.device
                newBufferWithLength:static_cast<NSUInteger>(rows * sizeof(float))
                            options:MTLResourceStorageModePrivate];
            if (!sums) {
                [encoder endEncoding];
                return 5;
            }
            bind_const(encoder, input, 0);
            [encoder setBuffer:sums offset:0 atIndex:1];
            [encoder setBytes:&params length:sizeof(params) atIndex:2];
            dispatch_threads(encoder, programs->row_sums, rows);
            [encoder setBuffer:sums offset:0 atIndex:0];
            bind_mutable(encoder, output, 1);
            [encoder setBytes:&params length:sizeof(params) atIndex:2];
            dispatch_threads(
                encoder, programs->row_broadcast, rows * output_width);
            [sums release];
        }
        [encoder endEncoding];
        return finish_command(encoding.command);
    }
}

extern "C" int math_native_metal_row_extrema(
    const void* input,
    void* output,
    void* winners,
    std::uint64_t rows,
    std::uint64_t width,
    std::uint64_t output_width,
    int maximum) {
    @autoreleasepool {
        if (!metal_rows_valid(input, output, rows, width, output_width) ||
            !metal_winners_valid(winners, input, rows))
            return 2;
        if (!const_buffer(input) || !mutable_buffer(output) ||
            !mutable_buffer(winners))
            return 3;
        Encoding encoding;
        if (!begin_encoding(input, encoding)) return 5;
        auto* programs = encoding.programs;
        id<MTLComputeCommandEncoder> encoder = encoding.encoder;
        SelectParams params{
            static_cast<std::uint32_t>(rows),
            static_cast<std::uint32_t>(width),
            static_cast<std::uint32_t>(output_width),
            maximum ? 1u : 0u};

        if (rows < 64 && width >= extrema_span_threshold) {
            // Few very wide rows: span winners first, then one merge per row.
            const std::uint64_t spans = std::min<std::uint64_t>(
                extrema_max_spans,
                (width + extrema_span_elements - 1) / extrema_span_elements);
            SpanParams span_params{
                static_cast<std::uint32_t>(rows),
                static_cast<std::uint32_t>(width),
                static_cast<std::uint32_t>(spans),
                static_cast<std::uint32_t>((width + spans - 1) / spans),
                maximum ? 1u : 0u, 0u, 0u, 0u};
            id<MTLBuffer> partial_values = [encoding.device
                newBufferWithLength:static_cast<NSUInteger>(
                                        rows * spans * sizeof(float))
                            options:MTLResourceStorageModePrivate];
            id<MTLBuffer> partial_indices = [encoding.device
                newBufferWithLength:static_cast<NSUInteger>(
                                        rows * spans * sizeof(std::int32_t))
                            options:MTLResourceStorageModePrivate];
            if (!partial_values || !partial_indices) {
                [partial_values release];
                [partial_indices release];
                [encoder endEncoding];
                return 5;
            }
            bind_const(encoder, input, 0);
            [encoder setBuffer:partial_values offset:0 atIndex:1];
            [encoder setBuffer:partial_indices offset:0 atIndex:2];
            [encoder setBytes:&span_params length:sizeof(span_params) atIndex:3];
            dispatch_groups(encoder, programs->extrema_partials, spans, rows);
            bind_const(encoder, input, 0);
            [encoder setBuffer:partial_values offset:0 atIndex:1];
            [encoder setBuffer:partial_indices offset:0 atIndex:2];
            bind_mutable(encoder, winners, 3);
            [encoder setBytes:&span_params length:sizeof(span_params) atIndex:4];
            dispatch_groups(encoder, programs->extrema_merge, rows, 1);
            // The command buffer retains bound buffers until it completes.
            [partial_values release];
            [partial_indices release];
        } else {
            bind_const(encoder, input, 0);
            bind_mutable(encoder, winners, 1);
            [encoder setBytes:&params length:sizeof(params) atIndex:2];
            if (width >= 128)
                dispatch_groups(encoder, programs->row_extrema_group, rows, 1);
            else
                dispatch_threads(encoder, programs->row_extrema, rows);
        }

        bind_const(encoder, input, 0);
        bind_mutable(encoder, winners, 1);
        bind_mutable(encoder, output, 2);
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        dispatch_threads(encoder, programs->row_select, rows * output_width);
        [encoder endEncoding];
        return finish_command(encoding.command);
    }
}

namespace {

int metal_winner_apply(
    const void* input,
    const void* winners,
    void* output,
    std::uint64_t rows,
    std::uint64_t input_width,
    std::uint64_t output_width,
    bool select) {
    @autoreleasepool {
        if (!metal_rows_valid(input, output, rows, input_width, output_width) ||
            !metal_winners_valid(winners, input, rows))
            return 2;
        if (!const_buffer(input) || !const_buffer(winners) ||
            !mutable_buffer(output))
            return 3;
        Encoding encoding;
        if (!begin_encoding(input, encoding)) return 5;
        SelectParams params{
            static_cast<std::uint32_t>(rows),
            static_cast<std::uint32_t>(input_width),
            static_cast<std::uint32_t>(output_width),
            0u};
        bind_const(encoding.encoder, input, 0);
        bind_const(encoding.encoder, winners, 1);
        bind_mutable(encoding.encoder, output, 2);
        [encoding.encoder setBytes:&params length:sizeof(params) atIndex:3];
        dispatch_threads(
            encoding.encoder,
            select ? encoding.programs->row_select
                   : encoding.programs->row_scatter,
            rows * output_width);
        [encoding.encoder endEncoding];
        return finish_command(encoding.command);
    }
}

} // namespace

extern "C" int math_native_metal_row_select(
    const void* input,
    const void* winners,
    void* output,
    std::uint64_t rows,
    std::uint64_t input_width,
    std::uint64_t output_width) {
    return metal_winner_apply(
        input, winners, output, rows, input_width, output_width, true);
}

extern "C" int math_native_metal_row_scatter(
    const void* input,
    const void* winners,
    void* output,
    std::uint64_t rows,
    std::uint64_t input_width,
    std::uint64_t output_width) {
    return metal_winner_apply(
        input, winners, output, rows, input_width, output_width, false);
}

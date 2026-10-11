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
#include <vector>

// Domain messages of the checked unary operations, defined in
// math_native.cpp beside the CPU and fake-GPU checks.
extern "C" const char* math_native_unary_domain_message(std::int32_t operation);

namespace {

struct MetalPrograms {
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> forward_plain = nil;
    id<MTLComputePipelineState> forward_checked = nil;
    id<MTLComputePipelineState> backward = nil;
    id<MTLComputePipelineState> second_backward = nil;
    id<MTLComputePipelineState> finite = nil;
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

kernel void finite_output(device const float* input [[buffer(0)]],
    device atomic_uint* status [[buffer(1)]],
    constant uint& count [[buffer(2)]], uint index [[thread_position_in_grid]]) {
    if (index < count && !isfinite(input[index]))
        atomic_store_explicit(status, 1u, memory_order_relaxed);
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
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        const char* fast_math = std::getenv("QUIDRA_METAL_FAST_MATH");
        options.fastMathEnabled = fast_math && std::strcmp(fast_math, "1") == 0;
#pragma clang diagnostic pop
        // Round each product before subsequent additions in unary derivatives.
        NSString* source = [@"#pragma METAL fp contract(off)\n"
            stringByAppendingString:math_metal_source()];
        NSError* error = nil;
        programs.library =
            [device newLibraryWithSource:source
                                 options:options
                                   error:&error];
        [options release];
        if (programs.library) {
            programs.forward_plain =
                make_pipeline(device, programs.library, @"unary_forward_plain");
            programs.forward_checked =
                make_pipeline(device, programs.library, @"unary_forward_checked");
            programs.backward =
                make_pipeline(device, programs.library, @"unary_backward");
            programs.second_backward =
                make_pipeline(device, programs.library, @"unary_second_backward");
            programs.finite = make_pipeline(device, programs.library, @"finite_output");
        }
    }
    if (!programs.library || !programs.forward_plain ||
        !programs.forward_checked || !programs.backward ||
        !programs.second_backward || !programs.finite)
        return nullptr;
    return &programs;
}

template <typename T>
T bridge_handle(std::uint64_t handle) {
    return (__bridge T)(reinterpret_cast<void*>(
        static_cast<std::uintptr_t>(handle)));
}

// The device's MTLDevice, for compiling Math's pipelines. A lookup that lends
// no queue, so it is allowed anywhere, also while Core's stream is held.
id<MTLDevice> native_device(long long device) {
    const auto handle = qcore_device_native_device(device);
    return handle == 0 ? nil : bridge_handle<id<MTLDevice>>(handle);
}

// A tensor buffer lent by Core, taken before the encoder scope opens: taking
// it may make Core work for it (a copy-on-write detach, an upload, storage of
// its own), which Core refuses inside the scope.
struct Borrowed {
    id<MTLBuffer> buffer = nil;
    NSUInteger offset = 0;

    explicit operator bool() const {
        return buffer != nil;
    }
};

Borrowed borrowed(const void* tensor, std::uint64_t handle) {
    if (handle == 0) return {};
    return {bridge_handle<id<MTLBuffer>>(handle),
            static_cast<NSUInteger>(qcore_tensor_device_offset_bytes(tensor))};
}

Borrowed borrow_const(const void* tensor) {
    return borrowed(tensor, qcore_tensor_device_handle_const(tensor));
}

Borrowed borrow_mutable(void* tensor) {
    return borrowed(tensor, qcore_tensor_device_handle(tensor));
}

// A write-only output: the kernel stores every element, so Core lends
// uninitialized storage with no fill (qcore_tensor_output_handle), and the
// output is marked written once the kernel is encoded.
Borrowed borrow_output(void* tensor) {
    return borrowed(tensor, qcore_tensor_output_handle(tensor));
}

Borrowed borrow_written(void* tensor, bool write_only) {
    return write_only ? borrow_output(tensor) : borrow_mutable(tensor);
}

bool mark_written(void* tensor) {
    return qcore_tensor_mark_written(tensor) == 0;
}

void bind(id<MTLComputeCommandEncoder> encoder, const Borrowed& value,
          NSUInteger index) {
    [encoder setBuffer:value.buffer offset:value.offset atIndex:index];
}

// One Math submission on a Metal device.
//
// On Core's command stream it follows the encode-scope protocol of
// quidra/native_extension.h: the constructor holds the device's stream, the
// caller then takes every tensor handle and scratch buffer, opens the scope
// (compute_encoder() or command_buffer()), sets state and dispatches, and
// close() ends scope and hold with one qcore_metal_note_work. The work joins
// Core's open command buffer, which Core commits with its own work and waits
// for only where the host reads, so Math never commits or waits.
//
// Without a Core stream (QUIDRA_METAL_STREAM=0) the same calls fill a command
// buffer of Math's own on Core's queue, and close() commits it without
// waiting; Core's synchronization covers command buffers committed on its
// queue.
//
// A submission that is not closed (an early return) ends the hold with no
// work, and an own command buffer is dropped uncommitted.
class MetalSubmission {
public:
    explicit MetalSubmission(long long device)
        : device_(device), held_(qcore_device_encode_begin(device) != 0) {}

    MetalSubmission(const MetalSubmission&) = delete;
    MetalSubmission& operator=(const MetalSubmission&) = delete;

    ~MetalSubmission() {
        if (closed_) return;
        if (held_) {
            qcore_metal_note_work(device_, 0, 0);
        } else if (own_encoder_) {
            [encoder_ endEncoding];
        }
    }

    // Opens a compute-encoder scope. On Core's stream this is Core's open
    // encoder (serial dispatch, tracked resources), which Math never ends.
    id<MTLComputeCommandEncoder> compute_encoder() {
        if (encoder_) return encoder_;
        if (held_) {
            const auto handle = qcore_device_compute_encoder(device_);
            if (handle != 0)
                encoder_ = bridge_handle<id<MTLComputeCommandEncoder>>(handle);
            return encoder_;
        }
        if (!own_command_buffer()) return nil;
        encoder_ = [command_ computeCommandEncoder];
        own_encoder_ = encoder_ != nil;
        return encoder_;
    }

    // Opens a command-buffer scope instead, for encoders Math creates itself
    // (and ends before close()). Scopes do not change kind.
    id<MTLCommandBuffer> command_buffer() {
        if (command_) return command_;
        if (held_) {
            const auto handle = qcore_metal_command_buffer(device_);
            if (handle != 0)
                command_ = bridge_handle<id<MTLCommandBuffer>>(handle);
            return command_;
        }
        return own_command_buffer() ? command_ : nil;
    }

    // Ends the submission after a successful encode. `dispatches` and
    // `bytes` feed Core's commit policy.
    int close(std::uint32_t dispatches, std::uint64_t bytes) {
        if (closed_) return 5;
        closed_ = true;
        if (held_) {
            qcore_metal_note_work(device_, dispatches, bytes);
            return 0;
        }
        if (own_encoder_) [encoder_ endEncoding];
        if (command_) [command_ commit];
        return 0;
    }

private:
    bool own_command_buffer() {
        if (command_) return true;
        const auto handle = qcore_device_queue_handle(device_);
        if (handle == 0) return false;
        command_ = [bridge_handle<id<MTLCommandQueue>>(handle) commandBuffer];
        return command_ != nil;
    }

    long long device_;
    bool held_;
    bool closed_ = false;
    bool own_encoder_ = false;
    id<MTLCommandBuffer> command_ = nil;
    id<MTLComputeCommandEncoder> encoder_ = nil;
};

// A Core status word for a checked kernel (sqrt and log domains). Take it
// before the encoder scope opens and bind it; after close() defer() hands
// the check to Core, which reports `message` with the statement that queued
// the kernel at the device's next synchronization point (a host read,
// gpu.sync or program exit) instead of waiting for the kernel now. A slot
// that was never deferred is released.
class DeviceStatus {
public:
    explicit DeviceStatus(long long device) : device_(device) {}
    DeviceStatus(const DeviceStatus&) = delete;
    DeviceStatus& operator=(const DeviceStatus&) = delete;
    ~DeviceStatus() {
        if (taken_) qcore_device_status_release(device_, slot_);
    }

    bool take() {
        std::uint64_t handle = 0;
        if (qcore_device_status_slot(device_, &handle, &offset_, &slot_) != 0)
            return false;
        taken_ = true;
        buffer_ = bridge_handle<id<MTLBuffer>>(handle);
        return buffer_ != nil;
    }

    id<MTLBuffer> buffer() const {
        return buffer_;
    }

    NSUInteger offset() const {
        return static_cast<NSUInteger>(offset_);
    }

    bool defer(const char* message) {
        if (!taken_) return false;
        if (qcore_device_defer_status(device_, slot_, message) == 0) {
            taken_ = false;
            return true;
        }
        // The encode has closed. Consume the slot only after its queued writes
        // complete when Core could not retain it for deferred validation.
        waited_ = qcore_device_status_wait(device_, slot_, &value_) == 0;
        taken_ = false;
        return false;
    }

    bool failed_check() const { return waited_ && value_ != 0; }

private:
    long long device_;
    bool taken_ = false;
    bool waited_ = false;
    std::uint32_t value_ = 0;
    std::uint64_t offset_ = 0;
    std::uint64_t slot_ = 0;
    id<MTLBuffer> buffer_ = nil;
};

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

void encode_dispatch(
    id<MTLComputeCommandEncoder> encoder,
    id<MTLComputePipelineState> pipeline,
    std::uint32_t count) {
    [encoder setComputePipelineState:pipeline];
    if (count != 0) {
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(thread_count(pipeline), 1, 1)];
    }
}

std::uint64_t float_bytes(std::uint64_t count) {
    return count * sizeof(float);
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

        const bool checked = operation == 2 || operation == 3;
        if (!checked && operation != 1 && operation != 4)
            return 2;

        const auto device = qcore_tensor_device(input);
        auto* programs = programs_for(native_device(device));
        if (!programs) return 5;

        MetalSubmission submission(device);
        const auto in = borrow_const(input);
        const auto out = borrow_output(output);
        if (!in || !out) return 3;
        DeviceStatus status(device);
        if (checked && !status.take()) return 5;
        id<MTLComputeCommandEncoder> encoder = submission.compute_encoder();
        if (!encoder) return 5;
        bind(encoder, in, 0);
        bind(encoder, out, 1);
        if (checked) {
            [encoder setBuffer:status.buffer()
                        offset:status.offset()
                       atIndex:2];
            [encoder setBytes:&count length:sizeof(count) atIndex:3];
            [encoder setBytes:&operation length:sizeof(operation) atIndex:4];
            encode_dispatch(encoder, programs->forward_checked, count);
        } else {
            [encoder setBytes:&count length:sizeof(count) atIndex:2];
            [encoder setBytes:&operation length:sizeof(operation) atIndex:3];
            encode_dispatch(encoder, programs->forward_plain, count);
        }
        const int closed = submission.close(1, 2 * float_bytes(count));
        if (closed != 0) return closed;
        if (!mark_written(output)) return 5;
        if (!checked) return 0;
        if (status.defer(math_native_unary_domain_message(operation))) return 0;
        // A refused handoff still carries the completed domain check.
        return status.failed_check() ? 4 : 5;
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

        const auto device = qcore_tensor_device(input);
        auto* programs = programs_for(native_device(device));
        if (!programs) return 5;

        MetalSubmission submission(device);
        const auto x = borrow_const(input);
        const auto y = borrow_const(output);
        const auto upstream = borrow_const(gradient_output);
        const auto result = borrow_mutable(gradient_input);
        if (!x || !y || !upstream || !result) return 3;
        id<MTLComputeCommandEncoder> encoder = submission.compute_encoder();
        if (!encoder) return 5;
        bind(encoder, x, 0);
        bind(encoder, y, 1);
        bind(encoder, upstream, 2);
        bind(encoder, result, 3);
        [encoder setBytes:&count length:sizeof(count) atIndex:4];
        [encoder setBytes:&operation length:sizeof(operation) atIndex:5];
        encode_dispatch(encoder, programs->backward, count);
        return submission.close(1, 4 * float_bytes(count));
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

        const auto device = qcore_tensor_device(input);
        auto* programs = programs_for(native_device(device));
        if (!programs) return 5;

        MetalSubmission submission(device);
        const auto x = borrow_const(input);
        const auto first = borrow_const(first_gradient);
        const auto upstream = borrow_const(gradient_output);
        const auto result = borrow_mutable(gradient_input);
        const auto result_first = borrow_mutable(gradient_first);
        if (!x || !first || !upstream || !result || !result_first) return 3;
        id<MTLComputeCommandEncoder> encoder = submission.compute_encoder();
        if (!encoder) return 5;
        bind(encoder, x, 0);
        bind(encoder, first, 1);
        bind(encoder, upstream, 2);
        bind(encoder, result, 3);
        bind(encoder, result_first, 4);
        [encoder setBytes:&count length:sizeof(count) atIndex:5];
        [encoder setBytes:&operation length:sizeof(operation) atIndex:6];
        encode_dispatch(encoder, programs->second_backward, count);
        return submission.close(1, 5 * float_bytes(count));
    }
}

// ===========================================================================
// Math-owned dense matrix products and reductions on Metal.
//
// Core lends tensor buffers, scratch storage and its command stream; Math
// owns the kernels and their numerical semantics. The library is compiled
// without fast-math so floating-point comparisons keep IEEE NaN/signed-zero
// behavior and divisions round correctly, like the CPU implementation.
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

// Scratch storage from Core's device buffer pool, for intermediate results
// that never leave one submission. Release it once the work that uses it has
// been encoded (after close()): Core hands the buffer out again only after
// the command buffer holding that work, or one Math committed on Core's queue
// before the release, has completed.
class Scratch {
public:
    Scratch() = default;
    Scratch(const Scratch&) = delete;
    Scratch& operator=(const Scratch&) = delete;
    Scratch(Scratch&& other) noexcept
        : token_(other.token_), buffer_(other.buffer_) {
        other.token_ = nullptr;
        other.buffer_ = nil;
    }
    ~Scratch() {
        if (token_) qcore_device_buffer_release(token_);
    }

    bool allocate(long long device, std::uint64_t bytes) {
        token_ = qcore_device_buffer_allocate(device, bytes);
        if (!token_) return false;
        const auto handle = qcore_device_buffer_handle(token_);
        if (handle != 0) buffer_ = bridge_handle<id<MTLBuffer>>(handle);
        return buffer_ != nil;
    }

    id<MTLBuffer> buffer() const {
        return buffer_;
    }

private:
    void* token_ = nullptr;
    id<MTLBuffer> buffer_ = nil;
};

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

// The buffers of one product, taken before the scope opens.
struct GemmBuffers {
    Borrowed left;
    Borrowed right;
    Borrowed output;
};

std::uint64_t gemm_bytes(const MathGemmTask& task) {
    return float_bytes(task.rows * task.inner + task.inner * task.columns +
                       task.rows * task.columns);
}

void encode_gemm(
    id<MTLComputeCommandEncoder> encoder,
    const LinearPrograms& programs,
    const MathGemmTask& task,
    const GemmBuffers& buffers) {
    GemmParams params{
        static_cast<std::uint32_t>(task.rows),
        static_cast<std::uint32_t>(task.inner),
        static_cast<std::uint32_t>(task.columns),
        static_cast<std::uint32_t>(task.left_row_stride),
        static_cast<std::uint32_t>(task.left_inner_stride),
        static_cast<std::uint32_t>(task.right_inner_stride),
        static_cast<std::uint32_t>(task.right_column_stride),
        0};
    bind(encoder, buffers.left, 0);
    bind(encoder, buffers.right, 1);
    bind(encoder, buffers.output, 2);
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

id mps_matrix(const Borrowed& value, const MpsOperand& operand) {
    auto& api = mps();
    MPSMatrixDescriptor* descriptor =
        [(id)api.descriptor
            matrixDescriptorWithRows:static_cast<NSUInteger>(operand.stored_rows)
                             columns:static_cast<NSUInteger>(operand.stored_columns)
                            rowBytes:static_cast<NSUInteger>(operand.row_bytes)
                            dataType:MPSDataTypeFloat32];
    if (!descriptor) return nil;
    return [(MPSMatrix*)[api.matrix alloc]
        initWithBuffer:value.buffer
                offset:value.offset
            descriptor:descriptor];
}


// True when MPS will serve this product (opted in, eligible shape and policy,
// framework available for the device).
bool mps_serves(const MathGemmTask& task, id<MTLDevice> device) {
    if (!mps_eligible(task)) return false;
    auto& api = mps();
    if (!api.ready || !api.supports(device)) return false;
    MpsOperand operand{};
    return mps_operand(task.rows, task.inner, task.left_row_stride,
                       task.left_inner_stride, operand) &&
           mps_operand(task.inner, task.columns, task.right_inner_stride,
                       task.right_column_stride, operand);
}

// Encodes the product with MPS; false leaves the task to the Math kernels
// (nothing has been encoded).
bool encode_mps(
    id<MTLCommandBuffer> command, id<MTLDevice> device,
    const MathGemmTask& task, const GemmBuffers& buffers) {
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
    id left_matrix = mps_matrix(buffers.left, left);
    id right_matrix = mps_matrix(buffers.right, right);
    id result_matrix = mps_matrix(buffers.output, result);
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

int gemm_outputs_written(
    const MathGemmTask* tasks, std::uint64_t task_count, bool write_only) {
    if (!write_only) return 0;
    for (std::uint64_t index = 0; index < task_count; ++index)
        if (!mark_written(tasks[index].output)) return 5;
    return 0;
}

} // namespace

// write_only: the outputs are fresh forward outputs, taken write-only and
// marked written; otherwise they are gradients Core lends.
extern "C" int math_native_metal_gemm(
    const MathGemmTask* tasks,
    std::uint64_t task_count,
    int write_only) {
    @autoreleasepool {
        if (!tasks || task_count == 0 || !tasks[0].output) return 1;
        const auto device = qcore_tensor_device(tasks[0].output);
        std::uint64_t bytes = 0;
        for (std::uint64_t index = 0; index < task_count; ++index) {
            if (!gemm_task_valid(tasks[index], device)) return 2;
            bytes += gemm_bytes(tasks[index]);
        }
        id<MTLDevice> metal_device = native_device(device);
        auto* programs = linear_programs_for(metal_device);
        if (!programs) return 5;
        // MPS encodes into a command buffer, so a call with an MPS product
        // takes Core's command buffer and creates its own encoders on it.
        bool with_mps = false;
        for (std::uint64_t index = 0; index < task_count; ++index)
            with_mps = with_mps || mps_serves(tasks[index], metal_device);

        MetalSubmission submission(device);
        std::vector<GemmBuffers> buffers(static_cast<std::size_t>(task_count));
        for (std::uint64_t index = 0; index < task_count; ++index) {
            auto& task = tasks[index];
            auto& bound = buffers[static_cast<std::size_t>(index)];
            bound.left = borrow_const(task.left);
            bound.right = borrow_const(task.right);
            bound.output = borrow_written(task.output, write_only != 0);
            if (!bound.left || !bound.right || !bound.output) return 3;
        }
        if (!with_mps) {
            id<MTLComputeCommandEncoder> encoder = submission.compute_encoder();
            if (!encoder) return 5;
            for (std::uint64_t index = 0; index < task_count; ++index)
                encode_gemm(encoder, *programs, tasks[index],
                            buffers[static_cast<std::size_t>(index)]);
            const int closed = submission.close(
                static_cast<std::uint32_t>(task_count), bytes);
            return closed != 0 ? closed
                : gemm_outputs_written(tasks, task_count, write_only != 0);
        }

        id<MTLCommandBuffer> command = submission.command_buffer();
        if (!command) return 5;
        id<MTLComputeCommandEncoder> encoder = nil;
        for (std::uint64_t index = 0; index < task_count; ++index) {
            const auto& task = tasks[index];
            const auto& bound = buffers[static_cast<std::size_t>(index)];
            if (mps_serves(task, metal_device)) {
                if (encoder) {
                    [encoder endEncoding];
                    encoder = nil;
                }
                if (encode_mps(command, metal_device, task, bound)) continue;
            }
            if (!encoder) {
                encoder = [command computeCommandEncoder];
                if (!encoder) return 5;
            }
            encode_gemm(encoder, *programs, task, bound);
        }
        if (encoder) [encoder endEncoding];
        const int closed =
            submission.close(static_cast<std::uint32_t>(task_count), bytes);
        return closed != 0 ? closed
            : gemm_outputs_written(tasks, task_count, write_only != 0);
    }
}

extern "C" int math_native_metal_row_reduce(
    const void* input,
    void* output,
    std::uint64_t rows,
    std::uint64_t input_width,
    std::uint64_t output_width,
    double divisor,
    int pairwise,
    int write_only) {
    @autoreleasepool {
        if (!metal_rows_valid(input, output, rows, input_width, output_width))
            return 2;
        const auto device = qcore_tensor_device(input);
        auto* programs = linear_programs_for(native_device(device));
        if (!programs) return 5;

        RowParams params{};
        params.rows = static_cast<std::uint32_t>(rows);
        params.input_width = static_cast<std::uint32_t>(input_width);
        params.output_width = static_cast<std::uint32_t>(output_width);
        params.divide = divisor != 1.0 ? 1u : 0u;
        params.divisor = static_cast<float>(divisor);
        const bool whole_tree =
            pairwise && rows == 1 && output_width == 1 && input_width > 1;
        const bool direct = input_width == 1 || output_width == 1 ||
                            input_width * output_width <= 4096;
        const std::uint64_t bytes =
            float_bytes(rows * input_width + rows * output_width);

        MetalSubmission submission(device);
        const auto in = borrow_const(input);
        const auto out = borrow_written(output, write_only != 0);
        if (!in || !out) return 3;
        const auto finish = [&](std::uint32_t dispatches) {
            const int closed = submission.close(dispatches, bytes);
            if (closed != 0) return closed;
            return write_only != 0 && !mark_written(output) ? 5 : 0;
        };

        if (whole_tree) {
            // Whole-tensor tree: block passes until one value remains. Every
            // pass but the last writes its block sums to scratch.
            std::vector<std::uint64_t> counts{input_width};
            while (counts.back() > pairwise_block_elements)
                counts.push_back(
                    (counts.back() + pairwise_block_elements - 1) /
                    pairwise_block_elements);
            std::vector<Scratch> levels(counts.size() - 1);
            for (std::size_t level = 1; level < counts.size(); ++level)
                if (!levels[level - 1].allocate(
                        device, float_bytes(counts[level])))
                    return 5;
            id<MTLComputeCommandEncoder> encoder = submission.compute_encoder();
            if (!encoder) return 5;
            for (std::size_t level = 0; level < counts.size(); ++level) {
                const bool last = level + 1 == counts.size();
                RowParams pass = params;
                pass.count = static_cast<std::uint32_t>(counts[level]);
                pass.divide = last ? params.divide : 0u;
                if (level == 0) bind(encoder, in, 0);
                else [encoder setBuffer:levels[level - 1].buffer() offset:0 atIndex:0];
                if (last) bind(encoder, out, 1);
                else [encoder setBuffer:levels[level].buffer() offset:0 atIndex:1];
                [encoder setBytes:&pass length:sizeof(pass) atIndex:2];
                const std::uint64_t groups =
                    (counts[level] + pairwise_block_elements - 1) /
                    pairwise_block_elements;
                dispatch_groups(encoder, programs->pairwise_block, groups, 1);
            }
            return finish(static_cast<std::uint32_t>(counts.size()));
        }
        if (direct) {
            id<MTLComputeCommandEncoder> encoder = submission.compute_encoder();
            if (!encoder) return 5;
            bind(encoder, in, 0);
            bind(encoder, out, 1);
            [encoder setBytes:&params length:sizeof(params) atIndex:2];
            dispatch_threads(
                encoder, programs->row_reduce_broadcast, rows * output_width);
            return finish(1);
        }
        Scratch sums;
        if (!sums.allocate(device, float_bytes(rows))) return 5;
        id<MTLComputeCommandEncoder> encoder = submission.compute_encoder();
        if (!encoder) return 5;
        bind(encoder, in, 0);
        [encoder setBuffer:sums.buffer() offset:0 atIndex:1];
        [encoder setBytes:&params length:sizeof(params) atIndex:2];
        dispatch_threads(encoder, programs->row_sums, rows);
        [encoder setBuffer:sums.buffer() offset:0 atIndex:0];
        bind(encoder, out, 1);
        [encoder setBytes:&params length:sizeof(params) atIndex:2];
        dispatch_threads(
            encoder, programs->row_broadcast, rows * output_width);
        return finish(2);
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
        const auto device = qcore_tensor_device(input);
        auto* programs = linear_programs_for(native_device(device));
        if (!programs) return 5;
        SelectParams params{
            static_cast<std::uint32_t>(rows),
            static_cast<std::uint32_t>(width),
            static_cast<std::uint32_t>(output_width),
            maximum ? 1u : 0u};
        const bool spans_needed = rows < 64 && width >= extrema_span_threshold;
        const std::uint64_t spans = !spans_needed ? 0 : std::min<std::uint64_t>(
            extrema_max_spans,
            (width + extrema_span_elements - 1) / extrema_span_elements);
        const std::uint64_t bytes =
            float_bytes(rows * width + rows * output_width + rows);

        MetalSubmission submission(device);
        const auto in = borrow_const(input);
        const auto out = borrow_output(output);
        const auto best = borrow_output(winners);
        if (!in || !out || !best) return 3;
        Scratch partial_values;
        Scratch partial_indices;
        if (spans_needed &&
            (!partial_values.allocate(device, rows * spans * sizeof(float)) ||
             !partial_indices.allocate(
                 device, rows * spans * sizeof(std::int32_t))))
            return 5;
        id<MTLComputeCommandEncoder> encoder = submission.compute_encoder();
        if (!encoder) return 5;

        std::uint32_t dispatches = 2;
        if (spans_needed) {
            // Few very wide rows: span winners first, then one merge per row.
            SpanParams span_params{
                static_cast<std::uint32_t>(rows),
                static_cast<std::uint32_t>(width),
                static_cast<std::uint32_t>(spans),
                static_cast<std::uint32_t>((width + spans - 1) / spans),
                maximum ? 1u : 0u, 0u, 0u, 0u};
            bind(encoder, in, 0);
            [encoder setBuffer:partial_values.buffer() offset:0 atIndex:1];
            [encoder setBuffer:partial_indices.buffer() offset:0 atIndex:2];
            [encoder setBytes:&span_params length:sizeof(span_params) atIndex:3];
            dispatch_groups(encoder, programs->extrema_partials, spans, rows);
            bind(encoder, in, 0);
            [encoder setBuffer:partial_values.buffer() offset:0 atIndex:1];
            [encoder setBuffer:partial_indices.buffer() offset:0 atIndex:2];
            bind(encoder, best, 3);
            [encoder setBytes:&span_params length:sizeof(span_params) atIndex:4];
            dispatch_groups(encoder, programs->extrema_merge, rows, 1);
            dispatches = 3;
        } else {
            bind(encoder, in, 0);
            bind(encoder, best, 1);
            [encoder setBytes:&params length:sizeof(params) atIndex:2];
            if (width >= 128)
                dispatch_groups(encoder, programs->row_extrema_group, rows, 1);
            else
                dispatch_threads(encoder, programs->row_extrema, rows);
        }

        bind(encoder, in, 0);
        bind(encoder, best, 1);
        bind(encoder, out, 2);
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        dispatch_threads(encoder, programs->row_select, rows * output_width);
        const int closed = submission.close(dispatches, bytes);
        if (closed != 0) return closed;
        return mark_written(output) && mark_written(winners) ? 0 : 5;
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
        const auto device = qcore_tensor_device(input);
        auto* programs = linear_programs_for(native_device(device));
        if (!programs) return 5;
        SelectParams params{
            static_cast<std::uint32_t>(rows),
            static_cast<std::uint32_t>(input_width),
            static_cast<std::uint32_t>(output_width),
            0u};

        MetalSubmission submission(device);
        const auto in = borrow_const(input);
        const auto best = borrow_const(winners);
        const auto out = borrow_mutable(output);
        if (!in || !best || !out) return 3;
        id<MTLComputeCommandEncoder> encoder = submission.compute_encoder();
        if (!encoder) return 5;
        bind(encoder, in, 0);
        bind(encoder, best, 1);
        bind(encoder, out, 2);
        [encoder setBytes:&params length:sizeof(params) atIndex:3];
        dispatch_threads(
            encoder,
            select ? programs->row_select : programs->row_scatter,
            rows * output_width);
        return submission.close(
            1, float_bytes(rows * input_width + rows * output_width + rows));
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

extern "C" int math_native_metal_validate_output(
    void* output, const char* operation, const char* message) {
    @autoreleasepool {
        if (!metal_tensor(output)) return 7;
        std::uint32_t count = 0;
        if (!count_u32(output, count)) return 2;
        if (count == 0) return 0;
        const auto device = qcore_tensor_device(output);
        auto* programs = programs_for(native_device(device));
        if (!programs) return 5;
        MetalSubmission submission(device);
        const auto value = borrow_const(output);
        DeviceStatus status(device);
        if (!value || !status.take()) return 5;
        auto encoder = submission.compute_encoder();
        if (!encoder) return 5;
        bind(encoder, value, 0);
        [encoder setBuffer:status.buffer() offset:status.offset() atIndex:1];
        [encoder setBytes:&count length:sizeof(count) atIndex:2];
        encode_dispatch(encoder, programs->finite, count);
        const int closed = submission.close(1, float_bytes(count));
        if (closed != 0) return closed;
        if (status.defer(message)) return 0;
        // The encode is closed and the valid slot has been consumed before
        // the failure bridge may unwind the calling Node.
        if (status.failed_check()) qcore_tensor_report_status(output, operation, 1);
        return 5;
    }
}

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <quidra/native_extension.h>

#include <cstdint>
#include <limits>
#include <map>
#include <mutex>

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

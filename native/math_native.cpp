#include <quidra/native_extension.h>

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <utility>

#ifdef __APPLE__
extern "C" int math_native_metal_tensor_unary_forward(
    const void* input, void* output, std::int32_t operation);
extern "C" int math_native_metal_tensor_unary_backward(
    const void* input, const void* output, const void* gradient_output,
    void* gradient_input, std::int32_t operation);
extern "C" int math_native_metal_tensor_unary_second_backward(
    const void* input, const void* first_gradient, const void* gradient_output,
    void* gradient_input, void* gradient_first, std::int32_t operation);
#endif

extern "C" int math_native_cuda_tensor_unary_forward(
    const void* input, void* output, std::int32_t operation);
extern "C" int math_native_cuda_tensor_unary_backward(
    const void* input, const void* output, const void* gradient_output,
    void* gradient_input, std::int32_t operation);
extern "C" int math_native_cuda_tensor_unary_second_backward(
    const void* input, const void* first_gradient, const void* gradient_output,
    void* gradient_input, void* gradient_first, std::int32_t operation);

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

const char* math_exact_atom_decimal(std::uint32_t opcode) {
    switch (opcode) {
        case 1:
            return "3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651328230664709384460955058223172535940812848111745028410270193852";
        case 2:
            return "2.71828182845904523536028747135266249775724709369995957496696762772407663035354759457138217852516642742746639193200305992181741359662904357290033429526059563073813232862794349";
        default:
            return nullptr;
    }
}

int math_exact_unary_evaluate(
    std::uint32_t opcode, double input, double* output) {
    if (!output || !std::isfinite(input)) return 0;
    switch (opcode) {
        case 1:
            if (input < 0.0) return 0;
            *output = std::sqrt(input);
            break;
        case 2: *output = std::sin(input); break;
        case 3: *output = std::cos(input); break;
        case 4: *output = std::tan(input); break;
        case 5:
            if (input <= 0.0) return 0;
            *output = std::log(input);
            break;
        case 6: *output = std::exp(input); break;
        default: return 0;
    }
    return std::isfinite(*output) ? 1 : 0;
}

std::uint32_t math_exact_unary_flags(std::uint32_t opcode) {
    switch (opcode) {
        case 1:
            return QCORE_EXACT_UNARY_DOMAIN_NONNEGATIVE |
                   QCORE_EXACT_UNARY_RESULT_NONNEGATIVE;
        case 2:
        case 3:
            return QCORE_EXACT_UNARY_TOTAL;
        case 4:
            return 0;
        case 5:
            return QCORE_EXACT_UNARY_DOMAIN_POSITIVE;
        case 6:
            return QCORE_EXACT_UNARY_TOTAL |
                   QCORE_EXACT_UNARY_RESULT_POSITIVE;
        default:
            return 0;
    }
}

struct MathExactProviderRegistration {
    MathExactProviderRegistration() {
        (void)qcore_exact_real_provider_register(
            "math", math_exact_atom_decimal);
        (void)qcore_exact_real_provider_register_unary(
            "math", math_exact_unary_evaluate, math_exact_unary_flags);
    }
};

const MathExactProviderRegistration math_exact_provider_registration{};

bool multiply_ok(std::size_t left, std::size_t right) {
    return right == 0 || left <= std::numeric_limits<std::size_t>::max() / right;
}

bool cpu_f32_contiguous(const void* tensor) {
    return tensor &&
           qcore_native_abi_version() == QUIDRA_NATIVE_ABI_VERSION &&
           qcore_tensor_dtype(tensor) == QCORE_DTYPE_FLOAT32 &&
           qcore_tensor_backend(tensor) == QCORE_BACKEND_CPU &&
           qcore_tensor_is_contiguous(tensor);
}

bool cuda_f32(const void* tensor) {
    return tensor &&
           qcore_native_abi_version() == QUIDRA_NATIVE_ABI_VERSION &&
           qcore_tensor_dtype(tensor) == QCORE_DTYPE_FLOAT32 &&
           qcore_tensor_backend(tensor) == QCORE_BACKEND_CUDA;
}

const void* cuda_pointer_const(const void* tensor) {
    const auto base = qcore_tensor_device_handle_const(tensor);
    if (base == 0) return nullptr;
    const auto offset = qcore_tensor_device_offset_bytes(tensor);
    if (offset > std::numeric_limits<std::uint64_t>::max() - base)
        return nullptr;
    return reinterpret_cast<const void*>(base + offset);
}

void* cuda_pointer(void* tensor) {
    const auto base = qcore_tensor_device_handle(tensor);
    if (base == 0) return nullptr;
    const auto offset = qcore_tensor_device_offset_bytes(tensor);
    if (offset > std::numeric_limits<std::uint64_t>::max() - base)
        return nullptr;
    return reinterpret_cast<void*>(base + offset);
}

class DynamicLibrary {
public:
    DynamicLibrary() = default;
    DynamicLibrary(const DynamicLibrary&) = delete;
    DynamicLibrary& operator=(const DynamicLibrary&) = delete;

    DynamicLibrary(DynamicLibrary&& other) noexcept
        : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    DynamicLibrary& operator=(DynamicLibrary&& other) noexcept {
        if (this == &other) return *this;
        close();
        handle_ = other.handle_;
        other.handle_ = nullptr;
        return *this;
    }

    ~DynamicLibrary() {
        close();
    }

    bool open(const char* name) {
        close();
#ifdef _WIN32
        handle_ = LoadLibraryA(name);
#else
        handle_ = dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
        return handle_ != nullptr;
    }

    template <typename T>
    T symbol(const char* name) const {
        if (!handle_) return nullptr;
#ifdef _WIN32
        return reinterpret_cast<T>(GetProcAddress(handle_, name));
#else
        return reinterpret_cast<T>(dlsym(handle_, name));
#endif
    }

private:
    void close() noexcept {
#ifdef _WIN32
        if (handle_) FreeLibrary(handle_);
#else
        if (handle_) dlclose(handle_);
#endif
        handle_ = nullptr;
    }

#ifdef _WIN32
    HMODULE handle_{};
#else
    void* handle_{};
#endif
};

using CublasStatus = int;
using CublasHandle = void*;
constexpr CublasStatus cublas_success = 0;
constexpr int cublas_op_n = 0;
constexpr int cublas_atomics_not_allowed = 0;
constexpr int cublas_default_math = 0;

struct CublasApi {
    DynamicLibrary library;
    CublasStatus (*create)(CublasHandle*){};
    CublasStatus (*destroy)(CublasHandle){};
    CublasStatus (*sgemm)(
        CublasHandle, int, int, int, int, int,
        const float*, const float*, int,
        const float*, int, const float*, float*, int){};
    CublasStatus (*set_atomics_mode)(CublasHandle, int){};
    CublasStatus (*set_math_mode)(CublasHandle, int){};

    bool ready() const {
        return create && destroy && sgemm;
    }
};

CublasApi load_cublas() {
#ifdef _WIN32
    const char* candidates[] = {
        "cublas64_13.dll",
        "cublas64_12.dll",
        "cublas64_11.dll"
    };
#else
    const char* candidates[] = {
        "libcublas.so.13",
        "libcublas.so.12",
        "libcublas.so.11",
        "libcublas.so"
    };
#endif
    for (const auto* name : candidates) {
        CublasApi api;
        if (!api.library.open(name)) continue;
        api.create =
            api.library.symbol<decltype(api.create)>("cublasCreate_v2");
        api.destroy =
            api.library.symbol<decltype(api.destroy)>("cublasDestroy_v2");
        api.sgemm =
            api.library.symbol<decltype(api.sgemm)>("cublasSgemm_v2");
        api.set_atomics_mode =
            api.library.symbol<decltype(api.set_atomics_mode)>(
                "cublasSetAtomicsMode");
        api.set_math_mode =
            api.library.symbol<decltype(api.set_math_mode)>(
                "cublasSetMathMode");
        if (api.ready()) return api;
    }
    return {};
}

CublasApi& cublas() {
    static CublasApi api = load_cublas();
    return api;
}

struct CublasSession {
    CublasApi* api{};
    CublasHandle handle{};

    ~CublasSession() {
        if (api && handle && api->destroy)
            (void)api->destroy(handle);
    }
};

bool validate_matmul_shape(
    const void* a,
    const void* b,
    const void* output,
    std::size_t& rows,
    std::size_t& inner,
    std::size_t& columns) {
    const auto a_rank = qcore_tensor_rank(a);
    const auto b_rank = qcore_tensor_rank(b);
    if (a_rank < 1 || (b_rank != 1 && b_rank != 2)) return false;

    const auto inner_ll = qcore_tensor_extent(a, a_rank - 1);
    const auto b_inner_ll = qcore_tensor_extent(b, 0);
    if (inner_ll <= 0 || b_inner_ll != inner_ll) return false;
    inner = static_cast<std::size_t>(inner_ll);

    const auto columns_ll =
        b_rank == 1 ? 1 : qcore_tensor_extent(b, 1);
    if (columns_ll <= 0) return false;
    columns = static_cast<std::size_t>(columns_ll);

    rows = 1;
    for (std::uint64_t axis = 0; axis + 1 < a_rank; ++axis) {
        const auto extent_ll = qcore_tensor_extent(a, axis);
        if (extent_ll <= 0) return false;
        const auto extent = static_cast<std::size_t>(extent_ll);
        if (!multiply_ok(rows, extent)) return false;
        rows *= extent;
    }

    if (qcore_tensor_rank(output) != (b_rank == 1 ? a_rank - 1 : a_rank))
        return false;
    for (std::uint64_t axis = 0; axis + 1 < a_rank; ++axis) {
        if (qcore_tensor_extent(output, axis) != qcore_tensor_extent(a, axis))
            return false;
    }
    if (b_rank == 2 &&
        qcore_tensor_extent(output, a_rank - 1) != columns_ll)
        return false;

    if (!multiply_ok(rows, inner) ||
        !multiply_ok(inner, columns) ||
        !multiply_ok(rows, columns))
        return false;

    return qcore_tensor_element_count(a) == rows * inner &&
           qcore_tensor_element_count(b) == inner * columns &&
           qcore_tensor_element_count(output) == rows * columns;
}

int empty_sum_backward(
    const void* const*,
    std::uint64_t,
    const void*,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void*,
    std::uint64_t) {
    // Core allocates custom-gradient outputs as initialized zeros. The
    // derivative of a sum over an empty tensor has zero elements, so there are
    // no payload bytes to write.
    return gradient_inputs && gradient_input_count == 1 ? 0 : 1;
}

int empty_sum_backward_tracked(
    const void* const*,
    std::uint64_t differentiable_input_count,
    const void* const*,
    std::uint64_t,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void*,
    std::uint64_t) {
    if (differentiable_input_count != 1 ||
        !gradient_output ||
        !gradient_inputs ||
        gradient_input_count != 1 ||
        !gradient_inputs[0]) {
        return 1;
    }

    // Higher-order provenance remains package-owned. This zero-valued empty
    // gradient gets a zero-Jacobian custom node whose parent is the incoming
    // scalar gradient.
    const void* parent = gradient_output;
    return qcore_tensor_attach_custom_autograd_ex(
        gradient_inputs[0],
        &parent,
        1,
        empty_sum_backward,
        empty_sum_backward_tracked,
        nullptr,
        0);
}


enum class TensorUnaryOperation : std::uint8_t {
    Abs = 1,
    Sqrt = 2,
    Log = 3,
    Exp = 4
};

bool same_tensor_shape(const void* left, const void* right) {
    if (qcore_tensor_rank(left) != qcore_tensor_rank(right) ||
        qcore_tensor_element_count(left) != qcore_tensor_element_count(right))
        return false;
    for (std::uint64_t axis = 0; axis < qcore_tensor_rank(left); ++axis) {
        if (qcore_tensor_extent(left, axis) != qcore_tensor_extent(right, axis))
            return false;
    }
    return true;
}

template <typename T>
const T* tensor_unary_read(const void* tensor) {
    const auto backend = qcore_tensor_backend(tensor);
    if (backend == QCORE_BACKEND_CPU)
        return static_cast<const T*>(qcore_tensor_cpu_data_const(tensor));
    if (backend == QCORE_BACKEND_TEST) {
        const auto handle = qcore_tensor_device_handle_const(tensor);
        if (handle == 0) return nullptr;
        const auto offset = qcore_tensor_device_offset_bytes(tensor);
        if (offset > std::numeric_limits<std::uint64_t>::max() - handle)
            return nullptr;
        return reinterpret_cast<const T*>(
            static_cast<std::uintptr_t>(handle + offset));
    }
    return nullptr;
}

template <typename T>
T* tensor_unary_write(void* tensor) {
    const auto backend = qcore_tensor_backend(tensor);
    if (backend == QCORE_BACKEND_CPU)
        return static_cast<T*>(qcore_tensor_cpu_data(tensor));
    if (backend == QCORE_BACKEND_TEST) {
        const auto handle = qcore_tensor_device_handle(tensor);
        if (handle == 0) return nullptr;
        const auto offset = qcore_tensor_device_offset_bytes(tensor);
        if (offset > std::numeric_limits<std::uint64_t>::max() - handle)
            return nullptr;
        return reinterpret_cast<T*>(
            static_cast<std::uintptr_t>(handle + offset));
    }
    return nullptr;
}

template <typename T>
T tensor_unary_forward(T value, TensorUnaryOperation operation, bool& domain_ok) {
    switch (operation) {
        case TensorUnaryOperation::Abs:
            return std::fabs(value);
        case TensorUnaryOperation::Sqrt:
            if (value < T(0)) {
                domain_ok = false;
                return T(0);
            }
            return std::sqrt(value);
        case TensorUnaryOperation::Log:
            if (!(value > T(0))) {
                domain_ok = false;
                return T(0);
            }
            return std::log(value);
        case TensorUnaryOperation::Exp:
            return std::exp(value);
    }
    domain_ok = false;
    return T(0);
}

template <typename T>
int tensor_unary_backward_typed(
    const void* input_raw,
    const void* output_raw,
    const void* gradient_output_raw,
    void* gradient_input_raw,
    TensorUnaryOperation operation) {
    const auto count = qcore_tensor_element_count(input_raw);
    if (qcore_tensor_element_count(output_raw) != count ||
        qcore_tensor_element_count(gradient_output_raw) != count ||
        qcore_tensor_element_count(gradient_input_raw) != count)
        return 2;

    const auto* input = tensor_unary_read<T>(input_raw);
    const auto* output = tensor_unary_read<T>(output_raw);
    const auto* gradient_output = tensor_unary_read<T>(gradient_output_raw);
    auto* gradient_input = tensor_unary_write<T>(gradient_input_raw);
    if (!input || !output || !gradient_output || !gradient_input)
        return 3;

    for (std::uint64_t index = 0; index < count; ++index) {
        T derivative = T(0);
        switch (operation) {
            case TensorUnaryOperation::Abs:
                derivative = input[index] < T(0)
                    ? T(-1)
                    : (input[index] > T(0) ? T(1) : T(0));
                break;
            case TensorUnaryOperation::Sqrt:
                derivative = T(1) / (T(2) * output[index]);
                break;
            case TensorUnaryOperation::Log:
                derivative = T(1) / input[index];
                break;
            case TensorUnaryOperation::Exp:
                derivative = output[index];
                break;
        }
        gradient_input[index] = gradient_output[index] * derivative;
    }
    return 0;
}

template <typename T>
T tensor_unary_first_derivative(
    T input,
    T output,
    TensorUnaryOperation operation) {
    switch (operation) {
        case TensorUnaryOperation::Abs:
            return input < T(0) ? T(-1) : (input > T(0) ? T(1) : T(0));
        case TensorUnaryOperation::Sqrt:
            return T(1) / (T(2) * output);
        case TensorUnaryOperation::Log:
            return T(1) / input;
        case TensorUnaryOperation::Exp:
            return output;
    }
    return T(0);
}

template <typename T>
T tensor_unary_second_derivative(
    T input,
    TensorUnaryOperation operation) {
    switch (operation) {
        case TensorUnaryOperation::Abs:
            return T(0);
        case TensorUnaryOperation::Sqrt:
            return T(-0.25) / (input * std::sqrt(input));
        case TensorUnaryOperation::Log:
            return T(-1) / (input * input);
        case TensorUnaryOperation::Exp:
            return std::exp(input);
    }
    return T(0);
}

int tensor_unary_backward(
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata,
    std::uint64_t metadata_size) {
    if (!saved_tensors || saved_tensor_count != 2 ||
        !gradient_output || !gradient_inputs || gradient_input_count != 1 ||
        !gradient_inputs[0] || !metadata || metadata_size != 1)
        return 1;

    const auto operation =
        static_cast<TensorUnaryOperation>(
            *static_cast<const std::uint8_t*>(metadata));
    const auto dtype = qcore_tensor_dtype(saved_tensors[0]);
    if (dtype != qcore_tensor_dtype(saved_tensors[1]) ||
        dtype != qcore_tensor_dtype(gradient_output) ||
        dtype != qcore_tensor_dtype(gradient_inputs[0]))
        return 2;

    const auto backend = qcore_tensor_backend(saved_tensors[0]);
    if (backend == QCORE_BACKEND_CPU || backend == QCORE_BACKEND_TEST) {
        if (dtype == QCORE_DTYPE_FLOAT32)
            return tensor_unary_backward_typed<float>(
                saved_tensors[0], saved_tensors[1], gradient_output,
                gradient_inputs[0], operation);
        if (dtype == QCORE_DTYPE_FLOAT64)
            return tensor_unary_backward_typed<double>(
                saved_tensors[0], saved_tensors[1], gradient_output,
                gradient_inputs[0], operation);
        return 2;
    }
    if (backend == QCORE_BACKEND_CUDA)
        return math_native_cuda_tensor_unary_backward(
            saved_tensors[0], saved_tensors[1], gradient_output,
            gradient_inputs[0], static_cast<std::int32_t>(operation));
#ifdef __APPLE__
    if (backend == QCORE_BACKEND_METAL)
        return math_native_metal_tensor_unary_backward(
            saved_tensors[0], saved_tensors[1], gradient_output,
            gradient_inputs[0], static_cast<std::int32_t>(operation));
#endif
    return 6;
}

template <typename T>
int tensor_unary_second_backward_typed(
    const void* input_raw,
    const void* first_gradient_raw,
    const void* gradient_output_raw,
    void* gradient_input_raw,
    void* gradient_first_raw,
    TensorUnaryOperation operation) {
    const auto count = qcore_tensor_element_count(input_raw);
    if (qcore_tensor_element_count(first_gradient_raw) != count ||
        qcore_tensor_element_count(gradient_output_raw) != count ||
        qcore_tensor_element_count(gradient_input_raw) != count ||
        qcore_tensor_element_count(gradient_first_raw) != count)
        return 2;

    const auto* input = tensor_unary_read<T>(input_raw);
    const auto* first_gradient = tensor_unary_read<T>(first_gradient_raw);
    const auto* gradient_output = tensor_unary_read<T>(gradient_output_raw);
    auto* gradient_input = tensor_unary_write<T>(gradient_input_raw);
    auto* gradient_first = tensor_unary_write<T>(gradient_first_raw);
    if (!input || !first_gradient || !gradient_output ||
        !gradient_input || !gradient_first)
        return 3;

    for (std::uint64_t index = 0; index < count; ++index) {
        bool domain_ok = true;
        const T output = tensor_unary_forward(
            input[index], operation, domain_ok);
        if (!domain_ok) return 4;
        const T first = tensor_unary_first_derivative(
            input[index], output, operation);
        const T second = tensor_unary_second_derivative(
            input[index], operation);
        gradient_input[index] =
            gradient_output[index] * first_gradient[index] * second;
        gradient_first[index] =
            gradient_output[index] * first;
    }
    return 0;
}

int tensor_unary_second_backward(
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata,
    std::uint64_t metadata_size) {
    if (!saved_tensors || saved_tensor_count != 2 ||
        !gradient_output || !gradient_inputs || gradient_input_count != 2 ||
        !gradient_inputs[0] || !gradient_inputs[1] ||
        !metadata || metadata_size != 1)
        return 1;

    const auto operation =
        static_cast<TensorUnaryOperation>(
            *static_cast<const std::uint8_t*>(metadata));
    const auto dtype = qcore_tensor_dtype(saved_tensors[0]);
    if (dtype != qcore_tensor_dtype(saved_tensors[1]) ||
        dtype != qcore_tensor_dtype(gradient_output) ||
        dtype != qcore_tensor_dtype(gradient_inputs[0]) ||
        dtype != qcore_tensor_dtype(gradient_inputs[1]))
        return 2;

    const auto backend = qcore_tensor_backend(saved_tensors[0]);
    if (backend == QCORE_BACKEND_CPU || backend == QCORE_BACKEND_TEST) {
        if (dtype == QCORE_DTYPE_FLOAT32)
            return tensor_unary_second_backward_typed<float>(
                saved_tensors[0], saved_tensors[1], gradient_output,
                gradient_inputs[0], gradient_inputs[1], operation);
        if (dtype == QCORE_DTYPE_FLOAT64)
            return tensor_unary_second_backward_typed<double>(
                saved_tensors[0], saved_tensors[1], gradient_output,
                gradient_inputs[0], gradient_inputs[1], operation);
        return 2;
    }
    if (backend == QCORE_BACKEND_CUDA)
        return math_native_cuda_tensor_unary_second_backward(
            saved_tensors[0], saved_tensors[1], gradient_output,
            gradient_inputs[0], gradient_inputs[1],
            static_cast<std::int32_t>(operation));
#ifdef __APPLE__
    if (backend == QCORE_BACKEND_METAL)
        return math_native_metal_tensor_unary_second_backward(
            saved_tensors[0], saved_tensors[1], gradient_output,
            gradient_inputs[0], gradient_inputs[1],
            static_cast<std::int32_t>(operation));
#endif
    return 6;
}

int tensor_unary_backward_tracked(
    const void* const* differentiable_inputs,
    std::uint64_t differentiable_input_count,
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata,
    std::uint64_t metadata_size) {
    if (!differentiable_inputs || differentiable_input_count != 1 ||
        !differentiable_inputs[0] ||
        !gradient_output || !gradient_inputs || gradient_input_count != 1 ||
        !gradient_inputs[0])
        return 1;

    const int status = tensor_unary_backward(
        saved_tensors, saved_tensor_count, gradient_output,
        gradient_inputs, gradient_input_count, metadata, metadata_size);
    if (status != 0) return status;

    const void* parents[] = {differentiable_inputs[0], gradient_output};
    const void* second_saved[] = {differentiable_inputs[0], gradient_output};
    return qcore_tensor_attach_custom_autograd_with_saved_ex(
        gradient_inputs[0],
        parents,
        2,
        second_saved,
        2,
        tensor_unary_second_backward,
        nullptr,
        metadata,
        metadata_size);
}

template <typename T>
int tensor_unary_forward_typed(
    const void* input_raw,
    void* output_raw,
    TensorUnaryOperation operation) {
    const auto count = qcore_tensor_element_count(input_raw);
    const auto* input = tensor_unary_read<T>(input_raw);
    auto* output = tensor_unary_write<T>(output_raw);
    if (!input || !output) return 3;

    bool domain_ok = true;
    for (std::uint64_t index = 0; index < count; ++index) {
        output[index] = tensor_unary_forward(input[index], operation, domain_ok);
        if (!domain_ok) return 4;
    }
    return 0;
}

} // namespace

extern "C" int math_native_tensor_unary(
    const void* input,
    void* output,
    std::int32_t operation_raw) {
    if (!input || !output ||
        qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION ||
        qcore_tensor_backend(input) != qcore_tensor_backend(output) ||
        qcore_tensor_device(input) != qcore_tensor_device(output) ||
        !qcore_tensor_is_contiguous(input) ||
        !qcore_tensor_is_contiguous(output) ||
        !same_tensor_shape(input, output) ||
        qcore_tensor_dtype(input) != qcore_tensor_dtype(output)) {
        return 1;
    }

    if (operation_raw < static_cast<std::int32_t>(TensorUnaryOperation::Abs) ||
        operation_raw > static_cast<std::int32_t>(TensorUnaryOperation::Exp))
        return 2;
    const auto operation =
        static_cast<TensorUnaryOperation>(operation_raw);
    const auto dtype = qcore_tensor_dtype(input);

    int status = 2;
    const auto backend = qcore_tensor_backend(input);
    if (backend == QCORE_BACKEND_CPU || backend == QCORE_BACKEND_TEST) {
        if (dtype == QCORE_DTYPE_FLOAT32)
            status = tensor_unary_forward_typed<float>(input, output, operation);
        else if (dtype == QCORE_DTYPE_FLOAT64)
            status = tensor_unary_forward_typed<double>(input, output, operation);
    }
    else if (backend == QCORE_BACKEND_CUDA) {
        status = math_native_cuda_tensor_unary_forward(
            input, output, operation_raw);
    }
#ifdef __APPLE__
    else if (backend == QCORE_BACKEND_METAL) {
        status = math_native_metal_tensor_unary_forward(
            input, output, operation_raw);
    }
#endif
    else {
        return 6;
    }
    if (status != 0) return status;

    const void* differentiable_inputs[] = {input};
    const void* saved_tensors[] = {input, output};
    const auto metadata = static_cast<std::uint8_t>(operation);
    const auto attach_status = qcore_tensor_attach_custom_autograd_with_saved_ex(
        output,
        differentiable_inputs,
        1,
        saved_tensors,
        2,
        tensor_unary_backward,
        tensor_unary_backward_tracked,
        &metadata,
        sizeof(metadata));
    return attach_status == 0 ? 0 : 5;
}

extern "C" int math_native_is_cpu_f32(const void* value) {
    return cpu_f32_contiguous(value) ? 1 : 0;
}

extern "C" int math_native_cuda_device_f32(const void* value) {
    if (!cuda_f32(value)) return -1;
    const auto device = qcore_tensor_device(value);
    if (device < 0 ||
        device > static_cast<long long>(
            std::numeric_limits<std::int32_t>::max()))
        return -1;
    return static_cast<int>(device);
}

extern "C" int math_native_matmul_f32(
    const void* a,
    const void* b,
    void* output) {
    if (!cpu_f32_contiguous(a) ||
        !cpu_f32_contiguous(b) ||
        !cpu_f32_contiguous(output)) {
        return 1;
    }

    std::size_t rows = 0;
    std::size_t inner = 0;
    std::size_t columns = 0;
    if (!validate_matmul_shape(
            a, b, output, rows, inner, columns))
        return 2;

    const auto* left =
        static_cast<const float*>(qcore_tensor_cpu_data_const(a));
    const auto* right =
        static_cast<const float*>(qcore_tensor_cpu_data_const(b));
    auto* destination =
        static_cast<float*>(qcore_tensor_cpu_data(output));
    if (!left || !right || !destination) return 3;

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t column = 0; column < columns; ++column) {
            float total = 0.0f;
            for (std::size_t k = 0; k < inner; ++k) {
                total += left[row * inner + k] *
                         right[k * columns + column];
            }
            destination[row * columns + column] = total;
        }
    }
    return 0;
}

extern "C" int math_native_matmul_cuda_f32(
    const void* a,
    const void* b,
    void* output) {
    if (!cuda_f32(a) || !cuda_f32(b) || !cuda_f32(output) ||
        !qcore_tensor_is_contiguous(a) ||
        !qcore_tensor_is_contiguous(b) ||
        !qcore_tensor_is_contiguous(output)) {
        return 1;
    }

    const auto device = qcore_tensor_device(a);
    if (device < 0 ||
        qcore_tensor_device(b) != device ||
        qcore_tensor_device(output) != device ||
        !qcore_device_activate(device)) {
        return 2;
    }

    std::size_t rows = 0;
    std::size_t inner = 0;
    std::size_t columns = 0;
    if (!validate_matmul_shape(
            a, b, output, rows, inner, columns))
        return 3;

    if (rows > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        inner > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        columns > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return 4;
    }

    const auto* left = static_cast<const float*>(cuda_pointer_const(a));
    const auto* right = static_cast<const float*>(cuda_pointer_const(b));
    auto* destination = static_cast<float*>(cuda_pointer(output));
    if (!left || !right || !destination) return 5;

    auto& api = cublas();
    if (!api.ready()) return 6;

    CublasSession session{&api};
    if (api.create(&session.handle) != cublas_success ||
        !session.handle) {
        return 7;
    }

    if ((qcore_execution_policy_get() == QCORE_EXECUTION_DETERMINISTIC)) {
        if (api.set_atomics_mode &&
            api.set_atomics_mode(
                session.handle,
                cublas_atomics_not_allowed) != cublas_success) {
            return 8;
        }
    }
    if (api.set_math_mode &&
        api.set_math_mode(
            session.handle,
            cublas_default_math) != cublas_success) {
        return 9;
    }

    const float alpha = 1.0F;
    const float beta = 0.0F;

    // cuBLAS is column-major. Interpreting the row-major inputs as transposed
    // column-major matrices computes C^T = B^T * A^T without materialization.
    const auto status = api.sgemm(
        session.handle,
        cublas_op_n,
        cublas_op_n,
        static_cast<int>(columns),
        static_cast<int>(rows),
        static_cast<int>(inner),
        &alpha,
        right,
        static_cast<int>(columns),
        left,
        static_cast<int>(inner),
        &beta,
        destination,
        static_cast<int>(columns));
    return status == cublas_success ? 0 : 10;
}

extern "C" int math_native_attach_empty_sum(
    const void* input,
    void* output) {
    if (!input || !output ||
        qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION ||
        qcore_tensor_element_count(input) != 0 ||
        qcore_tensor_element_count(output) != 1 ||
        qcore_tensor_rank(output) != 0 ||
        qcore_tensor_dtype(input) != qcore_tensor_dtype(output) ||
        qcore_tensor_device(input) != qcore_tensor_device(output)) {
        return 1;
    }

    const auto dtype = qcore_tensor_dtype(input);
    if (dtype != QCORE_DTYPE_FLOAT32 && dtype != QCORE_DTYPE_FLOAT64)
        return 2;

    const void* inputs[] = {input};
    return qcore_tensor_attach_custom_autograd_ex(
        output,
        inputs,
        1,
        empty_sum_backward,
        empty_sum_backward_tracked,
        nullptr,
        0);
}


extern "C" float math_native_sqrt_f32(float value) { return std::sqrt(value); }
extern "C" double math_native_sqrt_f64(double value) { return std::sqrt(value); }
extern "C" float math_native_sin_f32(float value) { return std::sin(value); }
extern "C" double math_native_sin_f64(double value) { return std::sin(value); }
extern "C" float math_native_cos_f32(float value) { return std::cos(value); }
extern "C" double math_native_cos_f64(double value) { return std::cos(value); }
extern "C" float math_native_tan_f32(float value) { return std::tan(value); }
extern "C" double math_native_tan_f64(double value) { return std::tan(value); }
extern "C" float math_native_log_f32(float value) { return std::log(value); }
extern "C" double math_native_log_f64(double value) { return std::log(value); }
extern "C" float math_native_exp_f32(float value) { return std::exp(value); }
extern "C" double math_native_exp_f64(double value) { return std::exp(value); }
extern "C" float math_native_pow_f32(float base, float exponent) {
    return std::pow(base, exponent);
}
extern "C" double math_native_pow_f64(double base, double exponent) {
    return std::pow(base, exponent);
}
extern "C" bool math_native_is_finite_f32(float value) {
    return std::isfinite(value);
}
extern "C" bool math_native_is_finite_f64(double value) {
    return std::isfinite(value);
}

namespace {

long double math_native_rounded_value(long double value, int operation) {
    switch (operation) {
        case 1: return std::trunc(value);
        case 2: return std::round(value);
        case 3: return std::floor(value);
        case 4: return std::ceil(value);
        default: return std::numeric_limits<long double>::quiet_NaN();
    }
}

template <typename T>
bool math_native_round_valid(T value, int operation) {
    if (!std::isfinite(value)) return false;
    const long double rounded =
        math_native_rounded_value(static_cast<long double>(value), operation);
    return std::isfinite(rounded) &&
           rounded >= static_cast<long double>(std::numeric_limits<long long>::min()) &&
           rounded <= static_cast<long double>(std::numeric_limits<long long>::max());
}

template <typename T>
long long math_native_round_value(T value, int operation) {
    if (!math_native_round_valid(value, operation)) return 0;
    return static_cast<long long>(
        math_native_rounded_value(static_cast<long double>(value), operation));
}

} // namespace

extern "C" std::int32_t math_native_round_valid_f32(float value, int operation) {
    return math_native_round_valid(value, operation) ? 1 : 0;
}
extern "C" std::int32_t math_native_round_valid_f64(double value, int operation) {
    return math_native_round_valid(value, operation) ? 1 : 0;
}
extern "C" long long math_native_round_value_f32(float value, int operation) {
    return math_native_round_value(value, operation);
}
extern "C" long long math_native_round_value_f64(double value, int operation) {
    return math_native_round_value(value, operation);
}

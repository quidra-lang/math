#include <quidra/native_extension.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

// One dense matrix product C = L * R executed by a Math backend. The output is
// a contiguous [rows, columns] tensor; both operands are read through element
// strides so transposed storage never needs a materialized copy:
//   C[i, j] = sum_k L[i * left_row_stride + k * left_inner_stride] *
//                   R[k * right_inner_stride + j * right_column_stride]
// The same POD layout is declared in math_metal.mm.
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

#ifdef __APPLE__
extern "C" int math_native_metal_tensor_unary_forward(
    const void* input, void* output, std::int32_t operation);
extern "C" int math_native_metal_tensor_unary_backward(
    const void* input, const void* output, const void* gradient_output,
    void* gradient_input, std::int32_t operation);
extern "C" int math_native_metal_tensor_unary_second_backward(
    const void* input, const void* first_gradient, const void* gradient_output,
    void* gradient_input, void* gradient_first, std::int32_t operation);
extern "C" int math_native_metal_gemm(
    const MathGemmTask* tasks, std::uint64_t task_count);
extern "C" int math_native_metal_row_reduce(
    const void* input, void* output, std::uint64_t rows,
    std::uint64_t input_width, std::uint64_t output_width,
    double divisor, int pairwise);
extern "C" int math_native_metal_row_extrema(
    const void* input, void* output, void* winners, std::uint64_t rows,
    std::uint64_t width, std::uint64_t output_width, int maximum);
extern "C" int math_native_metal_row_select(
    const void* input, const void* winners, void* output, std::uint64_t rows,
    std::uint64_t input_width, std::uint64_t output_width);
extern "C" int math_native_metal_row_scatter(
    const void* input, const void* winners, void* output, std::uint64_t rows,
    std::uint64_t input_width, std::uint64_t output_width);
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

extern "C" int math_native_cuda_device_f32(const void* value) {
    if (!cuda_f32(value)) return -1;
    const auto device = qcore_tensor_device(value);
    if (device < 0 ||
        device > static_cast<long long>(
            std::numeric_limits<std::int32_t>::max()))
        return -1;
    return static_cast<int>(device);
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

// ===========================================================================
// Math-owned dense matrix products and reductions.
//
// Core supplies opaque tensor borrows, the device queue, and the generic
// custom-autograd substrate. Everything below -- the matrix-product and
// reduction formulas, their derivatives, first-occurrence extrema, and every
// backend kernel -- is Math semantics. Unsupported placements (CUDA/HIP for
// these operations) and unsupported dtypes return a nonzero status so the
// Quidra layer keeps using its portable composition.
// ===========================================================================

// Opt-in dispatch trace for Math's own test suites. The native results are
// bit-identical to the portable composition by design, so values alone cannot
// show which path served a call. With QUIDRA_MATH_TEST_NATIVE_TRACE set, every
// native matrix/reduction forward, backward, and Metal GEMM kernel choice
// writes one line "quidra-math native <event> <backend> <dtype>" to stderr.
// Nothing else changes; unset, a call costs one cached flag check.
extern "C" void math_native_test_trace(const char* event, const void* tensor) {
    static const bool enabled =
        std::getenv("QUIDRA_MATH_TEST_NATIVE_TRACE") != nullptr;
    if (!enabled || !event || !tensor) return;
    const char* backend = "other";
    switch (qcore_tensor_backend(tensor)) {
        case QCORE_BACKEND_CPU: backend = "cpu"; break;
        case QCORE_BACKEND_CUDA: backend = "cuda"; break;
        case QCORE_BACKEND_HIP: backend = "hip"; break;
        case QCORE_BACKEND_METAL: backend = "metal"; break;
        case QCORE_BACKEND_TEST: backend = "test"; break;
        default: break;
    }
    const auto dtype = qcore_tensor_dtype(tensor);
    const char* type = dtype == QCORE_DTYPE_FLOAT32 ? "float32"
        : dtype == QCORE_DTYPE_FLOAT64 ? "float64" : "other";
    std::fprintf(stderr, "quidra-math native %s %s %s\n", event, backend, type);
}

namespace {

// Status codes shared by the linear/reduction bridges.
constexpr int status_ok = 0;
constexpr int status_invalid = 1;
constexpr int status_shape = 2;
constexpr int status_access = 3;
constexpr int status_backend = 5;
constexpr int status_unsupported_backend = 6;
constexpr int status_unsupported_dtype = 7;


enum class LinearBackend : std::uint8_t {
    Host,
    Metal,
    Unsupported
};

LinearBackend linear_backend(const void* tensor) {
    switch (qcore_tensor_backend(tensor)) {
        case QCORE_BACKEND_CPU:
        case QCORE_BACKEND_TEST:
            return LinearBackend::Host;
#ifdef __APPLE__
        case QCORE_BACKEND_METAL:
            return LinearBackend::Metal;
#endif
        default:
            return LinearBackend::Unsupported;
    }
}

bool linear_dtype_supported(LinearBackend backend, int dtype) {
    if (backend == LinearBackend::Host)
        return dtype == QCORE_DTYPE_FLOAT32 || dtype == QCORE_DTYPE_FLOAT64;
    if (backend == LinearBackend::Metal)
        return dtype == QCORE_DTYPE_FLOAT32;
    return false;
}

// Every tensor taking part in one native call must share backend, device,
// dtype, and contiguous layout. Returns a status code.
int same_placement(
    const void* const* tensors, std::size_t count, int dtype) {
    if (count == 0 || !tensors[0]) return status_invalid;
    const auto backend = qcore_tensor_backend(tensors[0]);
    const auto device = qcore_tensor_device(tensors[0]);
    for (std::size_t index = 0; index < count; ++index) {
        const auto* tensor = tensors[index];
        if (!tensor ||
            qcore_tensor_backend(tensor) != backend ||
            qcore_tensor_device(tensor) != device ||
            !qcore_tensor_is_contiguous(tensor))
            return status_invalid;
        if (dtype != 0 && qcore_tensor_dtype(tensor) != dtype)
            return status_invalid;
    }
    return status_ok;
}

// A value borrow must describe its differentiable parent element for element:
// same dtype, placement, and logical shape (the borrow itself is contiguous).
bool value_borrow_of(const void* data, const void* parent) {
    return qcore_tensor_dtype(data) == qcore_tensor_dtype(parent) &&
           qcore_tensor_backend(data) == qcore_tensor_backend(parent) &&
           qcore_tensor_device(data) == qcore_tensor_device(parent) &&
           same_tensor_shape(data, parent);
}

// The element count as the portable mean builds it: powers of two, doubled in
// T, added for every set bit from the lowest. For counts beyond T's integer
// precision this rounds like the portable composition instead of converting
// the count once.
template <typename T>
T element_count_divisor(std::uint64_t count) {
    T result = T(0);
    T power = T(1);
    while (count > 0) {
        if (count % 2 == 1) result = result + power;
        count /= 2;
        if (count > 0) power = power + power;
    }
    return result;
}

template <typename T>
const T* host_read(const void* tensor) {
    return tensor_unary_read<T>(tensor);
}

template <typename T>
T* host_write(void* tensor) {
    return tensor_unary_write<T>(tensor);
}

// ---------------------------------------------------------------------------
// Host GEMM. Products are rounded before accumulation (no FMA contraction) and
// every output accumulates k in ascending order, exactly like the portable
// composition "products, then inner-axis additions", so CPU results match the
// portable path bit for bit. Both operands are packed (row blocks and
// zero-padded column panels) so any stride pattern -- including the transposed
// operands of backward products -- streams through a register-blocked
// micro-kernel; the depth is blocked to stay cache resident.
//
// Accumulators start at -0, the exact identity of IEEE addition (-0 + x == x
// for every x, +0 and NaN included), so each output equals p0 + p1 + ... as
// the portable composition defines it, signed zeros included: products that
// are all -0 sum to -0, which a +0 seed would turn into +0.

#pragma clang fp contract(off)

template <typename T, std::size_t Rows, std::size_t Width>
void host_gemm_micro(
    const T* left_panel, const T* right_panel, std::uint64_t depth,
    bool first, T* output, std::uint64_t output_stride,
    std::size_t valid_rows, std::size_t valid_columns) {
    T accumulator[Rows][Width];
    for (std::size_t row = 0; row < Rows; ++row)
        for (std::size_t column = 0; column < Width; ++column)
            accumulator[row][column] =
                (!first && row < valid_rows && column < valid_columns)
                    ? output[row * output_stride + column] : -T(0);
    for (std::uint64_t k = 0; k < depth; ++k) {
        const T* a = left_panel + k * Rows;
        const T* b = right_panel + k * Width;
        for (std::size_t row = 0; row < Rows; ++row) {
            const T scale = a[row];
            for (std::size_t column = 0; column < Width; ++column) {
                const T product = scale * b[column];
                accumulator[row][column] = accumulator[row][column] + product;
            }
        }
    }
    for (std::size_t row = 0; row < valid_rows; ++row)
        for (std::size_t column = 0; column < valid_columns; ++column)
            output[row * output_stride + column] = accumulator[row][column];
}

template <typename T, std::size_t block_rows, std::size_t width>
void host_gemm_blocked(
    const T* left, const T* right, T* output, const MathGemmTask& task) {
    constexpr std::uint64_t depth_block = 256;
    // Left row blocks are packed a chunk at a time, so the scratch space stays
    // bounded instead of duplicating the whole left operand.
    constexpr std::size_t chunk_blocks = 64;
    const auto rows = task.rows;
    const auto inner = task.inner;
    const auto columns = task.columns;
    const std::size_t panels = (columns + width - 1) / width;
    const std::size_t row_blocks = (rows + block_rows - 1) / block_rows;

    std::vector<T> packed_right(panels * inner * width, T(0));
    for (std::size_t panel = 0; panel < panels; ++panel) {
        const std::size_t first = panel * width;
        const std::size_t count =
            std::min<std::size_t>(width, columns - first);
        T* destination = packed_right.data() + panel * inner * width;
        for (std::uint64_t k = 0; k < inner; ++k) {
            const T* source = right + k * task.right_inner_stride;
            for (std::size_t column = 0; column < count; ++column)
                destination[k * width + column] =
                    source[(first + column) * task.right_column_stride];
        }
    }
    std::vector<T> packed_left(
        std::min(row_blocks, chunk_blocks) * inner * block_rows, T(0));

    for (std::size_t chunk = 0; chunk < row_blocks; chunk += chunk_blocks) {
        const std::size_t blocks = std::min(chunk_blocks, row_blocks - chunk);
        for (std::size_t block = 0; block < blocks; ++block) {
            const std::size_t first = (chunk + block) * block_rows;
            const std::size_t count =
                std::min<std::size_t>(block_rows, rows - first);
            T* destination = packed_left.data() + block * inner * block_rows;
            if (count < block_rows)
                std::fill(destination, destination + inner * block_rows, T(0));
            const T* source = left + first * task.left_row_stride;
            for (std::uint64_t k = 0; k < inner; ++k) {
                const T* column = source + k * task.left_inner_stride;
                for (std::size_t row = 0; row < count; ++row)
                    destination[k * block_rows + row] =
                        column[row * task.left_row_stride];
            }
        }

        for (std::uint64_t k0 = 0; k0 < inner; k0 += depth_block) {
            const std::uint64_t depth = std::min(depth_block, inner - k0);
            for (std::size_t panel = 0; panel < panels; ++panel) {
                const std::size_t first_column = panel * width;
                const std::size_t valid_columns =
                    std::min<std::size_t>(width, columns - first_column);
                const T* right_panel =
                    packed_right.data() + (panel * inner + k0) * width;
                for (std::size_t block = 0; block < blocks; ++block) {
                    const std::size_t first_row = (chunk + block) * block_rows;
                    host_gemm_micro<T, block_rows, width>(
                        packed_left.data() + (block * inner + k0) * block_rows,
                        right_panel, depth, k0 == 0,
                        output + first_row * columns + first_column, columns,
                        std::min<std::size_t>(block_rows, rows - first_row),
                        valid_columns);
                }
            }
        }
    }
}

// Register blocks of 16 vector accumulators: narrow outputs (the usual NN
// weight widths) use taller row blocks instead of padding columns.
template <typename T>
void host_gemm(
    const T* left, const T* right, T* output, const MathGemmTask& task) {
    constexpr std::size_t lanes = 16 / sizeof(T);
    if (task.columns <= lanes)
        host_gemm_blocked<T, 8, lanes>(left, right, output, task);
    else if (task.columns <= 2 * lanes && sizeof(T) == sizeof(float))
        host_gemm_blocked<T, 8, 2 * lanes>(left, right, output, task);
    else
        host_gemm_blocked<T, 4, 4 * lanes>(left, right, output, task);
}

#pragma clang fp contract(on)

// Host tasks of one call share operands (the upstream gradient feeds both
// backward products), so each borrowed tensor is resolved only once.
template <typename T>
int host_gemm_tasks(const MathGemmTask* tasks, std::size_t count) {
    const void* keys[6] = {};
    const T* values[6] = {};
    std::size_t known = 0;
    const auto read = [&](const void* tensor) -> const T* {
        for (std::size_t index = 0; index < known; ++index)
            if (keys[index] == tensor) return values[index];
        const T* pointer = host_read<T>(tensor);
        if (known < 6) {
            keys[known] = tensor;
            values[known] = pointer;
            ++known;
        }
        return pointer;
    };
    for (std::size_t index = 0; index < count; ++index) {
        const auto& task = tasks[index];
        const auto* left = read(task.left);
        const auto* right = read(task.right);
        auto* output = host_write<T>(task.output);
        if (!left || !right || !output) return status_access;
        host_gemm<T>(left, right, output, task);
    }
    return status_ok;
}

// Runs every task on the tensors' backend. All tasks share one placement.
int run_gemm_tasks(const MathGemmTask* tasks, std::size_t count) {
    if (count == 0) return status_ok;
    const auto backend = linear_backend(tasks[0].output);
    const auto dtype = qcore_tensor_dtype(tasks[0].output);
    if (backend == LinearBackend::Host) {
        if (dtype == QCORE_DTYPE_FLOAT32)
            return host_gemm_tasks<float>(tasks, count);
        if (dtype == QCORE_DTYPE_FLOAT64)
            return host_gemm_tasks<double>(tasks, count);
        return status_unsupported_dtype;
    }
#ifdef __APPLE__
    if (backend == LinearBackend::Metal) {
        if (dtype != QCORE_DTYPE_FLOAT32) return status_unsupported_dtype;
        return math_native_metal_gemm(tasks, count);
    }
#endif
    return status_unsupported_backend;
}

// ---------------------------------------------------------------------------
// Matrix-product autograd.
//
// A product node computes C[M, N] = X[M, K] * Y[K, N]. saved[0] and saved[1]
// hold X and Y as contiguous snapshots, optionally stored transposed. The
// node's parents are the original differentiable tensors, whose logical
// layout is either the operand itself or its transpose; each gradient is
// written directly in its parent's layout. Because dX = G Y^T and dY = X^T G
// are products of the same family, tracked backward attaches product nodes
// to the gradients, so backward(track = true) stays differentiable to any
// order without a portable gather graph.

struct MatmulMetadata {
    std::uint64_t rows;
    std::uint64_t inner;
    std::uint64_t columns;
    std::uint8_t right_is_vector;
    std::uint8_t left_saved_transposed;
    std::uint8_t right_saved_transposed;
    std::uint8_t left_parent_transposed;
    std::uint8_t right_parent_transposed;
    std::uint8_t left_required;
    std::uint8_t right_required;
    std::uint8_t reserved;
};

static_assert(sizeof(MatmulMetadata) == 32, "stable Math metadata layout");

struct MatrixView {
    std::uint64_t row_stride;
    std::uint64_t column_stride;
};

// Strides of a logical [rows, columns] matrix stored contiguously either as
// itself or as its transpose.
MatrixView stored_matrix(
    std::uint64_t rows, std::uint64_t columns, bool transposed) {
    return transposed ? MatrixView{1, rows} : MatrixView{columns, 1};
}

MatrixView transposed_view(MatrixView view) {
    return {view.column_stride, view.row_stride};
}

MathGemmTask gemm_task(
    const void* left, MatrixView left_view,
    const void* right, MatrixView right_view,
    void* output,
    std::uint64_t rows, std::uint64_t inner, std::uint64_t columns) {
    return MathGemmTask{
        left, right, output, rows, inner, columns,
        left_view.row_stride, left_view.column_stride,
        right_view.row_stride, right_view.column_stride};
}

bool matmul_metadata(
    const void* metadata, std::uint64_t metadata_size,
    MatmulMetadata& result) {
    if (!metadata || metadata_size != sizeof(MatmulMetadata)) return false;
    std::memcpy(&result, metadata, sizeof(MatmulMetadata));
    return result.rows != 0 && result.inner != 0 && result.columns != 0;
}

// Computes the requested product gradients for one node.
int matmul_gradients(
    const MatmulMetadata& meta,
    const void* saved_left,
    const void* saved_right,
    const void* gradient_output,
    void* gradient_left,
    void* gradient_right,
    bool left_needed,
    bool right_needed) {
    const auto rows = meta.rows;
    const auto inner = meta.inner;
    const auto columns = meta.columns;
    const auto dtype = qcore_tensor_dtype(gradient_output);
    const void* placement[] = {
        saved_left, saved_right, gradient_output, gradient_left, gradient_right};
    if (same_placement(placement, 5, dtype) != status_ok)
        return status_invalid;
    if (qcore_tensor_element_count(saved_left) != rows * inner ||
        qcore_tensor_element_count(saved_right) != inner * columns ||
        qcore_tensor_element_count(gradient_output) != rows * columns ||
        qcore_tensor_element_count(gradient_left) != rows * inner ||
        qcore_tensor_element_count(gradient_right) != inner * columns)
        return status_shape;

    const auto left = stored_matrix(rows, inner, meta.left_saved_transposed);
    const auto right =
        stored_matrix(inner, columns, meta.right_saved_transposed);
    const auto gradient = stored_matrix(rows, columns, false);

    MathGemmTask tasks[2];
    std::size_t count = 0;
    if (left_needed) {
        if (!meta.left_parent_transposed) {
            // dX = G Y^T  -> [rows, inner]
            tasks[count++] = gemm_task(
                gradient_output, gradient, saved_right, transposed_view(right),
                gradient_left, rows, columns, inner);
        } else {
            // dX^T = Y G^T -> [inner, rows]
            tasks[count++] = gemm_task(
                saved_right, right, gradient_output, transposed_view(gradient),
                gradient_left, inner, columns, rows);
        }
    }
    if (right_needed) {
        if (!meta.right_parent_transposed) {
            // dY = X^T G -> [inner, columns]
            tasks[count++] = gemm_task(
                saved_left, transposed_view(left), gradient_output, gradient,
                gradient_right, inner, rows, columns);
        } else {
            // dY^T = G^T X -> [columns, inner]
            tasks[count++] = gemm_task(
                gradient_output, transposed_view(gradient), saved_left, left,
                gradient_right, columns, rows, inner);
        }
    }
    return run_gemm_tasks(tasks, count);
}

int matmul_backward(
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata,
    std::uint64_t metadata_size) {
    MatmulMetadata meta{};
    if (!saved_tensors || saved_tensor_count != 2 || !gradient_output ||
        !gradient_inputs || gradient_input_count != 2 ||
        !gradient_inputs[0] || !gradient_inputs[1] ||
        !matmul_metadata(metadata, metadata_size, meta))
        return status_invalid;
    // Gradients of untracked parents are discarded by Core, so their products
    // are skipped; Core already initialized those buffers to zero.
    const int status = matmul_gradients(
        meta, saved_tensors[0], saved_tensors[1], gradient_output,
        gradient_inputs[0], gradient_inputs[1],
        meta.left_required != 0, meta.right_required != 0);
    if (status == status_ok)
        math_native_test_trace("matmul-backward", gradient_output);
    return status;
}

int matmul_backward_tracked(
    const void* const* differentiable_inputs,
    std::uint64_t differentiable_input_count,
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata,
    std::uint64_t metadata_size);

int attach_matmul_node(
    void* output,
    const void* first_parent,
    const void* second_parent,
    const void* first_saved,
    const void* second_saved,
    const MatmulMetadata& meta) {
    const void* parents[] = {first_parent, second_parent};
    const void* saved[] = {first_saved, second_saved};
    return qcore_tensor_attach_custom_autograd_with_saved_ex(
        output, parents, 2, saved, 2,
        matmul_backward, matmul_backward_tracked,
        &meta, sizeof(meta));
}

int matmul_backward_tracked(
    const void* const* differentiable_inputs,
    std::uint64_t differentiable_input_count,
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata,
    std::uint64_t metadata_size) {
    MatmulMetadata meta{};
    if (!differentiable_inputs || differentiable_input_count != 2 ||
        !differentiable_inputs[0] || !differentiable_inputs[1] ||
        !saved_tensors || saved_tensor_count != 2 || !gradient_output ||
        !gradient_inputs || gradient_input_count != 2 ||
        !gradient_inputs[0] || !gradient_inputs[1] ||
        !matmul_metadata(metadata, metadata_size, meta))
        return status_invalid;

    // Every gradient needs provenance in tracked mode, so compute both.
    const int status = matmul_gradients(
        meta, saved_tensors[0], saved_tensors[1], gradient_output,
        gradient_inputs[0], gradient_inputs[1], true, true);
    if (status != status_ok) return status;

    const auto rows = meta.rows;
    const auto inner = meta.inner;
    const auto columns = meta.columns;
    const bool lst = meta.left_saved_transposed != 0;
    const bool rst = meta.right_saved_transposed != 0;
    const bool lpt = meta.left_parent_transposed != 0;
    const bool rpt = meta.right_parent_transposed != 0;
    const void* left_parent = differentiable_inputs[0];
    const void* right_parent = differentiable_inputs[1];
    const void* saved_left = saved_tensors[0];
    const void* saved_right = saved_tensors[1];

    MatmulMetadata left_node{};
    left_node.left_required = 1;
    left_node.right_required = 1;
    int attach = 0;
    if (!lpt) {
        // dX = G * Y^T: operands (G, Y^T), parents (G, right parent).
        left_node.rows = rows;
        left_node.inner = columns;
        left_node.columns = inner;
        left_node.left_saved_transposed = 0;
        left_node.right_saved_transposed = rst ? 0 : 1;
        left_node.left_parent_transposed = 0;
        left_node.right_parent_transposed = rpt ? 0 : 1;
        attach = attach_matmul_node(
            gradient_inputs[0], gradient_output, right_parent,
            gradient_output, saved_right, left_node);
    } else {
        // dX^T = Y * G^T: operands (Y, G^T), parents (right parent, G).
        left_node.rows = inner;
        left_node.inner = columns;
        left_node.columns = rows;
        left_node.left_saved_transposed = rst ? 1 : 0;
        left_node.right_saved_transposed = 1;
        left_node.left_parent_transposed = rpt ? 1 : 0;
        left_node.right_parent_transposed = 1;
        attach = attach_matmul_node(
            gradient_inputs[0], right_parent, gradient_output,
            saved_right, gradient_output, left_node);
    }
    if (attach != 0) return status_backend;

    MatmulMetadata right_node{};
    right_node.left_required = 1;
    right_node.right_required = 1;
    if (!rpt) {
        // dY = X^T * G: operands (X^T, G), parents (left parent, G).
        right_node.rows = inner;
        right_node.inner = rows;
        right_node.columns = columns;
        right_node.left_saved_transposed = lst ? 0 : 1;
        right_node.right_saved_transposed = 0;
        right_node.left_parent_transposed = lpt ? 0 : 1;
        right_node.right_parent_transposed = 0;
        attach = attach_matmul_node(
            gradient_inputs[1], left_parent, gradient_output,
            saved_left, gradient_output, right_node);
    } else {
        // dY^T = G^T * X: operands (G^T, X), parents (G, left parent).
        right_node.rows = columns;
        right_node.inner = rows;
        right_node.columns = inner;
        right_node.left_saved_transposed = 1;
        right_node.right_saved_transposed = lst ? 1 : 0;
        right_node.left_parent_transposed = 1;
        right_node.right_parent_transposed = lpt ? 1 : 0;
        attach = attach_matmul_node(
            gradient_inputs[1], gradient_output, left_parent,
            gradient_output, saved_left, right_node);
    }
    if (attach != 0) return status_backend;
    math_native_test_trace("matmul-backward-tracked", gradient_output);
    return status_ok;
}

// Layout bits passed by the Quidra layer.
constexpr std::int32_t layout_left_transposed = 1;
constexpr std::int32_t layout_right_transposed = 2;
constexpr std::int32_t layout_left_tracked = 4;
constexpr std::int32_t layout_right_tracked = 8;

bool same_extents(const void* left, const void* right) {
    return same_tensor_shape(left, right);
}

bool transposed_extents(const void* data, const void* logical) {
    return qcore_tensor_rank(data) == 2 && qcore_tensor_rank(logical) == 2 &&
           qcore_tensor_extent(data, 0) == qcore_tensor_extent(logical, 1) &&
           qcore_tensor_extent(data, 1) == qcore_tensor_extent(logical, 0);
}

// ---------------------------------------------------------------------------
// Row reductions: out[r, j] = (sum_k in[r, k]) / divisor for j < output_width.
// The family is closed under differentiation: the adjoint swaps the input and
// output widths. Whole-tensor sum/mean use the portable composition's
// level-by-level pairwise tree; last-axis sums add left to right.

struct RowReduceMetadata {
    std::uint64_t rows;
    std::uint64_t input_width;
    std::uint64_t output_width;
    double divisor;
    std::uint8_t pairwise;
    std::uint8_t reserved[7];
};

static_assert(sizeof(RowReduceMetadata) == 40, "stable Math metadata layout");

// The portable composition carries an odd trailing element to the next level
// as `x + x * 0`: the identity for finite values, NaN for infinities. Math
// keeps that definition on every path.
template <typename T>
T pairwise_carry(T value) {
    return value + value * T(0);
}

// Reproduces the tree built by the portable whole-tensor sum: adjacent pairs
// are added level by level and an odd trailing element is carried upward.
// Processing elements in order with a stack of power-of-two partial sums
// yields exactly that tree: the leftover partials are the carried tails, and a
// tail is carried whenever it has to climb to the level of the next larger
// partial before it is added.
template <typename T>
T pairwise_sum(const T* values, std::uint64_t count) {
    T stack_values[64];
    std::uint64_t stack_sizes[64];
    int top = 0;
    std::uint64_t index = 0;
    const auto push = [&](T value, std::uint64_t size) {
        while (top > 0 && stack_sizes[top - 1] == size) {
            value = stack_values[top - 1] + value;
            size *= 2;
            --top;
        }
        stack_values[top] = value;
        stack_sizes[top] = size;
        ++top;
    };
    for (; index + 8 <= count; index += 8) {
        const T* x = values + index;
        const T block =
            ((x[0] + x[1]) + (x[2] + x[3])) + ((x[4] + x[5]) + (x[6] + x[7]));
        push(block, 8);
    }
    for (; index < count; ++index) push(values[index], 1);
    T total = stack_values[top - 1];
    std::uint64_t total_size = stack_sizes[top - 1];
    for (int level = top - 2; level >= 0; --level) {
        if (stack_sizes[level] > total_size) total = pairwise_carry(total);
        total = stack_values[level] + total;
        total_size = 2 * stack_sizes[level];
    }
    return total;
}

template <typename T>
void host_row_reduce(
    const T* input, T* output, const RowReduceMetadata& meta) {
    const bool divide = meta.divisor != 1.0;
    const T divisor = static_cast<T>(meta.divisor);
    if (meta.pairwise && meta.rows == 1 && meta.output_width == 1) {
        const T total = pairwise_sum(input, meta.input_width);
        output[0] = divide ? total / divisor : total;
        return;
    }
    for (std::uint64_t row = 0; row < meta.rows; ++row) {
        const T* source = input + row * meta.input_width;
        T total = source[0];
        for (std::uint64_t k = 1; k < meta.input_width; ++k)
            total = total + source[k];
        if (divide) total = total / divisor;
        T* destination = output + row * meta.output_width;
        for (std::uint64_t column = 0; column < meta.output_width; ++column)
            destination[column] = total;
    }
}

template <typename T>
int host_row_reduce_typed(
    const void* input, void* output, const RowReduceMetadata& meta) {
    const auto* source = host_read<T>(input);
    auto* destination = host_write<T>(output);
    if (!source || !destination) return status_access;
    host_row_reduce<T>(source, destination, meta);
    return status_ok;
}

int run_row_reduce(
    const void* input, void* output, const RowReduceMetadata& meta) {
    const auto dtype = qcore_tensor_dtype(input);
    const void* placement[] = {input, output};
    if (same_placement(placement, 2, dtype) != status_ok)
        return status_invalid;
    if (meta.rows == 0 || meta.input_width == 0 || meta.output_width == 0 ||
        qcore_tensor_element_count(input) != meta.rows * meta.input_width ||
        qcore_tensor_element_count(output) != meta.rows * meta.output_width)
        return status_shape;
    const auto backend = linear_backend(input);
    if (!linear_dtype_supported(backend, dtype))
        return backend == LinearBackend::Unsupported
            ? status_unsupported_backend : status_unsupported_dtype;
    if (backend == LinearBackend::Host)
        return dtype == QCORE_DTYPE_FLOAT32
            ? host_row_reduce_typed<float>(input, output, meta)
            : host_row_reduce_typed<double>(input, output, meta);
#ifdef __APPLE__
    return math_native_metal_row_reduce(
        input, output, meta.rows, meta.input_width, meta.output_width,
        meta.divisor, meta.pairwise);
#else
    return status_unsupported_backend;
#endif
}

RowReduceMetadata row_reduce_adjoint(const RowReduceMetadata& meta) {
    RowReduceMetadata adjoint = meta;
    adjoint.input_width = meta.output_width;
    adjoint.output_width = meta.input_width;
    return adjoint;
}

bool row_reduce_metadata(
    const void* metadata, std::uint64_t metadata_size,
    RowReduceMetadata& result) {
    if (!metadata || metadata_size != sizeof(RowReduceMetadata)) return false;
    std::memcpy(&result, metadata, sizeof(RowReduceMetadata));
    return result.rows != 0 && result.input_width != 0 &&
           result.output_width != 0 && result.divisor != 0.0;
}

int row_reduce_backward(
    const void* const*,
    std::uint64_t,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata,
    std::uint64_t metadata_size) {
    RowReduceMetadata meta{};
    if (!gradient_output || !gradient_inputs || gradient_input_count != 1 ||
        !gradient_inputs[0] ||
        !row_reduce_metadata(metadata, metadata_size, meta))
        return status_invalid;
    const int status = run_row_reduce(
        gradient_output, gradient_inputs[0], row_reduce_adjoint(meta));
    if (status == status_ok)
        math_native_test_trace("reduce-backward", gradient_output);
    return status;
}

int row_reduce_backward_tracked(
    const void* const* differentiable_inputs,
    std::uint64_t differentiable_input_count,
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata,
    std::uint64_t metadata_size) {
    if (!differentiable_inputs || differentiable_input_count != 1)
        return status_invalid;
    const int status = row_reduce_backward(
        saved_tensors, saved_tensor_count, gradient_output, gradient_inputs,
        gradient_input_count, metadata, metadata_size);
    if (status != status_ok) return status;
    RowReduceMetadata meta{};
    (void)row_reduce_metadata(metadata, metadata_size, meta);
    const auto adjoint = row_reduce_adjoint(meta);
    const void* parents[] = {gradient_output};
    if (qcore_tensor_attach_custom_autograd_with_saved_ex(
            gradient_inputs[0], parents, 1, nullptr, 0,
            row_reduce_backward, row_reduce_backward_tracked,
            &adjoint, sizeof(adjoint)) != 0)
        return status_backend;
    math_native_test_trace("reduce-backward-tracked", gradient_output);
    return status_ok;
}

// ---------------------------------------------------------------------------
// First-occurrence extrema. The forward pass records each row's winning
// offset in a Math-owned int32 tensor. A "select" node copies the winner to
// every output position of its row; its adjoint "scatter" sends the row's
// summed upstream gradient only to the winner. The two kinds are adjoint to
// each other, so higher-order gradients remain available without host
// readback or per-element selection.

constexpr std::uint8_t extrema_select = 1;
constexpr std::uint8_t extrema_scatter = 2;

struct ExtremaMetadata {
    std::uint64_t rows;
    std::uint64_t input_width;
    std::uint64_t output_width;
    std::uint8_t kind;
    std::uint8_t reserved[7];
};

static_assert(sizeof(ExtremaMetadata) == 32, "stable Math metadata layout");

bool extrema_metadata(
    const void* metadata, std::uint64_t metadata_size,
    ExtremaMetadata& result) {
    if (!metadata || metadata_size != sizeof(ExtremaMetadata)) return false;
    std::memcpy(&result, metadata, sizeof(ExtremaMetadata));
    return result.rows != 0 && result.input_width != 0 &&
           result.output_width != 0 &&
           (result.kind == extrema_select || result.kind == extrema_scatter);
}

ExtremaMetadata extrema_adjoint(const ExtremaMetadata& meta) {
    ExtremaMetadata adjoint = meta;
    adjoint.input_width = meta.output_width;
    adjoint.output_width = meta.input_width;
    adjoint.kind = meta.kind == extrema_select ? extrema_scatter : extrema_select;
    return adjoint;
}

bool winners_valid(const void* winners, const void* reference,
                   std::uint64_t rows) {
    return winners &&
           qcore_tensor_dtype(winners) == QCORE_DTYPE_INT32 &&
           qcore_tensor_backend(winners) == qcore_tensor_backend(reference) &&
           qcore_tensor_device(winners) == qcore_tensor_device(reference) &&
           qcore_tensor_is_contiguous(winners) &&
           qcore_tensor_element_count(winners) == rows;
}

template <typename T>
int host_extrema_apply(
    const void* input_raw, const void* winners_raw, void* output_raw,
    const ExtremaMetadata& meta) {
    const auto* input = host_read<T>(input_raw);
    const auto* winners = host_read<std::int32_t>(winners_raw);
    auto* output = host_write<T>(output_raw);
    if (!input || !winners || !output) return status_access;
    for (std::uint64_t row = 0; row < meta.rows; ++row) {
        const auto winner = winners[row];
        if (winner < 0) return status_shape;
        const T* source = input + row * meta.input_width;
        T* destination = output + row * meta.output_width;
        if (meta.kind == extrema_select) {
            if (static_cast<std::uint64_t>(winner) >= meta.input_width)
                return status_shape;
            const T value = source[winner];
            for (std::uint64_t column = 0; column < meta.output_width; ++column)
                destination[column] = value;
        } else {
            if (static_cast<std::uint64_t>(winner) >= meta.output_width)
                return status_shape;
            T total = T(0);
            for (std::uint64_t k = 0; k < meta.input_width; ++k)
                total = total + source[k];
            for (std::uint64_t column = 0; column < meta.output_width; ++column)
                destination[column] = T(0);
            destination[winner] = total;
        }
    }
    return status_ok;
}

int run_extrema_apply(
    const void* input, const void* winners, void* output,
    const ExtremaMetadata& meta) {
    const auto dtype = qcore_tensor_dtype(input);
    const void* placement[] = {input, output};
    if (same_placement(placement, 2, dtype) != status_ok ||
        !winners_valid(winners, input, meta.rows))
        return status_invalid;
    if (qcore_tensor_element_count(input) != meta.rows * meta.input_width ||
        qcore_tensor_element_count(output) != meta.rows * meta.output_width)
        return status_shape;
    const auto backend = linear_backend(input);
    if (!linear_dtype_supported(backend, dtype))
        return backend == LinearBackend::Unsupported
            ? status_unsupported_backend : status_unsupported_dtype;
    if (backend == LinearBackend::Host)
        return dtype == QCORE_DTYPE_FLOAT32
            ? host_extrema_apply<float>(input, winners, output, meta)
            : host_extrema_apply<double>(input, winners, output, meta);
#ifdef __APPLE__
    return meta.kind == extrema_select
        ? math_native_metal_row_select(
              input, winners, output, meta.rows, meta.input_width,
              meta.output_width)
        : math_native_metal_row_scatter(
              input, winners, output, meta.rows, meta.input_width,
              meta.output_width);
#else
    return status_unsupported_backend;
#endif
}

int extrema_backward(
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata,
    std::uint64_t metadata_size) {
    ExtremaMetadata meta{};
    if (!saved_tensors || saved_tensor_count != 1 || !gradient_output ||
        !gradient_inputs || gradient_input_count != 1 ||
        !gradient_inputs[0] ||
        !extrema_metadata(metadata, metadata_size, meta))
        return status_invalid;
    const int status = run_extrema_apply(
        gradient_output, saved_tensors[0], gradient_inputs[0],
        extrema_adjoint(meta));
    if (status == status_ok)
        math_native_test_trace("extrema-backward", gradient_output);
    return status;
}

int extrema_backward_tracked(
    const void* const* differentiable_inputs,
    std::uint64_t differentiable_input_count,
    const void* const* saved_tensors,
    std::uint64_t saved_tensor_count,
    const void* gradient_output,
    void* const* gradient_inputs,
    std::uint64_t gradient_input_count,
    const void* metadata,
    std::uint64_t metadata_size) {
    if (!differentiable_inputs || differentiable_input_count != 1)
        return status_invalid;
    const int status = extrema_backward(
        saved_tensors, saved_tensor_count, gradient_output, gradient_inputs,
        gradient_input_count, metadata, metadata_size);
    if (status != status_ok) return status;
    ExtremaMetadata meta{};
    (void)extrema_metadata(metadata, metadata_size, meta);
    const auto adjoint = extrema_adjoint(meta);
    const void* parents[] = {gradient_output};
    const void* saved[] = {saved_tensors[0]};
    if (qcore_tensor_attach_custom_autograd_with_saved_ex(
            gradient_inputs[0], parents, 1, saved, 1,
            extrema_backward, extrema_backward_tracked,
            &adjoint, sizeof(adjoint)) != 0)
        return status_backend;
    math_native_test_trace("extrema-backward-tracked", gradient_output);
    return status_ok;
}

template <typename T>
int host_extrema_forward(
    const void* input_raw, void* output_raw, void* winners_raw,
    const ExtremaMetadata& meta, bool maximum) {
    const auto* input = host_read<T>(input_raw);
    auto* output = host_write<T>(output_raw);
    auto* winners = host_write<std::int32_t>(winners_raw);
    if (!input || !output || !winners) return status_access;
    for (std::uint64_t row = 0; row < meta.rows; ++row) {
        const T* source = input + row * meta.input_width;
        T best = source[0];
        std::uint64_t best_index = 0;
        for (std::uint64_t k = 1; k < meta.input_width; ++k) {
            const T candidate = source[k];
            const bool better =
                maximum ? candidate > best : candidate < best;
            if (better) {
                best = candidate;
                best_index = k;
            }
        }
        winners[row] = static_cast<std::int32_t>(best_index);
        T* destination = output + row * meta.output_width;
        for (std::uint64_t column = 0; column < meta.output_width; ++column)
            destination[column] = best;
    }
    return status_ok;
}

} // namespace

// Placement query used by the Quidra layer before it allocates native
// outputs: 1 when Math's native matrix/reduction kernels serve this tensor's
// backend and dtype, 0 when the portable composition must be used (CUDA/HIP
// keep their existing paths, float64 has no Metal kernel).
extern "C" int math_native_linear_placement(const void* value) {
    if (!value || qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION)
        return 0;
    return linear_dtype_supported(
        linear_backend(value), qcore_tensor_dtype(value)) ? 1 : 0;
}

// Backend-neutral native matrix product. left_data/right_data are contiguous
// value borrows (possibly the transposed storage of a transposed view, per
// the layout bits); left/right are the original differentiable tensors whose
// graphs become the node parents. The output is a fresh contiguous tensor with
// the public matmul shape.
extern "C" int math_native_matmul(
    const void* left_data,
    const void* right_data,
    const void* left,
    const void* right,
    void* output,
    std::int32_t layout) try {
    if (!left_data || !right_data || !left || !right || !output ||
        qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION ||
        layout < 0 || layout > 15)
        return status_invalid;

    const auto dtype = qcore_tensor_dtype(output);
    const auto backend = linear_backend(output);
    if (backend == LinearBackend::Unsupported)
        return status_unsupported_backend;
    if (!linear_dtype_supported(backend, dtype))
        return status_unsupported_dtype;
    const void* data[] = {left_data, right_data, output};
    if (same_placement(data, 3, dtype) != status_ok ||
        qcore_tensor_dtype(left) != dtype ||
        qcore_tensor_dtype(right) != dtype ||
        qcore_tensor_backend(left) != qcore_tensor_backend(output) ||
        qcore_tensor_backend(right) != qcore_tensor_backend(output) ||
        qcore_tensor_device(left) != qcore_tensor_device(output) ||
        qcore_tensor_device(right) != qcore_tensor_device(output))
        return status_invalid;

    std::size_t rows = 0;
    std::size_t inner = 0;
    std::size_t columns = 0;
    if (!validate_matmul_shape(left, right, output, rows, inner, columns))
        return status_shape;

    const bool left_transposed = (layout & layout_left_transposed) != 0;
    const bool right_transposed = (layout & layout_right_transposed) != 0;
    if (left_transposed ? !transposed_extents(left_data, left)
                        : !same_extents(left_data, left))
        return status_shape;
    if (right_transposed ? !transposed_extents(right_data, right)
                         : !same_extents(right_data, right))
        return status_shape;

    const auto left_view = stored_matrix(rows, inner, left_transposed);
    const auto right_view = stored_matrix(inner, columns, right_transposed);
    const auto task = gemm_task(
        left_data, left_view, right_data, right_view, output,
        rows, inner, columns);
    const int status = run_gemm_tasks(&task, 1);
    if (status != status_ok) return status;

    MatmulMetadata meta{};
    meta.rows = rows;
    meta.inner = inner;
    meta.columns = columns;
    meta.right_is_vector = qcore_tensor_rank(right) == 1 ? 1 : 0;
    meta.left_saved_transposed = left_transposed ? 1 : 0;
    meta.right_saved_transposed = right_transposed ? 1 : 0;
    meta.left_required = (layout & layout_left_tracked) != 0 ? 1 : 0;
    meta.right_required = (layout & layout_right_tracked) != 0 ? 1 : 0;
    if (attach_matmul_node(
            output, left, right, left_data, right_data, meta) != 0)
        return status_backend;
    math_native_test_trace("matmul", output);
    return status_ok;
} catch (...) {
    // Allocation failures (operand packing) report a status, so the Quidra
    // layer can fall back instead of unwinding through the C boundary.
    return status_backend;
}

// Whole-tensor and last-axis sums. operation: 1 = sum, 2 = mean, 3 = sum_last.
// `input` is a contiguous value borrow of `parent`, the differentiable tensor
// the node attaches to.
extern "C" int math_native_reduce(
    const void* input,
    const void* parent,
    void* output,
    std::int32_t operation) try {
    if (!input || !parent || !output ||
        qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION ||
        operation < 1 || operation > 3)
        return status_invalid;
    const auto backend = linear_backend(input);
    if (backend == LinearBackend::Unsupported)
        return status_unsupported_backend;
    if (!linear_dtype_supported(backend, qcore_tensor_dtype(input)))
        return status_unsupported_dtype;
    if (!value_borrow_of(input, parent)) return status_invalid;

    const auto count = qcore_tensor_element_count(input);
    if (count == 0) return status_shape;
    RowReduceMetadata meta{};
    meta.divisor = 1.0;
    if (operation == 1 || operation == 2) {
        if (qcore_tensor_rank(output) != 0) return status_shape;
        meta.rows = 1;
        meta.input_width = count;
        meta.output_width = 1;
        meta.pairwise = 1;
        if (operation == 2)
            meta.divisor = qcore_tensor_dtype(input) == QCORE_DTYPE_FLOAT32
                ? static_cast<double>(element_count_divisor<float>(count))
                : element_count_divisor<double>(count);
    } else {
        const auto rank = qcore_tensor_rank(input);
        if (rank == 0 || !same_tensor_shape(input, output)) return status_shape;
        const auto width = qcore_tensor_extent(input, rank - 1);
        if (width <= 0) return status_shape;
        meta.input_width = static_cast<std::uint64_t>(width);
        meta.output_width = meta.input_width;
        meta.rows = count / meta.input_width;
    }
    const int status = run_row_reduce(input, output, meta);
    if (status != status_ok) return status;
    const void* parents[] = {parent};
    if (qcore_tensor_attach_custom_autograd_with_saved_ex(
            output, parents, 1, nullptr, 0,
            row_reduce_backward, row_reduce_backward_tracked,
            &meta, sizeof(meta)) != 0)
        return status_backend;
    static const char* const names[] = {"sum", "mean", "sum_last"};
    math_native_test_trace(names[operation - 1], input);
    return status_ok;
} catch (...) {
    return status_backend;
}

// First-occurrence extrema with strict comparison.
// operation: 1 = max_last, 2 = min_last, 3 = max_all, 4 = min_all.
// `input` is a contiguous value borrow of `parent`, as for math_native_reduce.
extern "C" int math_native_select_extrema(
    const void* input,
    const void* parent,
    void* output,
    void* winners,
    std::int32_t operation) try {
    if (!input || !parent || !output || !winners ||
        qcore_native_abi_version() != QUIDRA_NATIVE_ABI_VERSION ||
        operation < 1 || operation > 4)
        return status_invalid;
    const auto dtype = qcore_tensor_dtype(input);
    const auto backend = linear_backend(input);
    if (backend == LinearBackend::Unsupported)
        return status_unsupported_backend;
    if (!linear_dtype_supported(backend, dtype))
        return status_unsupported_dtype;
    const void* placement[] = {input, output};
    if (same_placement(placement, 2, dtype) != status_ok ||
        !value_borrow_of(input, parent))
        return status_invalid;

    const auto count = qcore_tensor_element_count(input);
    if (count == 0) return status_shape;
    ExtremaMetadata meta{};
    meta.kind = extrema_select;
    const bool maximum = operation == 1 || operation == 3;
    if (operation <= 2) {
        const auto rank = qcore_tensor_rank(input);
        if (rank == 0 || !same_tensor_shape(input, output)) return status_shape;
        const auto width = qcore_tensor_extent(input, rank - 1);
        if (width <= 0) return status_shape;
        meta.input_width = static_cast<std::uint64_t>(width);
        meta.output_width = meta.input_width;
        meta.rows = count / meta.input_width;
    } else {
        if (qcore_tensor_rank(output) != 0) return status_shape;
        meta.rows = 1;
        meta.input_width = count;
        meta.output_width = 1;
    }
    if (meta.input_width >
            static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) ||
        !winners_valid(winners, input, meta.rows))
        return status_shape;

    int status = status_unsupported_backend;
    if (backend == LinearBackend::Host) {
        status = dtype == QCORE_DTYPE_FLOAT32
            ? host_extrema_forward<float>(input, output, winners, meta, maximum)
            : host_extrema_forward<double>(input, output, winners, meta, maximum);
    }
#ifdef __APPLE__
    else if (backend == LinearBackend::Metal) {
        status = math_native_metal_row_extrema(
            input, output, winners, meta.rows, meta.input_width,
            meta.output_width, maximum ? 1 : 0);
    }
#endif
    if (status != status_ok) return status;
    const void* parents[] = {parent};
    const void* saved[] = {winners};
    if (qcore_tensor_attach_custom_autograd_with_saved_ex(
            output, parents, 1, saved, 1,
            extrema_backward, extrema_backward_tracked,
            &meta, sizeof(meta)) != 0)
        return status_backend;
    static const char* const names[] = {
        "max_last", "min_last", "max_all", "min_all"};
    math_native_test_trace(names[operation - 1], input);
    return status_ok;
} catch (...) {
    return status_backend;
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

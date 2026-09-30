#include <quidra/native_extension.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

class DynamicLibrary {
public:
    DynamicLibrary() = default;
    DynamicLibrary(const DynamicLibrary&) = delete;
    DynamicLibrary& operator=(const DynamicLibrary&) = delete;
    ~DynamicLibrary() {
#ifdef _WIN32
        if (handle_) FreeLibrary(static_cast<HMODULE>(handle_));
#else
        if (handle_) dlclose(handle_);
#endif
    }
    bool open(const char* name) {
#ifdef _WIN32
        handle_ = static_cast<void*>(LoadLibraryA(name));
#else
        handle_ = dlopen(name, RTLD_NOW | RTLD_LOCAL);
#endif
        return handle_ != nullptr;
    }
    void* symbol(const char* name) const {
        if (!handle_) return nullptr;
#ifdef _WIN32
        return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
        return dlsym(handle_, name);
#endif
    }
private:
    void* handle_{};
};

template <typename T>
T load_symbol(const DynamicLibrary& library, const char* name) {
    return reinterpret_cast<T>(library.symbol(name));
}

struct CudaApi {
    using DevicePointer = std::uint64_t;
    using Module = void*;
    using Function = void*;
    using Stream = void*;
    using Result = int;

    DynamicLibrary library;
    Result (*init)(unsigned){};
    Result (*module_load_data)(Module*, const void*){};
    Result (*module_get_function)(Function*, Module, const char*){};
    Result (*launch_kernel)(Function, unsigned, unsigned, unsigned,
                            unsigned, unsigned, unsigned, unsigned,
                            Stream, void**, void**){};
    Result (*copy_d2h)(void*, DevicePointer, std::size_t){};
    Result (*memset_d32)(DevicePointer, unsigned, std::size_t){};
    bool ready{};

    CudaApi() {
#ifdef _WIN32
        if (!library.open("nvcuda.dll")) return;
#else
        if (!library.open("libcuda.so.1") && !library.open("libcuda.so")) return;
#endif
        init=load_symbol<decltype(init)>(library,"cuInit");
        module_load_data=load_symbol<decltype(module_load_data)>(library,"cuModuleLoadData");
        module_get_function=load_symbol<decltype(module_get_function)>(library,"cuModuleGetFunction");
        launch_kernel=load_symbol<decltype(launch_kernel)>(library,"cuLaunchKernel");
        copy_d2h=load_symbol<decltype(copy_d2h)>(library,"cuMemcpyDtoH_v2");
        if(!copy_d2h) copy_d2h=load_symbol<decltype(copy_d2h)>(library,"cuMemcpyDtoH");
        memset_d32=load_symbol<decltype(memset_d32)>(library,"cuMemsetD32_v2");
        if(!memset_d32) memset_d32=load_symbol<decltype(memset_d32)>(library,"cuMemsetD32");
        ready=init&&module_load_data&&module_get_function&&launch_kernel&&
              copy_d2h&&memset_d32&&init(0)==0;
    }
};

CudaApi& cuda(){ static CudaApi api; return api; }

std::uint64_t pointer_const(const void* tensor){
    const auto base=qcore_tensor_device_handle_const(tensor);
    const auto offset=qcore_tensor_device_offset_bytes(tensor);
    if(!base||offset>std::numeric_limits<std::uint64_t>::max()-base)return 0;
    return base+offset;
}
std::uint64_t pointer_mut(void* tensor){
    const auto base=qcore_tensor_device_handle(tensor);
    const auto offset=qcore_tensor_device_offset_bytes(tensor);
    if(!base||offset>std::numeric_limits<std::uint64_t>::max()-base)return 0;
    return base+offset;
}
bool cuda_f32(const void* tensor){
    return tensor&&qcore_tensor_backend(tensor)==QCORE_BACKEND_CUDA&&
           qcore_tensor_dtype(tensor)==QCORE_DTYPE_FLOAT32&&
           qcore_tensor_is_contiguous(tensor);
}
bool same_shape(const void* a,const void* b){
    if(!a||!b||qcore_tensor_rank(a)!=qcore_tensor_rank(b)||
       qcore_tensor_element_count(a)!=qcore_tensor_element_count(b))return false;
    for(std::uint64_t axis=0;axis<qcore_tensor_rank(a);++axis)
        if(qcore_tensor_extent(a,axis)!=qcore_tensor_extent(b,axis))return false;
    return true;
}
bool same_cuda(const void* a,const void* b){
    return cuda_f32(a)&&cuda_f32(b)&&
           qcore_tensor_device(a)==qcore_tensor_device(b)&&same_shape(a,b);
}

std::string prologue(std::string params,std::string regs){
    return ".version 6.0\n.target sm_30\n.address_size 64\n\n.visible .entry q_kernel(\n"+
        params+")\n{\n"+regs+
        "  mov.u32 %r1, %ctaid.x;\n  mov.u32 %r2, %ntid.x;\n"
        "  mov.u32 %r3, %tid.x;\n  mad.lo.s32 %r4, %r1, %r2, %r3;\n"
        "  cvt.u64.u32 %rd7, %r4;\n";
}

std::string forward_ptx(std::int32_t operation){
    std::string body;
    if(operation==1) body=
        "  abs.f32 %f2,%f1;\n  st.global.f32 [%rd10],%f2;\n";
    else if(operation==2) body=
        "  setp.lt.f32 %p2,%f1,0f00000000;\n  @%p2 bra INVALID;\n"
        "  sqrt.rn.f32 %f2,%f1;\n  st.global.f32 [%rd10],%f2;\n  bra DONE;\n"
        "INVALID:\n  mov.u32 %r5,1;\n  atom.global.exch.b32 %r6,[%rd4],%r5;\n"
        "  mov.f32 %f2,0f00000000;\n  st.global.f32 [%rd10],%f2;\n";
    else if(operation==3) body=
        "  setp.gt.f32 %p2,%f1,0f00000000;\n  @%p2 bra LOG_VALID;\n"
        "  mov.u32 %r5,1;\n  atom.global.exch.b32 %r6,[%rd4],%r5;\n"
        "  mov.f32 %f2,0f00000000;\n  st.global.f32 [%rd10],%f2;\n  bra DONE;\n"
        "LOG_VALID:\n  lg2.approx.f32 %f2,%f1;\n  mov.f32 %f3,0f3f317218;\n"
        "  mul.rn.f32 %f4,%f2,%f3;\n  st.global.f32 [%rd10],%f4;\n";
    else if(operation==4) body=
        "  mov.f32 %f2,0f3fb8aa3b;\n  mul.rn.f32 %f3,%f1,%f2;\n"
        "  ex2.approx.f32 %f4,%f3;\n  st.global.f32 [%rd10],%f4;\n";
    else return {};
    return prologue(
        "  .param .u64 p_out,\n  .param .u64 p_in,\n"
        "  .param .u64 p_count,\n  .param .u64 p_status\n",
        "  .reg .pred %p<5>;\n  .reg .b32 %r<16>;\n"
        "  .reg .b64 %rd<20>;\n  .reg .f32 %f<12>;\n")+
        "  ld.param.u64 %rd1,[p_out];\n  ld.param.u64 %rd2,[p_in];\n"
        "  ld.param.u64 %rd5,[p_count];\n  ld.param.u64 %rd4,[p_status];\n"
        "  setp.ge.u64 %p1,%rd7,%rd5;\n  @%p1 bra DONE;\n"
        "  mul.lo.u64 %rd8,%rd7,4;\n  add.u64 %rd9,%rd2,%rd8;\n"
        "  add.u64 %rd10,%rd1,%rd8;\n  ld.global.f32 %f1,[%rd9];\n"+
        body+"DONE:\n  ret;\n}\n";
}

std::string backward_ptx(std::int32_t operation){
    std::string body;
    if(operation==1) body=
        "  setp.gt.f32 %p2,%f1,0f00000000;\n  @%p2 bra POS;\n"
        "  setp.lt.f32 %p3,%f1,0f00000000;\n  @%p3 bra NEG;\n"
        "  mov.f32 %f4,0f00000000;\n  bra STORE;\n"
        "POS:\n  mov.f32 %f4,%f3;\n  bra STORE;\n"
        "NEG:\n  neg.f32 %f4,%f3;\nSTORE:\n";
    else if(operation==2) body=
        "  mov.f32 %f5,0f40000000;\n  mul.rn.f32 %f6,%f5,%f2;\n"
        "  div.rn.f32 %f4,%f3,%f6;\n";
    else if(operation==3) body="  div.rn.f32 %f4,%f3,%f1;\n";
    else if(operation==4) body="  mul.rn.f32 %f4,%f3,%f2;\n";
    else return {};
    return prologue(
        "  .param .u64 p_out,\n  .param .u64 p_input,\n"
        "  .param .u64 p_forward,\n  .param .u64 p_grad,\n"
        "  .param .u64 p_count\n",
        "  .reg .pred %p<5>;\n  .reg .b32 %r<12>;\n"
        "  .reg .b64 %rd<24>;\n  .reg .f32 %f<12>;\n")+
        "  ld.param.u64 %rd1,[p_out];\n  ld.param.u64 %rd2,[p_input];\n"
        "  ld.param.u64 %rd3,[p_forward];\n  ld.param.u64 %rd4,[p_grad];\n"
        "  ld.param.u64 %rd5,[p_count];\n  setp.ge.u64 %p1,%rd7,%rd5;\n"
        "  @%p1 bra DONE;\n  mul.lo.u64 %rd8,%rd7,4;\n"
        "  add.u64 %rd9,%rd2,%rd8;\n  add.u64 %rd10,%rd3,%rd8;\n"
        "  add.u64 %rd11,%rd4,%rd8;\n  add.u64 %rd12,%rd1,%rd8;\n"
        "  ld.global.f32 %f1,[%rd9];\n  ld.global.f32 %f2,[%rd10];\n"
        "  ld.global.f32 %f3,[%rd11];\n"+body+
        "  st.global.f32 [%rd12],%f4;\nDONE:\n  ret;\n}\n";
}

std::string second_ptx(std::int32_t operation){
    std::string body;
    if(operation==1) body=
        "  setp.gt.f32 %p2,%f1,0f00000000;\n  @%p2 bra SP;\n"
        "  setp.lt.f32 %p3,%f1,0f00000000;\n  @%p3 bra SN;\n"
        "  mov.f32 %f4,0f00000000;\n  bra SD;\n"
        "SP:\n  mov.f32 %f4,0f3f800000;\n  bra SD;\n"
        "SN:\n  mov.f32 %f4,0fbf800000;\nSD:\n  mov.f32 %f5,0f00000000;\n";
    else if(operation==2) body=
        "  sqrt.rn.f32 %f9,%f1;\n  mov.f32 %f10,0f40000000;\n"
        "  mul.rn.f32 %f11,%f10,%f9;\n  mov.f32 %f12,0f3f800000;\n"
        "  div.rn.f32 %f4,%f12,%f11;\n  mul.rn.f32 %f13,%f1,%f9;\n"
        "  mov.f32 %f14,0fbe800000;\n  div.rn.f32 %f5,%f14,%f13;\n";
    else if(operation==3) body=
        "  mov.f32 %f9,0f3f800000;\n  div.rn.f32 %f4,%f9,%f1;\n"
        "  mul.rn.f32 %f10,%f1,%f1;\n  mov.f32 %f11,0fbf800000;\n"
        "  div.rn.f32 %f5,%f11,%f10;\n";
    else if(operation==4) body=
        "  mov.f32 %f9,0f3fb8aa3b;\n  mul.rn.f32 %f10,%f1,%f9;\n"
        "  ex2.approx.f32 %f4,%f10;\n  mov.f32 %f5,%f4;\n";
    else return {};
    return prologue(
        "  .param .u64 p_gi,\n  .param .u64 p_gf,\n"
        "  .param .u64 p_input,\n  .param .u64 p_first,\n"
        "  .param .u64 p_upstream,\n  .param .u64 p_count\n",
        "  .reg .pred %p<5>;\n  .reg .b32 %r<12>;\n"
        "  .reg .b64 %rd<28>;\n  .reg .f32 %f<20>;\n")+
        "  ld.param.u64 %rd1,[p_gi];\n  ld.param.u64 %rd2,[p_gf];\n"
        "  ld.param.u64 %rd3,[p_input];\n  ld.param.u64 %rd4,[p_first];\n"
        "  ld.param.u64 %rd5,[p_upstream];\n  ld.param.u64 %rd6,[p_count];\n"
        "  setp.ge.u64 %p1,%rd7,%rd6;\n  @%p1 bra DONE;\n"
        "  mul.lo.u64 %rd8,%rd7,4;\n  add.u64 %rd9,%rd3,%rd8;\n"
        "  add.u64 %rd10,%rd4,%rd8;\n  add.u64 %rd11,%rd5,%rd8;\n"
        "  add.u64 %rd12,%rd1,%rd8;\n  add.u64 %rd13,%rd2,%rd8;\n"
        "  ld.global.f32 %f1,[%rd9];\n  ld.global.f32 %f2,[%rd10];\n"
        "  ld.global.f32 %f3,[%rd11];\n"+body+
        "  mul.rn.f32 %f6,%f3,%f2;\n  mul.rn.f32 %f7,%f6,%f5;\n"
        "  mul.rn.f32 %f8,%f3,%f4;\n  st.global.f32 [%rd12],%f7;\n"
        "  st.global.f32 [%rd13],%f8;\nDONE:\n  ret;\n}\n";
}

struct CachedKernel { CudaApi::Module module{}; CudaApi::Function function{}; };
std::mutex cache_mutex;
std::map<std::string,CachedKernel> cache;

CudaApi::Function kernel(long long device,const std::string& key,const std::string& ptx){
    auto& api=cuda();
    if(!api.ready||ptx.empty())return nullptr;
    const auto full=std::to_string(device)+":"+key;
    std::lock_guard<std::mutex> lock(cache_mutex);
    if(const auto it=cache.find(full);it!=cache.end())return it->second.function;
    CudaApi::Module module=nullptr; CudaApi::Function function=nullptr;
    if(api.module_load_data(&module,ptx.c_str())!=0||!module||
       api.module_get_function(&function,module,"q_kernel")!=0||!function)return nullptr;
    cache.emplace(full,CachedKernel{module,function});
    return function;
}

int launch(long long device,const std::string& key,const std::string& ptx,
           std::uint64_t count,void** arguments){
    if(count==0)return 0;
    auto& api=cuda();
    if(!api.ready||!qcore_device_activate(device))return 5;
    constexpr unsigned threads=256;
    const auto blocks64=(count+threads-1)/threads;
    if(blocks64>std::numeric_limits<unsigned>::max())return 2;
    auto function=kernel(device,key,ptx);
    if(!function)return 5;
    auto stream=reinterpret_cast<CudaApi::Stream>(
        static_cast<std::uintptr_t>(qcore_device_queue_handle(device)));
    return api.launch_kernel(function,static_cast<unsigned>(blocks64),1,1,
                             threads,1,1,0,stream,arguments,nullptr)==0?0:5;
}

} // namespace

extern "C" int math_native_cuda_tensor_unary_forward(
    const void* input,void* output,std::int32_t operation){
    if(!input||!output||!cuda_f32(input)||!cuda_f32(output)||
       qcore_tensor_device(input)!=qcore_tensor_device(output)||!same_shape(input,output))
        return input&&qcore_tensor_dtype(input)==QCORE_DTYPE_FLOAT64?7:1;
    if(operation<1||operation>4)return 2;
    std::uint64_t out=pointer_mut(output),in=pointer_const(input);
    std::uint64_t count=qcore_tensor_element_count(input),status=0;
    if(!out||!in)return 3;
    const auto device=qcore_tensor_device(input);
    void* scratch=nullptr;
    auto& api=cuda();
    if(operation==2||operation==3){
        if(!api.ready||!qcore_device_activate(device))return 5;
        scratch=qcore_device_buffer_allocate(device,sizeof(std::uint32_t));
        if(!scratch)return 5;
        status=qcore_device_buffer_handle(scratch);
        if(!status||api.memset_d32(status,0u,1u)!=0){
            qcore_device_buffer_release(scratch);return 5;
        }
    }
    void* args[]={&out,&in,&count,&status};
    const int launched=launch(device,"math-unary-forward-"+std::to_string(operation),
                              forward_ptx(operation),count,args);
    if(launched!=0){if(scratch)qcore_device_buffer_release(scratch);return launched;}
    if(scratch){
        std::uint32_t host=0;
        if(api.copy_d2h(&host,status,sizeof(host))!=0){
            qcore_device_buffer_release(scratch);return 5;
        }
        qcore_device_buffer_release(scratch);
        if(host!=0)return 4;
    }
    return 0;
}

extern "C" int math_native_cuda_tensor_unary_backward(
    const void* input,const void* output,const void* gradient_output,
    void* gradient_input,std::int32_t operation){
    if(!input||!output||!gradient_output||!gradient_input||
       !same_cuda(input,output)||!same_cuda(input,gradient_output)||
       !same_cuda(input,gradient_input))
        return input&&qcore_tensor_dtype(input)==QCORE_DTYPE_FLOAT64?7:1;
    if(operation<1||operation>4)return 2;
    std::uint64_t out=pointer_mut(gradient_input),in=pointer_const(input);
    std::uint64_t forward=pointer_const(output),grad=pointer_const(gradient_output);
    std::uint64_t count=qcore_tensor_element_count(input);
    if(!out||!in||!forward||!grad)return 3;
    void* args[]={&out,&in,&forward,&grad,&count};
    return launch(qcore_tensor_device(input),
                  "math-unary-backward-"+std::to_string(operation),
                  backward_ptx(operation),count,args);
}

extern "C" int math_native_cuda_tensor_unary_second_backward(
    const void* input,const void* first_gradient,const void* gradient_output,
    void* gradient_input,void* gradient_first,std::int32_t operation){
    if(!input||!first_gradient||!gradient_output||!gradient_input||!gradient_first||
       !same_cuda(input,first_gradient)||!same_cuda(input,gradient_output)||
       !same_cuda(input,gradient_input)||!same_cuda(input,gradient_first))
        return input&&qcore_tensor_dtype(input)==QCORE_DTYPE_FLOAT64?7:1;
    if(operation<1||operation>4)return 2;
    std::uint64_t gi=pointer_mut(gradient_input),gf=pointer_mut(gradient_first);
    std::uint64_t in=pointer_const(input),first=pointer_const(first_gradient);
    std::uint64_t upstream=pointer_const(gradient_output);
    std::uint64_t count=qcore_tensor_element_count(input);
    if(!gi||!gf||!in||!first||!upstream)return 3;
    void* args[]={&gi,&gf,&in,&first,&upstream,&count};
    return launch(qcore_tensor_device(input),
                  "math-unary-second-"+std::to_string(operation),
                  second_ptx(operation),count,args);
}

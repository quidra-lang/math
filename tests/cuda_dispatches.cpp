#include <cstdio>
#include <cstdlib>

#include "../native/math_cuda.cpp"

namespace {
std::uint64_t observed_dispatches = 0;
unsigned launch_calls = 0;
int launch_status = 0;
int module_status = 0;
bool activation_succeeds = true;

int load_module(void** module, const void*) {
    if (module_status != 0) return module_status;
    *module = reinterpret_cast<void*>(1);
    return 0;
}

int get_function(void** function, void*, const char*) {
    *function = reinterpret_cast<void*>(2);
    return 0;
}

int enqueue(void*, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned,
            unsigned, void*, void**, void**) {
    ++launch_calls;
    return launch_status;
}

void require(bool condition, const char* message) {
    if (condition) return;
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
}
}

extern "C" int qcore_device_activate(long long) {
    return activation_succeeds ? 1 : 0;
}

extern "C" std::uint64_t qcore_device_queue_handle(long long) {
    return 0;
}

extern "C" void qcore_counter_note_dispatches(std::uint64_t count) {
    observed_dispatches += count;
}

int main() {
    auto& api = cuda();
    api.ready = true;
    api.module_load_data = load_module;
    api.module_get_function = get_function;
    api.launch_kernel = enqueue;

    // Use the production launch/cache path with a driver that cannot submit
    // hardware work. Module lookup itself must not report a dispatch.
    require(kernel(0, "lookup", "ptx") != nullptr, "kernel lookup failed");
    require(observed_dispatches == 0 && launch_calls == 0,
            "kernel lookup was counted as a dispatch");
    require(launch(0, "lookup", "ptx", 1, nullptr) == 0,
            "successful launch returned a failure");
    require(observed_dispatches == 1 && launch_calls == 1,
            "successful CUDA enqueue was not counted exactly once");
    require(launch(0, "lookup", "ptx", 513, nullptr) == 0,
            "cached launch returned a failure");
    require(observed_dispatches == 2 && launch_calls == 2,
            "cached CUDA enqueue was not counted exactly once");

    launch_status = 1;
    require(launch(0, "lookup", "ptx", 1, nullptr) == 5,
            "failed launch status changed");
    require(observed_dispatches == 2 && launch_calls == 3,
            "failed CUDA enqueue changed the dispatch count");
    require(launch(0, "lookup", "ptx", 0, nullptr) == 0,
            "empty launch status changed");
    require(observed_dispatches == 2 && launch_calls == 3,
            "empty CUDA work changed the dispatch count");

    activation_succeeds = false;
    require(launch(0, "lookup", "ptx", 1, nullptr) == 5,
            "activation failure status changed");
    activation_succeeds = true;
    api.ready = false;
    require(launch(0, "lookup", "ptx", 1, nullptr) == 5,
            "unavailable backend status changed");
    api.ready = true;
    module_status = 1;
    require(launch(0, "load-failure", "ptx", 1, nullptr) == 5,
            "module load failure status changed");
    require(launch(0, "lookup", "ptx",
                   (std::uint64_t(std::numeric_limits<unsigned>::max()) + 1) * 256,
                   nullptr) == 2,
            "oversized launch status changed");
    require(observed_dispatches == 2 && launch_calls == 3,
            "refused CUDA work changed the dispatch count");
    std::puts("Math CUDA dispatch counters passed");
}

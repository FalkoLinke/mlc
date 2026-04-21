#include <iostream>
#include <chrono>

#include "benchmark_kernel.h"

typedef int(kernel_func_t)(int);

struct kernel_t {
public:
    kernel_func_t* const func;
    std::string const name;

    kernel_t(kernel_func_t* const func, std::string const name) : func(func), name(name) {

    }

    int call(int const repetitions) const {
        return func(repetitions);
    }
};

#define MAKE_KERNEL(FUNC) ( kernel_t( (FUNC), #FUNC ) )








void warmup_kernel(kernel_t const kernel, int const repetitions) {
    kernel.call(repetitions);
}

void handle_kernel_total(kernel_t const kernel, int const repetitions) {
    std::cout << "========================" << std::endl;
    std::cout << "Benchmarking \"" << kernel.name << "\"" << std::endl;

    int insts_count = 0;

    auto start = std::chrono::high_resolution_clock::now();

    insts_count = kernel.call(repetitions);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    double insts_per_second = insts_count / duration.count();

    std::cout << "Instructions per second: " << insts_per_second << std::endl;
}





int main() {
    kernel_t kernels[] = {
        MAKE_KERNEL(fmadd_kernel),
        MAKE_KERNEL(fmla_4s_kernel),
        MAKE_KERNEL(fmla_2s_kernel),
    };
    size_t kernels_count = sizeof(kernels) / sizeof(kernel_t);
    int repetitions = 4000;

    for (int i = 0; i < kernels_count; i++) {
        warmup_kernel(kernels[i], repetitions);
        handle_kernel_total(kernels[i], repetitions);
    }
    return 0;
}
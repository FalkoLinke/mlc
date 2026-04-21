#include <iostream>
#include <chrono>

#include "benchmark_kernel.h"

typedef int(kernel_func_t)(int);

struct kernel_t {
public:
    kernel_func_t* const func;
    std::string const name;
    int const fpops_per_inst;

    kernel_t(kernel_func_t* const func, std::string const name, int const fpops_per_inst) : func(func), name(name), fpops_per_inst(fpops_per_inst) {

    }

    int call(int const repetitions) const {
        return func(repetitions);
    }
};

#define MAKE_KERNEL(FUNC, OPS) ( kernel_t( (FUNC), #FUNC , (OPS)) )








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

    std::cout << "\tTime taken [s]: " << duration.count() << std::endl;
    std::cout << "\tTotal instructions: " << insts_count << std::endl;
    std::cout << "\tInstructions per second: " << insts_per_second << std::endl;
    std::cout << "\tFlops: " << insts_per_second * kernel.fpops_per_inst << std::endl;
}





int main() {
    kernel_t kernels[] = {
        MAKE_KERNEL(fmadd_kernel, 2),
        MAKE_KERNEL(fmla_4s_kernel, 2 * 4),
        MAKE_KERNEL(fmla_2s_kernel, 2 * 2),
    };
    size_t kernels_count = sizeof(kernels) / sizeof(kernel_t);
    int repetitions = 4000;

    for (int i = 0; i < kernels_count; i++) {
        warmup_kernel(kernels[i], repetitions);
        handle_kernel_total(kernels[i], repetitions);
    }
    return 0;
}
#include <iostream>
#include <chrono>

#include "benchmark_kernel.h"

/** 
 * The type of a kernel function to be measured.
 * 
 * Parameters:
 * - `repetitions`: The number of times that the kernel loops over it's measured instructions.
 * 
 * Returns:
 * The number of instructions executed. 
 * This does not have to be equal to the floating point operations executed by the function.
 */
typedef int(kernel_func_t)(int);

struct kernel_t {
public:
    /* A pointer to the kernel function. */
    kernel_func_t* const func;

    /* The name of the kernel. */
    std::string const name;

    /* The number of floating point operations executed per instruction. */
    int const fpops_per_inst;

    kernel_t(kernel_func_t* const func, std::string const name, int const fpops_per_inst) : func(func), name(name), fpops_per_inst(fpops_per_inst) {

    }

    /* Call the kernel function, passing `repetitions` as it's argument. */
    int call(int const repetitions) const {
        return func(repetitions);
    }
};

#define MAKE_KERNEL(FUNC, OPS) ( kernel_t( (FUNC), #FUNC , (OPS)) )






/**
 * Calls `kernel` with `repetitions`.
 */
void warmup_kernel(kernel_t const kernel, int const repetitions) {
    kernel.call(repetitions);
}

/**
 * Calls `kernel` with `repetitions` and measures it's time taken.
 * Writes the results of the evaluation to `stdout`.
 */
void handle_kernel_total(kernel_t const kernel, int const repetitions) {
    std::cout << "========================" << std::endl;
    std::cout << "Benchmarking \"" << kernel.name << "\"" << std::endl;

    int insts_count = 0;

    // measure
    auto start = std::chrono::high_resolution_clock::now();

    insts_count = kernel.call(repetitions);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    double insts_per_second = insts_count / duration.count();
    double gflops = insts_per_second * kernel.fpops_per_inst / std::giga::num;

    // results
    std::cout << "\tTime taken [s]: " << duration.count() << std::endl;
    std::cout << "\tTotal instructions: " << insts_count << std::endl;
    std::cout << "\tInstructions per second: " << insts_per_second << std::endl;
    std::cout << "\tGFlops: " << gflops << std::endl;
}





int main() {
    // The array of kernels to evaluate.
    kernel_t kernels[] = {
        MAKE_KERNEL(fmadd_kernel, 2),
        MAKE_KERNEL(fmla_4s_kernel, 2 * 4),
        MAKE_KERNEL(fmla_2s_kernel, 2 * 2),

        MAKE_KERNEL(fmadd_kernel_v2, 2),
        MAKE_KERNEL(fmla_4s_kernel_v2, 2 * 4),
        MAKE_KERNEL(fmla_2s_kernel_v2, 2 * 2),
    };
    size_t kernels_count = sizeof(kernels) / sizeof(kernel_t);
    int repetitions = 4000;

    // Evaluate each kernel.
    for (int i = 0; i < kernels_count; i++) {
        warmup_kernel(kernels[i], repetitions);
        handle_kernel_total(kernels[i], repetitions);
    }
    return 0;
}
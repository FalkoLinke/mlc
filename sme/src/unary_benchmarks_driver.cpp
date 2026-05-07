#include <iostream>
#include <chrono>

#include "identity_kernel.h"
#include "zero_kernel.h"
#include "relu_kernel.h"


/**
 * @brief       The function signature of a unary operation kernel.
 * @param a     The source matrix.
 * @param b     The target matrix.
 * @param ld_a  The leading parameter for a.
 * @param ld_b  The leading parameter for b.
 */
typedef void(kernel_func_t)(float const*, float*, int64_t, int64_t);

/**
 * @brief Kernel wrapper for a nontransposing `identity` operation. Conforms to `kernel_func_t`.
 */
void identity_16_16_kernel(float const* a, float* b, int64_t ld_a, int64_t ld_b) {
    identity_16_16(a, b, ld_a, ld_b , 0);
}

/**
 * @brief Kernel wrapper for a transposing `identity` operation. Conforms to `kernel_func_t`.
 */
void identity_16_16_trans_kernel(float const* a, float* b, int64_t ld_a, int64_t ld_b) {
    identity_16_16(a, b, ld_a, ld_b, 1);
}

/**
 * @brief Kernel wrapper for a `zero` operation. Conforms to `kernel_func_t`.
 */
void zero_16_16_kernel(float const* a, float* b, int64_t ld_a, int64_t ld_b) {
    zero_16_16(b, ld_b);
}

/**
 * @brief Kernel wrapper for a nontransposing `RELU` operation. Conforms to `kernel_func_t`.
 */
void relu_16_16_kernel(float const* a, float* b, int64_t ld_a, int64_t ld_b) {
    relu_16_16(a, b, ld_a, ld_b, 0);
}

/**
 * @brief Kernel wrapper for a transposing `RELU` operation. Conforms to `kernel_func_t`.
 */
void relu_16_16_trans_kernel(float const* a, float* b, int64_t ld_a, int64_t ld_b) {
    relu_16_16(a, b, ld_a, ld_b, 1);
}

/**
 * Stores a kernel function along with an identifying name.
 */
struct kernel_t {
public:
    /** The kernel function. Call this to execute the kernel. */
	kernel_func_t* const func;
    /** The identifying name. */
	std::string const name;

	kernel_t(kernel_func_t* const func, std::string const name) : func(func), name(name) {

	}

    /** Calls the kernel function with the given parameters. */
	void call(float const* a, float* b, int64_t ld_a, int64_t ld_b) const {
		return this->func(a, b, ld_a, ld_b);
	}
};

#define MAKE_KERNEL(FUNC) (kernel_t( (FUNC), #FUNC ))













/**
 * @brief Benchmark the given kernel and report the `GiBs` metric.
 * @param kernel        The kernel to evaluate.
 * 
 * This function allocates space in it's stack frame for two 16x16 FP32 matrices.
 * It then repeatedly executes the given kernel on those matrices, while measuring
 * the total amount of time taken.
 * From the total number of bytes processed it then determines and prints
 * the GiBs metric, along with other benchmark results.
 */
void benchmark_kernel(kernel_t kernel) {
    int const reps = 1000000;
    float a[16*16];
    float b[16*16];

    kernel_func_t* kernel_func = kernel.func;
    kernel_func(a, b, 16, 16);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < reps; i++) {
        kernel_func(a, b, 16, 16);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;


    double bytes_transferred = reps * 16 * 16 * sizeof(float);
    double gib_transferred = bytes_transferred / 1073741824.0f;
    double time_taken = duration.count();
    double gibs = gib_transferred / time_taken;


    std::cout << "=========================" << std::endl;
    std::cout << kernel.name << ":" << std::endl;
    std::cout << "\tDuration [s]: " << time_taken << std::endl;
    std::cout << "\tBytes transferred [GB]: " << bytes_transferred * 1e-9 << std::endl;
    std::cout << "\tGiBs: " << gibs << std::endl;
}











int main() {
    kernel_t kernels[] = {
        MAKE_KERNEL(identity_16_16_kernel),
        MAKE_KERNEL(identity_16_16_trans_kernel),
        MAKE_KERNEL(zero_16_16_kernel),
        MAKE_KERNEL(relu_16_16_kernel),
        MAKE_KERNEL(relu_16_16_trans_kernel),
    };
    size_t const kernels_count = sizeof(kernels) / sizeof(kernel_t);

    for (size_t i = 0; i < kernels_count; i++) {
        benchmark_kernel(kernels[i]);
    }

    return 0;
}

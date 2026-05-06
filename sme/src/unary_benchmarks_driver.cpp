#include <iostream>
#include <chrono>

#include "identity_kernel.h"
#include "zero_kernel.h"
#include "relu_kernel.h"



typedef void(kernel_func_t)(float const*, float*, int64_t, int64_t);

void identity_16_16_kernel(float const* a, float* b, int64_t ld_a, int64_t ld_b) {
    identity_16_16(a, b, ld_a, ld_b , 0);
}

void identity_16_16_trans_kernel(float const* a, float* b, int64_t ld_a, int64_t ld_b) {
    identity_16_16(a, b, ld_a, ld_b, 1);
}

void zero_16_16_kernel(float const* a, float* b, int64_t ld_a, int64_t ld_b) {
    zero_16_16(b, ld_b);
}

void relu_16_16_kernel(float const* a, float* b, int64_t ld_a, int64_t ld_b) {
    relu_16_16(a, b, ld_a, ld_b, 0);
}

void relu_16_16_trans_kernel(float const* a, float* b, int64_t ld_a, int64_t ld_b) {
    relu_16_16(a, b, ld_a, ld_b, 1);
}














void benchmark_kernel(kernel_func_t* kernel) {
    int const reps = 1000000;
    float a[16*16];
    float b[16*16];


    kernel(a, b, 16, 16);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < reps; i++) {
        kernel(a, b, 16, 16);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;


    double bytes_transferred = reps * 16 * 16 * sizeof(float);
    double time_taken = duration.count();
    double gibs = 1e-9 * bytes_transferred / time_taken;

    std::cout << "GiBs: " << gibs << std::endl;
}











int main() {
    kernel_func_t* kernels[] = {
        identity_16_16_kernel,
        identity_16_16_trans_kernel,
        zero_16_16_kernel,
        relu_16_16_kernel,
        relu_16_16_trans_kernel,
    };
    size_t const kernels_count = sizeof(kernels) / sizeof(kernel_func_t*);

    for (size_t i = 0; i < kernels_count; i++) {
        benchmark_kernel(kernels[i]);
    }

    return 0;
}

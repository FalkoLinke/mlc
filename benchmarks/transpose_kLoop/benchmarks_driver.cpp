#include <iostream>
#include <vector>
#include <cstdint>
#include <chrono>

#include "transpose_kernels.h"




typedef void(kernel_t)(float const*, float*, uint64_t, uint64_t);





void benchmark_kernel(kernel_t* kernel, uint64_t reps, uint64_t k_loop) {
    std::vector<float> a(16 * 16, 0.0f);
    std::vector<float> b(16 * 16, 0.0f);

    kernel(a.data(), b.data(), 16, 16);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < reps; i++) {
        kernel(a.data(), b.data(), 16, 16);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    uint64_t bytes_transferred = 2 * 16 * 16 * sizeof(float) * reps * k_loop;
    double gibs_transferred = bytes_transferred / (1024.0f * 1024.0f * 1024.0f);
    double gibs = gibs_transferred / duration.count();

    std::cout << gibs << "\t\t" << gibs_transferred << "\t\t" << duration.count() << std::endl;
}







int main() {
    uint64_t reps = 20000000;

    std::cout << "GiBs\t\tBytes trans\t\tDuration [s]" << std::endl;

    benchmark_kernel(&copy_sve, reps, 1);
    benchmark_kernel(&transpose_16x16_fp32_za, 400, 50000);
    benchmark_kernel(&transpose_16x16_fp32_tbl, 400, 50000);
    benchmark_kernel(&transpose_16x16_fp32_tbl_v2, 400, 50000);
    //benchmark_kernel(&transpose_16x16_fp32_simd, reps, 1);
    benchmark_kernel(&transpose_16x16_fp32_sme2, 400, 50000);

    return 0;
}
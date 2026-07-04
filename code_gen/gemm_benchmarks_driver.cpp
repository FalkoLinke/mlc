#include <iostream>
#include <chrono>

#include "Gemm.h"






void benchmark_gemm(uint32_t m, uint32_t n, uint32_t k, uint32_t trans_a, uint32_t trans_b, uint32_t trans_c, mini_jit::Gemm::dtype_t dtype, uint32_t lda = ~0, uint32_t ldb = ~0, uint32_t ldc = ~0) {
    using T = float;

    // generate kernel
    mini_jit::Gemm gemm;
    mini_jit::Gemm::error_t err = gemm.generate(m, n, k, trans_a, trans_b, trans_c, dtype);
    if (err != mini_jit::Gemm::error_t::success) {
        std::cerr << "Error for m=" << m << " n=" << n << " k=" << k << " trans_a=" << trans_a << " trans_b=" << trans_b << " trans_c=" << trans_c << std::endl;
        return;
    }
    mini_jit::Gemm::kernel_t kernel = gemm.get_kernel();
    gemm.write("test.bin");

    // allocate memory
    uint32_t a_d1 = trans_a ? k : m;
    uint32_t a_d2 = trans_a ? m : k;
    uint32_t b_d1 = trans_b ? n : k;
    uint32_t b_d2 = trans_b ? k : n;
    uint32_t c_d1 = trans_c ? n : m;
    uint32_t c_d2 = trans_c ? m : n;
    if (lda == ~0u) {
        lda = trans_a ? k : m;
    }
    if (ldb == ~0u) {
        ldb = trans_b ? n : k;
    }
    if (ldc == ~0u) {
        ldc = trans_c ? n : m;
    }
    std::vector<T> a(a_d2 * lda, 0.0f);
    std::vector<T> b(b_d2 * ldb, 0.0f);
    std::vector<T> c(c_d2 * ldc, 0.0f);


    // perform measurement
    uint64_t operations_per_rep = m * n * k * 2;
    double target_total_gflops = 4000;
    uint64_t reps = static_cast<uint64_t>((target_total_gflops / operations_per_rep) * 1e9);

    for (uint64_t i = 0; i < reps / 4; i++) {
        kernel(a.data(), b.data(), c.data(), lda, ldb, ldc);
    }
    auto start = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < reps; i++) {
        kernel(a.data(), b.data(), c.data(), lda, ldb, ldc);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_taken = end - start;

    // calculate and output results
    double total_gflops_computed = (reps * 1e-9) * operations_per_rep;
    double gflops = total_gflops_computed / time_taken.count();

    std::cout << m << "\t\t";
    std::cout << n << "\t\t";
    std::cout << k << "\t\t";
    std::cout << trans_a << "\t\t";
    std::cout << trans_b << "\t\t";
    std::cout << trans_c << "\t\t";
    std::cout << gflops << "\t\t";
    std::cout << time_taken.count() << "\t\t";
    std::cout << std::endl;
}








int main() {
    std::cout << "m\t\tn\t\tk\t\ttrans_a\t\ttrans_b\t\ttrans_c\t\tGFlops\t\tDuration [s]" << std::endl;

    benchmark_gemm(16, 16, 512, 0, 1, 0, mini_jit::Gemm::dtype_t::fp32);
    benchmark_gemm(16, 16, 512, 0, 1, 1, mini_jit::Gemm::dtype_t::fp32);
    benchmark_gemm(32, 32, 512, 0, 1, 0, mini_jit::Gemm::dtype_t::fp32);
    benchmark_gemm(32, 32, 512, 0, 1, 1, mini_jit::Gemm::dtype_t::fp32);
    benchmark_gemm(512, 512, 512, 0, 1, 0, mini_jit::Gemm::dtype_t::fp32);
    benchmark_gemm(512, 512, 512, 0, 1, 1, mini_jit::Gemm::dtype_t::fp32);

    return 0;
}

#include <chrono>
#include <cstdint>
#include <vector>
#include <iostream>

#include "gemm_kernels.h"



void benchmark_gemm_kernel(gemm_kernel_desc_t const desc, uint64_t const reps) {
    std::vector<float> in0(desc.m * desc.k, 0.0f);
    std::vector<float> in1(desc.n * desc.k, 0.0f);
    std::vector<float> out(desc.n * desc.m, 0.0f);

    uint64_t lda = desc.trans_a ? desc.k : desc.m;
    uint64_t ldb = desc.trans_b ? desc.k : desc.n;
    uint64_t ldc = desc.trans_c ? desc.n : desc.m;

    desc.func(in0.data(), in1.data(), out.data(), lda, ldb, ldc);
    auto start = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < reps; i++) {
        desc.func(in0.data(), in1.data(), out.data(), lda, ldb, ldc);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    uint64_t flops_per_rep = desc.m * desc.k * desc.n * 2;
    double gflops_computed = (reps * 1e-9) * flops_per_rep;
    double gflops = gflops_computed / duration.count();

    std::cout << desc.m << "\t";
    std::cout << desc.n << "\t";
    std::cout << desc.k << "\t";
    std::cout << desc.trans_a << "\t";
    std::cout << desc.trans_b << "\t";
    std::cout << desc.trans_c << "\t";
    std::cout << desc.desc << "\t";
    std::cout << gflops << "\t";
    std::cout << duration.count() << std::endl;
}





int main() {
    std::cout << "m\tn\tk\ttrans_a\ttrans_b\ttrans_c\tDescription\tGFlops\tDuration [s]" << std::endl;

    benchmark_gemm_kernel(desc_gemm_km_kn_nm_fp32_m16_n32_k1, 100000000);
    benchmark_gemm_kernel(desc_gemm_km_kn_nm_fp32_m16_n32_k16, 10000000);
    benchmark_gemm_kernel(desc_gemm_km_kn_nm_fp32_m16_n32_k512, 10000000);
    benchmark_gemm_kernel(desc_gemm_km_kn_nm_fp32_m16_n32_k512_v2, 10000000);
    benchmark_gemm_kernel(desc_gemm_km_kn_nm_fp32_m16_n64_k512, 10000000);
    benchmark_gemm_kernel(desc_gemm_mk_kn_nm_fp32_m16_n32_k16_za, 10000000);
    benchmark_gemm_kernel(desc_gemm_mk_kn_nm_fp32_m16_n32_k512_za, 1000000);
    benchmark_gemm_kernel(desc_gemm_mk_kn_nm_fp32_m16_n32_k512_tbl, 10000000);
    benchmark_gemm_kernel(desc_gemm_mk_kn_nm_fp32_m16_n32_k512_tbl_v2, 10000000);
    benchmark_gemm_kernel(desc_gemm_mk_nk_nm_fp32_m16_n16_k512_tbl_stack, 10000000);

    return 0;
}
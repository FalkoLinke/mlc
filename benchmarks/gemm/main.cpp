#include <vector>
#include <iostream>
#include "gemm_kernels.h"




void print_mat(float const* mat, uint64_t m, uint64_t n) {
    for (uint64_t r = 0; r < m; r++) {
        for (uint64_t c = 0; c < n; c++) {
            std::cout << mat[c * m + r] << " ";
        }
        std::cout << std::endl;
    }
}






void ref_gemm(gemm_kernel_desc_t desc, float const* in0, float const* in1, float* out, uint64_t lda, uint64_t ldb, uint64_t ldc) {
    uint64_t in0_sm = desc.trans_a ? lda : 1;
    uint64_t in0_sk = desc.trans_a ? 1 : lda;

    uint64_t in1_sk = desc.trans_b ? ldb : 1;
    uint64_t in1_sn = desc.trans_b ? 1 : ldb;

    uint64_t out_sm = desc.trans_c ? ldc : 1;
    uint64_t out_sn = desc.trans_c ? 1 : ldc;

    for (uint64_t xm = 0; xm < desc.m; xm++) {
        for (uint64_t xn = 0; xn < desc.n; xn++) {
            for (uint64_t xk = 0; xk < desc.k; xk++) {
                float const* in0_ptr = in0 + xm * in0_sm + xk * in0_sk;
                float const* in1_ptr = in1 + xn * in1_sn + xk * in1_sk;
                float* out_ptr       = out + xm * out_sm + xn * out_sn;
                *out_ptr += *in0_ptr * *in1_ptr;
            }
        }
    }
}



float max_abs_diff(float const* in0, float const* in1, uint64_t size) {
    float result = 0.0f;
    for (uint64_t i = 0; i < size; i++) {
        float diff = fabsf(in0[i] - in1[i]);
        result = fmaxf(result, diff);
    }
    return result;
}




bool verify_gemm(gemm_kernel_desc_t desc) {
    std::vector<float> in0(desc.m * desc.k, 0.0f);
    std::vector<float> in1(desc.n * desc.k, 0.0f);
    std::vector<float> out(desc.m * desc.n, 1.0f);
    std::vector<float> out_exp(desc.m * desc.n, 1.0f);
    for (uint64_t i = 0; i < in0.size(); i++) {
        in0[i] = (float)(i % 100) / 100.0f;
    }
    for (uint64_t i = 0; i < in1.size(); i++) {
        in1[i] = (float)((i + 50) % 100) / 100.0f;
    }

    uint64_t lda = desc.trans_a ? desc.k : desc.m;
    uint64_t ldb = desc.trans_b ? desc.n : desc.k;
    uint64_t ldc = desc.trans_c ? desc.n : desc.m;

    desc.func(in0.data(), in1.data(), out.data(), lda, ldb, ldc);
    ref_gemm(desc, in0.data(), in1.data(), out_exp.data(), lda, ldb, ldc);

    float diff = max_abs_diff(out.data(), out_exp.data(), out.size());
    
    std::cout << diff << std::endl;
    std::cout << std::endl;
    print_mat(out.data(), desc.m, desc.n);
    std::cout << std::endl;
    print_mat(out_exp.data(), desc.m, desc.n);
    std::cout << std::endl;
    
    return diff < 1e-3;
}






int main() {
    bool result = verify_gemm(desc_gemm_mk_kn_nm_fp32_m16_n32_k512_tbl);
    if (result) {
        std::cout << "SUCCESS" << std::endl;
    } else {
        std::cout << "FAILURE" << std::endl;
    }
    return 0;
}
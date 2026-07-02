#include <iostream>
#include <catch2/catch_test_macros.hpp>
#include "Gemm.h"
#include "mlc_common.hpp"





void reference_gemm(float const* a, float const* b, float* c, uint32_t lda, uint32_t ldb, uint32_t ldc, uint32_t m, uint32_t n, uint32_t k, uint32_t trans_a, uint32_t trans_b, uint32_t trans_c) {
    uint32_t a_sm = trans_a ? lda : 1;
    uint32_t a_sk = trans_a ? 1 : lda;

    uint32_t b_sn = trans_b ? 1 : ldb;
    uint32_t b_sk = trans_b ? ldb : 1;

    uint32_t c_sm = trans_c ? ldc : 1;
    uint32_t c_sn = trans_c ? 1 : ldc;

    for (uint32_t xm = 0; xm < m; xm++) {
        for (uint32_t xn = 0; xn < n; xn++) {
            for (uint32_t xk = 0; xk < k; xk++) {
                float const* a_ptr = a + a_sm * xm + a_sk * xk;
                float const* b_ptr = b + b_sn * xn + b_sk * xk;
                float* c_ptr = c + c_sm * xm + c_sn * xn;

                *c_ptr += *a_ptr * *b_ptr;
            }
        }
    }
}










void verify_gemm(uint32_t m, uint32_t n, uint32_t k, uint32_t lda, uint32_t ldb, uint32_t ldc, uint32_t trans_a, uint32_t trans_b, uint32_t trans_c) {
    using T = float;

    uint32_t a_d1 = trans_a ? k : m;
    uint32_t a_d2 = trans_a ? m : k;
    uint32_t b_d1 = trans_b ? n : k;
    uint32_t b_d2 = trans_b ? k : n;
    uint32_t c_d1 = trans_c ? n : m;
    uint32_t c_d2 = trans_c ? m : n;

    std::vector<T> a(a_d2 * lda, 0.0f);
    std::vector<T> b(b_d2 * ldb, 0.0f);
    std::vector<T> c(c_d2 * ldc, 0.0f);
    std::vector<T> exp(c_d2 * ldc, 0.0f);

    for (uint32_t i = 0; i < a.size(); i++) {
        a[i] = static_cast<float>(i % 100) / 100.0f;
    }
    for (uint32_t i = 0; i < b.size(); i++) {
        b[i] = static_cast<float>((i + 50) % 100) / 100.0f;
    }

    mini_jit::Gemm gemm;
    mini_jit::Gemm::error_t err = gemm.generate(m, n, k, trans_a, trans_b, trans_c, mini_jit::Gemm::dtype_t::fp32);
    if (err != mini_jit::Gemm::error_t::success) {
        FAIL("error during gemm kernel generation");
    }
    mini_jit::Gemm::kernel_t kernel = gemm.get_kernel();
    gemm.write("test.bin");

    reference_gemm(a.data(), b.data(), c.data(), lda, ldb, ldc, m, n, k, trans_a, trans_b, trans_c);
    kernel(a.data(), b.data(), c.data(), lda, ldb, ldc);

    float diff = max_abs_diff(c.data(), exp.data(), c.size());
    REQUIRE(diff < 1e-4);
}






TEST_CASE("test01", "[test]") {
    verify_gemm(32, 32, 32, 32, 32, 32, 0, 1, 0);
}




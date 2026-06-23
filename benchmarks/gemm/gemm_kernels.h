#ifndef GEMM_KERNELS_H
#define GEMM_KERNELS_H

#include <cstdint>



typedef void(gemm_kernel_t)(float const*, float const*, float*, uint64_t, uint64_t, uint64_t);




typedef struct gemm_kernel_desc {
    uint64_t m;
    uint64_t n;
    uint64_t k;
    bool trans_a;
    bool trans_b;
    bool trans_c;
    std::string desc;
    gemm_kernel_t* func;
} gemm_kernel_desc_t;







extern "C" {

    void gemm_km_kn_nm_fp32_m16_n32_k1(float const* in0, float const* in1, float* out, uint64_t lda, uint64_t ldb, uint64_t ldc);
    static gemm_kernel_desc_t desc_gemm_km_kn_nm_fp32_m16_n32_k1 = {
        .m = 16,
        .n = 32,
        .k = 1,
        .trans_a = false,
        .trans_b = true,
        .trans_c = false,
        .desc = "default",
        .func = gemm_km_kn_nm_fp32_m16_n32_k1,
    };

    void gemm_km_kn_nm_fp32_m16_n32_k16(float const* in0, float const* in1, float* out, uint64_t lda, uint64_t ldb, uint64_t ldc);
    static gemm_kernel_desc_t desc_gemm_km_kn_nm_fp32_m16_n32_k16 = {
        .m = 16,
        .n = 32,
        .k = 16,
        .trans_a = false,
        .trans_b = true,
        .trans_c = false,
        .desc = "default",
        .func = gemm_km_kn_nm_fp32_m16_n32_k16,
    };

    void gemm_km_kn_nm_fp32_m16_n32_k512(float const* in0, float const* in1, float* out, uint64_t lda, uint64_t ldb, uint64_t ldc);
    static gemm_kernel_desc_t desc_gemm_km_kn_nm_fp32_m16_n32_k512 = {
        .m = 16,
        .n = 32,
        .k = 512,
        .trans_a = false,
        .trans_b = true,
        .trans_c = false,
        .desc = "default",
        .func = gemm_km_kn_nm_fp32_m16_n32_k512,
    };

    void gemm_km_kn_nm_fp32_m16_n32_k512_v2(float const* in0, float const* in1, float* out, uint64_t lda, uint64_t ldb, uint64_t ldc);
    static gemm_kernel_desc_t desc_gemm_km_kn_nm_fp32_m16_n32_k512_v2 = {
        .m = 16,
        .n = 32,
        .k = 512,
        .trans_a = false,
        .trans_b = true,
        .trans_c = false,
        .desc = "default",
        .func = gemm_km_kn_nm_fp32_m16_n32_k512_v2,
    };

    void gemm_km_kn_nm_fp32_m16_n64_k512(float const* in0, float const* in1, float* out, uint64_t lda, uint64_t ldb, uint64_t ldc);
    static gemm_kernel_desc_t desc_gemm_km_kn_nm_fp32_m16_n64_k512 = {
        .m = 16,
        .n = 64,
        .k = 512,
        .trans_a = false,
        .trans_b = true,
        .trans_c = false,
        .desc = "default",
        .func = gemm_km_kn_nm_fp32_m16_n64_k512,
    };

    void gemm_mk_kn_nm_fp32_m16_n32_k16_za(float const* in0, float const* in1, float* out, uint64_t lda, uint64_t ldb, uint64_t ldc);
    static gemm_kernel_desc_t desc_gemm_mk_kn_nm_fp32_m16_n32_k16_za = {
        .m = 16,
        .n = 32,
        .k = 16,
        .trans_a = true,
        .trans_b = true,
        .trans_c = false,
        .desc = "ZA transpose",
        .func = gemm_mk_kn_nm_fp32_m16_n32_k16_za,
    };

    void gemm_mk_kn_nm_fp32_m16_n32_k512_za(float const* in0, float const* in1, float* out, uint64_t lda, uint64_t ldb, uint64_t ldc);
    static gemm_kernel_desc_t desc_gemm_mk_kn_nm_fp32_m16_n32_k512_za = {
        .m = 16,
        .n = 32,
        .k = 512,
        .trans_a = true,
        .trans_b = true,
        .trans_c = false,
        .desc = "ZA transpose",
        .func = gemm_mk_kn_nm_fp32_m16_n32_k512_za,
    };

    void gemm_mk_kn_nm_fp32_m16_n32_k512_tbl(float const* in0, float const* in1, float* out, uint64_t lda, uint64_t ldb, uint64_t ldc);
    static gemm_kernel_desc_t desc_gemm_mk_kn_nm_fp32_m16_n32_k512_tbl = {
        .m = 16,
        .n = 32,
        .k = 512,
        .trans_a = true,
        .trans_b = true,
        .trans_c = false,
        .desc = "TBL transpose",
        .func = gemm_mk_kn_nm_fp32_m16_n32_k512_tbl,
    };

    void gemm_mk_kn_nm_fp32_m16_n32_k512_tbl_v2(float const* in0, float const* in1, float* out, uint64_t lda, uint64_t ldb, uint64_t ldc);
    static gemm_kernel_desc_t desc_gemm_mk_kn_nm_fp32_m16_n32_k512_tbl_v2 = {
        .m = 16,
        .n = 32,
        .k = 512,
        .trans_a = true,
        .trans_b = true,
        .trans_c = false,
        .desc = "TBLV2 transpose",
        .func = gemm_mk_kn_nm_fp32_m16_n32_k512_tbl_v2,
    };

    void gemm_km_kn_nm_fp32_m16_n16_k512(float const* in0, float const* in1, float* out, uint64_t lda, uint64_t ldb, uint64_t ldc);
    static gemm_kernel_desc_t desc_gemm_km_kn_nm_fp32_m16_n16_k512 = {
        .m = 16,
        .n = 16,
        .k = 512,
        .trans_a = false,
        .trans_b = true,
        .trans_c = false,
        .desc = "default",
        .func = gemm_km_kn_nm_fp32_m16_n16_k512,
    };

    void gemm_mk_nk_nm_fp32_m16_n16_k512_tbl_stack(float const* in0, float const* in1, float* out, uint64_t lda, uint64_t ldb, uint64_t ldc);
    static gemm_kernel_desc_t desc_gemm_mk_nk_nm_fp32_m16_n16_k512_tbl_stack = {
        .m = 16,
        .n = 16,
        .k = 512,
        .trans_a = true,
        .trans_b = false,
        .trans_c = false,
        .desc = "TBL stack transpose",
        .func = gemm_mk_nk_nm_fp32_m16_n16_k512_tbl_stack,
    };

    void gemm_mk_nk_nm_fp32_m16_n16_k512_tbl_regs(float const* in0, float const* in1, float* out, uint64_t lda, uint64_t ldb, uint64_t ldc);
    static gemm_kernel_desc_t desc_gemm_mk_nk_nm_fp32_m16_n16_k512_tbl_regs = {
        .m = 16,
        .n = 16,
        .k = 512,
        .trans_a = true,
        .trans_b = false,
        .trans_c = false,
        .desc = "TBLV2 regs transpose",
        .func = gemm_mk_nk_nm_fp32_m16_n16_k512_tbl_regs,
    };


    void gemm_16_16_multiple_k(float const* in0, float const* in1, float* out, uint64_t lda, uint64_t ldb, uint64_t ldc);
    static gemm_kernel_desc_t desc_gemm_16_16_multiple_k = {
        .m = 16,
        .n = 16,
        .k = 512,
        .trans_a = false,
        .trans_b = true,
        .trans_c = false,
        .desc = "test",
        .func = gemm_16_16_multiple_k,
    };

    void gemm_16_16_ref_single_k(float const* in0, float const* in1, float* out, uint64_t lda, uint64_t ldb, uint64_t ldc);
    static gemm_kernel_desc_t desc_gemm_16_16_ref_single_k = {
        .m = 16,
        .n = 16,
        .k = 512,
        .trans_a = false,
        .trans_b = true,
        .trans_c = false,
        .desc = "test",
        .func = gemm_16_16_ref_single_k,
    };
}


#endif
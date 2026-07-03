#ifndef BENCHMARKS_TRANSPOSE_KERNELS_H
#define BENCHMARKS_TRANSPOSE_KERNELS_H

#include <cstdint>

extern "C" {
    void transpose_16x16_fp32_za(float const* in0, float* out, uint64_t ldi, uint64_t ldo);
    void transpose_16x16_fp32_tbl(float const* in0, float* out, uint64_t ldi, uint64_t ldo);
    void transpose_16x16_fp32_tbl_v2(float const* in0, float* out, uint64_t ldi, uint64_t ldo);
    void transpose_16x16_fp32_simd(float const* in0, float* out, uint64_t ldi, uint64_t ldo);
    void transpose_16x16_fp32_sme2(float const* in0, float* out, uint64_t ldi, uint64_t ldo);

    void copy_sve(float const* in0, float* out, uint64_t ldi, uint64_t ldo);
}


#endif
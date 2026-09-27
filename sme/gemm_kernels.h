#ifndef SME_GEMM_KERNELS_H
#define SME_GEMM_KERNELS_H

#include <cstdint>

extern "C" {

/**
 * @brief GEMM that computes C += A * B for the fixed sizes indicated by the kernel name
 *        (gemm_<m>_<n>_<k>). `gemm_16_16` uses m = n = 16 and k = 512,
 *        `gemm_M_N_K_16` uses m = n = k = 512.
 * @param a    Pointer to column-major matrix A.
 * @param b    Pointer to row-major matrix B.
 * @param c    Pointer to column-major matrix C.
 * @param ld_a Leading dimension of A.
 * @param ld_b Leading dimension of B.
 * @param ld_c Leading dimension of C.
 **/
void gemm_16_16( float const * a, float const * b, float * c, int64_t ld_a, int64_t ld_b, int64_t ld_c );

void gemm_32_32_1( float const * a, float const * b, float * c, int64_t ld_a, int64_t ld_b, int64_t ld_c );

void gemm_32_32_512( float const * a, float const * b, float * c, int64_t ld_a, int64_t ld_b, int64_t ld_c );

void gemm_512_32_512( float const * a, float const * b, float * c, int64_t ld_a, int64_t ld_b, int64_t ld_c );

void gemm_512_512_512( float const * a, float const * b, float * c, int64_t ld_a, int64_t ld_b, int64_t ld_c );

void gemm_M_N_K_16( float const * a, float const * b, float * c, int64_t ld_a, int64_t ld_b, int64_t ld_c );

}

#endif /*SME_GEMM_KERNELS_H*/

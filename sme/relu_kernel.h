#ifndef SME_RELU_KERNEL_H
#define SME_RELU_KERNEL_H

#include <cstdint>

extern "C" {

/*
* @brief Computes max(0,x) for every entry of column-major matrix A and writes the result to matrix B.
* @param a       Pointer to column-major matrix A.
* @param b       Pointer to matrix B.
* @param ld_a    Leading dimension of A.
* @param ld_b    Leading dimension of B.
* @param trans_b Column-major B if 0, row-major B if 1. 
**/
void relu_16_16( float const * a,
                float       * b,
                int64_t       ld_a,
                int64_t       ld_b,
                int32_t       trans_b );

}

#endif /*SME_RELU_KERNEL_H*/
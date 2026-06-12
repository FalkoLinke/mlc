#ifndef TRANSPOSE_KERNELS_H
#define TRANSPOSE_KERNELS_H

#include <cstdint>

extern "C" {
    void transpose_16x16_fp32_za(float const* in0, float* out, uint64_t ldi, uint64_t ldo);
    void transpose_16x16_fp32_tbl(float const* in0, float* out, uint64_t ldi, uint64_t ldo);
}


#endif
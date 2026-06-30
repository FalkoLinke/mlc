#ifndef SME_ZERO_KERNEL_H
#define SME_ZERO_KERNEL_H

#include <cstdint>

extern "C" {

/**
 * @brief Sets all entries of A to zero with size of A = 16x16. 
 * @param a    Pointer to column-major matrix A.
 * @param ld_a Leading dimension of A.
 **/
void zero_16_16( float* a,
                 int64_t ld_a );

}

#endif /*SME_ZERO_KERNEL_H*/
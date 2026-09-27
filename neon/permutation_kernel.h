#ifndef NEON_PERMUTATION_KERNEL_H
#define NEON_PERMUTATION_KERNEL_H

#include <cstdint>

extern "C" {

/**
 * @brief Permutation operation abc->cba with |a| = 8 and |b| = 4.
 *
 * Scalar reference version, moves one element per iteration.
 *
 * @param size_c Size of dimension c.
 * @param abc    Pointer to row-major tensor abc.
 * @param cba    Pointer to row-major tensor cba.
 **/
void perm_neon_abc_cba_scalar( int64_t       size_c,
                               float const * abc,
                               float       * cba );

/**
 * @brief Permutation operation abc->cba with |a| = 8 and |b| = 4.
 *
 * Optimized version, transposes two 4x4 blocks per iteration.
 *
 * @param size_c Size of dimension c.
 * @param abc    Pointer to row-major tensor abc.
 * @param cba    Pointer to row-major tensor cba.
 **/
void perm_neon_abc_cba( int64_t       size_c,
                        float const * abc,
                        float       * cba );

}

#endif /*NEON_PERMUTATION_KERNEL_H*/

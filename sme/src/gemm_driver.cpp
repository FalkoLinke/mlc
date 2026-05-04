#include <iostream>
#include "identity_kernel.h"

extern "C" {
    /**
    * @brief GEMM that computes: C+=AB.
    * @param a    Pointer to column-major matrix A.
    * @param b    Pointer to row-major matrix B.
    * @param c    Pointer to column-major matrix C.
    * @param ld_a Leading dimension of A.
    * @param ld_b Leading dimension of B.
    * @param ld_c Leading dimension of C.
    **/
   void gemm_32_32_1( float   const * a,
                      float   const * b,
                      float         * c,
                      int64_t         ld_a,
                      int64_t         ld_b,
                      int64_t         ld_c );

 /**
 * @brief Identity operation with m=16 and n=16.
 * @param a       Pointer to column-major matrix A.
 * @param b       Pointer to matrix B.
 * @param ld_a    Leading dimension of A.
 * @param ld_b    Leading dimension of B.
 * @param trans_b Column-major B if 0, row-major B if 1. 
 **/
void identity_16_16( float const * a,
                    float       * b,
                    int64_t       ld_a,
                    int64_t       ld_b,
                    int32_t       trans_b );

}

void print_mat(float const* mat, int m, int n) {
    for (int r = 0; r < m; r++) {
        for (int c = 0; c < n; c++) {
            std::cout << mat[c * m + r] << " ";
        }
        std::cout << std::endl;
    }
}

void fill_indices(float* buffer, int const size) {
    for (int i = 0; i < size; i++) {
        buffer[i] = (float)i;
    }
}


int main() {
    float a[32][32] = {0.0f};
    float b[32][32] = {0.0f};
    float c[32][32] = {0.0f};

    fill_indices(&a[0][0], 1024);
    fill_indices(&b[0][0], 1024);
    
    gemm_32_32_1(&a[0][0], &b[0][0], &c[0][0], 32, 32, 32);
    
    print_mat(&c[0][0], 32, 32);

    return 0;
}

#include <iostream>
#include <vector>

#include "transpose_kernels.h"




void print_mat(float const* mat, uint64_t m, uint64_t n) {
    for (uint64_t r = 0; r < m; r++) {
        for (uint64_t c = 0; c < n; c++) {
            std::cout << mat[c * m + r] << " ";
        }
        std::cout << std::endl;
    }
}







int main() {
    std::vector<float> a(16*16, 0.0f);
    std::vector<float> b(16*16, 0.0f);
    for (uint64_t i = 0; i < a.size(); i++) {
        a[i] = (float)i;
    }

    transpose_16x16_fp32_simd(a.data(), b.data(), 16, 16);

    print_mat(a.data(), 16, 16);
    std::cout << std::endl;
    print_mat(b.data(), 16, 16);
    return 0;
}



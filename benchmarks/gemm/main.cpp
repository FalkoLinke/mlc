#include <vector>
#include <iostream>
#include "gemm_kernels.h"




void print_mat(float const* mat, uint64_t m, uint64_t n) {
    for (uint64_t r = 0; r < m; r++) {
        for (uint64_t c = 0; c < n; c++) {
            std::cout << mat[c * m + r] << " ";
        }
        std::cout << std::endl;
    }
}


int main() {
    uint64_t m = 16;
    uint64_t n = 32;
    uint64_t k = 1;

    std::vector<float> in0(m * k, 0.0f);
    std::vector<float> in1(n * k, 0.0f);
    std::vector<float> out(n * m, 0.0f);
    for (uint64_t i = 0; i < in0.size(); i++) {
        in0[i] = (float)i;
    }
    for (uint64_t i = 0; i < in1.size(); i++) {
        in1[i] = (float)i;
    }

    gemm_km_kn_nm_fp32_m16_n32_k1(in0.data(), in1.data(), out.data(), m, n, m);

    print_mat(in0.data(), m, k);
    std::cout << std::endl;
    print_mat(in1.data(), n, k);
    std::cout << std::endl;
    print_mat(out.data(), m, n);

    return 0;
}
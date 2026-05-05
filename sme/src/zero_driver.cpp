#include <iostream>

#include "zero_kernel.h"

void print_mat(float const* mat, int const m, int const n) {
    for (int r = 0; r < m; r++) {
        for (int c = 0; c < n; c++) {
            std::cout << mat[c*m + r] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    float a[32*32];
    for (int i = 0; i < 32*32; i++) {
        a[i] = 5.0f;
    }

    zero_16_16(a + 8 + 8*32, 32);
    print_mat(a, 32, 32);

    return 0;
}
#include <iostream>
#include "identity_kernel.h"


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
    float a[256] = {0.0f};
    float b[256] = {0.0f};

    fill_indices(a, 256);
    identity_16_16(a, b, 16, 16, 1);
    print_mat(b, 16, 16);

    return 0;
}

#include <iostream>
#include "relu_kernel.h"

void print_mat(float const* mat, int m, int n) {
    for (int r = 0; r < m; r++) {
        for (int c = 0; c < n; c++) {
            std::cout << mat[c * m + r] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    float a[16*16] = {0.0f};
    float b[16*16] = {0.0f};

    for (int i = 0; i < 16*16; i++) {
	a[i] = 5.0f;
    }
    for (int i = 0; i < 16; i++) {
        a[i*16 + i] = -1.0f;
    }

    relu_16_16(a, b, 16*sizeof(float), 16*sizeof(float), 0);
    print_mat(b, 16, 16);

    return 0;
}

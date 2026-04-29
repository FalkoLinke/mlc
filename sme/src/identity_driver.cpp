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





int main() {
    float a[] = {
        0, 1, 2, 3,
        4, 5, 6, 7,
        8, 9, 10, 11,
        12, 13, 14, 15
    };
    float b[16] = {0.0f};

    identity_4_4(a, b, 4 * sizeof(float), 4 * sizeof(float), 1);
    print_mat(b, 4, 4);

    return 0;
}
#include <iostream>

#include "Kernel.h"
#include "KernelFactory.h"
#include "Unary.h"

using mini_jit::Kernel;
using mini_jit::KernelFactory;
using mini_jit::Unary;




int main() {
    Unary unary;
    unary.generate(16, 16, 0, Unary::dtype_t::fp32, Unary::ptype_t::identity);
    unary.write("test.bin");
    Unary::kernel_t kernel = unary.get_kernel();

    float a[16*16];
    float b[16*16];
    int64_t lda = 16;
    int64_t ldb = 16;
    for (int i = 0; i < 16*16; i++) {
        a[i] = /*(i % 2 == 0 ? -1.0 : 1.0) * */(float)i;
    }

    kernel(a, b, lda, ldb);

    for (int r = 0; r < 16; r++) {
        for (int c = 0; c < 16; c++) {
            std::cout << b[c * 16 + r] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
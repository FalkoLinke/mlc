#include <iostream>

#include "Kernel.h"
#include "KernelFactory.h"

using mini_jit::Kernel;
using mini_jit::KernelFactory;



int main() {
    Kernel kernel;
    KernelFactory factory;
    KernelFactory::ReluKernel relu_16_16 = factory.generate_relu_16_16(kernel);

    float a[16*16];
    float b[16*16];
    for (int i = 0; i < 16*16; i++) {
        a[i] = (i % 2 == 0 ? -1.0 : 1.0) * (float)i;
    }

    relu_16_16(a, b, 16, 16, 1);

    for (int r = 0; r < 16; r++) {
        for (int c = 0; c < 16; c++) {
            std::cout << b[c * 16 + r] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
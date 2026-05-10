#include <iostream>

#include "Kernel.h"
#include "KernelFactory.h"

using mini_jit::Kernel;
using mini_jit::KernelFactory;



int main() {
    Kernel kernel;
    KernelFactory factory;
    KernelFactory::ZeroKernel zero_16_16 = factory.generate_zero_16_16(kernel);
    kernel.write("test.bin");

    float a[16*16];
    float b[16*16];
    for (int i = 0; i < 16*16; i++) {
        b[i] = (i % 2 == 0 ? -1.0 : 1.0) * (float)i;
    }

    zero_16_16(b, 16);

    for (int r = 0; r < 16; r++) {
        for (int c = 0; c < 16; c++) {
            std::cout << b[c * 16 + r] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
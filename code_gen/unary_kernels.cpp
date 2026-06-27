#include "unary_kernels.h"




static mini_jit::Kernel kernel_identity_16_16;
static mini_jit::Kernel kernel_zero_16_16;
static mini_jit::Kernel kernel_relu_16_16;

mini_jit::KernelFactory::IdentityKernel identity_16_16;
mini_jit::KernelFactory::ZeroKernel zero_16_16;
mini_jit::KernelFactory::ReluKernel relu_16_16;


void generate_kernels() {
    mini_jit::KernelFactory kf;

    identity_16_16 = kf.generate_identity_16_16(kernel_identity_16_16);
    zero_16_16 = kf.generate_zero_16_16(kernel_zero_16_16);
    relu_16_16 = kf.generate_relu_16_16(kernel_relu_16_16);
}

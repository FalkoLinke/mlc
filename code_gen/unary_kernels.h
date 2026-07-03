#ifndef CODEGEN_UNARY_KERNELS_H
#define CODEGEN_UNARY_KERNELS_H


#include "Kernel.h"
#include "KernelFactory.h"


extern mini_jit::KernelFactory::IdentityKernel identity_16_16;
extern mini_jit::KernelFactory::ZeroKernel zero_16_16;
extern mini_jit::KernelFactory::ReluKernel relu_16_16;



void generate_kernels();


#endif /*CODEGEN_UNARY_KERNELS_H*/
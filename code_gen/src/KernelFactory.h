#ifndef KERNEL_FACTORY_H
#define KERNEL_FACTORY_H

#include <cstdint>
#include "Kernel.h"

namespace mini_jit {
    class KernelFactory;
}

class mini_jit::KernelFactory {
public:
    using IdentityKernel = void(*)(float const*, float*, int64_t, int64_t, int32_t);
    using ZeroKernel = void(*)(float*, int64_t);
    using ReluKernel = void(*)(float const*, float*, int64_t, int64_t, int32_t);


    IdentityKernel generate_identity_16_16(mini_jit::Kernel& kernel);
    ZeroKernel generate_zero_16_16(mini_jit::Kernel& kernel);
    ReluKernel generate_relu_16_16(mini_jit::Kernel& kernel);
};

#endif /*KERNEL_FACTORY_H*/
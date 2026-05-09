#include "KernelFactory.h"

using mini_jit::KernelFactory;
using mini_jit::Kernel;

KernelFactory::IdentityKernel KernelFactory::generate_identity_16_16(Kernel& kernel) {
    kernel.add_instr( 0xa9bf7bfd );
    kernel.add_instr( 0x910003fd );
    kernel.add_instr( 0xd503477f );
    kernel.add_instr( 0xb4000044 );
    kernel.add_instr( 0x1400000b );
    kernel.add_instr( 0x2598e3e0 );
    kernel.add_instr( 0xd2800205 );
    kernel.add_instr( 0xb40000e5 );
    kernel.add_instr( 0xa540a000 );
    kernel.add_instr( 0xe540e020 );
    kernel.add_instr( 0x8b020800 );
    kernel.add_instr( 0x8b030821 );
    kernel.add_instr( 0xf10004a5 );
    kernel.add_instr( 0x17fffffa );
    kernel.add_instr( 0x1400001e );
    kernel.add_instr( 0x2598e3e0 );
    kernel.add_instr( 0x5280000c );
    kernel.add_instr( 0xd2800085 );
    kernel.add_instr( 0xb4000185 );
    kernel.add_instr( 0xe09f0000 );
    kernel.add_instr( 0x8b020800 );
    kernel.add_instr( 0xe09f0001 );
    kernel.add_instr( 0x8b020800 );
    kernel.add_instr( 0xe09f0002 );
    kernel.add_instr( 0x8b020800 );
    kernel.add_instr( 0xe09f0003 );
    kernel.add_instr( 0x8b020800 );
    kernel.add_instr( 0x1100118c );
    kernel.add_instr( 0xf10004a5 );
    kernel.add_instr( 0x17fffff5 );
    kernel.add_instr( 0x5280000c );
    kernel.add_instr( 0xd2800085 );
    kernel.add_instr( 0xb4000185 );
    kernel.add_instr( 0xe0bf8020 );
    kernel.add_instr( 0x8b030821 );
    kernel.add_instr( 0xe0bf8021 );
    kernel.add_instr( 0x8b030821 );
    kernel.add_instr( 0xe0bf8022 );
    kernel.add_instr( 0x8b030821 );
    kernel.add_instr( 0xe0bf8023 );
    kernel.add_instr( 0x8b030821 );
    kernel.add_instr( 0x1100118c );
    kernel.add_instr( 0xf10004a5 );
    kernel.add_instr( 0x17fffff5 );
    kernel.add_instr( 0xd503467f );
    kernel.add_instr( 0xa8c17bfd );
    kernel.add_instr( 0xd65f03c0 );

    kernel.set_kernel();

    IdentityKernel result = (IdentityKernel)kernel.get_kernel();
    return result;
}




KernelFactory::ZeroKernel KernelFactory::generate_zero_16_16(Kernel& kernel) {
    kernel.add_instr( 0xd503477f );
    kernel.add_instr( 0xc00800ff );
    kernel.add_instr( 0x5280000c );
    kernel.add_instr( 0x2598e3e0 );
    kernel.add_instr( 0xd2800202 );
    kernel.add_instr( 0xb40000a2 );
    kernel.add_instr( 0xe0bf0000 );
    kernel.add_instr( 0x8b010800 );
    kernel.add_instr( 0xf1000442 );
    kernel.add_instr( 0x17fffffc );
    kernel.add_instr( 0xd503467f );
    kernel.add_instr( 0xd65f03c0 );

    kernel.set_kernel();
    ZeroKernel result = (ZeroKernel)kernel.get_kernel();
    return result;
}


KernelFactory::ReluKernel KernelFactory::generate_relu_16_16(Kernel& kernel) {
    kernel.add_instr( 0xa9bf7bfd );
    kernel.add_instr( 0x910003fd );
    kernel.add_instr( 0xd503477f );
    kernel.add_instr( 0xb4000044 );
    kernel.add_instr( 0x1400000c );
    kernel.add_instr( 0x2598e3e0 );
    kernel.add_instr( 0xd2800205 );
    kernel.add_instr( 0xb4000105 );
    kernel.add_instr( 0xa540a000 );
    kernel.add_instr( 0x659e8000 );
    kernel.add_instr( 0xe540e020 );
    kernel.add_instr( 0x8b020800 );
    kernel.add_instr( 0x8b030821 );
    kernel.add_instr( 0xf10004a5 );
    kernel.add_instr( 0x17fffff9 );
    kernel.add_instr( 0x14000026 );
    kernel.add_instr( 0x2598e3e0 );
    kernel.add_instr( 0x5280000c );
    kernel.add_instr( 0xd2800085 );
    kernel.add_instr( 0xb4000285 );
    kernel.add_instr( 0xa540a000 );
    kernel.add_instr( 0x8b020800 );
    kernel.add_instr( 0x659e8000 );
    kernel.add_instr( 0xc0800000 );
    kernel.add_instr( 0xa540a001 );
    kernel.add_instr( 0x8b020800 );
    kernel.add_instr( 0x659e8001 );
    kernel.add_instr( 0xc0800021 );
    kernel.add_instr( 0xa540a002 );
    kernel.add_instr( 0x8b020800 );
    kernel.add_instr( 0x659e8002 );
    kernel.add_instr( 0xc0800042 );
    kernel.add_instr( 0xa540a003 );
    kernel.add_instr( 0x8b020800 );
    kernel.add_instr( 0x659e8003 );
    kernel.add_instr( 0xc0800063 );
    kernel.add_instr( 0x1100118c );
    kernel.add_instr( 0xf10004a5 );
    kernel.add_instr( 0x17ffffed );
    kernel.add_instr( 0x5280000c );
    kernel.add_instr( 0xd2800085 );
    kernel.add_instr( 0xb4000185 );
    kernel.add_instr( 0xe0bf8020 );
    kernel.add_instr( 0x8b030821 );
    kernel.add_instr( 0xe0bf8021 );
    kernel.add_instr( 0x8b030821 );
    kernel.add_instr( 0xe0bf8022 );
    kernel.add_instr( 0x8b030821 );
    kernel.add_instr( 0xe0bf8023 );
    kernel.add_instr( 0x8b030821 );
    kernel.add_instr( 0x1100118c );
    kernel.add_instr( 0xf10004a5 );
    kernel.add_instr( 0x17fffff5 );
    kernel.add_instr( 0xd503467f );
    kernel.add_instr( 0xa8c17bfd );
    kernel.add_instr( 0xd65f03c0 );

    kernel.set_kernel();
    ReluKernel result = (ReluKernel)kernel.get_kernel();
    return result;
}
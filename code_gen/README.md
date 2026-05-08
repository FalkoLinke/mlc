# Code Generation

## Kernel creation

Implement your SSVE and SME kernels using just-in-time code generation.

### Tasks

   1. Convert the mnemonics of your `identity_16_16`, `zero_16_16`, `relu_16_16`, and `gemm_512_512_512` kernels to instruction words.
   2. Verify and benchmark your kernels.

## Code Generation

The provided header files [Unary.h](data/Unary.h) and [Gemm.h](data/Gemm.h) specify the interface for the kernel generator. The `generate` function generates a kernel, and `get_kernel` returns a function pointer to the kernel.

### Unary Primitives

The unfinished unary code generator uses the C++ namespace `mini_jit::Unary`. 

#### Tasks

1. Implement the `generate` function to instantiate the unary primitives using SSVE instructions. The code generator should support multiples of 16 as sizes for M and N.
2. Test and benchmark representative kernels. Include at least settings with dimension sizes 64, 128, and 512, that is, 9 settings in total. Report performance in GiB/s.

### GEMM Primitive

The unfinished GEMM code generator uses the C++ namespace `mini_jit::Gemm`.

#### Tasks

1. Implement the `generate` function to instantiate the GEMM primitive using SME instructions. The code generator should support multiples of 16 as sizes for M and N and an arbitrary size for K.
2. Test the kernel generation. Include at least settings with dimension sizes 64, 128, and 512, that is, 27 settings in total. Report performance in GFLOPS.

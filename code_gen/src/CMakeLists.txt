cmake_minimum_required(VERSION 4.0)


Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.8.1 # or a later release
)

FetchContent_MakeAvailable(Catch2)


set(CXX_FLAGS -Wall -Wextra -march=armv9.2-a+nosve+sme -O3)


add_executable(kernel_examples.out kernel_examples.cpp Kernel.cpp InstGen.cpp LabeledBranch.cpp BranchRef.cpp)
target_compile_options(kernel_examples.out PRIVATE ${CXX_FLAGS})

add_executable(instgen_examples.out instgen_examples.cpp InstGen.cpp LabeledBranch.cpp)
target_compile_options(instgen_examples.out PRIVATE ${CXX_FLAGS})

add_executable(main.out main.cpp Kernel.cpp KernelFactory.cpp InstGen.cpp LabeledBranch.cpp BranchRef.cpp Unary.cpp)
target_compile_options(main.out PRIVATE ${CXX_FLAGS})

add_executable(unary_tests.out unary_tests.cpp Kernel.cpp KernelFactory.cpp unary_kernels.cpp InstGen.cpp LabeledBranch.cpp BranchRef.cpp)
target_compile_options(unary_tests.out PRIVATE ${CXX_FLAGS})
target_link_libraries(unary_tests.out PRIVATE Catch2::Catch2)

add_executable(unary_benchmarks_driver.out unary_benchmarks_driver.cpp Kernel.cpp KernelFactory.cpp unary_kernels.cpp InstGen.cpp LabeledBranch.cpp BranchRef.cpp)
target_compile_options(unary_benchmarks_driver.out PRIVATE ${CXX_FLAGS})


add_executable(gemm_driver_512_512_512.out 
    gemm_driver_512_512_512.cpp 
    gemm_kernel_512_512_512.cpp 
    Kernel.cpp 
    InstGen.cpp 
    LabeledBranch.cpp 
    BranchRef.cpp
)
target_compile_options(gemm_driver_512_512_512.out PRIVATE ${CXX_FLAGS})


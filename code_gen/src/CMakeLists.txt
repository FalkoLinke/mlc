cmake_minimum_required(VERSION 4.0)


Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.8.1 # or a later release
)

FetchContent_MakeAvailable(Catch2)




add_executable(kernel_examples.out kernel_examples.cpp Kernel.cpp InstGen.cpp LabeledBranch.cpp BranchRef.cpp)

add_executable(instgen_examples.out instgen_examples.cpp InstGen.cpp LabeledBranch.cpp)

add_executable(main.out main.cpp Kernel.cpp KernelFactory.cpp InstGen.cpp LabeledBranch.cpp BranchRef.cpp)

add_executable(unary_tests.out unary_tests.cpp Kernel.cpp KernelFactory.cpp unary_kernels.cpp InstGen.cpp LabeledBranch.cpp BranchRef.cpp)
target_link_libraries(unary_tests.out PRIVATE Catch2::Catch2)

add_executable(unary_benchmarks_driver.out unary_benchmarks_driver.cpp Kernel.cpp KernelFactory.cpp unary_kernels.cpp InstGen.cpp LabeledBranch.cpp BranchRef.cpp)



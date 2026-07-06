cmake_minimum_required(VERSION 4.0)
project(CODE_GEN C ASM CXX)




set(CXX_FLAGS -Wall -Wextra -march=armv9.2-a+nosve+sme+sme2 -O3)






add_library(code_gen Kernel.cpp InstGen.cpp Gemm.cpp Unary.cpp InsRef.cpp LabeledInstruction.cpp)
target_compile_options(code_gen PRIVATE ${CXX_FLAGS})
target_include_directories(code_gen PUBLIC .)








add_executable(kernel_examples.out kernel_examples.cpp)
target_compile_options(kernel_examples.out PRIVATE ${CXX_FLAGS})
target_link_libraries(kernel_examples.out PRIVATE code_gen mlc_common)

add_executable(instgen_examples.out instgen_examples.cpp)
target_compile_options(instgen_examples.out PRIVATE ${CXX_FLAGS})
target_link_libraries(instgen_examples.out PRIVATE code_gen mlc_common)





add_executable(main.out main.cpp KernelFactory.cpp)
target_compile_options(main.out PRIVATE ${CXX_FLAGS})
target_link_libraries(main.out PRIVATE code_gen mlc_common)

add_executable(unary_tests.out unary_tests.cpp KernelFactory.cpp unary_kernels.cpp)
target_compile_options(unary_tests.out PRIVATE ${CXX_FLAGS})
target_link_libraries(unary_tests.out PRIVATE Catch2::Catch2 code_gen mlc_common)

add_executable(unary_benchmarks_driver.out unary_benchmarks_driver.cpp KernelFactory.cpp unary_kernels.cpp)
target_compile_options(unary_benchmarks_driver.out PRIVATE ${CXX_FLAGS})
target_link_libraries(unary_benchmarks_driver.out PRIVATE code_gen mlc_common)

add_executable(unary_jit_benchmarks_driver.out unary_jit_benchmarks_driver.cpp KernelFactory.cpp unary_kernels.cpp)
target_compile_options(unary_jit_benchmarks_driver.out PRIVATE ${CXX_FLAGS})
target_link_libraries(unary_jit_benchmarks_driver.out PRIVATE code_gen mlc_common)








add_executable(gemm_driver_512_512_512.out gemm_driver_512_512_512.cpp gemm_kernel_512_512_512.cpp)
target_compile_options(gemm_driver_512_512_512.out PRIVATE ${CXX_FLAGS})
target_link_libraries(gemm_driver_512_512_512.out PRIVATE code_gen mlc_common)

add_executable(gemm_driver_M_N_K.out gemm_driver_M_N_K.cpp)
target_compile_options(gemm_driver_M_N_K.out PRIVATE ${CXX_FLAGS})
target_link_libraries(gemm_driver_M_N_K.out PRIVATE code_gen mlc_common)

add_executable(gemm_tests.out gemm_tests.cpp)
target_compile_options(gemm_tests.out PRIVATE ${CXX_FLAGS})
target_link_libraries(gemm_tests.out PRIVATE code_gen mlc_common Catch2::Catch2WithMain)

add_executable(gemm_benchmarks_driver.out gemm_benchmarks_driver.cpp)
target_compile_options(gemm_benchmarks_driver.out PRIVATE ${CXX_FLAGS})
target_link_libraries(gemm_benchmarks_driver.out PRIVATE code_gen mlc_common)




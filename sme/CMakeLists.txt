cmake_minimum_required(VERSION 3.10)
project(SME ASM C CXX)




set(CXX_FLAGS -Wall -Wextra -O3 -march=armv9.2-a+nosve+sme+sme2)



set(KERNELS 
    gemm_16_16_kernel.s
    gemm_16_16_Rest.s
    gemm_32_32_1_kernel_old.s
    gemm_32_32_1_kernel.s
    gemm_32_32_512_kernel.s
    gemm_512_32_512_kernel.s
    gemm_512_512_512_kernel_first.s
    gemm_512_512_512_kernel.s
    gemm_M_N_K_16th_kernel.s
    identity_kernel.s
    relu_kernel.s
    zero_kernel.s
)
set_source_files_properties(${KERNELS} PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")






add_executable(identity_driver identity_driver.cpp identity_kernel.s)
target_compile_options(identity_driver PRIVATE ${CXX_FLAGS})

add_executable(relu_driver relu_driver.cpp relu_kernel.s)
target_compile_options(relu_driver PRIVATE ${CXX_FLAGS})

add_executable(zero_driver zero_driver.cpp zero_kernel.s)
target_compile_options(zero_driver PRIVATE ${CXX_FLAGS})



add_executable(unary_benchmarks_driver unary_benchmarks_driver.cpp identity_kernel.s relu_kernel.s zero_kernel.s)
target_compile_options(unary_benchmarks_driver PRIVATE ${CXX_FLAGS})

add_executable(unary_tests unary_tests.cpp identity_kernel.s relu_kernel.s zero_kernel.s)
target_compile_options(unary_tests PRIVATE ${CXX_FLAGS})





if(OpenMP_FOUND)
    add_executable(gemm_driver_16_16_multi gemm_driver_16_16_multi.cpp gemm_16_16_kernel.s)
    target_compile_options(gemm_driver_16_16_multi PRIVATE ${CXX_FLAGS})
    target_link_libraries(gemm_driver_16_16_multi PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(gemm_driver_16_16 gemm_driver_16_16.cpp gemm_16_16_kernel.s)
target_compile_options(gemm_driver_16_16 PRIVATE ${CXX_FLAGS})

add_executable(gemm_driver_32_32_1 gemm_driver_32_32_1.cpp gemm_32_32_1_kernel.s)
target_compile_options(gemm_driver_32_32_1 PRIVATE ${CXX_FLAGS})

add_executable(gemm_driver_32_32_512 gemm_driver_32_32_512.cpp gemm_32_32_512_kernel.s)
target_compile_options(gemm_driver_32_32_512 PRIVATE ${CXX_FLAGS})

add_executable(gemm_driver_512_32_512 gemm_driver_512_32_512.cpp gemm_512_32_512_kernel.s)
target_compile_options(gemm_driver_512_32_512 PRIVATE ${CXX_FLAGS})

add_executable(gemm_driver_512_512_512 gemm_driver_512_512_512.cpp gemm_512_512_512_kernel.s)
target_compile_options(gemm_driver_512_512_512 PRIVATE ${CXX_FLAGS})

add_executable(gemm_driver_M_N_K_16th gemm_driver_M_N_K_16th.cpp gemm_M_N_K_16th_kernel.s)
target_compile_options(gemm_driver_M_N_K_16th PRIVATE ${CXX_FLAGS})











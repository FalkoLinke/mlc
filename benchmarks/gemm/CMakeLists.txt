cmake_minimum_required(VERSION 3.10)
project(GEMM C ASM CXX)





set(CXX_FLAGS -Wall -Wextra -march=armv9.2-a+nosve+sme+sme2 -O3)



add_executable(main main.cpp gemm_km_kn_nm_fp32_m16_n32_k1.s gemm_km_kn_nm_fp32_m16_n32_k512.s)
target_compile_options(main PRIVATE ${CXX_FLAGS})
set_property(SOURCE gemm_km_kn_nm_fp32_m16_n32_k1.s gemm_km_kn_nm_fp32_m16_n32_k512.s PROPERTY COMPILE_FLAGS "-x assembler-with-cpp")




add_executable(benchmarks_driver benchmarks_driver.cpp gemm_km_kn_nm_fp32_m16_n32_k1.s gemm_km_kn_nm_fp32_m16_n32_k512.s)
target_compile_options(benchmarks_driver PRIVATE ${CXX_FLAGS})







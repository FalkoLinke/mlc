cmake_minimum_required(VERSION 3.10)
project(GEMM C ASM CXX)





set(CXX_FLAGS -Wall -Wextra -march=armv9.2-a+nosve+sme+sme2 -O3)



set(KERNELS 
    gemm_km_kn_nm_fp32_m16_n32_k1.s 
    gemm_km_kn_nm_fp32_m16_n32_k16.s 
    gemm_km_kn_nm_fp32_m16_n16_k512.s 
    gemm_km_kn_nm_fp32_m16_n32_k512.s 
    gemm_km_kn_nm_fp32_m16_n32_k512_v2.s
    gemm_km_kn_nm_fp32_m16_n64_k512.s
    gemm_mk_kn_nm_fp32_m16_n32_k16_za.s
    gemm_mk_kn_nm_fp32_m16_n32_k512_za.s
    gemm_mk_kn_nm_fp32_m16_n32_k512_tbl.s
    gemm_mk_kn_nm_fp32_m16_n32_k512_tbl_v2.s
    gemm_mk_nk_nm_fp32_m16_n16_k512_tbl_stack.s
    gemm_mk_nk_nm_fp32_m16_n16_k512_tbl_regs.s
    gemm_16_16_multiple_k.s
    gemm_16_16_multiple_k_v2.s
    gemm_16_16_ref_single_k.s
    gemm_16_16_trSME_mk.s
    gemm_16_16_trZA_mk.s
    gemm_512_512_trZA.s
    gemm_512_512_trSME.s
)
set_source_files_properties(${KERNELS} PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")


add_executable(gemm_main main.cpp ${KERNELS})
target_compile_options(gemm_main PRIVATE ${CXX_FLAGS})

add_executable(gemm_benchmarks_driver benchmarks_driver.cpp ${KERNELS})
target_compile_options(gemm_benchmarks_driver PRIVATE ${CXX_FLAGS})

add_executable(gemm_benchmarks_driver_16_16 benchmarks_driver_16_16.cpp ${KERNELS})
target_compile_options(gemm_benchmarks_driver_16_16 PRIVATE ${CXX_FLAGS})







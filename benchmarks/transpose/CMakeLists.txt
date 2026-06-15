cmake_minimum_required(VERSION 3.10)
project(TRANSPOSE C ASM CXX)





set(CXX_FLAGS -Wall -Wextra -march=armv9.2-a+nosve+sme+sme2 -O3)



add_executable(main main.cpp transpose_za.s transpose_tbl.s transpose_simd.s copy_sve.s)
target_compile_options(main PRIVATE ${CXX_FLAGS})
set_property(SOURCE transpose_za.s transpose_tbl.s transpose_simd.s copy_sve.s PROPERTY COMPILE_FLAGS "-x assembler-with-cpp")




add_executable(benchmarks_driver benchmarks_driver.cpp transpose_za.s transpose_tbl.s transpose_simd.s copy_sve.s)
target_compile_options(benchmarks_driver PRIVATE ${CXX_FLAGS})







cmake_minimum_required(VERSION 3.10)
project(TRANSPOSE C ASM CXX)





set(CXX_FLAGS -Wall -Wextra -march=armv9.2-a+nosve+sme+sme2 -O3)



add_executable(main main.cpp transpose_za.s transpose_tbl.s)
target_compile_options(main PRIVATE ${CXX_FLAGS})
set_property(SOURCE transpose_za.s transpose_tbl.s PROPERTY COMPILE_FLAGS "-x assembler-with-cpp")









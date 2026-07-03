cmake_minimum_required(VERSION 3.10)
project(COMMON ASM C CXX)




set(CXX_FLAGS -Wall -Wextra -O3 -march=armv9.2-a+nosve+sme+sme2)


add_library(mlc_common INTERFACE)
target_include_directories(mlc_common INTERFACE .)




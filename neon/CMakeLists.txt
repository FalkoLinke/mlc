cmake_minimum_required(VERSION 3.10)
project(NEON ASM C CXX)




set(CXX_FLAGS -Wall -Wextra -O3)



set(KERNELS 
    benchmark_kernel.s
    permutation_kernel.s
)
set_source_files_properties(${KERNELS} PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")



add_executable(benchmark_driver benchmark_driver.cpp benchmark_kernel.s)
target_compile_options(benchmark_driver PRIVATE ${CXX_FLAGS})

# add_executable(permutation_driver permutation_driver.cpp permutation_kernel.s)
# target_compile_options(permutation_driver PRIVATE ${CXX_FLAGS})









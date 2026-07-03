cmake_minimum_required(VERSION 3.10)
project(ASSEMBLY ASM C CXX)




set(CXX_FLAGS -Wall -Wextra -O3)



set(KERNELS 
    base_math.s
)
set_source_files_properties(${KERNELS} PROPERTIES COMPILE_FLAGS "-x assembler-with-cpp")



add_executable(base_math_driver base_math_driver.cpp base_math.cpp ${KERNELS})
target_compile_options(base_math_driver PRIVATE ${CXX_FLAGS})
target_link_libraries(base_math_driver PRIVATE Catch2::Catch2WithMain)







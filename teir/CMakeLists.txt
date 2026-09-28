cmake_minimum_required(VERSION 4.0)
project(TEIR C ASM CXX)



set(CXX_FLAGS -Wall -Wextra -march=armv9.2-a+nosve+sme+sme2 -O2)


# runtime: interpreter and compiler
add_library(teir_runtime
  teir.cpp
  teir_lowering.cpp
  teir_interpreter.cpp
  teir_compiler.cpp
  UnaryCache.cpp
  GemmCache.cpp
)
target_compile_options(teir_runtime PRIVATE ${CXX_FLAGS})
target_include_directories(teir_runtime PUBLIC .)
target_link_libraries(teir_runtime PUBLIC code_gen mlc_common)
if(OpenMP_CXX_FOUND)
  target_link_libraries(teir_runtime PUBLIC OpenMP::OpenMP_CXX)
else()
  message(WARNING "OpenMP not found: parallel TEIR iteration nodes are executed sequentially.")
endif()



add_executable(teir_interpreter_tests.out teir_interpreter_tests.cpp)
target_compile_options(teir_interpreter_tests.out PRIVATE ${CXX_FLAGS})
target_link_libraries(teir_interpreter_tests.out PRIVATE Catch2::Catch2WithMain teir_runtime)
mlc_add_sme_tests(teir_interpreter_tests.out)

add_executable(teir_compiler_tests.out teir_compiler_tests.cpp)
target_compile_options(teir_compiler_tests.out PRIVATE ${CXX_FLAGS})
target_link_libraries(teir_compiler_tests.out PRIVATE Catch2::Catch2WithMain teir_runtime)
mlc_add_sme_tests(teir_compiler_tests.out)

add_executable(teir_runtime_tests.out teir_runtime_tests.cpp)
target_compile_options(teir_runtime_tests.out PRIVATE ${CXX_FLAGS})
target_link_libraries(teir_runtime_tests.out PRIVATE Catch2::Catch2WithMain teir_runtime)
mlc_add_sme_tests(teir_runtime_tests.out)



add_executable(teir_benchmarks.out teir_benchmarks.cpp)
target_compile_options(teir_benchmarks.out PRIVATE ${CXX_FLAGS})
target_link_libraries(teir_benchmarks.out PRIVATE teir_runtime)

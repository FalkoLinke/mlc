cmake_minimum_required(VERSION 4.0)
project(TEIR C ASM CXX)



set(CXX_FLAGS -Wall -Wextra -march=armv9.2-a+nosve+sme+sme2 -O0)


add_executable(teir_interpreter_tests.out 
  teir_interpreter_tests.cpp 
  teir.cpp 
  teir_interpreter.cpp 
  UnaryCache.cpp
  GemmCache.cpp
)
target_compile_options(teir_interpreter_tests.out PRIVATE ${CXX_FLAGS})
target_link_libraries(teir_interpreter_tests.out PRIVATE Catch2::Catch2WithMain code_gen mlc_common)



add_executable(teir_compiler_tests.out
  teir_compiler_tests.cpp
  teir.cpp
  teir_compiler.cpp
  UnaryCache.cpp
  GemmCache.cpp
)
target_compile_options(teir_compiler_tests.out PRIVATE ${CXX_FLAGS})
target_link_libraries(teir_compiler_tests.out PRIVATE Catch2::Catch2WithMain code_gen mlc_common)



add_executable(benchmark_contraction.out
  benchmark_contraction.cpp
  teir_interpreter.cpp 
  teir.cpp
  teir_compiler.cpp
  UnaryCache.cpp
  GemmCache.cpp
)
target_compile_options(benchmark_contraction.out PRIVATE ${CXX_FLAGS})
target_link_libraries(benchmark_contraction.out PRIVATE code_gen mlc_common)

add_executable(benchmark_matmul.out
  benchmark_matmul.cpp
  teir_interpreter.cpp 
  teir.cpp
  teir_compiler.cpp
  UnaryCache.cpp
  GemmCache.cpp
)
target_compile_options(benchmark_matmul.out PRIVATE ${CXX_FLAGS})
target_link_libraries(benchmark_matmul.out PRIVATE code_gen mlc_common)

add_executable(benchmark_transposition.out
  benchmark_transposition.cpp
  teir_interpreter.cpp 
  teir.cpp
  teir_compiler.cpp
  UnaryCache.cpp
  GemmCache.cpp
)
target_compile_options(benchmark_transposition.out PRIVATE ${CXX_FLAGS})
target_link_libraries(benchmark_transposition.out PRIVATE code_gen mlc_common)




cmake_minimum_required(VERSION 4.0)
project(TEIR)


Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.8.1 # or a later release
)

FetchContent_MakeAvailable(Catch2)


set(CXX_FLAGS -Wall -Wextra -march=armv9.2-a+nosve+sme+sme2 -O0)


add_executable(teir_interpreter_tests.out 
  teir_interpreter_tests.cpp 
  teir.cpp 
  teir_interpreter.cpp 
  InsRef.cpp 
  Gemm.cpp 
  InstGen.cpp 
  Kernel.cpp 
  LabeledInstruction.cpp
  Unary.cpp
  UnaryCache.cpp
  GemmCache.cpp
)
target_compile_options(teir_interpreter_tests.out PRIVATE ${CXX_FLAGS})
target_link_libraries(teir_interpreter_tests.out PRIVATE Catch2::Catch2WithMain)



add_executable(teir_compiler_tests.out
  teir_compiler_tests.cpp
  teir.cpp
  teir_compiler.cpp
  InsRef.cpp
  Gemm.cpp
  InstGen.cpp
  Kernel.cpp
  LabeledInstruction.cpp
  Unary.cpp
  UnaryCache.cpp
  GemmCache.cpp
)
target_compile_options(teir_compiler_tests.out PRIVATE ${CXX_FLAGS})
target_link_libraries(teir_compiler_tests.out PRIVATE Catch2::Catch2WithMain)



add_executable(benchmark_contraction.out
  benchmark_contraction.cpp
  teir_interpreter.cpp 
  teir.cpp
  teir_compiler.cpp
  InsRef.cpp
  Gemm.cpp
  InstGen.cpp
  Kernel.cpp
  LabeledInstruction.cpp
  Unary.cpp
  UnaryCache.cpp
  GemmCache.cpp
)
target_compile_options(benchmark_contraction.out PRIVATE ${CXX_FLAGS})

add_executable(benchmark_matmul.out
  benchmark_matmul.cpp
  teir_interpreter.cpp 
  teir.cpp
  teir_compiler.cpp
  InsRef.cpp
  Gemm.cpp
  InstGen.cpp
  Kernel.cpp
  LabeledInstruction.cpp
  Unary.cpp
  UnaryCache.cpp
  GemmCache.cpp
)
target_compile_options(benchmark_matmul.out PRIVATE ${CXX_FLAGS})

add_executable(benchmark_transposition.out
  benchmark_transposition.cpp
  teir_interpreter.cpp 
  teir.cpp
  teir_compiler.cpp
  InsRef.cpp
  Gemm.cpp
  InstGen.cpp
  Kernel.cpp
  LabeledInstruction.cpp
  Unary.cpp
  UnaryCache.cpp
  GemmCache.cpp
)
target_compile_options(benchmark_transposition.out PRIVATE ${CXX_FLAGS})




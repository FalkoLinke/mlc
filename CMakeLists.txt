cmake_minimum_required(VERSION 3.10)
project(MLC C ASM CXX)


# prepare Catch2
Include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.8.1 # or a later release
)
FetchContent_MakeAvailable(Catch2)


# prepare OpenMP
find_package(OpenMP)




add_subdirectory(common)
add_subdirectory(assembly)
add_subdirectory(neon)
add_subdirectory(sme)
add_subdirectory(code_gen)
add_subdirectory(teir)
add_subdirectory(benchmarks/gemm)
add_subdirectory(benchmarks/transpose)
add_subdirectory(benchmarks/transpose_kLoop)


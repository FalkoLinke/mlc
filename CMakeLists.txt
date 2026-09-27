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


# prepare tests
enable_testing()
list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
include(Catch)


# SME kernels and JIT generated code can only be tested on processors supporting SME
set(MLC_HOST_HAS_SME OFF)
if(APPLE)
  execute_process(COMMAND sysctl -n hw.optional.arm.FEAT_SME
                  OUTPUT_VARIABLE MLC_SYSCTL_SME
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET)
  if(MLC_SYSCTL_SME STREQUAL "1")
    set(MLC_HOST_HAS_SME ON)
  endif()
elseif(EXISTS /proc/cpuinfo)
  file(READ /proc/cpuinfo MLC_CPUINFO)
  if(MLC_CPUINFO MATCHES " sme ")
    set(MLC_HOST_HAS_SME ON)
  endif()
endif()
option(MLC_SME_TESTS "Register tests which execute SME instructions" ${MLC_HOST_HAS_SME})
if(NOT MLC_SME_TESTS)
  message(WARNING "Host does not support SME: SME and JIT tests are built but not registered with ctest.")
endif()

# registers the Catch2 test cases of `target` with ctest
function(mlc_add_tests target)
  catch_discover_tests(${target} DISCOVERY_MODE PRE_TEST TEST_PREFIX "${target}: ")
endfunction()

# registers the Catch2 test cases of `target` with ctest, if SME tests are enabled
function(mlc_add_sme_tests target)
  if(MLC_SME_TESTS)
    mlc_add_tests(${target})
  endif()
endfunction()


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


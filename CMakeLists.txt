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






add_subdirectory(assembly)
add_subdirectory(neon)



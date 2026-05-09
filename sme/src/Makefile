
CXXFLAGS_COMMON = -Wall -Wextra

TOOLCHAIN ?= standard 

ifeq ($(TOOLCHAIN),standard)

CXX = clang++
CXXFLAGS = $(CXXFLAGS_COMMON) -march=armv9.2-a+sme -O0
ASFLAGS = $(CXXFLAGS) -x assembler-with-cpp

else

ifeq ($(TOOLCHAIN),homebrew)

LLVM_PREFIX = /opt/homebrew/opt/llvm
CXX = $(LLVM_PREFIX)/bin/clang++
CC = $(LLVM_PREFIX)/bin/clang
CXXFLAGS = $(CXXFLAGS_COMMON) -fopenmp -march=armv9.2-a+sme -O0
ASFLAGS = $(CXXFLAGS) -x assembler-with-cpp

else

$(error Invalid toolchain. Select either "TOOLCHAIN=standard" or "TOOLCHAIN=homebrew")

endif

endif











all: identity_driver relu_driver zero_driver unary_benchmarks_driver unary_tests gemm_driver_32_32_1 gemm_driver_32_32_512 gemm_driver_16_16 gemm_driver_512_32_512 gemm_driver_512_512_512 gemm_driver_16_16_multi sve_stats


clean:
	rm -f identity_driver
	rm -f relu_driver
	rm -f gemm_driver_32_32_1
	rm -f gemm_driver_32_32_512
	rm -f gemm_driver_16_16
	rm -f gemm_driver_512_32_512
	rm -f gemm_driver_512_512_512
	rm -f gemm_driver_16_16_multi
	rm -f zero_driver
	rm -f unary_benchmarks_driver
	rm -f unary_tests
	rm -f sve_stats
	rm -f *.o


sve_stats: sve_stats.o
	$(CXX) $^ -o sve_stats

identity_driver: identity_driver.o identity_kernel.o
	$(CXX) $^ -o identity_driver

relu_driver: relu_driver.o relu_kernel.o identity_kernel.o
	$(CXX) $^ -o relu_driver

gemm_driver_32_32_1: gemm_driver_32_32_1.o gemm_kernel_32_32_1.o
	$(CXX) $^ -o gemm_driver_32_32_1

gemm_driver_32_32_512: gemm_driver_32_32_512.o gemm_kernel_32_32_512.o
	$(CXX) $^ -o gemm_driver_32_32_512

gemm_driver_512_32_512: gemm_driver_512_32_512.o gemm_kernel_512_32_512.o
	$(CXX) $^ -o gemm_driver_512_32_512

gemm_driver_512_512_512: gemm_driver_512_512_512.o gemm_kernel_512_512_512.o
	$(CXX) $^ -o gemm_driver_512_512_512
	
gemm_driver_16_16: gemm_driver_16_16.o gemm_kernel_16_16.o
	$(CXX) $^ -o gemm_driver_16_16

gemm_driver_16_16_multi: gemm_driver_16_16_multi.o gemm_kernel_16_16.o
	$(CXX) $^ -fopenmp  -o gemm_driver_16_16_multi

zero_driver: zero_driver.o zero_kernel.o
	$(CXX) $^ -o zero_driver

unary_benchmarks_driver: unary_benchmarks_driver.o identity_kernel.o zero_kernel.o relu_kernel.o
	$(CXX) $^ -o unary_benchmarks_driver

unary_tests: unary_tests.o identity_kernel.o zero_kernel.o relu_kernel.o
	$(CXX) $^ -o unary_tests

sve_stats.o: sve_stats.cpp
	$(CXX) -c $(CXXFLAGS) $^ -o sve_stats.o

identity_driver.o: identity_driver.cpp
	$(CXX) -c $(CXXFLAGS) $^ -o identity_driver.o

relu_driver.o: relu_driver.cpp
	$(CXX) -c $(CXXFLAGS) $^ -o relu_driver.o

gemm_driver_32_32_1.o: gemm_driver_32_32_1.cpp
	$(CXX) -c $(CXXFLAGS) $^ -o gemm_driver_32_32_1.o

gemm_driver_16_16.o: gemm_driver_16_16.cpp
	$(CXX) -c $(CXXFLAGS) $^ -o gemm_driver_16_16.o

gemm_driver_16_16_multi.o: gemm_driver_16_16_multi.cpp
	$(CXX) -c $(CXXFLAGS) $^ -o gemm_driver_16_16_multi.o

gemm_driver_32_32_512.o: gemm_driver_32_32_512.cpp
	$(CXX) -c $(CXXFLAGS) $^ -o gemm_driver_32_32_512.o

gemm_driver_512_32_512.o: gemm_driver_512_32_512.cpp
	$(CXX) -c $(CXXFLAGS) $^ -o gemm_driver_512_32_512.o

gemm_driver_512_512_512.o: gemm_driver_512_512_512.cpp
	$(CXX) -c $(CXXFLAGS) $^ -o gemm_driver_512_512_512.o


zero_driver.o: zero_driver.cpp
	$(CXX) -c $(CXXFLAGS) $^ -o zero_driver.o

unary_benchmarks_driver.o: unary_benchmarks_driver.cpp
	$(CXX) -c $(CXXFLAGS) $^ -o unary_benchmarks_driver.o

unary_tests.o: unary_tests.cpp
	$(CXX) -c $(CXXFLAGS) $^ -o unary_tests.o

relu_kernel.o: relu_kernel.s
	$(CXX) -c $(ASFLAGS) $^ -o relu_kernel.o

identity_kernel.o: identity_kernel.s
	$(CXX) -c $(ASFLAGS) $^ -o identity_kernel.o

gemm_kernel_32_32_1.o: gemm_32_32_1_kernel.s
	$(CXX) -c $(ASFLAGS) $^ -o gemm_kernel_32_32_1.o

gemm_kernel_16_16.o: gemm_16_16_kernel.s
	$(CXX) -c $(ASFLAGS) $^ -o gemm_kernel_16_16.o

gemm_kernel_32_32_512.o: gemm_32_32_512_kernel.s
	$(CXX) -c $(ASFLAGS) $^ -o gemm_kernel_32_32_512.o
zero_kernel.o: zero_kernel.s
	$(CXX) -c $(ASFLAGS) $^ -o zero_kernel.o

gemm_kernel_512_32_512.o: gemm_512_32_512_kernel.s
	$(CXX) -c $(ASFLAGS) $^ -o gemm_kernel_512_32_512.o

gemm_kernel_512_512_512.o: gemm_512_512_512_kernel.s
	$(CXX) -c $(ASFLAGS) $^ -o gemm_kernel_512_512_512.o

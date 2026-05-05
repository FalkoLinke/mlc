
CXXFLAGS_COMMON = -Wall -Wextra

TOOLCHAIN ?= standard 

ifeq ($(TOOLCHAIN),standard)

CXX = clang++
CXXFLAGS = $(CXXFLAGS_COMMON) -march=armv9.2-a+nosve+sme -O0
ASFLAGS = $(CXXFLAGS) -x assembler-with-cpp

else

ifeq ($(TOOLCHAIN),homebrew)

LLVM_PREFIX = /opt/homebrew/opt/llvm
CXX = $(LLVM_PREFIX)/bin/clang++
CC = $(LLVM_PREFIX)/bin/clang
CXXFLAGS = $(CXXFLAGS_COMMON) -march=armv9.2-a+nosve+sme -O3
ASFLAGS = $(CXXFLAGS) -x assembler-with-cpp

else

$(error Invalid toolchain. Select either "TOOLCHAIN=standard" or "TOOLCHAIN=homebrew")

endif

endif











all: identity_driver relu_driver gemm_driver_32_32_1 gemm_driver_32_32_512 gemm_driver_16_16



clean:
	rm -f identity_driver
	rm -f relu_driver
	rm -f gemm_driver_32_32_1
	rm -f gemm_driver_32_32_512
	rm -f gemm_driver_16_16
	rm -f *.o



identity_driver: identity_driver.o identity_kernel.o
	$(CXX) $^ -o identity_driver

relu_driver: relu_driver.o relu_kernel.o identity_kernel.o
	$(CXX) $^ -o relu_driver

gemm_driver_32_32_1: gemm_driver_32_32_1.o gemm_kernel_32_32_1.o identity_kernel.o
	$(CXX) $^ -o gemm_driver_32_32_1

gemm_driver_32_32_512: gemm_driver_32_32_512.o gemm_kernel_32_32_512.o identity_kernel.o
	$(CXX) $^ -o gemm_driver_32_32_512
	
gemm_driver_16_16: gemm_driver_16_16.o gemm_kernel_16_16.o identity_kernel.o
	$(CXX) $^ -o gemm_driver_16_16
	


identity_driver.o: identity_driver.cpp
	$(CXX) -c $(CXXFLAGS) $^ -o identity_driver.o

relu_driver.o: relu_driver.cpp
	$(CXX) -c $(CXXFLAGS) $^ -o relu_driver.o

gemm_driver_32_32_1.o: gemm_driver_32_32_1.cpp
	$(CXX) -c $(CXXFLAGS) $^ -o gemm_driver_32_32_1.o

gemm_driver_16_16.o: gemm_driver_16_16.cpp
	$(CXX) -c $(CXXFLAGS) $^ -o gemm_driver_16_16.o

gemm_driver_32_32_512.o: gemm_driver_32_32_512.cpp
	$(CXX) -c $(CXXFLAGS) $^ -o gemm_driver_32_32_512.o



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

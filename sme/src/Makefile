
CXXFLAGS_COMMON = -Wall -Wextra

TOOLCHAIN ?= gnu

ifeq ($(TOOLCHAIN),gnu)

CXX = g++
CXXFLAGS = $(CXXFLAGS_COMMON) -march=armv9-a+sve -O0
ASFLAGS = $(CXXFLAGS) -x assembler-with-cpp

else

ifeq ($(TOOLCHAIN),llvm)

LLVM_PREFIX = /opt/homebrew/opt/llvm
CXX = $(LLVM_PREFIX)/bin/clang++
CC = $(LLVM_PREFIX)/bin/clang
CXXFLAGS = $(CXXFLAGS_COMMON) -march=armv9.2-a+nosve+sme -O3
ASFLAGS = $(CXXFLAGS) -x assembler-with-cpp

else

$(error Invalid toolchain. Select either "TOOLCHAIN=gnu" or "TOOLCHAIN=llvm")

endif

endif











all: identity_driver relu_driver



clean:
	rm -f identity_driver
	rm -f relu_driver
	rm -f *.o



identity_driver: identity_driver.o identity_kernel.o
	$(CXX) $^ -o identity_driver

relu_driver: relu_driver.o relu_kernel.o identity_kernel.o
	$(CXX) $^ -o relu_driver




identity_driver.o: identity_driver.cpp
	$(CXX) -c $(CXXFLAGS) $^ -o identity_driver.o

relu_driver.o: relu_driver.cpp
	$(CXX) -c $(CXXFLAGS) $^ -o relu_driver.o

relu_kernel.o: relu_kernel.s
	$(CXX) -c $(ASFLAGS) $^ -o relu_kernel.o

identity_kernel.o: identity_kernel.s
	$(CXX) -c $(ASFLAGS) $^ -o identity_kernel.o





# clang++
# /opt/homebrew/opt/llvm

# ASFLAGS ?= -march=armv9.2-a+sme


TOOLCHAIN?=standard

ifeq ($(TOOLCHAIN),standard)

CXX = clang++
CXXFLAGS = -Wall -Wextra -march=armv9.2-a+nosve+sme -O3
ASFLAGS = $(CXXFLAGS) -x assembler-with-cpp

else

ifeq ($(TOOLCHAIN),homebrew)

LLVM_PREFIX = /opt/homebrew/opt/llvm
CXX = $(LLVM_PREFIX)/bin/clang++
CC = $(LLVM_PREFIX)/bin/clang
CXXFLAGS = -Wall -Wextra -fopenmp -march=armv9.2-a+nosve+sme -O3
ASFLAGS = $(CXXFLAGS) -x assembler-with-cpp

else
$(error Invalid toolchain. Select either "TOOLCHAIN=standard" or "TOOLCHAIN=homebrew")
endif

endif









all: kernel_examples.out main.out instgen_examples.out

clean:
	rm -f *.bin
	rm -f *.out
	rm -f *.o




kernel_examples.out: kernel_examples.cpp Kernel.cpp InstGen.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

main.out: main.cpp Kernel.cpp KernelFactory.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

instgen_examples.out: instgen_examples.cpp InstGen.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@


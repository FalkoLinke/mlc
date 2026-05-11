
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









all: kernel_examples.out main.out instgen_examples.out unary_tests.out unary_benchmarks_driver.out

clean:
	rm -f *.bin
	rm -f *.out
	rm -f *.o




kernel_examples.out: kernel_examples.cpp Kernel.cpp InstGen.cpp BranchRef.cpp LabeledBranch.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

main.out: main.cpp Kernel.cpp KernelFactory.cpp BranchRef.cpp LabeledBranch.cpp InstGen.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

instgen_examples.out: instgen_examples.cpp InstGen.cpp BranchRef.cpp LabeledBranch.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

unary_tests.out: unary_tests.cpp unary_kernels.cpp KernelFactory.cpp Kernel.cpp InstGen.cpp BranchRef.cpp LabeledBranch.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

unary_benchmarks_driver.out: unary_benchmarks_driver.cpp unary_kernels.cpp KernelFactory.cpp Kernel.cpp InstGen.cpp BranchRef.cpp LabeledBranch.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@




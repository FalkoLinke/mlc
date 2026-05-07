
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











all: identity_driver relu_driver zero_driver unary_benchmarks_driver unary_tests



clean:
	rm -f identity_driver
	rm -f relu_driver
	rm -f zero_driver
	rm -f unary_benchmarks_driver
	rm -f unary_tests
	rm -f *.o



identity_driver: identity_driver.o identity_kernel.o
	$(CXX) $^ -o identity_driver

relu_driver: relu_driver.o relu_kernel.o identity_kernel.o
	$(CXX) $^ -o relu_driver

zero_driver: zero_driver.o zero_kernel.o
	$(CXX) $^ -o zero_driver

unary_benchmarks_driver: unary_benchmarks_driver.o identity_kernel.o zero_kernel.o relu_kernel.o
	$(CXX) $^ -o unary_benchmarks_driver

unary_tests: unary_tests.o identity_kernel.o zero_kernel.o relu_kernel.o
	$(CXX) $^ -o unary_tests


identity_driver.o: identity_driver.cpp
	$(CXX) -c $(CXXFLAGS) $^ -o identity_driver.o

relu_driver.o: relu_driver.cpp
	$(CXX) -c $(CXXFLAGS) $^ -o relu_driver.o

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

zero_kernel.o: zero_kernel.s
	$(CXX) -c $(ASFLAGS) $^ -o zero_kernel.o



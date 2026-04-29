CPP = g++
CPPFLAGS = -march=armv9-a+sve -O0
ASFLAGS = $(CPPFLAGS) -x assembler-with-cpp



all: identity_driver relu_driver



clean:
	rm -f identity_driver
	rm -f relu_driver
	rm -f *.o



identity_driver: identity_driver.o identity_kernel.o
	g++ $^ -o identity_driver

relu_driver: relu_driver.o relu_kernel.o identity_kernel.o
	g++ $^ -o relu_driver




identity_driver.o: identity_driver.cpp
	g++ -c $(CPPFLAGS) $^ -o identity_driver.o

relu_driver.o: relu_driver.cpp
	g++ -c $(CPPFLAGS) $^ -o relu_driver.o

relu_kernel.o: relu_kernel.s
	g++ -c $(ASFLAGS) $^ -o relu_kernel.o

identity_kernel.o: identity_kernel.s
	g++ -c $(ASFLAGS) $^ -o identity_kernel.o





# clang++
# /opt/homebrew/opt/llvm

# ASFLAGS ?= -march=armv9.2-a+sme

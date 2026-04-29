all: identity_driver

clean:
	rm -f identity_driver

identity_driver: identity_driver.cpp identity_kernel.s
	g++ -march=armv9-a+sve -O0 identity_driver.cpp -x assembler-with-cpp identity_kernel.s -o identity_driver


# clang++
# /opt/homebrew/opt/llvm

# ASFLAGS ?= -march=armv9.2-a+sme

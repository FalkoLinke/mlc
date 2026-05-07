#include <iostream>
#include <chrono>

#include "identity_kernel.h"
#include "zero_kernel.h"
#include "relu_kernel.h"



typedef void(kernel_func_t)(float const*, float*, int64_t, int64_t);

void identity_16_16_kernel(float const* a, float* b, int64_t ld_a, int64_t ld_b) {
    identity_16_16(a, b, ld_a, ld_b , 0);
}

void identity_16_16_trans_kernel(float const* a, float* b, int64_t ld_a, int64_t ld_b) {
    identity_16_16(a, b, ld_a, ld_b, 1);
}

void zero_16_16_kernel(float const* a, float* b, int64_t ld_a, int64_t ld_b) {
    zero_16_16(b, ld_b);
}

void relu_16_16_kernel(float const* a, float* b, int64_t ld_a, int64_t ld_b) {
    relu_16_16(a, b, ld_a, ld_b, 0);
}

void relu_16_16_trans_kernel(float const* a, float* b, int64_t ld_a, int64_t ld_b) {
    relu_16_16(a, b, ld_a, ld_b, 1);
}

struct kernel_t {
public:
	kernel_func_t* const func;
	std::string const name;

	kernel_t(kernel_func_t* const func, std::string const name) : func(func), name(name) {

	}

	void call(float const* a, float* b, int64_t ld_a, int64_t ld_b) const {
		return this->func(a, b, ld_a, ld_b);
	}
};

#define MAKE_KERNEL(FUNC) (kernel_t( (FUNC), #FUNC ))














void benchmark_kernel(kernel_t kernel) {
    int const reps = 1000000;
    float a[16*16];
    float b[16*16];

    kernel_func_t* kernel_func = kernel.func;
    kernel_func(a, b, 16, 16);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < reps; i++) {
        kernel_func(a, b, 16, 16);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;


    double bytes_transferred = reps * 16 * 16 * sizeof(float);
    double gib_transferred = bytes_transferred / 1073741824.0f;
    double time_taken = duration.count();
    double gibs = gib_transferred / time_taken;


    std::cout << "=========================" << std::endl;
    std::cout << kernel.name << ":" << std::endl;
    std::cout << "\tDuration [s]: " << time_taken << std::endl;
    std::cout << "\tBytes transferred [GB]: " << bytes_transferred * 1e-9 << std::endl;
    std::cout << "\tGiBs: " << gibs << std::endl;
}











int main() {
    kernel_t kernels[] = {
        MAKE_KERNEL(identity_16_16_kernel),
        MAKE_KERNEL(identity_16_16_trans_kernel),
        MAKE_KERNEL(zero_16_16_kernel),
        MAKE_KERNEL(relu_16_16_kernel),
        MAKE_KERNEL(relu_16_16_trans_kernel),
    };
    size_t const kernels_count = sizeof(kernels) / sizeof(kernel_t);

    for (size_t i = 0; i < kernels_count; i++) {
        benchmark_kernel(kernels[i]);
    }

    return 0;
}

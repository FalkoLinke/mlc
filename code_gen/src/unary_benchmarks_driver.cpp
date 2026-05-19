#include <iostream>
#include <chrono>

#include "unary_kernels.h"
#include "Unary.h"


using mini_jit::Unary;

int counter = 0;




/**
 * @brief       The function signature of a unary operation kernel.
 * @param a     The source matrix.
 * @param b     The target matrix.
 * @param ld_a  The leading parameter for a.
 * @param ld_b  The leading parameter for b.
 */
typedef void(kernel_func_t)(float const*, float*, int64_t, int64_t);

/**
 * @brief Kernel wrapper for a nontransposing `identity` operation. Conforms to `kernel_func_t`.
 */
void identity_16_16_kernel(float const* a, float* b, int64_t ld_a, int64_t ld_b) {
    identity_16_16(a, b, ld_a, ld_b , 0);
}

/**
 * @brief Kernel wrapper for a transposing `identity` operation. Conforms to `kernel_func_t`.
 */
void identity_16_16_trans_kernel(float const* a, float* b, int64_t ld_a, int64_t ld_b) {
    identity_16_16(a, b, ld_a, ld_b, 1);
}

/**
 * @brief Kernel wrapper for a `zero` operation. Conforms to `kernel_func_t`.
 */
void zero_16_16_kernel(float const* a, float* b, int64_t ld_a, int64_t ld_b) {
    zero_16_16(b, ld_b);
}

/**
 * @brief Kernel wrapper for a nontransposing `RELU` operation. Conforms to `kernel_func_t`.
 */
void relu_16_16_kernel(float const* a, float* b, int64_t ld_a, int64_t ld_b) {
    relu_16_16(a, b, ld_a, ld_b, 0);
}

/**
 * @brief Kernel wrapper for a transposing `RELU` operation. Conforms to `kernel_func_t`.
 */
void relu_16_16_trans_kernel(float const* a, float* b, int64_t ld_a, int64_t ld_b) {
    relu_16_16(a, b, ld_a, ld_b, 1);
}

/**
 * Stores a kernel function along with an identifying name.
 */
struct kernel_t {
public:
    /** The kernel function. Call this to execute the kernel. */
	kernel_func_t* const func;
    /** The identifying name. */
	std::string const name;

	kernel_t(kernel_func_t* const func, std::string const name) : func(func), name(name) {

	}

    /** Calls the kernel function with the given parameters. */
	void call(float const* a, float* b, int64_t ld_a, int64_t ld_b) const {
		return this->func(a, b, ld_a, ld_b);
	}
};

#define MAKE_KERNEL(FUNC) (kernel_t( (FUNC), #FUNC ))













/**
 * @brief Benchmark the given kernel and report the `GiBs` metric.
 * @param kernel        The kernel to evaluate.
 * 
 * This function allocates space in it's stack frame for two 16x16 FP32 matrices.
 * It then repeatedly executes the given kernel on those matrices, while measuring
 * the total amount of time taken.
 * From the total number of bytes processed it then determines and prints
 * the GiBs metric, along with other benchmark results.
 */
void benchmark_kernel(kernel_t kernel, long const reps) {
    float a[16*16];
    float b[16*16];

    kernel_func_t* kernel_func = kernel.func;
    kernel_func(a, b, 16, 16);

    auto start = std::chrono::high_resolution_clock::now();

    for (long i = 0; i < reps; i++) {
        kernel_func(a, b, 16, 16);
        counter += 1;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;


    double gib_transferred = (reps / 1073741824.0f) * (16 * 16 * sizeof(float));
    double time_taken = duration.count();
    double gibs = gib_transferred / time_taken;


    std::cout << "=========================" << std::endl;
    std::cout << kernel.name << ":" << std::endl;
    std::cout << "\tDuration [s]: " << time_taken << std::endl;
    std::cout << "\tBytes transferred [GiB]: " << gib_transferred << std::endl;
    std::cout << "\tGiBs: " << gibs << std::endl;
}





void benchmark_jit_kernel(uint64_t m, uint64_t n, Unary::ptype_t op, bool trans_b, long const reps) {
    uint64_t ma = m;
    uint64_t na = n;
    uint64_t mb = trans_b ? n : m;
    uint64_t nb = trans_b ? m : n;

    int64_t lda = ma;
    int64_t ldb = mb;

    std::vector<float> a(na * lda, 0.0f);
    std::vector<float> b(nb * ldb, 0.0f);

    Unary unary;
    unary.generate(m, n, trans_b, Unary::dtype_t::fp32, op);
    Unary::kernel_t kernel_func = unary.get_kernel();

    kernel_func(a.data(), b.data(), lda, ldb);

    auto start = std::chrono::high_resolution_clock::now();

    for (long i = 0; i < reps; i++) {
        kernel_func(a.data(), b.data(), lda, ldb);
        counter += 1;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;


    double gib_transferred = (reps / 1073741824.0f) * (m * n * sizeof(float));
    double time_taken = duration.count();
    double gibs = gib_transferred / time_taken;


    std::cout << "=========================" << std::endl;
    std::cout << "\tOp: " << (uint32_t)op << std::endl;
    std::cout << "\t(m, n): (" << m << ", " << n << ")" << std::endl;
    std::cout << "\ttrans_b: " << trans_b << std::endl;
    std::cout << "\tDuration [s]: " << time_taken << std::endl;
    std::cout << "\tBytes transferred [GiB]: " << gib_transferred << std::endl;
    std::cout << "\tGiBs: " << gibs << std::endl;
}














void benchmark_kernels(long const reps) {
    kernel_t kernels[] = {
        MAKE_KERNEL(identity_16_16_kernel),
        MAKE_KERNEL(identity_16_16_trans_kernel),
        MAKE_KERNEL(zero_16_16_kernel),
        MAKE_KERNEL(relu_16_16_kernel),
        MAKE_KERNEL(relu_16_16_trans_kernel),
    };
    size_t const kernels_count = sizeof(kernels) / sizeof(kernel_t);

    for (size_t i = 0; i < kernels_count; i++) {
        benchmark_kernel(kernels[i], reps);
    }
}

void benchmark_jit_kernels(long const reps) {
    std::vector<uint64_t> ms = {64, 128, 512};
    std::vector<uint64_t> ns = {64, 128, 512};
    std::vector<bool> trans_bs = {false, true};
    std::vector<Unary::ptype_t> ops = {Unary::ptype_t::identity, Unary::ptype_t::zero, Unary::ptype_t::relu};

    for (Unary::ptype_t op : ops) {
        for (bool trans_b : trans_bs) {
            for (uint64_t m : ms) {
                for (uint64_t n : ns) {
                    benchmark_jit_kernel(m, n, op, trans_b, reps);
                }
            }
        }
    }
}

int main() {
    generate_kernels();
    //long const reps = 500000000;
    long const reps = 1000000;

    benchmark_kernels(reps);
    benchmark_jit_kernels(reps);

    std::cerr << counter << std::endl;
    return 0;
}

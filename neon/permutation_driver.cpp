#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

#include "permutation_kernel.h"


/* The sizes of the a and b dimensions are fixed by the kernels. */
static constexpr int64_t SIZE_A = 8;
static constexpr int64_t SIZE_B = 4;

/* The sizes of the c dimension to benchmark. */
static constexpr int64_t SIZES_C[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384};

/* The target duration of a single measurement in seconds. */
static constexpr double TARGET_DURATION = 0.5;


typedef void (kernel_func_t)(int64_t, float const*, float*);

struct kernel_t {
    kernel_func_t* const func;
    std::string const name;
};


/**
 * @brief Reference implementation of the permutation operation abc->cba.
 */
void ref_perm_abc_cba(int64_t size_c, float const* abc, float* cba) {
    for (int64_t a = 0; a < SIZE_A; a++) {
        for (int64_t b = 0; b < SIZE_B; b++) {
            for (int64_t c = 0; c < size_c; c++) {
                cba[(c * SIZE_B + b) * SIZE_A + a] = abc[(a * SIZE_B + b) * size_c + c];
            }
        }
    }
}


/**
 * @brief Benchmarks `kernel` for the given size of the c dimension.
 *
 * Executes the kernel repeatedly until at least `TARGET_DURATION` has elapsed and
 * reports the achieved bandwidth. Each call of the kernel reads and writes the
 * entire tensor once, therefore the number of bytes moved per call is
 * `2 * |a| * |b| * |c| * sizeof(float)`.
 *
 * @return `false` if the kernel produced a wrong result, `true` otherwise.
 */
bool benchmark_kernel(kernel_t const& kernel, int64_t size_c) {
    uint64_t const elements = SIZE_A * SIZE_B * size_c;

    std::vector<float> abc(elements);
    std::vector<float> cba(elements, 0.0f);
    std::vector<float> exp(elements, 0.0f);
    for (uint64_t i = 0; i < elements; i++) {
        abc[i] = static_cast<float>(i);
    }

    // verify before measuring
    kernel.func(size_c, abc.data(), cba.data());
    ref_perm_abc_cba(size_c, abc.data(), exp.data());
    if (cba != exp) {
        std::cout << std::setw(26) << kernel.name
                  << std::setw(10) << size_c
                  << "   WRONG RESULT" << std::endl;
        return false;
    }

    // determine the number of repetitions needed for a stable measurement
    uint64_t repetitions = 16;
    double duration = 0.0;
    while (true) {
        auto start = std::chrono::high_resolution_clock::now();
        for (uint64_t i = 0; i < repetitions; i++) {
            kernel.func(size_c, abc.data(), cba.data());
        }
        auto end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration<double>(end - start).count();

        if (duration >= TARGET_DURATION) {
            break;
        }
        repetitions *= 2;
    }

    double const bytes = 2.0 * static_cast<double>(elements) * sizeof(float) * static_cast<double>(repetitions);
    double const gibs = bytes / (1024.0 * 1024.0 * 1024.0) / duration;

    std::cout << std::setw(26) << kernel.name
              << std::setw(10) << size_c
              << std::setw(14) << repetitions
              << std::setw(12) << std::fixed << std::setprecision(4) << duration
              << std::setw(12) << std::fixed << std::setprecision(2) << gibs
              << std::endl;
    return true;
}


int main() {
    kernel_t const kernels[] = {
        { perm_neon_abc_cba_scalar, "perm_neon_abc_cba_scalar" },
        { perm_neon_abc_cba,        "perm_neon_abc_cba" },
    };

    std::cout << "Permutation abc->cba, |a| = " << SIZE_A << ", |b| = " << SIZE_B << std::endl;
    std::cout << std::setw(26) << "kernel"
              << std::setw(10) << "size_c"
              << std::setw(14) << "repetitions"
              << std::setw(12) << "time [s]"
              << std::setw(12) << "GiB/s"
              << std::endl;

    bool ok = true;
    for (kernel_t const& kernel : kernels) {
        for (int64_t size_c : SIZES_C) {
            ok &= benchmark_kernel(kernel, size_c);
        }
        std::cout << std::endl;
    }

    return ok ? 0 : 1;
}

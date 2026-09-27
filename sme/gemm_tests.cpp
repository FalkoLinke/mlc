#include <cstdint>
#include <random>
#include <string>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "mlc_common.hpp"
#include "gemm_kernels.h"


/* The largest accepted relative deviation from the reference implementation. */
static constexpr double TOLERANCE = 1e-4;


typedef void (gemm_kernel_t)(float const*, float const*, float*, int64_t, int64_t, int64_t);


/**
 * @brief Fills `values` with random values in [-1, 1].
 */
void fill_random(std::vector<float>& values, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& v : values) {
        v = dist(gen);
    }
}


/**
 * @brief Executes `kernel` and the reference implementation on random matrices and
 *        compares their results.
 *
 * A is column-major with m rows and k columns, B is row-major with k rows and n columns
 * and C is column-major with m rows and n columns. The kernels accumulate into C,
 * therefore C is initialized with random values as well.
 *
 * @param kernel The kernel to test.
 * @param m      The number of rows of A and C.
 * @param n      The number of columns of B and C.
 * @param k      The number of columns of A and rows of B.
 * @param ld_a   The leading dimension of A.
 * @param ld_b   The leading dimension of B.
 * @param ld_c   The leading dimension of C.
 */
void check_gemm(gemm_kernel_t* kernel,
                uint64_t m, uint64_t n, uint64_t k,
                int64_t ld_a, int64_t ld_b, int64_t ld_c) {
    std::vector<float> a(static_cast<uint64_t>(ld_a) * k);
    std::vector<float> b(static_cast<uint64_t>(ld_b) * k);
    std::vector<float> c(static_cast<uint64_t>(ld_c) * n);

    fill_random(a, 1);
    fill_random(b, 2);
    fill_random(c, 3);
    std::vector<float> exp = c;

    kernel(a.data(), b.data(), c.data(), ld_a, ld_b, ld_c);
    gemm_ref(a.data(), b.data(), exp.data(), m, n, k, ld_a, ld_b, ld_c);

    REQUIRE(max_rel_diff(c.data(), exp.data(), m, n, ld_c, ld_c) <= TOLERANCE);
}


/*
 * The kernels of this week are written for fixed matrix shapes and derive their loop
 * bounds from the leading dimensions, therefore A and B have to be stored densely
 * (ld_a = m, ld_b = n). Only `gemm_512_32_512` supports a leading dimension of C
 * that is larger than m.
 */

TEST_CASE("gemm_32_32_1 computes C += A * B", "[sme][gemm]") {
    check_gemm(gemm_32_32_1, 32, 32, 1, 32, 32, 32);
}

TEST_CASE("gemm_32_32_512 computes C += A * B", "[sme][gemm]") {
    check_gemm(gemm_32_32_512, 32, 32, 512, 32, 32, 32);
}

TEST_CASE("gemm_512_32_512 computes C += A * B", "[sme][gemm]") {
    check_gemm(gemm_512_32_512, 512, 32, 512, 512, 32, 512);
}

TEST_CASE("gemm_512_32_512 respects the leading dimension of C", "[sme][gemm]") {
    check_gemm(gemm_512_32_512, 512, 32, 512, 512, 32, 1024);
}

TEST_CASE("gemm_512_512_512 computes C += A * B", "[sme][gemm]") {
    check_gemm(gemm_512_512_512, 512, 512, 512, 512, 512, 512);
}

TEST_CASE("gemm_16_16 computes C += A * B", "[sme][gemm]") {
    check_gemm(gemm_16_16, 16, 16, 512, 16, 16, 16);
}

TEST_CASE("gemm_M_N_K_16 computes C += A * B", "[sme][gemm]") {
    check_gemm(gemm_M_N_K_16, 512, 512, 512, 512, 512, 512);
}

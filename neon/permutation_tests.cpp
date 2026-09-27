#include <cstdint>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "permutation_kernel.h"


/* The sizes of the a and b dimensions are fixed by the kernels. */
static constexpr int64_t SIZE_A = 8;
static constexpr int64_t SIZE_B = 4;


/**
 * @brief Reference implementation of the permutation operation abc->cba.
 * @param size_c Size of dimension c.
 * @param abc    Pointer to row-major tensor abc.
 * @param cba    Pointer to row-major tensor cba.
 **/
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
 * @brief Creates an input tensor whose elements are their own linear index.
 */
std::vector<float> make_input(int64_t size_c) {
    std::vector<float> abc(SIZE_A * SIZE_B * size_c);
    for (size_t i = 0; i < abc.size(); i++) {
        abc[i] = static_cast<float>(i);
    }
    return abc;
}

/**
 * @brief Runs `kernel` and the reference implementation and compares their results.
 *
 * The output buffers carry two guard elements at their end in order to detect
 * out of bounds writes of the kernel.
 */
void check_kernel(void (*kernel)(int64_t, float const*, float*), int64_t size_c) {
    std::vector<float> abc = make_input(size_c);
    std::vector<float> cba(abc.size() + 2, -1.0f);
    std::vector<float> exp(abc.size() + 2, -1.0f);

    kernel(size_c, abc.data(), cba.data());
    ref_perm_abc_cba(size_c, abc.data(), exp.data());

    REQUIRE(cba == exp);
}


TEST_CASE("perm_neon_abc_cba_scalar matches the reference", "[neon][permutation]") {
    int64_t size_c = GENERATE(0, 1, 2, 3, 4, 5, 7, 8, 12, 15, 16, 31, 32, 64, 100, 128, 512);
    check_kernel(perm_neon_abc_cba_scalar, size_c);
}

TEST_CASE("perm_neon_abc_cba matches the reference", "[neon][permutation]") {
    int64_t size_c = GENERATE(0, 1, 2, 3, 4, 5, 7, 8, 12, 15, 16, 31, 32, 64, 100, 128, 512);
    check_kernel(perm_neon_abc_cba, size_c);
}

TEST_CASE("perm_neon_abc_cba moves the corner elements correctly", "[neon][permutation]") {
    constexpr int64_t size_c = 8;
    std::vector<float> abc = make_input(size_c);
    std::vector<float> cba(abc.size(), -1.0f);

    perm_neon_abc_cba(size_c, abc.data(), cba.data());

    // (a, b, c) = (0, 0, 0) stays in place
    REQUIRE(cba[0] == abc[0]);
    // (a, b, c) = (7, 3, 7): last element of the input is the last element of the output
    REQUIRE(cba[abc.size() - 1] == abc[abc.size() - 1]);
    // (a, b, c) = (1, 0, 0) is located at (c, b, a) = (0, 0, 1)
    REQUIRE(cba[1] == abc[SIZE_B * size_c]);
}

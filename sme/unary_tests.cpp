#include <iostream>
#include <math.h>
#include <catch2/catch_test_macros.hpp>

#include "mlc_common.hpp"

#include "identity_kernel.h"
#include "zero_kernel.h"
#include "relu_kernel.h"


























/**
 * @brief Reference implementation for an `identity` operation with m=16 and n=16.
 * @param a       Pointer to column-major matrix A.
 * @param b       Pointer to matrix B.
 * @param ld_a    Leading dimension of A.
 * @param ld_b    Leading dimension of B.
 * @param trans_b Column-major B if 0, row-major B if 1. 
 **/
void ref_identity_16_16(float const* a, float *b, int64_t ld_a, int64_t ld_b, int32_t trans_b) {
    identity(a, b, 16, 16, ld_a, ld_b, trans_b);
}

/**
 * @brief Reference implementation for the `zero` operation.
 * @param a    Pointer to column-major matrix A.
 * @param ld_a Leading dimension of A.
 **/
void ref_zero_16_16(float* a, int64_t ld_a) {
    zero(a, 16, 16, ld_a);
}

/*
* @brief Reference implementation for the `RELU` operation.
* @param a       Pointer to column-major matrix A.
* @param b       Pointer to matrix B.
* @param ld_a    Leading dimension of A.
* @param ld_b    Leading dimension of B.
* @param trans_b Column-major B if 0, row-major B if 1. 
**/
void ref_relu_16_16(float const* a, float* b, int64_t ld_a, int64_t ld_b, int32_t trans_b) {
    relu(a, b, 16, 16, ld_a, ld_b, trans_b);
}























TEST_CASE("identity_16_16 copies a dense matrix", "[sme][unary]") {
    float a[16 * 16];
    float b[16 * 16];
    float exp[16 * 16];

    fill_indices(a, 16*16);
    fill_indices(exp, 16*16);

    identity_16_16(a, b, 16, 16, 0);

    bool result = mats_equal(b, exp, 16, 16);
    REQUIRE(result);
}

TEST_CASE("zero_16_16 clears a dense matrix", "[sme][unary]") {
    float a[16 * 16];
    float b[16 * 16];

    fill_indices(a, 16 * 16);
    fill_indices(b, 16 * 16);

    zero_16_16(a, 16);
    ref_zero_16_16(b, 16);

    bool result = mats_equal(a, b, 16, 16);
    REQUIRE(result);
}

TEST_CASE("identity_16_16 matches the reference", "[sme][unary]") {
    float a[16 * 16];
    float b[16 * 16];
    float exp[16 * 16];

    fill_indices(a, 16 * 16);

    identity_16_16(a, b, 16, 16, 0);
    ref_identity_16_16(a, exp, 16, 16, 0);

    bool result = mats_equal(b, exp, 16, 16);
    REQUIRE(result);
}

TEST_CASE("transposing identity_16_16 matches the reference", "[sme][unary]") {
    float a[16 * 16];
    float b[16 * 16];
    float exp[16 * 16];

    fill_indices(a, 16 * 16);

    identity_16_16(a, b, 16, 16, 1);
    ref_identity_16_16(a, exp, 16, 16, 1);

    bool result = mats_equal(b, exp, 16, 16);
    REQUIRE(result);
}

TEST_CASE("relu_16_16 matches the reference", "[sme][unary]") {
    float a[16 * 16];
    float b[16 * 16];
    float exp[16 * 16];

    fill_indices(a, 16 * 16);
    for (int i = 0; i < 16; i++) {
        a[i*16] = -1.0;
    }

    relu_16_16(a, b, 16, 16, 0);
    ref_relu_16_16(a, exp, 16, 16, 0);

    bool result = mats_equal(b, exp, 16, 16);
    REQUIRE(result);
} 

TEST_CASE("transposing relu_16_16 matches the reference", "[sme][unary]") {
    float a[16 * 16];
    float b[16 * 16];
    float exp[16 * 16];

    fill_indices(a, 16 * 16);
    for (int i = 0; i < 16; i++) {
        a[i*16] = -1.0;
    }

    relu_16_16(a, b, 16, 16, 1);
    ref_relu_16_16(a, exp, 16, 16, 1);

    bool result = mats_equal(b, exp, 16, 16);
    REQUIRE(result);
} 

TEST_CASE("zero_16_16 only clears its submatrix", "[sme][unary]") {
    int const rows = 512;
    float a[rows * rows];
    float b[rows * rows];

    fill_indices(a, rows * rows);
    fill_indices(b, rows * rows);

    float* a_sub = a + rows / 4 + rows / 4 * rows;
    float* b_sub = b + rows / 4 + rows / 4 * rows;
    zero_16_16(a_sub, rows);
    ref_zero_16_16(b_sub, rows);

    bool result = mats_equal(a, b, rows, rows);
    REQUIRE(result);
}

TEST_CASE("identity_16_16 only writes its submatrix", "[sme][unary]") {
    int const rows = 512;
    float a[rows * rows];
    float b[rows * rows];
    float exp[rows * rows];

    fill_indices(a, rows * rows);
    fill_const(b, rows * rows, -5.0f);
    fill_const(exp, rows * rows, -5.0f);

    int sub_off = rows / 4 + rows / 4 * rows;
    identity_16_16(a + sub_off, b + sub_off, rows, rows, 0);
    ref_identity_16_16(a + sub_off, exp + sub_off, rows, rows, 0);

    bool result = mats_equal(b, exp, rows, rows);
    REQUIRE(result);
}

TEST_CASE("transposing identity_16_16 only writes its submatrix", "[sme][unary]") {
    int const rows = 512;
    float a[rows * rows];
    float b[rows * rows];
    float exp[rows * rows];

    fill_indices(a, rows * rows);
    fill_const(b, rows * rows, -5.0f);
    fill_const(exp, rows * rows, -5.0f);

    int sub_off = rows / 4 + rows / 4 * rows;
    identity_16_16(a + sub_off, b + sub_off, rows, rows, 1);
    ref_identity_16_16(a + sub_off, exp + sub_off, rows, rows, 1);

    bool result = mats_equal(b, exp, rows, rows);
    REQUIRE(result);
}

TEST_CASE("relu_16_16 only writes its submatrix", "[sme][unary]") {
    int const rows = 512;
    int sub_off = rows / 4 + rows / 4 * rows;

    float a[rows * rows];
    float b[rows * rows];
    float exp[rows * rows];

    fill_indices(a, rows * rows);
    fill_const(b, rows * rows, -5.0f);
    fill_const(exp, rows * rows, -5.0f);
    for (int i = 0; i < 16; i++) {
        (a + sub_off)[i*16] = -1.0;
    }

    relu_16_16(a + sub_off, b + sub_off, rows, rows, 0);
    ref_relu_16_16(a + sub_off, exp + sub_off, rows, rows, 0);

    bool result = mats_equal(b, exp, rows, rows);
    REQUIRE(result);
} 

TEST_CASE("transposing relu_16_16 only writes its submatrix", "[sme][unary]") {
    int const rows = 512;
    int sub_off = rows / 4 + rows / 4 * rows;
    
    float a[rows * rows];
    float b[rows * rows];
    float exp[rows * rows];

    fill_indices(a, rows * rows);
    fill_const(b, rows * rows, -5.0f);
    fill_const(exp, rows * rows, -5.0f);
    for (int i = 0; i < 16; i++) {
        (a + sub_off)[i*16] = -1.0;
    }

    relu_16_16(a + sub_off, b + sub_off, rows, rows, 1);
    ref_relu_16_16(a + sub_off, exp + sub_off, rows, rows, 1);

    bool result = mats_equal(b, exp, rows, rows);
    REQUIRE(result);
} 















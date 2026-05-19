#include <iostream>
#include <math.h>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "unary_kernels.h"
#include "code_gen_common.hpp"
#include "Unary.h"

using mini_jit::Unary;































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























TEST_CASE( "nontransposing identity 01", "[test]") {
    float a[16 * 16];
    float b[16 * 16];
    float exp[16 * 16];

    fill_indices(a, 16*16);
    fill_indices(exp, 16*16);

    identity_16_16(a, b, 16, 16, 0);

    bool result = mats_equal(b, exp, 16, 16);
    REQUIRE(result);
}

TEST_CASE( "zero 01", "[test]") {
    float a[16 * 16];
    float b[16 * 16];

    fill_indices(a, 16 * 16);
    fill_indices(b, 16 * 16);

    zero_16_16(a, 16);
    ref_zero_16_16(b, 16);

    bool result = mats_equal(a, b, 16, 16);
    REQUIRE(result);
}

TEST_CASE( "nontransposing identity 02", "[test]") {
    float a[16 * 16];
    float b[16 * 16];
    float exp[16 * 16];

    fill_indices(a, 16 * 16);

    identity_16_16(a, b, 16, 16, 0);
    ref_identity_16_16(a, exp, 16, 16, 0);

    bool result = mats_equal(b, exp, 16, 16);
    REQUIRE(result);
}

TEST_CASE("transposing identity 01", "[test]") {
    float a[16 * 16];
    float b[16 * 16];
    float exp[16 * 16];

    fill_indices(a, 16 * 16);

    identity_16_16(a, b, 16, 16, 1);
    ref_identity_16_16(a, exp, 16, 16, 1);

    bool result = mats_equal(b, exp, 16, 16);
    REQUIRE(result);
}

TEST_CASE("nontransposing relu 01", "[test]") {
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

TEST_CASE("transposing relu 01", "[test]") {
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

TEST_CASE("zero submatrix 01", "[test]") {
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

TEST_CASE("nontransposing identity submatrix 01", "[test]") {
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

TEST_CASE("transposing identity submatrix 01", "[test]") {
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

TEST_CASE("nontransposing relu submatrix 01", "[test]") {
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

TEST_CASE("transposing relu submatrix 01", "[test]") {
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





template <typename T>
struct RefUnary {
    private:
        Unary::ptype_t const op;
        bool const trans_b;
        uint64_t const m;
        uint64_t const n;

    public:
        RefUnary(Unary::ptype_t op, bool trans_b, uint64_t m, uint64_t n) : op(op), trans_b(trans_b), m(m), n(n) {

        }
        ~RefUnary() = default;

        RefUnary( RefUnary const & ) = default;
        RefUnary & operator=( RefUnary const & ) = default;
        RefUnary( RefUnary && ) noexcept = delete;
        RefUnary & operator=( RefUnary && ) noexcept = delete;

        void operator()(T const* a, T* b, int64_t lda, int64_t ldb) {
            uint64_t ma = m;
            uint64_t na = n;
            uint64_t mb = trans_b ? n : m;
            uint64_t nb = trans_b ? m : n;

            if (op == Unary::ptype_t::identity) {
                identity(a, b, ma, na, lda, ldb, trans_b);

            } else if (op == Unary::ptype_t::zero) {
                zero(b, mb, nb, ldb);

            } else if (op == Unary::ptype_t::relu) {
                relu(a, b, ma, na, lda, ldb, trans_b);

            }
        }
};

void test_unary_op(uint64_t m, uint64_t n, int64_t lda, int64_t ldb, bool trans_b, Unary::ptype_t op) {
    uint64_t ma = m;
    uint64_t na = n;
    uint64_t mb = trans_b ? n : m;
    uint64_t nb = trans_b ? m : n;

    // make sure leading info is correct
    if (!(0 <= lda && 0 <= ldb)) {
        return;
    }
    if (!(static_cast<uint64_t>(lda) >= ma)) {
        return;
    }
    if (!(static_cast<uint64_t>(ldb) >= mb)) {
        return;
    }
    INFO("m=" << m << ", n=" << n << ", lda=" << lda << ", ldb=" << ldb << ", trans_b=" << trans_b << ", op=" << (uint32_t)op);

    // generate kernel under test
    Unary unary;
    Unary::error_t err = unary.generate(m, n, trans_b, Unary::dtype_t::fp32, op);
    REQUIRE(err == Unary::error_t::success);
    Unary::kernel_t kernel = unary.get_kernel();

    // obtain corresponding reference implementation
    RefUnary<float> ref_kernel(op, trans_b, m, n);

    // allocate and initialize memory
    std::vector<float> a(na * lda, 0);
    std::vector<float> b(nb * ldb, -1.0f);
    std::vector<float> exp(nb * ldb, -1.0f);
    fill_indices(a.data(), ma, na, lda);

    // execute kernels
    kernel(a.data(), b.data(), lda, ldb);
    ref_kernel(a.data(), exp.data(), lda, ldb);

    // check if correct 
    bool result = mats_equal<float>(b.data(), exp.data(), mb, nb, ldb, ldb);
    REQUIRE(result);
}

TEST_CASE("codegen kernels", "[test]") {
    std::vector<uint64_t> ms = {16, 32, 48, 64};
    std::vector<uint64_t> ns = {16, 32, 48, 64};
    std::vector<int64_t> ldas = {16, 32, 48, 64, 80, 96, 112, 128};
    std::vector<int64_t> ldbs = {16, 32, 48, 64, 80, 96, 112, 128};
    std::vector<Unary::ptype_t> ops = { Unary::ptype_t::identity, Unary::ptype_t::zero, Unary::ptype_t::relu };
    std::vector<bool> trans_bs = { false, true };

    for (uint64_t m : ms) {
        for (uint64_t n : ns) {
            for (int64_t lda : ldas) {
                for (int64_t ldb : ldbs) {
                    for (Unary::ptype_t op : ops) {
                        for (bool trans_b : trans_bs) {
                            test_unary_op(m, n, lda, ldb, trans_b, op);
                        }
                    }
                }
            }
        }
    }
}











int main(int argc, char** argv) {
    generate_kernels();

    int result = Catch::Session().run(argc, argv);

    return result;
}



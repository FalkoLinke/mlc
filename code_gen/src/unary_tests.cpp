#include <iostream>
#include <math.h>

#include "unary_kernels.h"






typedef bool(test_func_t)();

struct unit_test_t {
public:
	test_func_t* const test_func;
	std::string const test_name;

	unit_test_t(test_func_t* const test_func, std::string const test_name) : test_func(test_func), test_name(test_name) {

	}

	bool call() const {
		return this->test_func();
	}
};

#define MAKE_TEST(FUNC) (unit_test_t( (FUNC), #FUNC ))













/**
 * @brief Fills the buffer with the given value. 
 * @param buffer    Pointer to the buffer to fill.
 * @param size      The number of floats in the buffer.
 * @param value     The value to set each entry of the buffer to.
 **/
void fill_const(float* buffer, size_t const size, float const value) {
    for (size_t i = 0; i < size; i++) {
        buffer[i] = value;
    }
}

/**
 * @brief Sets each entry of the buffer to it's index. 
 * @param buffer    Pointer to the buffer to fill.
 * @param size      The number of floats in the buffer.
 **/
void fill_indices(float *buffer, size_t const size) {
    for (size_t i = 0; i < size; i++) {
        buffer[i] = (float)i;
    }
}

/**
 * @brief Returns `true` if the two matrices are equal. 
 * @param a         The first matrix.
 * @param b         The second matrix.
 * @param m         The number of rows of a and b.
 * @param n         The number of columns of a and b.
 * @return          `true` if all elements are equal, `false` otherwise.
 **/
bool mats_equal(float const* a, float const* b, int m, int n) {
    for (int r = 0; r < m; r++) {
        for (int c = 0; c < n; c++) {
            if (a[r * n + c] != b[r * n + c]) {
                return false;
            }
        }
    }
    return true;
}



















/**
 * @brief Reference implementation for an `identity` operation with m=16 and n=16.
 * @param a       Pointer to column-major matrix A.
 * @param b       Pointer to matrix B.
 * @param ld_a    Leading dimension of A.
 * @param ld_b    Leading dimension of B.
 * @param trans_b Column-major B if 0, row-major B if 1. 
 **/
void ref_identity_16_16(float const* a, float *b, int64_t ld_a, int64_t ld_b, int32_t trans_b) {
    int off_a_outer = ld_a;
    int off_a_inner = 1;

    int off_b_outer = trans_b == 0 ? ld_b : 1;
    int off_b_inner = trans_b == 0 ? 1 : ld_b;

    float const* ptr_a_outer = a;
    float* ptr_b_outer = b;
    for (int c = 0; c < 16; c++) {

        float const* ptr_a_inner = ptr_a_outer;
        float* ptr_b_inner = ptr_b_outer;
        for (int r = 0; r < 16; r++) {
            *ptr_b_inner = *ptr_a_inner;

            ptr_a_inner += off_a_inner;
            ptr_b_inner += off_b_inner;
        }

        ptr_a_outer += off_a_outer;
        ptr_b_outer += off_b_outer;
    }
}

/**
 * @brief Reference implementation for the `zero` operation.
 * @param a    Pointer to column-major matrix A.
 * @param ld_a Leading dimension of A.
 **/
void ref_zero_16_16(float* a, int64_t ld_a) {
    for (int c = 0; c < 16; c++) {
        for (int r = 0; r < 16; r++) {
            a[r] = 0.0;
        }
        a += ld_a;
    }
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
    ref_identity_16_16(a, b, ld_a, ld_b, trans_b);

    for (int c = 0; c < 16; c++) {
        for (int r = 0; r < 16; r++) {
            b[r] = fmaxf(b[r], 0.0);
        }
        b += ld_b;
    }
}
























bool test01() {
    float a[16 * 16];
    float b[16 * 16];
    float exp[16 * 16];

    fill_indices(a, 16*16);
    fill_indices(exp, 16*16);

    identity_16_16(a, b, 16, 16, 0);

    bool result = mats_equal(b, exp, 16, 16);
    return result;
}

bool test02() {
    float a[16 * 16];
    float b[16 * 16];

    fill_indices(a, 16 * 16);
    fill_indices(b, 16 * 16);

    zero_16_16(a, 16);
    ref_zero_16_16(b, 16);

    bool result = mats_equal(a, b, 16, 16);
    return result;
}

bool test03() {
    float a[16 * 16];
    float b[16 * 16];
    float exp[16 * 16];

    fill_indices(a, 16 * 16);

    identity_16_16(a, b, 16, 16, 0);
    ref_identity_16_16(a, exp, 16, 16, 0);

    bool result = mats_equal(b, exp, 16, 16);
    return result;
}

bool test04() {
    float a[16 * 16];
    float b[16 * 16];
    float exp[16 * 16];

    fill_indices(a, 16 * 16);

    identity_16_16(a, b, 16, 16, 1);
    ref_identity_16_16(a, exp, 16, 16, 1);

    bool result = mats_equal(b, exp, 16, 16);
    return result;
}

bool test05() {
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
    return result;
} 

bool test06() {
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
    return result;
} 

bool test07() {
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
    return result;
}

bool test08() {
    int const rows = 512;
    float a[rows * rows];
    float b[rows * rows];
    float exp[rows * rows];

    fill_indices(a, rows * rows);
    fill_const(b, rows * rows, -5.0);
    fill_const(exp, rows * rows, -5.0);

    int sub_off = rows / 4 + rows / 4 * rows;
    identity_16_16(a + sub_off, b + sub_off, rows, rows, 0);
    ref_identity_16_16(a + sub_off, exp + sub_off, rows, rows, 0);

    bool result = mats_equal(b, exp, rows, rows);
    return result;
}

bool test09() {
    int const rows = 512;
    float a[rows * rows];
    float b[rows * rows];
    float exp[rows * rows];

    fill_indices(a, rows * rows);
    fill_const(b, rows * rows, -5.0);
    fill_const(exp, rows * rows, -5.0);

    int sub_off = rows / 4 + rows / 4 * rows;
    identity_16_16(a + sub_off, b + sub_off, rows, rows, 1);
    ref_identity_16_16(a + sub_off, exp + sub_off, rows, rows, 1);

    bool result = mats_equal(b, exp, rows, rows);
    return result;
}

bool test10() {
    int const rows = 512;
    int sub_off = rows / 4 + rows / 4 * rows;

    float a[rows * rows];
    float b[rows * rows];
    float exp[rows * rows];

    fill_indices(a, rows * rows);
    fill_const(b, rows * rows, -5.0);
    fill_const(exp, rows * rows, -5.0);
    for (int i = 0; i < 16; i++) {
        (a + sub_off)[i*16] = -1.0;
    }

    relu_16_16(a + sub_off, b + sub_off, rows, rows, 0);
    ref_relu_16_16(a + sub_off, exp + sub_off, rows, rows, 0);

    bool result = mats_equal(b, exp, rows, rows);
    return result;
} 

bool test11() {
    int const rows = 512;
    int sub_off = rows / 4 + rows / 4 * rows;
    
    float a[rows * rows];
    float b[rows * rows];
    float exp[rows * rows];

    fill_indices(a, rows * rows);
    fill_const(b, rows * rows, -5.0);
    fill_const(exp, rows * rows, -5.0);
    for (int i = 0; i < 16; i++) {
        (a + sub_off)[i*16] = -1.0;
    }

    relu_16_16(a + sub_off, b + sub_off, rows, rows, 1);
    ref_relu_16_16(a + sub_off, exp + sub_off, rows, rows, 1);

    bool result = mats_equal(b, exp, rows, rows);
    return result;
} 

















int main() {
    // initialize global JIT-compiled kernel function pointers
    generate_kernels();

	// The array of the tests to run. 
	unit_test_t tests[] = {
		MAKE_TEST(test01),
        MAKE_TEST(test02),
        MAKE_TEST(test03),
        MAKE_TEST(test04),
        MAKE_TEST(test05),
        MAKE_TEST(test06),
        MAKE_TEST(test07),
        MAKE_TEST(test08),
        MAKE_TEST(test09),
        MAKE_TEST(test10),
        MAKE_TEST(test11),
	};
	int tests_count = sizeof(tests) / sizeof(unit_test_t);

	// Run all tests.
	int failure_count = 0;
	int success_count = 0;
	bool all_successful = true;
	for (int i = 0; i < tests_count; i++) {
		bool success = tests[i].call();
		if (!success) {
			std::cout << "\"" << tests[i].test_name << "\": FAIL" << std::endl;
			failure_count += 1;
		} else {
			std::cout << "\"" << tests[i].test_name << "\": SUCCESS" << std::endl;
			success_count += 1;
		}
		all_successful &= success;
	}

	// Print results.
	std::cout << "Successes: " << success_count << " / " << tests_count << std::endl;
	std::cout << "Failures: " << failure_count << " / " << tests_count << std::endl;
	return all_successful ? 0 : 1;
}





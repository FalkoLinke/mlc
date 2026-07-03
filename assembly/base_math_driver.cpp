#include <iostream>
#include <catch2/catch_test_macros.hpp>

#include "base_math_cpp.h"
#include "base_math_s.h"













/**
 * Prints a matrix to `stdout`.
 * 
 * Parameters:
 * - `a`: The matrix to print stored in row-major order.
 * - `m`: The number of rows of `a`.
 * - `n`: The number of columns of `a`.
 */
void print_mat(uint64_t const* a, uint32_t const m, uint32_t const n) {
	for (uint32_t r = 0; r < m; r++) {
		for (uint32_t c = 0; c < n; c++) {
			std::cout << a[r * n + c] << " ";
		}
		std::cout << std::endl;
	}
}

/**
 * Checks whether all elements at equal indices of two matrices are equal.
 * 
 * Parameters:
 * - `a`: The first matrix in row-major order.
 * - `b`: The second matrix in row-major order.
 * - `m`: The number of rows of `a` and `b`.
 * - `n`: The number of columns of `a` and `b`.
 * 
 * Returns: 
 * - `true` if all elements at equal indices are equal, `false` otherwise.
 */
bool mats_equal(uint64_t const* a, uint64_t const* b, uint32_t const m, uint32_t const n) {
	for (uint32_t r = 0; r < m; r++) {
		for (uint32_t c = 0; c < n; c++) {
			if (a[r * n + c] != b[r * n + c]) {
				return false;
			}
		}
	}
	return true;
}




















TEST_CASE("test01", "[test]") {
	int a = 5;
	int b = 7;
	int c = add(a, b);
	int exp = 12;
	REQUIRE(c == exp);
}

/* Test inner product. */
TEST_CASE("test02", "[test]") {
	uint32_t a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
	uint32_t b[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
	uint32_t size = sizeof(a) / sizeof(uint32_t);
	int c = inner_product(a, b, size);
	int exp = 36;
	REQUIRE(c == exp);
}

/* Test inner product. */
TEST_CASE("test03", "[test]") {
	uint32_t a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
	uint32_t b[] = {0, 1, 0, 1, 0, 1, 0, 1, 0};
	uint32_t size = sizeof(a) / sizeof(uint32_t);
	int c = inner_product(a, b, size);
	int exp = 16;
	REQUIRE(c == exp);
}

/* Test outer product. */
TEST_CASE("test04", "[test]") {
	uint32_t a[] = {1, 2, 3};
	uint32_t b[] = {1, 2, 3};
	uint64_t c[9] = {0};
	uint32_t size = sizeof(a) / sizeof(uint32_t);
	uint64_t exp[9] = {
		1, 2, 3,
		2, 4, 6,
		3, 6, 9,
	};
	outer_product(a, b, size, c);
	REQUIRE(mats_equal(c, exp, size, size));
}

/* Test outer product. */
TEST_CASE("test05", "[test]") {
	uint32_t a[] = {2, 4, 2, 4};
	uint32_t b[] = {1, 3, 1, 3};
	uint64_t c[16] = {0};
	uint64_t exp[16] = {0};
	uint32_t size = sizeof(a) / sizeof(uint32_t);
	
	outer_product(a, b, size, c);
	outer_product_cpp(a, b, size, exp);
	REQUIRE(mats_equal(c, exp, size, size));
}

/* Test squared magnitude using inner product. */
TEST_CASE("test06", "[test]") {
	uint32_t a[] = {2, 4, 6};
	uint32_t size = sizeof(a) / sizeof(uint32_t);
	int64_t c = inner_product(a, a, size);
	int64_t exp = 56;
	REQUIRE(exp == c);
}

/* Test empty inner product. */
TEST_CASE("test07", "[test]") {
	uint32_t a[] = {};
	uint32_t b[] = {};
	uint32_t size = sizeof(a) / sizeof(uint32_t);
	int64_t c = inner_product(a, b, size);
	int64_t exp = 0;
	REQUIRE(exp == c);
}

/* Test empty outer product. */
TEST_CASE("test08", "[test]") {
	uint32_t a[] = {};
	uint32_t b[] = {};
	uint32_t size = sizeof(a) / sizeof(uint32_t);
	uint64_t c[] = {};
	uint64_t exp[] = {};
	outer_product(a, b, size, c);
	REQUIRE(mats_equal(c, exp, size, size));
}












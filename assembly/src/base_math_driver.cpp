#include <iostream>
#include "base_math_cpp.h"
#include "base_math_s.h"




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
 * Prints a matrix to `stdout`.
 * 
 * Parameters:
 * - `a`: The matrix to print stored in row-major order.
 * - `m`: The number of rows of `a`.
 * - `n`: The number of columns of `a`.
 */
void print_mat(uint64_t const* a, uint32_t const m, uint32_t const n) {
	for (int r = 0; r < m; r++) {
		for (int c = 0; c < n; c++) {
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
	for (int r = 0; r < m; r++) {
		for (int c = 0; c < n; c++) {
			if (a[r * n + c] != b[r * n + c]) {
				return false;
			}
		}
	}
	return true;
}





















bool test01() {
	int a = 5;
	int b = 7;
	int c = add(a, b);
	int exp = 12;
	return c == exp;
}

/* Test inner product. */
bool test02() {
	uint32_t a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
	uint32_t b[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
	uint32_t size = sizeof(a) / sizeof(uint32_t);
	int c = inner_product(a, b, size);
	int exp = 36;
	return c == exp;
}

/* Test inner product. */
bool test03() {
	uint32_t a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
	uint32_t b[] = {0, 1, 0, 1, 0, 1, 0, 1, 0};
	uint32_t size = sizeof(a) / sizeof(uint32_t);
	int c = inner_product(a, b, size);
	int exp = 16;
	return c == exp;
}

/* Test outer product. */
bool test04() {
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
	return mats_equal(c, exp, size, size);
}

/* Test outer product. */
bool test05() {
	uint32_t a[] = {2, 4, 2, 4};
	uint32_t b[] = {1, 3, 1, 3};
	uint64_t c[16] = {0};
	uint64_t exp[16] = {0};
	uint32_t size = sizeof(a) / sizeof(uint32_t);
	
	outer_product(a, b, size, c);
	outer_product_cpp(a, b, size, exp);
	return mats_equal(c, exp, size, size);
}

/* Test squared magnitude using inner product. */
bool test06() {
	uint32_t a[] = {2, 4, 6};
	uint32_t size = sizeof(a) / sizeof(uint32_t);
	int64_t c = inner_product(a, a, size);
	int64_t exp = 56;
	return exp == c;
}

/* Test empty inner product. */
bool test07() {
	uint32_t a[] = {};
	uint32_t b[] = {};
	uint32_t size = sizeof(a) / sizeof(uint32_t);
	int64_t c = inner_product(a, b, size);
	int64_t exp = 0;
	return exp == c;
}

/* Test empty outer product. */
bool test08() {
	uint32_t a[] = {};
	uint32_t b[] = {};
	uint32_t size = sizeof(a) / sizeof(uint32_t);
	uint64_t c[] = {};
	uint64_t exp[] = {};
	outer_product(a, b, size, c);
	return mats_equal(c, exp, size, size);
}












int main() {
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





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












void print_mat(const uint64_t* a, const uint32_t m, const uint32_t n) {
	for (int r = 0; r < m; r++) {
		for (int c = 0; c < n; c++) {
			std::cout << a[r * n + c] << " ";
		}
		std::cout << std::endl;
	}
}

bool mats_equal(const uint64_t* a, const uint64_t* b, const uint32_t m, const uint32_t n) {
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

bool test02() {
	uint32_t a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
	uint32_t b[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
	uint32_t size = sizeof(a) / sizeof(uint32_t);
	int c = inner_product(a, b, size);
	int exp = 36;
	return c == exp;
}

bool test03() {
	uint32_t a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
	uint32_t b[] = {0, 1, 0, 1, 0, 1, 0, 1, 0};
	uint32_t size = sizeof(a) / sizeof(uint32_t);
	int c = inner_product(a, b, size);
	int exp = 16;
	return c == exp;
}

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



















int main() {
	unit_test_t tests[] = {
		MAKE_TEST(test01),
		MAKE_TEST(test02),
		MAKE_TEST(test03),
		MAKE_TEST(test04),
	};
	int tests_count = sizeof(tests) / sizeof(unit_test_t);

	bool all_successful = true;
	for (int i = 0; i < tests_count; i++) {
		bool success = tests[i].call();
		if (!success) {
			std::cout << "\"" << tests[i].test_name << "\": FAIL" << std::endl;
		} else {
			std::cout << "\"" << tests[i].test_name << "\": SUCCESS" << std::endl;
		}
		all_successful &= success;
	}

	return all_successful ? 0 : 1;
}





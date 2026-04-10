#include <iostream>
#include "base_math_cpp.h"
#include "base_math_s.h"


void print_mat(const uint64_t* a, const uint32_t m, const uint32_t n) {
	for (int r = 0; r < m; r++) {
		for (int c = 0; c < n; c++) {
			std::cout << a[r * n + c] << " ";
		}
		std::cout << std::endl;
	}
}





void test01() {
	int a = 5;
	int b = 7;
	int c = add(a, b);
	std::cout << "Result: " << c << std::endl;
}

void test02() {
	uint32_t a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
	uint32_t b[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
	uint32_t size = sizeof(a) / sizeof(uint32_t);
	int c = inner_product(a, b, size);
	std::cout << "Result: " << c << std::endl;	
}

void test03() {
	uint32_t a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
	uint32_t b[] = {0, 1, 0, 1, 0, 1, 0, 1, 0};
	uint32_t size = sizeof(a) / sizeof(uint32_t);
	int c = inner_product(a, b, size);
	std::cout << "Result: " << c << std::endl;
}

void test04() {
	uint32_t a[] = {1, 2, 3};
	uint32_t b[] = {1, 2, 3};
	uint64_t c[9] = {0};
	uint32_t size = sizeof(a) / sizeof(uint32_t);
	outer_product(a, b, size, c);
	print_mat(c, size, size);
}

int main() {
	test01();
	test02();
	test03();
	test04();
	return 0;
}





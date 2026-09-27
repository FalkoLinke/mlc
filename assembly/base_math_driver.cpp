#include <cstdint>
#include <random>
#include <vector>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "base_math_cpp.h"
#include "base_math_s.h"




/**
 * Fills `size` elements with random values in `[0, max]`.
 *
 * Parameters:
 * - `size`: The number of elements.
 * - `max`: The largest value that may be generated.
 * - `seed`: The seed of the random number generator.
 *
 * Returns:
 * - A vector holding the random values.
 */
std::vector<uint32_t> random_vector(uint32_t const size, uint32_t const max, uint32_t const seed) {
	std::mt19937 gen(seed);
	std::uniform_int_distribution<uint32_t> dist(0, max);
	std::vector<uint32_t> v(size);
	for (uint32_t& e : v) {
		e = dist(gen);
	}
	return v;
}




TEST_CASE("add adds two integers", "[assembly][add]") {
	REQUIRE(add(5, 7) == 12);
	REQUIRE(add(-5, 7) == 2);
	REQUIRE(add(0, 0) == 0);
}




TEST_CASE("inner_product of fixed vectors", "[assembly][inner_product]") {
	uint32_t a[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};

	SECTION("with vector of ones") {
		uint32_t b[] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
		REQUIRE(inner_product(a, b, 9) == 36);
	}

	SECTION("with alternating vector") {
		uint32_t b[] = {0, 1, 0, 1, 0, 1, 0, 1, 0};
		REQUIRE(inner_product(a, b, 9) == 16);
	}

	SECTION("squared magnitude") {
		uint32_t c[] = {2, 4, 6};
		REQUIRE(inner_product(c, c, 3) == 56);
	}

	SECTION("single element") {
		uint32_t c[] = {7};
		uint32_t d[] = {6};
		REQUIRE(inner_product(c, d, 1) == 42);
	}
}

TEST_CASE("inner_product of empty vectors is zero", "[assembly][inner_product]") {
	REQUIRE(inner_product(nullptr, nullptr, 0) == 0);
}

TEST_CASE("inner_product does not overflow 32 bits", "[assembly][inner_product]") {
	uint32_t a[] = {100000, 100000};
	uint32_t b[] = {100000, 100000};
	REQUIRE(inner_product(a, b, 2) == int64_t{20000000000});
}

TEST_CASE("inner_product matches C++ reference", "[assembly][inner_product]") {
	uint32_t size = GENERATE(1, 2, 3, 7, 16, 100, 1000);
	// values below 2^15 keep the reference implementation free of 32 bit overflows
	std::vector<uint32_t> a = random_vector(size, 1 << 15, size);
	std::vector<uint32_t> b = random_vector(size, 1 << 15, size + 1);

	REQUIRE(inner_product(a.data(), b.data(), size) == inner_product_cpp(a.data(), b.data(), size));
}




TEST_CASE("outer_product of fixed vectors", "[assembly][outer_product]") {
	uint32_t a[] = {1, 2, 3};
	uint32_t b[] = {1, 2, 3};
	std::vector<uint64_t> c(9, 0);
	std::vector<uint64_t> exp = {
		1, 2, 3,
		2, 4, 6,
		3, 6, 9,
	};
	outer_product(a, b, 3, c.data());
	REQUIRE(c == exp);
}

TEST_CASE("outer_product is not symmetric for different vectors", "[assembly][outer_product]") {
	uint32_t a[] = {1, 2};
	uint32_t b[] = {3, 4};
	std::vector<uint64_t> c(4, 0);
	std::vector<uint64_t> exp = {
		3, 4,
		6, 8,
	};
	outer_product(a, b, 2, c.data());
	REQUIRE(c == exp);
}

TEST_CASE("outer_product of empty vectors does not write", "[assembly][outer_product]") {
	uint64_t sentinel = 0xDEADBEEF;
	outer_product(nullptr, nullptr, 0, &sentinel);
	REQUIRE(sentinel == 0xDEADBEEF);
}

TEST_CASE("outer_product does not overflow 32 bits", "[assembly][outer_product]") {
	uint32_t a[] = {0xFFFFFFFF};
	uint32_t b[] = {0xFFFFFFFF};
	uint64_t c = 0;
	outer_product(a, b, 1, &c);
	REQUIRE(c == uint64_t{0xFFFFFFFF} * uint64_t{0xFFFFFFFF});
}

TEST_CASE("outer_product matches C++ reference", "[assembly][outer_product]") {
	uint32_t size = GENERATE(1, 2, 3, 7, 16, 100);
	// values below 2^15 keep the reference implementation free of 32 bit overflows
	std::vector<uint32_t> a = random_vector(size, 1 << 15, size);
	std::vector<uint32_t> b = random_vector(size, 1 << 15, size + 1);
	std::vector<uint64_t> c(size * size, 0);
	std::vector<uint64_t> exp(size * size, 0);

	outer_product(a.data(), b.data(), size, c.data());
	outer_product_cpp(a.data(), b.data(), size, exp.data());
	REQUIRE(c == exp);
}

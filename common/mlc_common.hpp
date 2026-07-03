#ifndef COMMON_MLC_COMMON_HPP
#define COMMON_MLC_COMMON_HPP


#include <cstdint>
#include <iostream>
#include <math.h>









template <typename T>
void fill_const(T* buffer, uint64_t size, T value) {
    for (uint64_t i = 0; i < size; i++) {
        buffer[i] = value;
    }
}

template <typename T>
void fill_const(T* mat, uint64_t m, uint64_t n, T value) {
    fill_const(mat, m * n, value);
}

template <typename T>
void fill_const(T* mat, uint64_t m, uint64_t n, int64_t ld, T value) {
    for (uint64_t c = 0; c < n; c++) {
        for (uint64_t r = 0; r < m; r++) {
            *(mat + c * ld + r) = value;
        }
    }
}







template <typename T>
void fill_indices(T* buffer, uint64_t size) {
    for (uint64_t i = 0; i < size; i++) {
        buffer[i] = static_cast<T>(i);
    }
}

template <typename T>
void fill_indices(T* mat, uint64_t m, uint64_t n) {
    fill_indices(mat, m * n);
}

template <typename T>
void fill_indices(T* mat, uint64_t m, uint64_t n, int64_t ld) {
    uint64_t i = 0;
    for (uint64_t c = 0; c < n; c++) {
        for (uint64_t r = 0; r < m; r++) {
            *(mat + c * ld + r) = static_cast<T>(i);
            i += 1;
        }
    }
}







/**
 * @brief Checks if all of the elements of A and B are equal using the default equality operator.
 * @param a: Column major matrix A.
 * @param b: Column major matrix B.
 * @param m: Number of rows of A.
 * @param n: Number of columns of B.
 * @param lda: Leading dimension of A.
 * @param ldb: Leading dimension of B.
 */
template <typename T>
bool mats_equal(T const* a, T const* b, uint64_t m, uint64_t n, int64_t lda, int64_t ldb) {
    for (uint64_t c = 0; c < n; c++) {
        for (uint64_t r = 0; r < m; r++) {
            if (!(*(a + c * lda + r) == *(b + c * ldb + r))) {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Checks if all of the elements of A and B are equal using the default equality operator.
 * @param a: Column major matrix A.
 * @param b: Column major matrix B.
 * @param m: Number of rows of A.
 * @param n: Number of columns of B.
 */
template <typename T>
bool mats_equal(T const* a, T const* b, uint64_t m, uint64_t n) {
    return mats_equal<T>(a, b, m, n, m, m);
}

template <typename T>
double max_abs_diff(T const* a, T const* b, uint64_t size) {
    double result = 0.0;
    for (uint64_t i = 0; i < size; i++) {
        double va = static_cast<double>(a[i]);
        double vb = static_cast<double>(b[i]);
        result = fmax(result, fabs(va - vb));
    }
    return result;
}









/**
 * @brief Performs an identity operation from A into B.
 * 
 * `a` and `b` may point to the same matrix.
 * 
 * @param a: The column-major matrix A.
 * @param b: The matrix B.
 * @param m: Number of rows of A.
 * @param n: Number of columns of A.
 * @param lda: The leading dimension of A.
 * @param ldb: The leading dimension of B.
 * @param trans_b: Column-major matrix B if `false`, row-major otherwise.
 */
template <typename T>
void identity(T const* a, T* b, uint64_t m, uint64_t n, int64_t lda, int64_t ldb, bool trans_b) {
    int64_t sbc = trans_b ? 1 : ldb;
    int64_t sbr = trans_b ? ldb : 1;

    for (uint64_t c = 0; c < n; c++) {
        for (uint64_t r = 0; r < m; r++) {
            *(b + c * sbc + r * sbr) = *(a + c * lda + r);
        }
    }
}

/**
 * @brief Performs an zero operation on A.
 * 
 * @param a: The column-major matrix A.
 * @param m: Number of rows of A.
 * @param n: Number of columns of A.
 * @param lda: The leading dimension of A.
 */
template <typename T>
void zero(T* a, uint64_t m, uint64_t n, int64_t lda) {
    fill_const(a, m, n, lda, static_cast<T>(0));
}


/**
 * @brief Performs an relu operation from A into B.
 * 
 * `a` and `b` may point to the same matrix.
 * 
 * @param a: The column-major matrix A.
 * @param b: The matrix B.
 * @param m: Number of rows of A.
 * @param n: Number of columns of A.
 * @param lda: The leading dimension of A.
 * @param ldb: The leading dimension of B.
 * @param trans_b: Column-major matrix B if `false`, row-major otherwise.
 */
template <typename T>
void relu(T const* a, T* b, uint64_t m, uint64_t n, int64_t lda, int64_t ldb, bool trans_b) {
    int64_t sbc = trans_b ? 1 : ldb;
    int64_t sbr = trans_b ? ldb : 1;

    for (uint64_t c = 0; c < n; c++) {
        for (uint64_t r = 0; r < m; r++) {
            *(b + c * sbc + r * sbr) = std::max(*(a + c * lda + r), static_cast<T>(0));
        }
    }
}














/**
 * @brief Prints the m x n matrix A to stdout.
 * @param a: The column major matrix A.
 * @param m: The number of rows of A.
 * @param n: The number of columns of A.
 * @param lda: The leading dimension of A.
 */
template <typename T>
void print_mat(T const* a, uint64_t m, uint64_t n, int64_t lda) {
    for (uint64_t r = 0; r < m; r++) {
        for (uint64_t c = 0; c < n; c++) {
            std::cout << *(a + c * lda + r) << " ";
        }
        std::cout << std::endl;
    }
}



#endif /*COMMON_MLC_COMMON_HPP*/
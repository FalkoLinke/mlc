#ifndef COMMON_MLC_COMMON_HPP
#define COMMON_MLC_COMMON_HPP


#include <cstdint>
#include <iostream>
#include <math.h>
#include <cmath>
#include <algorithm>



/**
 * @brief Checks that `func(arg)` preserves the callee-saved registers d8 - d15 (AAPCS64).
 *
 * Kernels which switch to streaming mode have to save these registers,
 * because smstart and smstop zero all vector registers.
 */
inline bool callee_saved_fp_preserved(void (*func)(void*), void* arg) {
    double out[8] = {0.0};
    asm volatile(
        "mov x19, %[out]\n"
        "mov x0, %[arg]\n"
        "fmov d8, #1.0\n"
        "fmov d9, #2.0\n"
        "fmov d10, #3.0\n"
        "fmov d11, #4.0\n"
        "fmov d12, #5.0\n"
        "fmov d13, #6.0\n"
        "fmov d14, #7.0\n"
        "fmov d15, #8.0\n"
        "blr %[func]\n"
        "stp d8, d9, [x19]\n"
        "stp d10, d11, [x19, #16]\n"
        "stp d12, d13, [x19, #32]\n"
        "stp d14, d15, [x19, #48]\n"
        :
        : [func] "r"(func), [arg] "r"(arg), [out] "r"(out)
        : "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15",
          "x16", "x17", "x19", "x30",
          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15",
          "v16", "v17", "v18", "v19", "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31",
          "memory", "cc"
    );
    for (int i = 0; i < 8; i++) {
        if (out[i] != (double)(i + 1)) {
            return false;
        }
    }
    return true;
}









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




/**
 * @brief Reference implementation of a GEMM computing C += A * B.
 *
 * The accumulation is performed in `double` in order to provide a more accurate
 * result than the kernels, which accumulate in FP32.
 *
 * @param a: The column major matrix A with m rows and k columns.
 * @param b: The row major matrix B with k rows and n columns.
 * @param c: The column major matrix C with m rows and n columns.
 * @param m: The number of rows of A and C.
 * @param n: The number of columns of B and C.
 * @param k: The number of columns of A and rows of B.
 * @param lda: The leading dimension of A.
 * @param ldb: The leading dimension of B.
 * @param ldc: The leading dimension of C.
 */
template <typename T>
void gemm_ref(T const* a, T const* b, T* c,
              uint64_t m, uint64_t n, uint64_t k,
              int64_t lda, int64_t ldb, int64_t ldc) {
    for (uint64_t col = 0; col < n; col++) {
        for (uint64_t row = 0; row < m; row++) {
            double sum = 0.0;
            for (uint64_t i = 0; i < k; i++) {
                sum += static_cast<double>(*(a + i * lda + row)) * static_cast<double>(*(b + i * ldb + col));
            }
            *(c + col * ldc + row) += static_cast<T>(sum);
        }
    }
}


/**
 * @brief Determines the largest relative deviation between the m x n matrices A and B.
 *
 * Elements whose reference value is close to zero are compared absolutely.
 *
 * @param a: The column major matrix A.
 * @param b: The column major reference matrix B.
 * @param m: The number of rows of A and B.
 * @param n: The number of columns of A and B.
 * @param lda: The leading dimension of A.
 * @param ldb: The leading dimension of B.
 */
template <typename T>
double max_rel_diff(T const* a, T const* b, uint64_t m, uint64_t n, int64_t lda, int64_t ldb) {
    double max_diff = 0.0;
    for (uint64_t c = 0; c < n; c++) {
        for (uint64_t r = 0; r < m; r++) {
            double const va = static_cast<double>(*(a + c * lda + r));
            double const vb = static_cast<double>(*(b + c * ldb + r));
            double const scale = std::abs(vb) > 1.0 ? std::abs(vb) : 1.0;
            double const diff = std::abs(va - vb) / scale;
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
    }
    return max_diff;
}


#endif /*COMMON_MLC_COMMON_HPP*/
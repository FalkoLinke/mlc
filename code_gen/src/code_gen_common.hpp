#ifndef CODE_GEN_COMMON_H
#define CODE_GEN_COMMON_H


#include <cstdint>




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

template <typename T>
bool mats_equal(T const* a, T const* b, uint64_t m, uint64_t n) {
    return mats_equal<T>(a, b, m, n, m, m);
}











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

template <typename T>
void zero(T* a, uint64_t m, uint64_t n, int64_t lda) {
    fill_const(a, m, n, lda, static_cast<T>(0));
}

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















template <typename T>
void print_mat(T const* a, uint64_t m, uint64_t n, int64_t lda) {
    for (uint64_t r = 0; r < m; r++) {
        for (uint64_t c = 0; c < n; c++) {
            std::cout << *(a + c * lda + r) << " ";
        }
        std::cout << std::endl;
    }
}








#endif /*CODE_GEN_COMMON_H*/
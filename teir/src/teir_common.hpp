#ifndef TEIR_COMMON_HPP
#define TEIR_COMMON_HPP


#include <cstdint>
#include <iostream>




template <typename T>
void teir_ab_ba_v1(T const* in0, T* out, uint64_t da, uint64_t db, int64_t in0_sa, int64_t in0_sb, int64_t out_sa, int64_t out_sb) {
    for (uint64_t ia = 0; ia < da; ia++) {
        for (uint64_t ib = 0; ib < db; ib++) {
            T const* in0_ptr = in0 + ia * in0_sa + ib * in0_sb;
            T* out_ptr       = out + ia * out_sa + ib * out_sb;
            *out_ptr = *in0_ptr;
        }
    }
}











template <typename T>
void teir_abcd_dcba_v1(T const* in0, T* out, uint64_t da, uint64_t db, uint64_t dc, uint64_t dd) {
    int64_t in0_sd = 1;
    int64_t in0_sc = dd * in0_sd;
    int64_t in0_sb = dc * in0_sc;
    int64_t in0_sa = db * in0_sb;

    int64_t out_sa = 1;
    int64_t out_sb = da * out_sa;
    int64_t out_sc = db * out_sb;
    int64_t out_sd = dc * out_sc;

    for (uint64_t ia = 0; ia < da; ia++) {
        for (uint64_t ib = 0; ib < db; ib++) {
            for (uint64_t ic = 0; ic < dc; ic++) {
                for (uint64_t id = 0; id < dd; id++) {
                    T const* in0_ptr = in0 + in0_sa * ia + in0_sb * ib + in0_sc * ic + in0_sd * id;
                    T* out_ptr       = out + out_sa * ia + out_sb * ib + out_sc * ic + out_sd * id;
                    *out_ptr = *in0_ptr;
                }
            }
        }
    }
}


template <typename T>
void teir_abcd_dcba_v2(T const* in0, T* out, uint64_t da, uint64_t db, uint64_t dc, uint64_t dd) {
    int64_t in0_sd = 1;
    int64_t in0_sc = dd * in0_sd;
    int64_t in0_sb = dc * in0_sc;
    int64_t in0_sa = db * in0_sb;

    int64_t out_sa = 1;
    int64_t out_sb = da * out_sa;
    int64_t out_sc = db * out_sb;
    int64_t out_sd = dc * out_sc;

    for (uint64_t ib = 0; ib < db; ib++) {
        for (uint64_t ic = 0; ic < dc; ic++) {
            T const* in0_ptr = in0 + in0_sb * ib + in0_sc * ic;
            T* out_ptr       = out + out_sb * ib + out_sc * ic;
            teir_ab_ba_v1(in0_ptr, out_ptr, da, dd, in0_sa, in0_sd, out_sa, out_sd);
        }
    }
    /*
    for (uint64_t ib = 0; ib < db; ib++) {
        for (uint64_t ic = 0; ic < dc; ic++) {
            T const* in0_ptr1 = in0 + in0_sb * ib + in0_sc * ic;
            T* out_ptr1       = out + out_sb * ib + out_sc * ic;

            for (uint64_t ia = 0; ia < da; ia++) {
                for (uint64_t id = 0; id < dd; id++) {
                    T const* in0_ptr2 = in0_ptr1 + in0_sa * ia + in0_sd * id;
                    T* out_ptr2       = out_ptr1 + out_sa * ia + out_sd * id;
                    *out_ptr2 = *in0_ptr2;
                }
            }
        }
    }
    */
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



#endif /*TEIR_COMMON_HPP*/
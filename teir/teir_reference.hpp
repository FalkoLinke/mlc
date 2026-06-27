#ifndef TEIR_COMMON_HPP
#define TEIR_COMMON_HPP


#include <cstdint>
#include <iostream>
#include <math.h>



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
void teir_abc_acb(T const* in0, T* out, uint64_t da, uint64_t db, uint64_t dc) {
    uint64_t in0_sc = 1;
    uint64_t in0_sb = dc * in0_sc;
    uint64_t in0_sa = db * in0_sb;

    uint64_t out_sb = 1;
    uint64_t out_sc = db * out_sb;
    uint64_t out_sa = dc * out_sc;

    for (uint64_t ia = 0; ia < da; ia++) {
        for (uint64_t ib = 0; ib < db; ib++) {
            for (uint64_t ic = 0; ic < dc; ic++) {
                T const* in0_ptr = in0 + ia * in0_sa + ib * in0_sb + ic * in0_sc;
                T* out_ptr       = out + ia * out_sa + ib * out_sb + ic * out_sc;
                *out_ptr = *in0_ptr;
            }
        }
    }
}













template <typename T>
void teir_bkm_bkn_bnm(T const* in0, T const* in1, T* out, uint64_t db, uint64_t dk, uint64_t dm, uint64_t dn) {
    uint64_t const in0_sm = 1;
    uint64_t const in0_sk = dm * in0_sm;
    uint64_t const in0_sb = dk * in0_sk;
    uint64_t const in0_sn = 0;
    
    uint64_t const in1_sn = 1;
    uint64_t const in1_sk = dn * in1_sn;
    uint64_t const in1_sb = dk * in1_sk;
    uint64_t const in1_sm = 0;

    uint64_t const out_sm = 1;
    uint64_t const out_sn = dm * out_sm;
    uint64_t const out_sb = dn * out_sn;
    uint64_t const out_sk = 0;

    for (uint64_t ib = 0; ib < db; ib++) {
        for (uint64_t im = 0; im < dm; im++) {
            for (uint64_t in = 0; in < dn; in++) {
                for (uint64_t ik = 0; ik < dk; ik++) {
                    T const* in0_ptr = in0 + ib * in0_sb + im * in0_sm + in * in0_sn + ik * in0_sk;
                    T const* in1_ptr = in1 + ib * in1_sb + im * in1_sm + in * in1_sn + ik * in1_sk;
                    T* out_ptr       = out + ib * out_sb + im * out_sm + in * out_sn + ik * out_sk;
                    *out_ptr += *in0_ptr * *in1_ptr;
                }
            }
        }
    }
}













#endif /*TEIR_COMMON_HPP*/
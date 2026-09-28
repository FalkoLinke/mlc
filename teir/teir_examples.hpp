#ifndef TEIR_TEIR_EXAMPLES_HPP
#define TEIR_TEIR_EXAMPLES_HPP

/**
 * Builders for the example operations of `teir/data` and a batched GEMM operation.
 *
 * The strides are derived from the extents in the same way as in the `.teir` files,
 * so the examples can be instantiated with smaller extents for testing.
 * `parallel = false` replaces all `parallel` policies by `sequential`.
 */

#include <cstdint>
#include <random>
#include <vector>

#include "teir.h"


static constexpr uint64_t teir_fp32 = sizeof(float);

inline teir_policy_t teir_example_policy(bool parallel) {
    return parallel ? teir_policy_t::policy_parallel : teir_policy_t::policy_sequential;
}


/** Number of floats of a tensor whose largest stride is `outer_stride` bytes along an axis of extent `outer_extent`. */
inline uint64_t teir_example_floats(uint64_t outer_extent, uint64_t outer_stride) {
    return outer_extent * outer_stride / teir_fp32;
}


/**
 * `data/matmul.teir`: mk,kn->mn as m0k0m1k1,k0n0k1n1->m0n0m1n1.
 * File extents: m0=256, m1=32, n0=128, n1=64, k0=16, k1=512.
 * Tensor order: in0, in1, out.
 */
inline teir_operation teir_example_matmul(uint64_t m0, uint64_t m1, uint64_t n0, uint64_t n1, uint64_t k0, uint64_t k1, bool parallel = true) {
    // in0: m0 k0 m1 k1
    uint64_t in0_k1 = teir_fp32;
    uint64_t in0_m1 = k1 * in0_k1;
    uint64_t in0_k0 = m1 * in0_m1;
    uint64_t in0_m0 = k0 * in0_k0;
    // in1: k0 n0 k1 n1
    uint64_t in1_n1 = teir_fp32;
    uint64_t in1_k1 = n1 * in1_n1;
    uint64_t in1_n0 = k1 * in1_k1;
    uint64_t in1_k0 = n0 * in1_n0;
    // out: m0 n0 m1 n1
    uint64_t out_n1 = teir_fp32;
    uint64_t out_m1 = n1 * out_n1;
    uint64_t out_n0 = m1 * out_m1;
    uint64_t out_m0 = n0 * out_n0;

    return teir_operation(
        "matmul",
        {
            teir_tensor("in0", teir_dtype_t::dtype_fp32),
            teir_tensor("in1", teir_dtype_t::dtype_fp32),
            teir_tensor("out", teir_dtype_t::dtype_fp32),
        },
        {
            teir_axis("m0", m0, {in0_m0, 0,      out_m0}, {0, 0, 0}),
            teir_axis("m1", m1, {in0_m1, 0,      out_m1}, {0, 0, 0}),
            teir_axis("n0", n0, {0,      in1_n0, out_n0}, {0, 0, 0}),
            teir_axis("n1", n1, {0,      in1_n1, out_n1}, {0, 0, 0}),
            teir_axis("k0", k0, {in0_k0, in1_k0, 0     }, {0, 0, 0}),
            teir_axis("k1", k1, {in0_k1, in1_k1, 0     }, {0, 0, 0}),
        },
        {
            teir_primitive("zero", teir_ptype_t::ptype_zero,     {"out"},               {{"M", {"m1"}}, {"N", {"n1"}}},                  {{"data_type", "f32"}}),
            teir_primitive("gemm", teir_ptype_t::ptype_contract, {"in0", "in1", "out"}, {{"M", {"m1"}}, {"N", {"n1"}}, {"K", {"k1"}}}, {{"data_type", "f32"}}),
        },
        teir_schedule(
            {"iter_k0"},
            {
                teir_iter_node("iter_k0", "k0", teir_policy_t::policy_sequential, {"inv_zero", "iter_m0"}),
                teir_iter_node("iter_m0", "m0", teir_example_policy(parallel),    {"iter_n0"}),
                teir_iter_node("iter_n0", "n0", teir_example_policy(parallel),    {"inv_gemm"}),
            },
            {
                teir_inv_node("inv_zero", "zero"),
                teir_inv_node("inv_gemm", "gemm"),
            }
        )
    );
}


/**
 * `data/contraction.teir`: pqtu,trus->pqrs.
 * File extents: p=128, q=96, r=96, s=64, t=32, u=256.
 * Tensor order: in0, in1, out.
 */
inline teir_operation teir_example_contraction(uint64_t p, uint64_t q, uint64_t r, uint64_t s, uint64_t t, uint64_t u, bool parallel = true) {
    // in0: p q t u
    uint64_t in0_u = teir_fp32;
    uint64_t in0_t = u * in0_u;
    uint64_t in0_q = t * in0_t;
    uint64_t in0_p = q * in0_q;
    // in1: t r u s
    uint64_t in1_s = teir_fp32;
    uint64_t in1_u = s * in1_s;
    uint64_t in1_r = u * in1_u;
    uint64_t in1_t = r * in1_r;
    // out: p q r s
    uint64_t out_s = teir_fp32;
    uint64_t out_r = s * out_s;
    uint64_t out_q = r * out_r;
    uint64_t out_p = q * out_q;

    return teir_operation(
        "contraction",
        {
            teir_tensor("in0", teir_dtype_t::dtype_fp32),
            teir_tensor("in1", teir_dtype_t::dtype_fp32),
            teir_tensor("out", teir_dtype_t::dtype_fp32),
        },
        {
            teir_axis("p", p, {in0_p, 0,     out_p}, {0, 0, 0}),
            teir_axis("q", q, {in0_q, 0,     out_q}, {0, 0, 0}),
            teir_axis("r", r, {0,     in1_r, out_r}, {0, 0, 0}),
            teir_axis("s", s, {0,     in1_s, out_s}, {0, 0, 0}),
            teir_axis("t", t, {in0_t, in1_t, 0    }, {0, 0, 0}),
            teir_axis("u", u, {in0_u, in1_u, 0    }, {0, 0, 0}),
        },
        {
            teir_primitive("zero", teir_ptype_t::ptype_zero,     {"out"},               {{"M", {"q"}}, {"N", {"s"}}},                 {{"data_type", "f32"}}),
            teir_primitive("gemm", teir_ptype_t::ptype_contract, {"in0", "in1", "out"}, {{"M", {"q"}}, {"N", {"s"}}, {"K", {"u"}}}, {{"data_type", "f32"}}),
        },
        teir_schedule(
            {"iter_p"},
            {
                teir_iter_node("iter_p", "p", teir_example_policy(parallel),    {"iter_r"}),
                teir_iter_node("iter_r", "r", teir_example_policy(parallel),    {"iter_t"}),
                teir_iter_node("iter_t", "t", teir_policy_t::policy_sequential, {"inv_zero", "inv_gemm"}),
            },
            {
                teir_inv_node("inv_zero", "zero", {teir_guard(teir_guard_kind::first, "t")}),
                teir_inv_node("inv_gemm", "gemm"),
            }
        )
    );
}


/**
 * `data/transposition.teir`: abcd->dbac.
 * File extents: a=96, b=128, c=48, d=32.
 * Tensor order: in, out.
 *
 * The output strides of the file for `b` (17664) and `d` (2260992) assume 92 instead of 96 blocks of `a`,
 * so different iterations write the same output elements. The builder uses the dense strides of a
 * dbac output tensor instead (18432 and 2359296 for the file extents).
 * The file executes all iteration nodes sequentially; `parallel = true` parallelizes `a` and `b`.
 */
inline teir_operation teir_example_transposition(uint64_t a, uint64_t b, uint64_t c, uint64_t d, bool parallel = false) {
    // in: a b c d
    uint64_t in_d = teir_fp32;
    uint64_t in_c = d * in_d;
    uint64_t in_b = c * in_c;
    uint64_t in_a = b * in_b;
    // out: d b a c
    uint64_t out_c = teir_fp32;
    uint64_t out_a = c * out_c;
    uint64_t out_b = a * out_a;
    uint64_t out_d = b * out_b;

    return teir_operation(
        "transposition",
        {
            teir_tensor("in",  teir_dtype_t::dtype_fp32),
            teir_tensor("out", teir_dtype_t::dtype_fp32),
        },
        {
            teir_axis("a", a, {in_a, out_a}, {0, 0}),
            teir_axis("b", b, {in_b, out_b}, {0, 0}),
            teir_axis("c", c, {in_c, out_c}, {0, 0}),
            teir_axis("d", d, {in_d, out_d}, {0, 0}),
        },
        {
            teir_primitive("copy", teir_ptype_t::ptype_copy, {"in", "out"}, {{"M", {"c"}}, {"N", {"d"}}}, {{"data_type", "f32"}}),
        },
        teir_schedule(
            {"iter_a"},
            {
                teir_iter_node("iter_a", "a", teir_example_policy(parallel), {"iter_b"}),
                teir_iter_node("iter_b", "b", teir_example_policy(parallel), {"inv_copy"}),
            },
            {
                teir_inv_node("inv_copy", "copy"),
            }
        )
    );
}


/**
 * Batched GEMM with an outer sequential K-blocking loop: out[b] = sum_{k0} in0[b, k0] * in1[b, k0].
 *
 * Schedule: iter_k0 (sequential) -> iter_b (parallel or sequential) -> [zero guarded by first(k0), gemm].
 * The zero invocation checks a guard on an axis outside of the parallel region.
 *
 * `trans_a`, `trans_b` and `trans_c` select the storage format of the matrices like the flags of `mini_jit::Gemm`:
 * A (m x k) is column-major for `trans_a = 0`, B (k x n) is row-major for `trans_b = 1`, C (m x n) is column-major for `trans_c = 0`.
 * Tensor order: in0, in1, out. Tensor sizes in floats: `db * dk0 * dm * dk`, `db * dk0 * dk * dn`, `db * dm * dn`.
 */
inline teir_operation teir_example_batched_gemm(uint64_t db, uint64_t dk0, uint64_t dm, uint64_t dn, uint64_t dk, uint32_t trans_a, uint32_t trans_b, uint32_t trans_c, bool parallel) {
    uint64_t in0_m = trans_a ? dk * teir_fp32 : teir_fp32;
    uint64_t in0_k = trans_a ? teir_fp32 : dm * teir_fp32;
    uint64_t in0_k0 = dm * dk * teir_fp32;
    uint64_t in0_b = dk0 * in0_k0;

    uint64_t in1_k = trans_b ? dn * teir_fp32 : teir_fp32;
    uint64_t in1_n = trans_b ? teir_fp32 : dk * teir_fp32;
    uint64_t in1_k0 = dk * dn * teir_fp32;
    uint64_t in1_b = dk0 * in1_k0;

    uint64_t out_m = trans_c ? dn * teir_fp32 : teir_fp32;
    uint64_t out_n = trans_c ? teir_fp32 : dm * teir_fp32;
    uint64_t out_b = dm * dn * teir_fp32;

    return teir_operation(
        "batched_gemm",
        {
            teir_tensor("in0", teir_dtype_t::dtype_fp32),
            teir_tensor("in1", teir_dtype_t::dtype_fp32),
            teir_tensor("out", teir_dtype_t::dtype_fp32),
        },
        {
            teir_axis("b",  db,  {in0_b,  in1_b,  out_b}, {0, 0, 0}),
            teir_axis("k0", dk0, {in0_k0, in1_k0, 0    }, {0, 0, 0}),
            teir_axis("m",  dm,  {in0_m,  0,      out_m}, {0, 0, 0}),
            teir_axis("n",  dn,  {0,      in1_n,  out_n}, {0, 0, 0}),
            teir_axis("k",  dk,  {in0_k,  in1_k,  0    }, {0, 0, 0}),
        },
        {
            teir_primitive("zero", teir_ptype_t::ptype_zero,     {"out"},               {{"M", {"m"}}, {"N", {"n"}}},                 {}),
            teir_primitive("gemm", teir_ptype_t::ptype_contract, {"in0", "in1", "out"}, {{"M", {"m"}}, {"N", {"n"}}, {"K", {"k"}}}, {}),
        },
        teir_schedule(
            {"iter_k0"},
            {
                teir_iter_node("iter_k0", "k0", teir_policy_t::policy_sequential, {"iter_b"}),
                teir_iter_node("iter_b",  "b",  teir_example_policy(parallel),    {"inv_zero", "inv_gemm"}),
            },
            {
                teir_inv_node("inv_zero", "zero", {teir_guard(teir_guard_kind::first, "k0")}),
                teir_inv_node("inv_gemm", "gemm"),
            }
        )
    );
}

/** Reference implementation of `teir_example_batched_gemm`, overwriting `out`. */
inline void teir_example_batched_gemm_ref(float const* in0, float const* in1, float* out, uint64_t db, uint64_t dk0, uint64_t dm, uint64_t dn, uint64_t dk, uint32_t trans_a, uint32_t trans_b, uint32_t trans_c) {
    for (uint64_t ib = 0; ib < db; ib++) {
        for (uint64_t im = 0; im < dm; im++) {
            for (uint64_t in = 0; in < dn; in++) {
                double sum = 0.0;
                for (uint64_t ik0 = 0; ik0 < dk0; ik0++) {
                    float const* a = in0 + (ib * dk0 + ik0) * dm * dk;
                    float const* b = in1 + (ib * dk0 + ik0) * dk * dn;
                    for (uint64_t ik = 0; ik < dk; ik++) {
                        float va = trans_a ? a[im * dk + ik] : a[ik * dm + im];
                        float vb = trans_b ? b[ik * dn + in] : b[in * dk + ik];
                        sum += (double)va * (double)vb;
                    }
                }
                float* c = out + ib * dm * dn;
                c[trans_c ? im * dn + in : in * dm + im] = (float)sum;
            }
        }
    }
}


/** Fills `data` with uniformly distributed random values in [-1, 1). */
inline void teir_example_fill_random(std::vector<float>& data, uint32_t seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& value : data) {
        value = dist(gen);
    }
}


#endif

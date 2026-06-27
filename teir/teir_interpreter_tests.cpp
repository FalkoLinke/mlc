#include <iostream>
#include <catch2/catch_test_macros.hpp>

#include "mlc_common.hpp"
#include "teir_reference.hpp"
#include "teir.h"
#include "teir_interpreter.h"



TEST_CASE( "abcd->dcba", "[test]") {
    uint32_t const da = 64;
    uint32_t const db = 32;
    uint32_t const dc = 128;
    uint32_t const dd = 16;

    std::vector<float> a(da * db * dc * dd, 0.0f);
    std::vector<float> b(da * db * dc * dd, 0.0f);
    std::vector<float> c(da * db * dc * dd, 0.0f);
    fill_indices(a.data(), a.size());

    teir_abcd_dcba_v1(a.data(), b.data(), da, db, dc, dd);
    teir_abcd_dcba_v2(a.data(), c.data(), da, db, dc, dd);

    bool result = b == c;
    REQUIRE(result);
}



TEST_CASE( "ab->ab", "[test]") {
    uint64_t const da = 8;
    uint64_t const db = 8;
    uint64_t const in0_sb = sizeof(float);
    uint64_t const in0_sa = db * in0_sb;
    uint64_t const out_sb = sizeof(float);
    uint64_t const out_sa = db * out_sb;

    std::vector<float> a(da * db, 0.0f);
    std::vector<float> b(da * db, 0.0f);
    std::vector<float> c(da * db, 0.0f);
    fill_indices(a.data(), a.size());

    teir_operation ab_ab_op(
        "ab->ba",
        {
            teir_tensor("in0", teir_dtype_t::dtype_fp32),
            teir_tensor("out", teir_dtype_t::dtype_fp32),
        },
        {
            teir_axis("a", da, {in0_sa, out_sa}, {0, 0}),
            teir_axis("b", db, {in0_sb, out_sb}, {0, 0}),
        },
        {
            teir_primitive(
                "copy",
                teir_ptype_t::ptype_copy,
                { "in0", "out" },
                {{"M", {}}, {"N", {}}},
                {}
            )
        },
        teir_schedule(
            {"iter_a"},
            {
                teir_iter_node("iter_a", "a", teir_policy_t::policy_sequential, {"iter_b"}),
                teir_iter_node("iter_b", "b", teir_policy_t::policy_sequential, {"inv_copy"}),
            },
            {
                teir_inv_node("inv_copy", "copy")
            }
        )
    );

    std::vector<void*> args = {a.data(), b.data()};
    teir_interpreter interpreter(ab_ab_op, args);
    interpreter.run();
    teir_ab_ba_v1(a.data(), c.data(), da, db, in0_sa/4, in0_sb/4, out_sa/4, out_sb/4);

    bool result = b == c;
    REQUIRE(result);
}

TEST_CASE( "ab->ba", "[test]") {
    uint64_t const da = 8;
    uint64_t const db = 8;
    uint64_t const in0_sb = sizeof(float);
    uint64_t const in0_sa = db * in0_sb;
    uint64_t const out_sa = sizeof(float);
    uint64_t const out_sb = da * out_sa;

    std::vector<float> a(da * db, 0.0f);
    std::vector<float> b(da * db, 0.0f);
    std::vector<float> c(da * db, 0.0f);
    fill_indices(a.data(), a.size());

    teir_operation ab_ab_op(
        "ab->ba",
        {
            teir_tensor("in0", teir_dtype_t::dtype_fp32),
            teir_tensor("out", teir_dtype_t::dtype_fp32),
        },
        {
            teir_axis("a", da, {in0_sa, out_sa}, {0, 0}),
            teir_axis("b", db, {in0_sb, out_sb}, {0, 0}),
        },
        {
            teir_primitive(
                "copy",
                teir_ptype_t::ptype_copy,
                { "in0", "out" },
                {{"M", {}}, {"N", {}}},
                {}
            )
        },
        teir_schedule(
            {"iter_a"},
            {
                teir_iter_node("iter_a", "a", teir_policy_t::policy_sequential, {"iter_b"}),
                teir_iter_node("iter_b", "b", teir_policy_t::policy_sequential, {"inv_copy"}),
            },
            {
                teir_inv_node("inv_copy", "copy")
            }
        )
    );

    std::vector<void*> args = {a.data(), b.data()};
    teir_interpreter interpreter(ab_ab_op, args);
    interpreter.run();
    teir_ab_ba_v1(a.data(), c.data(), da, db, in0_sa/4, in0_sb/4, out_sa/4, out_sb/4);

    bool result = b == c;
    REQUIRE(result);
}








TEST_CASE( "abc->abc", "[test]") {
    uint64_t const da = 64;
    uint64_t const db = 64;
    uint64_t const dc = 64;
    
    uint64_t const in0_sc = sizeof(float);
    uint64_t const in0_sb = dc * in0_sc;
    uint64_t const in0_sa = db * in0_sb;

    uint64_t const out_sc = sizeof(float);
    uint64_t const out_sb = dc * out_sc;
    uint64_t const out_sa = db * out_sb;

    std::vector<float> a(da * db * dc, 0.0f);
    std::vector<float> b(da * db * dc, 0.0f);
    std::vector<float> c(da * db * dc, 0.0f);
    fill_indices(a.data(), a.size());

    teir_operation abc_abc_op(
        "abc->abc",
        {
            teir_tensor("in0", teir_dtype_t::dtype_fp32),
            teir_tensor("out", teir_dtype_t::dtype_fp32),
        },
        {
            teir_axis("a", da, {in0_sa, out_sa}, {0, 0}),
            teir_axis("b", db, {in0_sb, out_sb}, {0, 0}),
            teir_axis("c", dc, {in0_sc, out_sc}, {0, 0}),
        },
        {
            teir_primitive(
                "copy",
                teir_ptype_t::ptype_copy,
                { "in0", "out" },
                {{"M", {"c"}}, {"N", {"b"}}},
                {}
            )
        },
        teir_schedule(
            {"iter_a"},
            {
                teir_iter_node("iter_a", "a", teir_policy_t::policy_sequential, {"inv_copy"}),
            },
            {
                teir_inv_node("inv_copy", "copy")
            }
        )
    );

    std::vector<void*> args = {a.data(), b.data()};
    teir_interpreter interpreter(abc_abc_op, args);
    interpreter.run();
    c = std::vector<float>(a.begin(), a.end());

    bool result = b == c;
    REQUIRE(result);
}

TEST_CASE( "abc->acb", "[test]") {
    uint64_t const da = 64;
    uint64_t const db = 64;
    uint64_t const dc = 64;
    
    uint64_t const in0_sc = sizeof(float);
    uint64_t const in0_sb = dc * in0_sc;
    uint64_t const in0_sa = db * in0_sb;

    uint64_t const out_sb = sizeof(float);
    uint64_t const out_sc = db * out_sb;
    uint64_t const out_sa = dc * out_sc;

    std::vector<float> a(da * db * dc, 0.0f);
    std::vector<float> b(da * db * dc, 0.0f);
    std::vector<float> c(da * db * dc, 0.0f);
    fill_indices(a.data(), a.size());

    teir_operation abc_acb_op(
        "abc->acb",
        {
            teir_tensor("in0", teir_dtype_t::dtype_fp32),
            teir_tensor("out", teir_dtype_t::dtype_fp32),
        },
        {
            teir_axis("a", da, {in0_sa, out_sa}, {0, 0}),
            teir_axis("b", db, {in0_sb, out_sb}, {0, 0}),
            teir_axis("c", dc, {in0_sc, out_sc}, {0, 0}),
        },
        {
            teir_primitive(
                "copy",
                teir_ptype_t::ptype_copy,
                { "in0", "out" },
                {{"M", {"c"}}, {"N", {"b"}}},
                {}
            )
        },
        teir_schedule(
            {"iter_a"},
            {
                teir_iter_node("iter_a", "a", teir_policy_t::policy_sequential, {"inv_copy"}),
            },
            {
                teir_inv_node("inv_copy", "copy")
            }
        )
    );

    std::vector<void*> args = {a.data(), b.data()};
    teir_interpreter interpreter(abc_acb_op, args);
    interpreter.run();
    teir_abc_acb(a.data(), c.data(), da, db, dc);

    bool result = b == c;
    REQUIRE(result);
}







TEST_CASE( "RELU abc->acb", "[test]") {
    uint64_t const da = 64;
    uint64_t const db = 64;
    uint64_t const dc = 64;
    
    uint64_t const in0_sc = sizeof(float);
    uint64_t const in0_sb = dc * in0_sc;
    uint64_t const in0_sa = db * in0_sb;

    uint64_t const out_sb = sizeof(float);
    uint64_t const out_sc = db * out_sb;
    uint64_t const out_sa = dc * out_sc;

    std::vector<float> a(da * db * dc, 0.0f);
    std::vector<float> b(da * db * dc, 0.0f);
    std::vector<float> c(da * db * dc, 0.0f);
    for (uint64_t i = 0; i < a.size(); i++) {
        a[i] = (i % 2 == 0) ? i : -i;
    }

    teir_operation relu_abc_acb_op(
        "RELU abc->acb",
        {
            teir_tensor("in0", teir_dtype_t::dtype_fp32),
            teir_tensor("out", teir_dtype_t::dtype_fp32),
        },
        {
            teir_axis("a", da, {in0_sa, out_sa}, {0, 0}),
            teir_axis("b", db, {in0_sb, out_sb}, {0, 0}),
            teir_axis("c", dc, {in0_sc, out_sc}, {0, 0}),
        },
        {
            teir_primitive(
                "relu",
                teir_ptype_t::ptype_relu,
                { "in0", "out" },
                {{"M", {"c"}}, {"N", {"b"}}},
                {}
            )
        },
        teir_schedule(
            {"iter_a"},
            {
                teir_iter_node("iter_a", "a", teir_policy_t::policy_sequential, {"inv_relu"}),
            },
            {
                teir_inv_node("inv_relu", "relu")
            }
        )
    );

    std::vector<void*> args = {a.data(), b.data()};
    teir_interpreter interpreter(relu_abc_acb_op, args);
    interpreter.run();

    teir_abc_acb(a.data(), c.data(), da, db, dc);
    for (uint64_t i = 0; i < c.size(); i++) {
        c[i] = fmax(c[i], 0.0f);
    }

    bool result = b == c;
    REQUIRE(result);
}












TEST_CASE( "abc->abc with guard", "[test]") {
    uint64_t const da = 64;
    uint64_t const db = 64;
    uint64_t const dc = 64;
    
    uint64_t const in0_sc = sizeof(float);
    uint64_t const in0_sb = dc * in0_sc;
    uint64_t const in0_sa = db * in0_sb;

    uint64_t const out_sc = sizeof(float);
    uint64_t const out_sb = dc * out_sc;
    uint64_t const out_sa = db * out_sb;

    std::vector<float> a(da * db * dc, 0.0f);
    std::vector<float> b(da * db * dc, 0.0f);
    std::vector<float> c(da * db * dc, 0.0f);
    fill_indices(a.data(), a.size());

    teir_operation abc_abc_op(
        "abc->abc",
        {
            teir_tensor("in0", teir_dtype_t::dtype_fp32),
            teir_tensor("out", teir_dtype_t::dtype_fp32),
        },
        {
            teir_axis("a", da, {in0_sa, out_sa}, {0, 0}),
            teir_axis("b", db, {in0_sb, out_sb}, {0, 0}),
            teir_axis("c", dc, {in0_sc, out_sc}, {0, 0}),
        },
        {
            teir_primitive(
                "copy",
                teir_ptype_t::ptype_copy,
                { "in0", "out" },
                {{"M", {"c"}}, {"N", {"b"}}},
                {}
            )
        },
        teir_schedule(
            {"iter_a"},
            {
                teir_iter_node("iter_a", "a", teir_policy_t::policy_sequential, {"inv_copy"}),
            },
            {
                teir_inv_node("inv_copy", "copy", {teir_guard(teir_guard_kind::first, "a")})
            }
        )
    );

    std::vector<void*> args = {a.data(), b.data()};
    teir_interpreter interpreter(abc_abc_op, args);
    interpreter.run();

    identity(a.data(), c.data(), dc, db, db, db, false);

    bool result = b == c;
    REQUIRE(result);
}








TEST_CASE( "bkm,bkn->bnm", "[test]") {
    uint64_t const db = 32;
    uint64_t const dk = 32;
    uint64_t const dm = 32;
    uint64_t const dn = 32;

    uint64_t const in0_sm = sizeof(float);
    uint64_t const in0_sk = dm * in0_sm;
    uint64_t const in0_sb = dk * in0_sk;
    uint64_t const in0_sn = 0;
    
    uint64_t const in1_sn = sizeof(float);
    uint64_t const in1_sk = dn * in1_sn;
    uint64_t const in1_sb = dk * in1_sk;
    uint64_t const in1_sm = 0;

    uint64_t const out_sm = sizeof(float);
    uint64_t const out_sn = dm * out_sm;
    uint64_t const out_sb = dn * out_sn;
    uint64_t const out_sk = 0;

    std::vector<float> a(db * dk * dm, 0.0f);
    std::vector<float> b(db * dk * dn, 0.0f);
    std::vector<float> c(db * dn * dm, 0.0f);
    std::vector<float> d(db * dn * dm, 0.0f);
    for (uint64_t i = 0; i < a.size(); i++) {
        a[i] = i % (dk * dm);
    }
    for (uint64_t ib = 0; ib < db; ib++) {
        for (uint64_t d = 0; d < std::min(dn, dk); d++) {
            b[ib * in1_sb / 4 + d * in1_sn / 4 + d * in1_sk / 4] = 1.0;
        }
    }

    teir_operation bkm_bkn_bnm(
        "bkm_bkn_bnm",
        {
            teir_tensor("in0", teir_dtype_t::dtype_fp32),
            teir_tensor("in1", teir_dtype_t::dtype_fp32),
            teir_tensor("out", teir_dtype_t::dtype_fp32),
        },
        {
            teir_axis("b", db, {in0_sb, in1_sb, out_sb}, {0, 0, 0}),
            teir_axis("m", dm, {in0_sm, in1_sm, out_sm}, {0, 0, 0}),
            teir_axis("n", dn, {in0_sn, in1_sn, out_sn}, {0, 0, 0}),
            teir_axis("k", dk, {in0_sk, in1_sk, out_sk}, {0, 0, 0}),
        },
        {
            teir_primitive(
                "gemm",
                teir_ptype_t::ptype_contract,
                { "in0", "in1", "out" },
                {{"M", {"m"}}, {"N", {"n"}}, {"K", {"k"}}},
                {}
            )
        },
        teir_schedule(
            {"iter_b"},
            {
                teir_iter_node("iter_b", "b", teir_policy_t::policy_sequential, {"inv_gemm"}),
            },
            {
                teir_inv_node("inv_gemm", "gemm")
            }
        )
    );

    std::vector<void*> args = {a.data(), b.data(), c.data()};
    teir_interpreter interpreter(bkm_bkn_bnm, args);
    interpreter.run();

    teir_bkm_bkn_bnm(a.data(), b.data(), d.data(), db, dk, dm, dn);

    double err = max_abs_diff(c.data(), d.data(), c.size());
    REQUIRE(err < 1e-6);
}
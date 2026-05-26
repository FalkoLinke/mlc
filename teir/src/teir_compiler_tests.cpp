#include <iostream>
#include <catch2/catch_test_macros.hpp>

#include "teir.h"
#include "teir_common.hpp"
#include "teir_compiler.h"







TEST_CASE("zero abc", "[test]") {
    uint64_t da = 16;
    uint64_t db = 16;
    uint64_t dc = 16;

    uint64_t out_sc = sizeof(float);
    uint64_t out_sb = dc * out_sc;
    uint64_t out_sa = db * out_sb;

    std::vector<float> a(da * db * dc, -1.0f);
    std::vector<float> b(da * db * dc, 0.0f);

    teir_operation zero_abc(
        "zero abc",
        {
            teir_tensor("out", teir_dtype_t::dtype_fp32),
        },
        {
            teir_axis("a", da, {out_sa}, {0}),
            teir_axis("b", db, {out_sb}, {0}),
            teir_axis("c", dc, {out_sc}, {0}),
        },
        {
            teir_primitive(
                "zero",
                teir_ptype_t::ptype_zero,
                {"out"},
                {{"M", {"c"}}, {"N", {"b"}}},
                {}
            ),
        },
        teir_schedule(
            {"iter_a"},
            {
                teir_iter_node("iter_a", "a", teir_policy_t::policy_sequential, {"inv_zero"}),
            },
            {
                teir_inv_node("inv_zero", "zero"),
            }
        )
    );

    teir_compiler compiler;
    compiler.compile(zero_abc);
    teir_compiler::teir_function_t func = compiler.get_function();

    std::vector<void*> args = {a.data()};
    func(args.data());

    bool result = a == b;
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

    
    teir_compiler compiler;
    compiler.compile(ab_ab_op);
    teir_compiler::teir_function_t func = compiler.get_function();
    std::vector<void*> args = {a.data(), b.data()};
    func(args.data());
    
    teir_ab_ba_v1(a.data(), c.data(), da, db, in0_sa/4, in0_sb/4, out_sa/4, out_sb/4);

    bool result = b == c;
    REQUIRE(result);
}








TEST_CASE( "abc->abc", "[test]") {
    uint64_t const da = 16;
    uint64_t const db = 16;
    uint64_t const dc = 16;
    
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

    teir_compiler compiler;
    compiler.compile(abc_abc_op);
    teir_compiler::teir_function_t func = compiler.get_function();
    std::vector<void*> args = {a.data(), b.data()};
    func(args.data());

    c = std::vector<float>(a.begin(), a.end());

    bool result = b == c;
    REQUIRE(result);
}

TEST_CASE( "abc->acb", "[test]") {
    uint64_t const da = 16;
    uint64_t const db = 16;
    uint64_t const dc = 16;
    
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

    teir_compiler compiler;
    compiler.compile(abc_acb_op);
    teir_compiler::teir_function_t func = compiler.get_function();
    std::vector<void*> args = {a.data(), b.data()};
    func(args.data());

    teir_abc_acb(a.data(), c.data(), da, db, dc);

    bool result = b == c;
    REQUIRE(result);
}






















































TEST_CASE( "16x16 identity fp32 notrans", "[test]") {
    uint64_t da = 16;
    uint64_t db = 16;

    uint64_t in0_sb = sizeof(float);
    uint64_t in0_sa = db * in0_sb;
    uint64_t out_sb = sizeof(float);
    uint64_t out_sa = db * out_sb;

    std::vector<float> a(da * db, 0.0f);
    std::vector<float> b(da * db, 0.0f);
    std::vector<float> c(da * db, 0.0f);
    fill_indices(a.data(), a.size());

    teir_operation op(
        "16x16_identity_fp32_notrans",
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
                {"in0", "out"},
                {{"M", {"b"}}, {"N", {"a"}}},
                {}
            )
        },
        teir_schedule(
            {"inv_copy"},
            {},
            {
                teir_inv_node("inv_copy", "copy")
            }
        )
    );
    teir_compiler compiler;
    compiler.compile(op);
    teir_compiler::teir_function_t func = compiler.get_function();

    std::vector<void*> args = {a.data(), b.data()};
    func(args.data());
    identity(a.data(), c.data(), db, da, da, da, false);

    bool result = b == c;
    REQUIRE(result);
}

TEST_CASE( "16x16 identity fp32 trans", "[test]") {
    uint64_t da = 16;
    uint64_t db = 16;

    uint64_t in0_sb = sizeof(float);
    uint64_t in0_sa = db * in0_sb;
    uint64_t out_sa = sizeof(float);
    uint64_t out_sb = da * out_sa;

    std::vector<float> a(da * db, 0.0f);
    std::vector<float> b(da * db, 0.0f);
    std::vector<float> c(da * db, 0.0f);
    fill_indices(a.data(), a.size());

    teir_operation op(
        "16x16_identity_fp32_trans",
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
                {"in0", "out"},
                {{"M", {"b"}}, {"N", {"a"}}},
                {}
            )
        },
        teir_schedule(
            {"inv_copy"},
            {},
            {
                teir_inv_node("inv_copy", "copy")
            }
        )
    );
    teir_compiler compiler;
    compiler.compile(op);
    teir_compiler::teir_function_t func = compiler.get_function();

    std::vector<void*> args = {a.data(), b.data()};
    func(args.data());
    identity(a.data(), c.data(), db, da, da, db, true);

    bool result = b == c;
    REQUIRE(result);
}

TEST_CASE( "16x16 relu fp32 notrans", "[test]") {
    uint64_t da = 16;
    uint64_t db = 16;

    uint64_t in0_sb = sizeof(float);
    uint64_t in0_sa = db * in0_sb;
    uint64_t out_sb = sizeof(float);
    uint64_t out_sa = db * out_sb;

    std::vector<float> a(da * db, 0.0f);
    std::vector<float> b(da * db, 0.0f);
    std::vector<float> c(da * db, 0.0f);
    for (uint64_t i = 0; i < a.size(); i++) {
        a[i] = (i % 2 == 0) ? (float)i : (float)-i;
    }

    teir_operation op(
        "16x16_relu_fp32_notrans",
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
                "relu",
                teir_ptype_t::ptype_relu,
                {"in0", "out"},
                {{"M", {"b"}}, {"N", {"a"}}},
                {}
            )
        },
        teir_schedule(
            {"inv_relu"},
            {},
            {
                teir_inv_node("inv_relu", "relu")
            }
        )
    );
    teir_compiler compiler;
    compiler.compile(op);
    teir_compiler::teir_function_t func = compiler.get_function();

    std::vector<void*> args = {a.data(), b.data()};
    func(args.data());
    relu(a.data(), c.data(), db, da, da, da, false);

    bool result = b == c;
    REQUIRE(result);
}

TEST_CASE( "16x16 relu fp32 trans", "[test]") {
    uint64_t da = 16;
    uint64_t db = 16;

    uint64_t in0_sb = sizeof(float);
    uint64_t in0_sa = db * in0_sb;
    uint64_t out_sa = sizeof(float);
    uint64_t out_sb = da * out_sa;

    std::vector<float> a(da * db, 0.0f);
    std::vector<float> b(da * db, 0.0f);
    std::vector<float> c(da * db, 0.0f);
    for (uint64_t i = 0; i < a.size(); i++) {
        a[i] = (i % 2 == 0) ? (float)i : (float)-i;
    }

    teir_operation op(
        "16x16_relu_fp32_notrans",
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
                "relu",
                teir_ptype_t::ptype_relu,
                {"in0", "out"},
                {{"M", {"b"}}, {"N", {"a"}}},
                {}
            )
        },
        teir_schedule(
            {"inv_relu"},
            {},
            {
                teir_inv_node("inv_relu", "relu")
            }
        )
    );
    teir_compiler compiler;
    compiler.compile(op);
    teir_compiler::teir_function_t func = compiler.get_function();

    std::vector<void*> args = {a.data(), b.data()};
    func(args.data());
    relu(a.data(), c.data(), db, da, da, db, true);

    bool result = b == c;
    REQUIRE(result);
}

TEST_CASE( "16x16 zero fp32", "[test]") {
    uint64_t da = 16;
    uint64_t db = 16;

    uint64_t out_sb = sizeof(float);
    uint64_t out_sa = db * out_sb;

    std::vector<float> a(da * db, -1.0f);
    std::vector<float> b(da * db, -1.0f);

    teir_operation op(
        "16x16_zero_fp32",
        {
            teir_tensor("out", teir_dtype_t::dtype_fp32),
        },
        {
            teir_axis("a", da, {out_sa}, {0}),
            teir_axis("b", db, {out_sb}, {0}),
        },
        {
            teir_primitive(
                "zero",
                teir_ptype_t::ptype_zero,
                {"out"},
                {{"M", {"b"}}, {"N", {"a"}}},
                {}
            )
        },
        teir_schedule(
            {"inv_zero"},
            {},
            {
                teir_inv_node("inv_zero", "zero")
            }
        )
    );
    teir_compiler compiler;
    compiler.compile(op);
    teir_compiler::teir_function_t func = compiler.get_function();

    std::vector<void*> args = {a.data()};
    func(args.data());
    zero(b.data(), db, da, da);

    bool result = a == b;
    REQUIRE(result);
}


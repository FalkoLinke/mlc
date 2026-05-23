#include <iostream>
#include <catch2/catch_test_macros.hpp>

#include "teir_common.hpp"
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
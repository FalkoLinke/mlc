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
    compiler.write("test.bin");

    std::vector<void*> args = {a.data()};
    func(args.data());

    bool result = a == b;
    REQUIRE(result);
}









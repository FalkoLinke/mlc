#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "teir.h"
#include "mlc_common.hpp"
#include "teir_compiler.h"
#include "teir_interpreter.h"
#include "teir_examples.hpp"


static constexpr double TOLERANCE = 1e-5;

static double max_rel_diff(std::vector<float> const& result, std::vector<float> const& expected) {
    return max_rel_diff(result.data(), expected.data(), result.size(), 1, result.size(), expected.size());
}

/** Executes `operation` either with the compiler or the interpreter. */
static void execute(teir_operation const& operation, std::vector<void*> const& args, bool compiled) {
    if (compiled) {
        teir_compiler compiler;
        compiler.compile(operation);
        std::vector<void*> tensors = args;
        compiler.get_function()(tensors.data());
    } else {
        teir_interpreter interpreter(operation, args);
        interpreter.run();
    }
}



TEST_CASE("batched gemm in all storage formats matches the reference", "[teir][runtime][parallel]") {
    uint32_t const trans_a = GENERATE(0, 1);
    uint32_t const trans_b = GENERATE(0, 1);
    uint32_t const trans_c = GENERATE(0, 1);
    bool const parallel = GENERATE(false, true);
    bool const compiled = GENERATE(false, true);

    uint64_t const db = 7;
    uint64_t const dk0 = 3;
    uint64_t const dm = 32;
    uint64_t const dn = 48;
    uint64_t const dk = 24;

    CAPTURE(trans_a, trans_b, trans_c, parallel, compiled);

    std::vector<float> in0(db * dk0 * dm * dk);
    std::vector<float> in1(db * dk0 * dk * dn);
    std::vector<float> out(db * dm * dn);
    std::vector<float> expected(out.size());
    teir_example_fill_random(in0, 1);
    teir_example_fill_random(in1, 2);
    // the operation zeroes the output itself
    teir_example_fill_random(out, 3);

    teir_operation operation = teir_example_batched_gemm(db, dk0, dm, dn, dk, trans_a, trans_b, trans_c, parallel);
    execute(operation, {in0.data(), in1.data(), out.data()}, compiled);

    teir_example_batched_gemm_ref(in0.data(), in1.data(), expected.data(), db, dk0, dm, dn, dk, trans_a, trans_b, trans_c);
    REQUIRE(max_rel_diff(out, expected) < 1e-4);
}



TEST_CASE("transposition example matches the reference", "[teir][runtime][parallel]") {
    bool const parallel = GENERATE(false, true);
    bool const compiled = GENERATE(false, true);
    CAPTURE(parallel, compiled);

    uint64_t const da = 5;
    uint64_t const db = 7;
    uint64_t const dc = 48;
    uint64_t const dd = 32;

    std::vector<float> in(da * db * dc * dd);
    std::vector<float> out(in.size(), 0.0f);
    teir_example_fill_random(in, 4);

    execute(teir_example_transposition(da, db, dc, dd, parallel), {in.data(), out.data()}, compiled);

    // abcd -> dbac
    bool equal = true;
    for (uint64_t a = 0; a < da; a++) {
        for (uint64_t b = 0; b < db; b++) {
            for (uint64_t c = 0; c < dc; c++) {
                for (uint64_t d = 0; d < dd; d++) {
                    float expected = in[((a * db + b) * dc + c) * dd + d];
                    float result = out[((d * db + b) * da + a) * dc + c];
                    equal = equal && expected == result;
                }
            }
        }
    }
    REQUIRE(equal);
}



/**
 * Executes the example operation with parallel policies (compiled and interpreted)
 * and compares the output to the interpreted execution with sequential policies.
 */
template <typename build_t>
static void check_parallel_example(build_t build, uint64_t in0_size, uint64_t in1_size, uint64_t out_size) {
    std::vector<float> in0(in0_size);
    std::vector<float> in1(in1_size);
    std::vector<float> out_init(out_size);
    teir_example_fill_random(in0, 5);
    teir_example_fill_random(in1, 6);
    teir_example_fill_random(out_init, 7);

    std::vector<float> expected = out_init;
    execute(build(false), {in0.data(), in1.data(), expected.data()}, false);

    for (bool compiled : {false, true}) {
        CAPTURE(compiled);
        std::vector<float> out = out_init;
        execute(build(true), {in0.data(), in1.data(), out.data()}, compiled);
        REQUIRE(max_rel_diff(out, expected) < TOLERANCE);
    }
}

TEST_CASE("parallel matmul example matches the sequential interpreter", "[teir][runtime][parallel]") {
    uint64_t const m0 = 3, m1 = 32, n0 = 5, n1 = 64, k0 = 2, k1 = 48;
    check_parallel_example(
        [&](bool parallel) { return teir_example_matmul(m0, m1, n0, n1, k0, k1, parallel); },
        m0 * k0 * m1 * k1,
        k0 * n0 * k1 * n1,
        m0 * n0 * m1 * n1
    );
}

TEST_CASE("parallel contraction example matches the sequential interpreter", "[teir][runtime][parallel]") {
    uint64_t const p = 3, q = 32, r = 5, s = 64, t = 2, u = 48;
    check_parallel_example(
        [&](bool parallel) { return teir_example_contraction(p, q, r, s, t, u, parallel); },
        p * q * t * u,
        t * r * u * s,
        p * q * r * s
    );
}



TEST_CASE("compiled functions preserve the callee-saved registers", "[teir][runtime][abi]") {
    bool const parallel = GENERATE(false, true);
    CAPTURE(parallel);

    struct call_t {
        teir_compiler::teir_function_t function;
        void** tensors;
    };

    uint64_t const db = 2, dk0 = 2, dm = 32, dn = 32, dk = 16;
    std::vector<float> in0(db * dk0 * dm * dk, 1.0f);
    std::vector<float> in1(db * dk0 * dk * dn, 1.0f);
    std::vector<float> out(db * dm * dn, 0.0f);
    std::vector<void*> tensors = {in0.data(), in1.data(), out.data()};

    teir_compiler compiler;
    compiler.compile(teir_example_batched_gemm(db, dk0, dm, dn, dk, 0, 1, 0, parallel));
    call_t call = {compiler.get_function(), tensors.data()};

    auto invoke = [](void* arg) {
        call_t* call = (call_t*)arg;
        call->function(call->tensors);
    };
    REQUIRE(callee_saved_preserved(invoke, &call));
}

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "teir.h"
#include "mlc_common.hpp"
#include "teir_compiler.h"
#include "teir_examples.hpp"
#include "teir_interpreter.h"
#include "teir_passes.h"
#include "teir_transform.h"


namespace {

double max_rel_diff(std::vector<float> const& result, std::vector<float> const& expected) {
    return ::max_rel_diff(result.data(), expected.data(), result.size(), 1, result.size(), expected.size());
}

/** Random tensors for an operation with inputs of the given sizes and an output of size `out_size`. */
struct tensors_t {
    std::vector<std::vector<float>> data;

    tensors_t(std::vector<uint64_t> const& sizes) {
        for (uint64_t i = 0; i < sizes.size(); i++) {
            data.emplace_back(sizes[i]);
            teir_example_fill_random(data.back(), 10 + i);
        }
    }

    /** Executes the operation on a copy of the tensors and returns the output. */
    std::vector<float> execute(teir_operation const& operation, bool compiled) const {
        std::vector<std::vector<float>> copy = data;
        std::vector<void*> args;
        for (std::vector<float>& tensor : copy) {
            args.push_back(tensor.data());
        }
        if (compiled) {
            teir_compiler compiler;
            compiler.compile(operation);
            compiler.get_function()(args.data());
        } else {
            teir_interpreter interpreter(operation, args);
            interpreter.run();
        }
        return copy.back();
    }
};

// small instances of the operations
constexpr uint64_t gb = 3, gk0 = 4, gm = 64, gn = 32, gk = 24;

teir_operation batched_gemm() {
    return teir_example_batched_gemm(gb, gk0, gm, gn, gk, 0, 1, 0, false);
}

tensors_t batched_gemm_tensors() {
    return tensors_t({gb * gk0 * gm * gk, gb * gk0 * gk * gn, gb * gm * gn});
}

constexpr uint64_t ea = 2, eb = 2, ec = 2, es = 4, ep = 8, ex = 64, ey = 96;

teir_operation einsum() {
    return teir_example_einsum(ea, eb, ec, es, ep, ex, ey);
}

tensors_t einsum_tensors() {
    return tensors_t({ea * ec * es * ep * ex, eb * es * ep * ey, ea * eb * ec * ey * ex});
}

std::vector<float> einsum_reference(tensors_t const& tensors) {
    std::vector<float> out(tensors.data[2].size());
    teir_example_einsum_ref(tensors.data[0].data(), tensors.data[1].data(), out.data(), ea, eb, ec, es, ep, ex, ey);
    return out;
}

/** A small target, so that the passes split and parallelize the small operations. */
teir_target small_target() {
    teir_target target;
    target.l1d_bytes = 4096;           // K blocks of at most 16
    target.l2_bytes = 32 * 1024;       // 8192 floats for the blocks of A, B and C
    target.cores_per_l2 = 1;
    target.threads = 4;
    target.iterations_per_thread = 2;  // at least 8 parallel iterations
    return target;
}

} // namespace



TEST_CASE("split of an iterated axis translates the guards", "[teir][transform]") {
    tensors_t tensors = batched_gemm_tensors();
    teir_operation operation = teir_split(batched_gemm(), "k0", 2);

    // the zero is guarded by first(k0) = first(k0_o), first(k0_i)
    teir_inv_node const* zero = operation.schedule.resolve_inv_id("inv_zero");
    REQUIRE(zero->guards.size() == 2);
    REQUIRE(zero->guards[0].axis_id == "k0_o");
    REQUIRE(zero->guards[1].axis_id == "k0_i");
    REQUIRE(operation.resolve_axis_id("k0_o")->extent == 2);

    REQUIRE(max_rel_diff(tensors.execute(operation, true), tensors.execute(batched_gemm(), false)) < 1e-5);
}

TEST_CASE("split of a primitive axis wraps the invocations into a loop", "[teir][transform]") {
    tensors_t tensors = batched_gemm_tensors();
    teir_operation operation = teir_split(batched_gemm(), "m", 32);

    // zero and gemm are consecutive siblings and share one loop
    teir_iter_node const* loop = operation.schedule.resolve_iter_id("iter_m_o");
    REQUIRE(loop != nullptr);
    REQUIRE(loop->children == std::vector<std::string>{"inv_zero", "inv_gemm"});
    REQUIRE(operation.resolve_primitive_id("gemm")->axes.at("M") == std::vector<std::string>{"m_i"});

    REQUIRE(max_rel_diff(tensors.execute(operation, true), tensors.execute(batched_gemm(), false)) < 1e-5);
}

TEST_CASE("split rejects extents which do not divide the axis", "[teir][transform]") {
    REQUIRE_THROWS_AS(teir_split(batched_gemm(), "k0", 3), teir_transform_error);
    REQUIRE_THROWS_AS(teir_split(batched_gemm(), "k0", 4), teir_transform_error);
}

TEST_CASE("fuse undoes a split", "[teir][transform]") {
    tensors_t tensors = batched_gemm_tensors();

    // iterated axis: both parts are iterated
    teir_operation fused = teir_fuse(teir_split(batched_gemm(), "k0", 2), "k0_o", "k0_i");
    REQUIRE(fused.resolve_axis_id("k0_ok0_i")->extent == gk0);
    REQUIRE(max_rel_diff(tensors.execute(fused, true), tensors.execute(batched_gemm(), false)) < 1e-5);

    // primitive axis: the outer part is iterated and has to be promoted before fusing
    teir_operation split = teir_split(batched_gemm(), "m", 32);
    REQUIRE_THROWS_AS(teir_fuse(split, "m_o", "m_i"), teir_transform_error);
    fused = teir_fuse(teir_promote(split, "iter_m_o", "M"), "m_o", "m_i");
    REQUIRE(fused.resolve_primitive_id("zero")->axes.at("M") == std::vector<std::string>{"m_om_i"});
    REQUIRE(fused.resolve_primitive_id("gemm")->axes.at("M") == std::vector<std::string>{"m_om_i"});
    REQUIRE(max_rel_diff(tensors.execute(fused, true), tensors.execute(batched_gemm(), false)) < 1e-5);
}

TEST_CASE("fuse rejects axes with incompatible strides", "[teir][transform]") {
    REQUIRE_THROWS_AS(teir_fuse(batched_gemm(), "b", "k0"), teir_transform_error);
    REQUIRE_THROWS_AS(teir_fuse(einsum(), "s", "p"), teir_transform_error);  // s is iterated, p is a primitive axis
}

TEST_CASE("promote and fuse move a contraction loop into the primitive", "[teir][transform]") {
    tensors_t tensors = einsum_tensors();
    teir_operation promoted = teir_promote(einsum(), "iter_s", "K");

    // the zero only depends on the removed loop through its guard first(s), which is dropped
    REQUIRE(promoted.schedule.resolve_iter_id("iter_s") == nullptr);
    REQUIRE(promoted.schedule.resolve_inv_id("inv_zero")->guards.empty());
    REQUIRE(promoted.resolve_primitive_id("gemm")->axes.at("K") == std::vector<std::string>{"s", "p"});

    teir_operation fused = teir_fuse(promoted, "s", "p");
    REQUIRE(fused.resolve_primitive_id("gemm")->axes.at("K") == std::vector<std::string>{"sp"});
    REQUIRE(fused.resolve_axis_id("sp")->extent == es * ep);

    for (bool compiled : {false, true}) {
        CAPTURE(compiled);
        REQUIRE(max_rel_diff(tensors.execute(fused, compiled), einsum_reference(tensors)) < 1e-4);
    }
}

TEST_CASE("promote rejects mismatching dimensions and unguarded invocations", "[teir][transform]") {
    // s is read by both inputs, so it can only become a K axis
    REQUIRE_THROWS_AS(teir_promote(einsum(), "iter_s", "M"), teir_transform_error);
    // the children of iter_c are iteration nodes
    REQUIRE_THROWS_AS(teir_promote(einsum(), "iter_c", "M"), teir_transform_error);
    // without its guard first(s) the zero is executed in every iteration of s and can not be moved out of the loop
    teir_operation unguarded = einsum();
    unguarded.schedule.invocation_nodes[0].guards.clear();
    REQUIRE_THROWS_AS(teir_promote(unguarded, "iter_s", "K"), teir_transform_error);
}

TEST_CASE("reorder interchanges a reduction and a free loop", "[teir][transform]") {
    uint64_t const m0 = 3, m1 = 32, n0 = 2, n1 = 64, k0 = 3, k1 = 16;
    tensors_t tensors({m0 * k0 * m1 * k1, k0 * n0 * k1 * n1, m0 * n0 * m1 * n1});
    teir_operation baseline = teir_example_matmul(m0, m1, n0, n1, k0, k1, false, true);

    teir_operation reordered = teir_reorder(teir_reorder(baseline, "iter_k0"), "iter_k0");
    REQUIRE(reordered.schedule.roots == std::vector<std::string>{"iter_m0"});
    REQUIRE(reordered.schedule.resolve_iter_id("iter_k0")->children == std::vector<std::string>{"inv_zero", "inv_gemm"});

    REQUIRE(max_rel_diff(tensors.execute(reordered, true), tensors.execute(baseline, false)) < 1e-5);
}

TEST_CASE("reorder rejects interchanging two reduction loops", "[teir][transform]") {
    teir_operation split = teir_split(batched_gemm(), "k0", 2);
    // k0_o -> k0_i are both reduction loops of the output
    REQUIRE_THROWS_AS(teir_reorder(split, "iter_k0"), teir_transform_error);
}

TEST_CASE("set_policy rejects parallel reduction loops", "[teir][transform]") {
    REQUIRE_THROWS_AS(teir_set_policy(batched_gemm(), "iter_k0", teir_policy_t::policy_parallel), teir_transform_error);
    teir_operation parallel = teir_set_policy(batched_gemm(), "iter_b", teir_policy_t::policy_parallel);
    REQUIRE(parallel.schedule.resolve_iter_id("iter_b")->policy == teir_policy_t::policy_parallel);
}

TEST_CASE("swap_operands computes the same contraction", "[teir][transform]") {
    tensors_t tensors = batched_gemm_tensors();
    teir_operation swapped = teir_swap_operands(batched_gemm(), "gemm");
    REQUIRE(swapped.resolve_primitive_id("gemm")->tensors == std::vector<std::string>{"in1", "in0", "out"});
    REQUIRE(max_rel_diff(tensors.execute(swapped, true), tensors.execute(batched_gemm(), false)) < 1e-5);
}

TEST_CASE("teir_to_string prints the schedule", "[teir][transform]") {
    std::string text = teir_to_string(einsum());
    REQUIRE(text.find("iter @iter_s axis @s policy sequential children [@inv_zero, @inv_gemm]") != std::string::npos);
    REQUIRE(text.find("invoke @inv_zero primitive @zero  guard first(@s)") != std::string::npos);
    REQUIRE(text.find("primitive @gemm : Contraction axes { M: [@x], N: [@y], K: [@p] }") != std::string::npos);
}



TEST_CASE("every combination of passes preserves the einsum", "[teir][passes]") {
    uint32_t const passes = GENERATE(Catch::Generators::range(0u, 16u));
    bool const compiled = GENERATE(false, true);
    CAPTURE(passes, compiled);

    tensors_t tensors = einsum_tensors();
    teir_operation optimized = teir_optimize(einsum(), small_target(), passes);
    REQUIRE(max_rel_diff(tensors.execute(optimized, compiled), einsum_reference(tensors)) < 1e-4);
}

TEST_CASE("all passes transform the einsum as intended", "[teir][passes]") {
    std::vector<std::string> log;
    teir_operation optimized = teir_optimize(einsum(), small_target(), teir_opt_all, &log);

    // K = s * p = 32 is split into blocks of 16, M = 64 into blocks of 32, N = 96 fits
    teir_primitive const& gemm = *optimized.resolve_primitive_id("gemm");
    REQUIRE(optimized.resolve_axis_id(gemm.axes.at("K")[0])->extent == 16);
    REQUIRE(optimized.resolve_axis_id(gemm.axes.at("M")[0])->extent == 32);
    REQUIRE(optimized.resolve_axis_id(gemm.axes.at("N")[0])->extent == 96);
    // a, b and c give 8 parallel iterations
    for (char const* node : {"iter_a", "iter_b", "iter_c"}) {
        REQUIRE(optimized.schedule.resolve_iter_id(node)->policy == teir_policy_t::policy_parallel);
    }
    REQUIRE(log.size() == 3);
}

TEST_CASE("all passes preserve the matmul and contraction examples", "[teir][passes]") {
    bool const compiled = GENERATE(false, true);
    CAPTURE(compiled);

    {
        uint64_t const m0 = 3, m1 = 32, n0 = 5, n1 = 64, k0 = 2, k1 = 48;
        tensors_t tensors({m0 * k0 * m1 * k1, k0 * n0 * k1 * n1, m0 * n0 * m1 * n1});
        teir_operation baseline = teir_example_matmul(m0, m1, n0, n1, k0, k1, true, true);
        teir_operation optimized = teir_optimize(baseline, small_target(), teir_opt_all);
        // the operand order pass swaps the inputs, since the output is row-major
        REQUIRE(optimized.resolve_primitive_id("gemm")->tensors[0] == "in1");
        REQUIRE(max_rel_diff(tensors.execute(optimized, compiled), tensors.execute(baseline, false)) < 1e-5);
    }
    {
        uint64_t const p = 3, q = 32, r = 5, s = 64, t = 2, u = 48;
        tensors_t tensors({p * q * t * u, t * r * u * s, p * q * r * s});
        teir_operation baseline = teir_example_contraction(p, q, r, s, t, u, true);
        teir_operation optimized = teir_optimize(baseline, small_target(), teir_opt_all);
        REQUIRE(max_rel_diff(tensors.execute(optimized, compiled), tensors.execute(baseline, false)) < 1e-5);
    }
}

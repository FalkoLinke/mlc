/**
 * benchmark_matmul.cpp
 *
 * Hardcoded benchmark for the TEIR @matmul operation using teir_interpreter.
 *
 */

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <algorithm>

#include "teir.h"
#include "teir_interpreter.h"

// ---------------------------------------------------------------------------
// Tensor sizes derived from the axis extents and strides (in floats)
// ---------------------------------------------------------------------------
static constexpr uint64_t M0 = 256;
static constexpr uint64_t M1 = 32;
static constexpr uint64_t N0 = 128;
static constexpr uint64_t N1 = 64;
static constexpr uint64_t K0 = 16;
static constexpr uint64_t K1 = 512;

// Max byte stride calculations divided by 4 (sizeof(f32)) yield these exact sizes:
static constexpr uint64_t IN0_FLOATS = 67108864; // M0 * M1 * K0 * K1
static constexpr uint64_t IN1_FLOATS = 67108864; // K0 * K1 * N0 * N1
static constexpr uint64_t OUT_FLOATS = 67108864; // M0 * M1 * N0 * N1

// ---------------------------------------------------------------------------
// GFLOP calculation
// ---------------------------------------------------------------------------
static double compute_gflops(double elapsed_s) {
    constexpr double flops =
        2.0 * (double)M0 * (double)M1 * (double)N0 *
              (double)N1 * (double)K0 * (double)K1;
    return (flops / elapsed_s) / 1e9;
}

// ---------------------------------------------------------------------------
// Build the teir_operation for @matmul
// ---------------------------------------------------------------------------
static teir_operation build_matmul() {

    // ---- tensors -----------------------------------------------------------
    std::vector<teir_tensor> tensors = {
        teir_tensor("in0", teir_dtype_t::dtype_fp32),
        teir_tensor("in1", teir_dtype_t::dtype_fp32),
        teir_tensor("out", teir_dtype_t::dtype_fp32),
    };


    // ---- axes --------------------------------------------------------------
    // order: [in0, in1, out]
    std::vector<teir_axis> axes = {
        teir_axis("m0", M0, { 1048576, 0,        1048576 }, { 0, 0, 0 }),
        teir_axis("m1", M1, { 4,       0,        4       }, { 0, 0, 0 }), // m1 stride is now 4 for in0 and out
        teir_axis("n0", N0, { 0,       131072,   8192    }, { 0, 0, 0 }),
        teir_axis("n1", N1, { 0,       4,        128     }, { 0, 0, 0 }), // n1 stride scales by M1 for out
        teir_axis("k0", K0, { 65536,   16777216, 0       }, { 0, 0, 0 }),
        teir_axis("k1", K1, { 128,     256,      0       }, { 0, 0, 0 }), // k1 stride scales by M1 for in0
    };
    // ---- axes --------------------------------------------------------------
    // // order: [in0, in1, out]
    // std::vector<teir_axis> axes = {
    //     teir_axis("m0", M0, { 1048576, 0,        1048576 }, { 0, 0, 0 }),
    //     teir_axis("m1", M1, { 2048,    0,        256     }, { 0, 0, 0 }),
    //     teir_axis("n0", N0, { 0,       131072,   8192    }, { 0, 0, 0 }),
    //     teir_axis("n1", N1, { 0,       4,        4       }, { 0, 0, 0 }),
    //     teir_axis("k0", K0, { 65536,   16777216, 0       }, { 0, 0, 0 }),
    //     teir_axis("k1", K1, { 4,       256,      0       }, { 0, 0, 0 }),
    // };

    // ---- primitives --------------------------------------------------------
    std::vector<teir_primitive> primitives = {
        teir_primitive(
            "zero",
            teir_ptype_t::ptype_zero,
            { "out" },
            { {"M", {"m1"}}, {"N", {"n1"}} },
            { {"data_type", "f32"} }
        ),
        teir_primitive(
            "gemm",
            teir_ptype_t::ptype_contract,
            { "in0", "in1", "out" },
            { {"M", {"m1"}}, {"N", {"n1"}}, {"K", {"k1"}} },
            { {"data_type", "f32"} }
        ),
    };

    // ---- schedule ----------------------------------------------------------
    teir_schedule schedule(
        /* roots */ { "iter_k0" },

        /* iteration_nodes */ {
            teir_iter_node("iter_k0", "k0", teir_policy_t::policy_sequential,
                { "inv_zero", "iter_m0" }),
            teir_iter_node("iter_m0", "m0", teir_policy_t::policy_parallel,
                { "iter_n0" }),
            teir_iter_node("iter_n0", "n0", teir_policy_t::policy_parallel,
                { "inv_gemm" }),
        },

        /* invocation_nodes */ {
            teir_inv_node("inv_zero", "zero"),
            teir_inv_node("inv_gemm", "gemm"),
        }
    );

    return teir_operation("matmul", tensors, axes, primitives, schedule);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    int repetitions = 10;
    if (argc >= 2) {
        repetitions = std::stoi(argv[1]);
        if (repetitions < 1) repetitions = 1;
    }

    std::cout << "=== TEIR @matmul Interpreter Benchmark ===\n\n";
    std::cout << "Tensor shapes:\n";
    std::cout << "  %in0  [m0=" << M0 << " k0=" << K0 << " m1=" << M1 << " k1=" << K1 << "]"
              << "  " << (IN0_FLOATS * sizeof(float) / 1024 / 1024) << " MiB\n";
    std::cout << "  %in1  [k0=" << K0 << " n0=" << N0 << " k1=" << K1 << " n1=" << N1 << "]"
              << "  " << (IN1_FLOATS * sizeof(float) / 1024 / 1024) << " MiB\n";
    std::cout << "  %out  [m0=" << M0 << " n0=" << N0 << " m1=" << M1 << " n1=" << N1 << "]"
              << "  " << (OUT_FLOATS * sizeof(float) / 1024 / 1024) << " MiB\n";
    std::cout << "\nFLOPs per run: "
              << std::fixed << std::setprecision(3)
              << (2.0 * M0 * M1 * N0 * N1 * K0 * K1 / 1e12) << " TFLOP\n";
    std::cout << "Repetitions:   " << repetitions << "\n\n";

    std::vector<float> in0(IN0_FLOATS);
    std::vector<float> in1(IN1_FLOATS);
    std::vector<float> out(OUT_FLOATS, 0.0f);

    for (uint64_t i = 0; i < IN0_FLOATS; i++) in0[i] = static_cast<float>(i % 256) * 0.001f;
    for (uint64_t i = 0; i < IN1_FLOATS; i++) in1[i] = static_cast<float>(i % 256) * 0.001f;

    teir_operation op = build_matmul();

    std::vector<void*> args = {
        static_cast<void*>(in0.data()),
        static_cast<void*>(in1.data()),
        static_cast<void*>(out.data()),
    };

    std::cout << "Warmup ... " << std::flush;
    {
        std::fill(out.begin(), out.end(), 0.0f);
        teir_interpreter interp(op, args);
        interp.run();
    }
    std::cout << "done\n\n";

    std::vector<double> times_s(repetitions);

    for (int rep = 0; rep < repetitions; rep++) {
        std::fill(out.begin(), out.end(), 0.0f);
        teir_interpreter interp(op, args);

        auto t0 = std::chrono::high_resolution_clock::now();
        interp.run();
        auto t1 = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        times_s[rep] = elapsed;

        std::cout << "  rep " << std::setw(2) << (rep + 1) << "/" << repetitions
                  << "  time: " << std::fixed << std::setprecision(3) << elapsed << " s"
                  << "   GFLOP/s: " << std::setprecision(2) << compute_gflops(elapsed)
                  << "\n";
    }

    double avg  = std::accumulate(times_s.begin(), times_s.end(), 0.0) / repetitions;
    double best = *std::min_element(times_s.begin(), times_s.end());
    double worst= *std::max_element(times_s.begin(), times_s.end());

    std::cout << "\n--- Summary ---\n" << std::fixed;
    std::cout << "  best   " << std::setprecision(3) << best  << " s  ->  "
              << std::setprecision(2) << compute_gflops(best)  << " GFLOP/s\n";
    std::cout << "  avg    " << std::setprecision(3) << avg   << " s  ->  "
              << std::setprecision(2) << compute_gflops(avg)   << " GFLOP/s\n";
    std::cout << "  worst  " << std::setprecision(3) << worst << " s  ->  "
              << std::setprecision(2) << compute_gflops(worst) << " GFLOP/s\n";

    std::cout << "\nSpot-check: out[0] = " << out[0] << "  (should be > 0)\n";

    return 0;
}
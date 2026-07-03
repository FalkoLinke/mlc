/**
 * benchmark_transposition.cpp
 *
 * Hardcoded benchmark for the TEIR @transposition operation using teir_interpreter.
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
// Extents
// ---------------------------------------------------------------------------
static constexpr uint64_t A = 96;
static constexpr uint64_t B = 128;
static constexpr uint64_t C = 48;
static constexpr uint64_t D = 32;

// Calculated exact sizes from the maximal combinations of byte-strides / 4
static constexpr uint64_t IN_FLOATS  = 18874368;
static constexpr uint64_t OUT_FLOATS = 18088128;

// ---------------------------------------------------------------------------
// Memory Bandwidth (GB/s) calculation
// ---------------------------------------------------------------------------
static double compute_gbps(double elapsed_s) {
    // Both Read and Write amounts
    constexpr double bytes = 2.0 * A * B * C * D * sizeof(float);
    return (bytes / elapsed_s) / 1e9;
}

// ---------------------------------------------------------------------------
// Build the teir_operation for @transposition
// ---------------------------------------------------------------------------
static teir_operation build_transposition() {

    // ---- tensors -----------------------------------------------------------
    std::vector<teir_tensor> tensors = {
        teir_tensor("in", teir_dtype_t::dtype_fp32),
        teir_tensor("out", teir_dtype_t::dtype_fp32),
    };

    // ---- axes --------------------------------------------------------------
    // order: [in, out]
    std::vector<teir_axis> axes = {
        teir_axis("a", A, { 786432, 192     }, { 0, 0 }),
        teir_axis("b", B, { 6144,   17664   }, { 0, 0 }),
        teir_axis("c", C, { 128,    4       }, { 0, 0 }),
        teir_axis("d", D, { 4,      2260992 }, { 0, 0 }),
    };

    // ---- primitives --------------------------------------------------------
    std::vector<teir_primitive> primitives = {
        teir_primitive(
            "copy",
            teir_ptype_t::ptype_copy,
            { "in", "out" },
            { {"M", {"d"}}, {"N", {"c"}} }, // Swap "c" and "d" here
            { {"data_type", "f32"} }
        ),
    };

    // ---- schedule ----------------------------------------------------------
    teir_schedule schedule(
        /* roots */ { "iter_a" },

        /* iteration_nodes */ {
            teir_iter_node("iter_a", "a", teir_policy_t::policy_sequential, { "iter_b" }),
            teir_iter_node("iter_b", "b", teir_policy_t::policy_sequential, { "inv_copy" }),
        },

        /* invocation_nodes */ {
            teir_inv_node("inv_copy", "copy"),
        }
    );

    return teir_operation("transposition", tensors, axes, primitives, schedule);
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

    std::cout << "=== TEIR @transposition Interpreter Benchmark ===\n\n";
    std::cout << "Tensor shapes:\n";
    std::cout << "  %in   [a=" << A << " b=" << B << " c=" << C << " d=" << D << "]"
              << "  " << (IN_FLOATS * sizeof(float) / 1024 / 1024) << " MiB\n";
    std::cout << "  %out  (Stride overlaps)"
              << "                " << (OUT_FLOATS * sizeof(float) / 1024 / 1024) << " MiB\n";
    std::cout << "\nMemory processed per run: "
              << std::fixed << std::setprecision(3)
              << (2.0 * A * B * C * D * sizeof(float) / 1e6) << " MB\n";
    std::cout << "Repetitions:   " << repetitions << "\n\n";

    std::vector<float> in(IN_FLOATS);
    std::vector<float> out(OUT_FLOATS, 0.0f);

    for (uint64_t i = 0; i < IN_FLOATS; i++) in[i] = static_cast<float>(i % 256) * 0.001f;

    teir_operation op = build_transposition();

    std::vector<void*> args = {
        static_cast<void*>(in.data()),
        static_cast<void*>(out.data()),
    };

    // Helferfunktion für das Enum
    auto print_teir_error = [](teir_interpreter_error_t e) {
        switch (e) {
            case teir_err_unresolved_axis_id: return "teir_err_unresolved_axis_id";
            case teir_err_unresolved_primitive_id: return "teir_err_unresolved_primitive_id";
            case teir_err_unresolved_iter_node_id: return "teir_err_unresolved_iter_node_id";
            case teir_err_unresolved_inv_node_id: return "teir_err_unresolved_inv_node_id";
            case teir_err_unresolved_schedule_node_id: return "teir_err_unresolved_schedule_node_id";
            case teir_err_unresolved_tensor_id: return "teir_err_unresolved_tensor_id";
            case teir_err_missing_primitive_lowering: return "teir_err_missing_primitive_lowering";
            case teir_err_missing_strides: return "teir_err_missing_strides";
            case teir_err_missing_offsets: return "teir_err_missing_offsets";
            case teir_err_invalid_guard: return "teir_err_invalid_guard";
            default: return "UNKNOWN_ERROR_CODE";
        }
    };

    std::cout << "Warmup ... " << std::flush;
    try {
        std::fill(out.begin(), out.end(), 0.0f);
        teir_interpreter interp(op, args);
        interp.run();
        std::cout << "done\n\n";
    } 
    catch (teir_interpreter_error_t e) {
        std::cerr << "\n[TEIR ERROR während Warmup]: " << print_teir_error(e) << "\n";
        return 1;
    }
    catch (const std::exception& e) {
        std::cerr << "\n[C++ ERROR]: " << e.what() << "\n";
        return 1;
    }

    std::vector<double> times_s(repetitions);

    for (int rep = 0; rep < repetitions; rep++) {
        std::fill(out.begin(), out.end(), 0.0f);
        
        try {
            teir_interpreter interp(op, args);

            auto t0 = std::chrono::high_resolution_clock::now();
            interp.run();
            auto t1 = std::chrono::high_resolution_clock::now();

            double elapsed = std::chrono::duration<double>(t1 - t0).count();
            times_s[rep] = elapsed;

            std::cout << "  rep " << std::setw(2) << (rep + 1) << "/" << repetitions
                      << "  time: " << std::fixed << std::setprecision(3) << elapsed << " s"
                      << "   GB/s: " << std::setprecision(2) << compute_gbps(elapsed)
                      << "\n";
        } 
        catch (teir_interpreter_error_t e) {
            std::cerr << "\n[TEIR ERROR in rep " << rep+1 << "]: " << print_teir_error(e) << "\n";
            return 1;
        }
    }

    double avg  = std::accumulate(times_s.begin(), times_s.end(), 0.0) / repetitions;
    double best = *std::min_element(times_s.begin(), times_s.end());
    double worst= *std::max_element(times_s.begin(), times_s.end());

    std::cout << "\n--- Summary ---\n" << std::fixed;
    std::cout << "  best   " << std::setprecision(3) << best  << " s  ->  "
              << std::setprecision(2) << compute_gbps(best)  << " GB/s\n";
    std::cout << "  avg    " << std::setprecision(3) << avg   << " s  ->  "
              << std::setprecision(2) << compute_gbps(avg)   << " GB/s\n";
    std::cout << "  worst  " << std::setprecision(3) << worst << " s  ->  "
              << std::setprecision(2) << compute_gbps(worst) << " GB/s\n";

    std::cout << "\nSpot-check: out[0] = " << out[0] << "  (should be > 0)\n";

    return 0;
}
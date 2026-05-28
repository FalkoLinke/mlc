/**
 * benchmark_contraction.cpp
 *
 * Hardcoded benchmark for the TEIR @contraction operation using teir_interpreter.
 *
 * TEIR specification:
 *   teir @contraction {
 *     tensor %in0 : f32
 *     tensor %in1 : f32
 *     tensor %out : f32
 *     axis @p  extent 128  strides { in0: 3145728, out: 2359296 }
 *     axis @q  extent 96   strides { in0: 32768,   out: 24576   }
 *     axis @r  extent 96   strides { in1: 65536,   out: 256     }
 *     axis @s  extent 64   strides { in1: 4,       out: 4       }
 *     axis @t  extent 32   strides { in0: 1024,    in1: 6291456 }
 *     axis @u  extent 256  strides { in0: 4,       in1: 256     }
 *     primitive @zero : Zero        axes { M: [@q], N: [@s]          }
 *     primitive @gemm : Contraction axes { M: [@q], N: [@s], K: [@u] }
 *     schedule {
 *       roots [@iter_p]
 *       iter @iter_p axis @p policy parallel   children [@iter_r]
 *       iter @iter_r axis @r policy parallel   children [@iter_t]
 *       iter @iter_t axis @t policy sequential children [@inv_zero, @inv_gemm]
 *       invoke @inv_zero primitive @zero  guard first(@t)
 *       invoke @inv_gemm primitive @gemm
 *     }
 *   }
 *
 */

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "teir.h"
#include "teir_interpreter.h"

// ---------------------------------------------------------------------------
// Tensor sizes derived from the axis extents and strides
//
//   %in0  [p=128, q=96, t=32, u=256]   stride-check: u*4=in0_u  t*4*256=in0_t  etc.
//   %in1  [t=32,  r=96, u=256, s=64]
//   %out  [p=128, q=96, r=96,  s=64]
// ---------------------------------------------------------------------------
static constexpr uint64_t P = 128;
static constexpr uint64_t Q =  96;
static constexpr uint64_t R =  96;
static constexpr uint64_t S =  64;
static constexpr uint64_t T =  32;
static constexpr uint64_t U = 256;

static constexpr uint64_t IN0_FLOATS = P * Q * T * U;  // 100 663 296
static constexpr uint64_t IN1_FLOATS = T * R * U * S;  //  50 331 648
static constexpr uint64_t OUT_FLOATS = P * Q * R * S;  //  75 497 472

// ---------------------------------------------------------------------------
// GFLOP calculation
//
//   out[p,q,r,s] += sum_t sum_u  in0[p,q,t,u] * in1[t,r,u,s]
//
//   FLOPs = 2 * P * Q * R * S * T * U  (one mul + one add per K-element)
// ---------------------------------------------------------------------------
static double compute_gflops(double elapsed_s) {
    constexpr double flops =
        2.0 * (double)P * (double)Q * (double)R *
              (double)S * (double)T * (double)U;
    return (flops / elapsed_s) / 1e9;
}

// ---------------------------------------------------------------------------
// Build the teir_operation for @contraction
// ---------------------------------------------------------------------------
static teir_operation build_contraction() {

    // ---- tensors -----------------------------------------------------------
    // index 0 = in0, 1 = in1, 2 = out  (order must match args[] later)
    std::vector<teir_tensor> tensors = {
        teir_tensor("in0", teir_dtype_t::dtype_fp32),
        teir_tensor("in1", teir_dtype_t::dtype_fp32),
        teir_tensor("out", teir_dtype_t::dtype_fp32),
    };

    // ---- axes --------------------------------------------------------------
    // strides / offsets vectors: [in0, in1, out]  (same order as tensors)
    // A stride of 0 means the axis does not stride over that tensor.
    std::vector<teir_axis> axes = {
        // @p  extent 128   in0:3145728  in1:0        out:2359296
        teir_axis("p", P,
            /* strides */ { 3145728, 0,       2359296 },
            /* offsets */ { 0,       0,       0       }),

        // @q  extent 96    in0:32768    in1:0        out:24576
        teir_axis("q", Q,
            /* strides */ { 32768, 0,     24576 },
            /* offsets */ { 0,     0,     0     }),

        // @r  extent 96    in0:0        in1:65536    out:256
        teir_axis("r", R,
            /* strides */ { 0, 65536, 256 },
            /* offsets */ { 0, 0,     0   }),

        // @s  extent 64    in0:0        in1:4        out:4
        teir_axis("s", S,
            /* strides */ { 0, 4, 4 },
            /* offsets */ { 0, 0, 0 }),

        // @t  extent 32    in0:1024     in1:6291456  out:0
        teir_axis("t", T,
            /* strides */ { 1024, 6291456, 0 },
            /* offsets */ { 0,    0,       0 }),

        // @u  extent 256   in0:4        in1:256      out:0
        teir_axis("u", U,
            /* strides */ { 4, 256, 0 },
            /* offsets */ { 0, 0,   0 }),
    };

    // ---- primitives --------------------------------------------------------
    std::vector<teir_primitive> primitives = {
        // @zero : Zero   axes { M: [@q], N: [@s] }   tensors: [out]
        teir_primitive(
            "zero",
            teir_ptype_t::ptype_zero,
            /* tensors  */ { "out" },
            /* axes     */ { {"M", {"q"}}, {"N", {"s"}} },
            /* metadata */ { {"data_type", "f32"} }
        ),

        // @gemm : Contraction   axes { M: [@q], N: [@s], K: [@u] }   tensors: [in0, in1, out]
        teir_primitive(
            "gemm",
            teir_ptype_t::ptype_contract,
            /* tensors  */ { "in0", "in1", "out" },
            /* axes     */ { {"M", {"q"}}, {"N", {"s"}}, {"K", {"u"}} },
            /* metadata */ { {"data_type", "f32"} }
        ),
    };

    // ---- schedule ----------------------------------------------------------
    teir_schedule schedule(
        /* roots */ { "iter_p" },

        /* iteration_nodes */ {
            // iter @iter_p  axis @p  policy parallel   children [@iter_r]
            teir_iter_node("iter_p", "p", teir_policy_t::policy_parallel,
                /* children */ { "iter_r" }),

            // iter @iter_r  axis @r  policy parallel   children [@iter_t]
            teir_iter_node("iter_r", "r", teir_policy_t::policy_parallel,
                /* children */ { "iter_t" }),

            // iter @iter_t  axis @t  policy sequential  children [@inv_zero, @inv_gemm]
            teir_iter_node("iter_t", "t", teir_policy_t::policy_sequential,
                /* children */ { "inv_zero", "inv_gemm" }),
        },

        /* invocation_nodes */ {
            // invoke @inv_zero  primitive @zero   guard first(@t)
            teir_inv_node("inv_zero", "zero",
                /* guards */ { teir_guard(teir_guard_kind::first, "t") }),

            // invoke @inv_gemm  primitive @gemm   (no guard)
            teir_inv_node("inv_gemm", "gemm"),
        }
    );

    return teir_operation("contraction", tensors, axes, primitives, schedule);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    int repetitions = 50000;
    if (argc >= 2) {
        repetitions = std::stoi(argv[1]);
        if (repetitions < 1) repetitions = 1;
    }

    std::cout << "=== TEIR @contraction Interpreter Benchmark ===\n\n";
    std::cout << "Tensor shapes:\n";
    std::cout << "  %in0  [p=" << P << " q=" << Q << " t=" << T << " u=" << U << "]"
              << "  " << (IN0_FLOATS * sizeof(float) / 1024 / 1024) << " MiB\n";
    std::cout << "  %in1  [t=" << T << " r=" << R << " u=" << U << " s=" << S << "]"
              << "  " << (IN1_FLOATS * sizeof(float) / 1024 / 1024) << " MiB\n";
    std::cout << "  %out  [p=" << P << " q=" << Q << " r=" << R << " s=" << S << "]"
              << "  " << (OUT_FLOATS * sizeof(float) / 1024 / 1024) << " MiB\n";
    std::cout << "\nFLOPs per run: "
              << std::fixed << std::setprecision(3)
              << (2.0 * P * Q * R * S * T * U / 1e12) << " TFLOP\n";
    std::cout << "Repetitions:   " << repetitions << "\n\n";

    // ---- allocate tensors --------------------------------------------------
    std::vector<float> in0(IN0_FLOATS);
    std::vector<float> in1(IN1_FLOATS);
    std::vector<float> out(OUT_FLOATS, 0.0f);

    for (uint64_t i = 0; i < IN0_FLOATS; i++) in0[i] = static_cast<float>(i % 256) * 0.001f;
    for (uint64_t i = 0; i < IN1_FLOATS; i++) in1[i] = static_cast<float>(i % 256) * 0.001f;

    // ---- build operation ---------------------------------------------------
    teir_operation op = build_contraction();

    std::vector<void*> args = {
        static_cast<void*>(in0.data()),
        static_cast<void*>(in1.data()),
        static_cast<void*>(out.data()),
    };

    // ---- warmup ------------------------------------------------------------
    std::cout << "Warmup ... " << std::flush;
    {
        std::fill(out.begin(), out.end(), 0.0f);
        teir_interpreter interp(op, args);
        interp.run();
    }
    std::cout << "done\n\n";

    // ---- timed runs --------------------------------------------------------
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

    // ---- summary -----------------------------------------------------------
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

    // ---- spot-check --------------------------------------------------------
    std::cout << "\nSpot-check: out[0] = " << out[0] << "  (should be > 0)\n";

    return 0;
}
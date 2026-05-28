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
 *     primitive @zero : Zero        axes { M: [@q], N: [@s]           }
 *     primitive @gemm : Contraction axes { M: [@q], N: [@s], K: [@u]  }
 *     schedule {
 *       roots [@iter_p]
 *       iter @iter_p axis @p  children [@iter_r]
 *       iter @iter_r axis @r  children [@iter_t]
 *       iter @iter_t axis @t  children [@inv_zero, @inv_gemm]
 *       invoke @inv_zero primitive @zero  guard first(@t)
 *       invoke @inv_gemm primitive @gemm
 *     }
 *   }
 *
 * Compile (example):
 *   g++ -O2 -std=c++17 benchmark_contraction.cpp \
 *       -I<path/to/teir/include> -L<path/to/teir/lib> -lteir \
 *       -o benchmark_contraction
 *
 * Run:
 *   ./benchmark_contraction [repetitions]
 *   ./benchmark_contraction 5
 */

#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "teir.h"
#include "teir_interpreter.h"

// ---------------------------------------------------------------------------
// Tensor dimensions derived from the strides / extents
//
//   %in0  shape: [p=128, q=96, t=32, u=256]  (row-major f32)
//         stride check: u-stride=4 (1 float), t-stride=1024 (256 floats = u),
//                       q-stride=32768 (8192 floats = t*u),
//                       p-stride=3145728 (786432 floats = q*t*u)  ✓
//
//   %in1  shape: [t=32, r=96, u=256, s=64]   (row-major f32)
//         stride check: s-stride=4 (1 float), u-stride=256 (64 floats = s),
//                       r-stride=65536 (16384 floats = u*s),
//                       t-stride=6291456 (1572864 floats = r*u*s)  ✓
//
//   %out  shape: [p=128, q=96, r=96, s=64]   (row-major f32)
//         stride check: s-stride=4 (1 float), r-stride=256 (64 floats = s),
//                       q-stride=24576 (6144 floats = r*s),
//                       p-stride=2359296 (589824 floats = q*r*s)  ✓
// ---------------------------------------------------------------------------

static constexpr uint64_t P = 128;
static constexpr uint64_t Q =  96;
static constexpr uint64_t R =  96;
static constexpr uint64_t S =  64;
static constexpr uint64_t T =  32;
static constexpr uint64_t U = 256;

static constexpr uint64_t IN0_FLOATS = P * Q * T * U;   // 128*96*32*256 = 100663296
static constexpr uint64_t IN1_FLOATS = T * R * U * S;   //  32*96*256*64 = 50331648
static constexpr uint64_t OUT_FLOATS = P * Q * R * S;   // 128*96*96*64  = 75497472

// ---------------------------------------------------------------------------
// Build the teir_operation struct for @contraction
// ---------------------------------------------------------------------------
static teir_operation build_contraction() {
    teir_operation op;

    // ---- tensors -----------------------------------------------------------
    // Order matters: index 0 = in0, 1 = in1, 2 = out
    op.tensors = {
        teir_tensor{"in0"},
        teir_tensor{"in1"},
        teir_tensor{"out"},
    };

    // ---- axes --------------------------------------------------------------
    // strides vector order must match tensors: [in0, in1, out]
    // A stride of 0 means the axis does not apply to that tensor.

    // @p  extent 128   in0:3145728   in1:0   out:2359296
    {
        teir_axis a;
        a.id      = "p";
        a.extent  = P;
        a.strides = { 3145728, 0,       2359296 };
        a.offsets = { 0,       0,       0       };
        op.axes.push_back(a);
    }

    // @q  extent 96    in0:32768   in1:0   out:24576
    {
        teir_axis a;
        a.id      = "q";
        a.extent  = Q;
        a.strides = { 32768, 0,     24576 };
        a.offsets = { 0,     0,     0     };
        op.axes.push_back(a);
    }

    // @r  extent 96    in0:0   in1:65536   out:256
    {
        teir_axis a;
        a.id      = "r";
        a.extent  = R;
        a.strides = { 0, 65536, 256 };
        a.offsets = { 0, 0,     0   };
        op.axes.push_back(a);
    }

    // @s  extent 64    in0:0   in1:4   out:4
    {
        teir_axis a;
        a.id      = "s";
        a.extent  = S;
        a.strides = { 0, 4, 4 };
        a.offsets = { 0, 0, 0 };
        op.axes.push_back(a);
    }

    // @t  extent 32    in0:1024   in1:6291456   out:0
    {
        teir_axis a;
        a.id      = "t";
        a.extent  = T;
        a.strides = { 1024, 6291456, 0 };
        a.offsets = { 0,    0,       0 };
        op.axes.push_back(a);
    }

    // @u  extent 256   in0:4   in1:256   out:0
    {
        teir_axis a;
        a.id      = "u";
        a.extent  = U;
        a.strides = { 4, 256, 0 };
        a.offsets = { 0, 0,   0 };
        op.axes.push_back(a);
    }

    // ---- primitives --------------------------------------------------------

    // @zero : Zero  axes { M: [@q], N: [@s] }  tensors: [out]
    {
        teir_primitive prim;
        prim.id    = "zero";
        prim.ptype = teir_ptype_t::ptype_zero;
        prim.tensors = { "out" };
        prim.axes["M"] = { "q" };
        prim.axes["N"] = { "s" };
        op.primitives.push_back(prim);
    }

    // @gemm : Contraction  axes { M: [@q], N: [@s], K: [@u] }  tensors: [in0, in1, out]
    {
        teir_primitive prim;
        prim.id    = "gemm";
        prim.ptype = teir_ptype_t::ptype_contraction;
        prim.tensors = { "in0", "in1", "out" };
        prim.axes["M"] = { "q" };
        prim.axes["N"] = { "s" };
        prim.axes["K"] = { "u" };
        op.primitives.push_back(prim);
    }

    // ---- schedule ----------------------------------------------------------
    {
        teir_schedule sched;
        sched.roots = { "iter_p" };

        // iter @iter_p  axis @p  children [@iter_r]
        {
            teir_iter_node n;
            n.id       = "iter_p";
            n.axis     = "p";
            n.children = { "iter_r" };
            sched.iter_nodes.push_back(n);
        }

        // iter @iter_r  axis @r  children [@iter_t]
        {
            teir_iter_node n;
            n.id       = "iter_r";
            n.axis     = "r";
            n.children = { "iter_t" };
            sched.iter_nodes.push_back(n);
        }

        // iter @iter_t  axis @t  children [@inv_zero, @inv_gemm]
        {
            teir_iter_node n;
            n.id       = "iter_t";
            n.axis     = "t";
            n.children = { "inv_zero", "inv_gemm" };
            sched.iter_nodes.push_back(n);
        }

        // invoke @inv_zero  primitive @zero  guard first(@t)
        {
            teir_inv_node n;
            n.id        = "inv_zero";
            n.primitive = "zero";
            teir_guard g;
            g.axis_id = "t";
            g.kind    = teir_guard_kind::first;
            n.guards.push_back(g);
            sched.inv_nodes.push_back(n);
        }

        // invoke @inv_gemm  primitive @gemm
        {
            teir_inv_node n;
            n.id        = "inv_gemm";
            n.primitive = "gemm";
            sched.inv_nodes.push_back(n);
        }

        op.schedule = sched;
    }

    return op;
}

// ---------------------------------------------------------------------------
// GFLOP calculation
//
// The contraction computes:
//   out[p, q, r, s] = sum_t sum_u in0[p, q, t, u] * in1[t, r, u, s]
//
// FLOPs per output element: T * U multiply-adds = 2 * T * U
// Total output elements:    P * Q * R * S
// Total FLOPs:              2 * P * Q * R * S * T * U
// ---------------------------------------------------------------------------
static double compute_gflops(double elapsed_seconds) {
    // 2 * 128 * 96 * 96 * 64 * 32 * 256 = 2 * 4.0265318e12 ≈ 8.053e12
    constexpr double flops =
        2.0 * (double)P * (double)Q * (double)R *
              (double)S * (double)T * (double)U;
    return (flops / elapsed_seconds) / 1e9;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    int repetitions = 3;
    if (argc >= 2) {
        repetitions = std::stoi(argv[1]);
        if (repetitions < 1) repetitions = 1;
    }

    std::cout << "=== TEIR @contraction Interpreter Benchmark ===\n\n";
    std::cout << "Tensor sizes:\n";
    std::cout << "  %in0  [" << P << " x " << Q << " x " << T << " x " << U << "]"
              << "  " << (IN0_FLOATS * 4 / 1024 / 1024) << " MiB\n";
    std::cout << "  %in1  [" << T << " x " << R << " x " << U << " x " << S << "]"
              << "  " << (IN1_FLOATS * 4 / 1024 / 1024) << " MiB\n";
    std::cout << "  %out  [" << P << " x " << Q << " x " << R << " x " << S << "]"
              << "  " << (OUT_FLOATS * 4 / 1024 / 1024) << " MiB\n";
    std::cout << "\nTotal FLOPs per run: "
              << std::fixed << std::setprecision(3)
              << (2.0 * P * Q * R * S * T * U / 1e12) << " TFLOP\n";
    std::cout << "Repetitions: " << repetitions << "\n\n";

    // ---- allocate tensors --------------------------------------------------
    std::vector<float> in0(IN0_FLOATS);
    std::vector<float> in1(IN1_FLOATS);
    std::vector<float> out(OUT_FLOATS, 0.0f);

    // initialise inputs with simple values so the benchmark is reproducible
    for (uint64_t i = 0; i < IN0_FLOATS; i++) {
        in0[i] = static_cast<float>(i % 256) * 0.001f;
    }
    for (uint64_t i = 0; i < IN1_FLOATS; i++) {
        in1[i] = static_cast<float>(i % 256) * 0.001f;
    }

    // ---- build operation ---------------------------------------------------
    teir_operation op = build_contraction();

    // ---- argument pointer array --------------------------------------------
    std::vector<void*> args = {
        static_cast<void*>(in0.data()),
        static_cast<void*>(in1.data()),
        static_cast<void*>(out.data()),
    };

    // ---- warmup run --------------------------------------------------------
    std::cout << "Warmup run ... " << std::flush;
    {
        std::fill(out.begin(), out.end(), 0.0f);
        teir_interpreter interp(op, args);
        interp.run();
    }
    std::cout << "done\n\n";

    // ---- timed runs --------------------------------------------------------
    std::vector<double> times_s(repetitions);

    for (int rep = 0; rep < repetitions; rep++) {
        // reset output so @zero guard takes effect each run
        std::fill(out.begin(), out.end(), 0.0f);

        teir_interpreter interp(op, args);

        auto t0 = std::chrono::high_resolution_clock::now();
        interp.run();
        auto t1 = std::chrono::high_resolution_clock::now();

        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        times_s[rep] = elapsed;

        double gflops = compute_gflops(elapsed);
        std::cout << "  rep " << std::setw(2) << (rep + 1) << "/" << repetitions
                  << "  time: " << std::fixed << std::setprecision(3)
                  << elapsed << " s"
                  << "   GFLOP/s: " << std::setprecision(2) << gflops
                  << "\n";
    }

    // ---- summary -----------------------------------------------------------
    double sum  = std::accumulate(times_s.begin(), times_s.end(), 0.0);
    double avg  = sum / repetitions;
    double best = *std::min_element(times_s.begin(), times_s.end());
    double worst= *std::max_element(times_s.begin(), times_s.end());

    std::cout << "\n--- Summary ---\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  best   time: " << best  << " s   -> "
              << std::setprecision(2) << compute_gflops(best)  << " GFLOP/s\n";
    std::cout << std::setprecision(3);
    std::cout << "  avg    time: " << avg   << " s   -> "
              << std::setprecision(2) << compute_gflops(avg)   << " GFLOP/s\n";
    std::cout << std::setprecision(3);
    std::cout << "  worst  time: " << worst << " s   -> "
              << std::setprecision(2) << compute_gflops(worst) << " GFLOP/s\n";

    // ---- quick correctness spot-check --------------------------------------
    // out[0,0,0,0] should be non-zero after the contraction
    std::cout << "\nSpot-check: out[0,0,0,0] = " << out[0] << "  (should be > 0)\n";

    return 0;
}

/**
 * Tensorized Neural Network 
 *
 * Benchmark für das 2-Layer Tensorized Neural Network (MPO / TT-Matrix).
 * Führt die Inferenz über 5000 Samples in Tiles von je 512 Samples aus.
 */

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>
#include <fstream>
#include <cfenv>
#include <algorithm>

#include "teir.h"
#include "teir_interpreter.h"

static constexpr uint64_t TILE = 512;
static constexpr uint64_t TOTAL_SAMPLES = 5000;

// ---------------------------------------------------------------------------
// Helpers für File I/O
// ---------------------------------------------------------------------------
void load_bin(const std::string& path, std::vector<float>& buf) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::cerr << "Warning: could not open " << path << "\n";
        return;
    }
    ifs.read(reinterpret_cast<char*>(buf.data()), buf.size() * sizeof(float));
}

void load_bin_int(const std::string& path, std::vector<int32_t>& buf) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::cerr << "Warning: could not open " << path << "\n";
        return;
    }
    ifs.read(reinterpret_cast<char*>(buf.data()), buf.size() * sizeof(int32_t));
}

// ---------------------------------------------------------------------------
// c1: bqp, psc -> bqsc
// Iteration: b, s  |  Primitive GEMM: M=[c], N=[q], K=[p]
// ---------------------------------------------------------------------------
static teir_operation build_c1() {
    std::vector<teir_tensor> tensors = {
        teir_tensor("X", teir_dtype_t::dtype_fp32),
        teir_tensor("G1a", teir_dtype_t::dtype_fp32),
        teir_tensor("Y1", teir_dtype_t::dtype_fp32),
    };
    std::vector<teir_axis> axes = {
        teir_axis("b", TILE, { 512,    0, 131072 }, { 0, 0, 0 }),
        teir_axis("s", 16,   {   0,  512,    512 }, { 0, 0, 0 }),
        teir_axis("q", 16,   {  32,    0,   8192 }, { 0, 0, 0 }),
        teir_axis("c", 512,  {   0,    1,      1 }, { 0, 0, 0 }),
        teir_axis("p", 32,   {   1, 8192,      0 }, { 0, 0, 0 }),
    };
    std::vector<teir_primitive> primitives = {
        teir_primitive("zero", teir_ptype_t::ptype_zero, {"Y1"}, {{"M", {"c"}}, {"N", {"q"}}}, {{"data_type", "f32"}}),
        teir_primitive("gemm", teir_ptype_t::ptype_contract, {"X", "G1a", "Y1"}, {{"M", {"c"}}, {"N", {"q"}}, {"K", {"p"}}}, {{"data_type", "f32"}}),
    };
    teir_schedule schedule({"iter_b"}, {
        teir_iter_node("iter_b", "b", teir_policy_t::policy_parallel, {"iter_s"}),
        teir_iter_node("iter_s", "s", teir_policy_t::policy_parallel, {"inv_zero", "inv_gemm"}),
    }, {
        teir_inv_node("inv_zero", "zero"),
        teir_inv_node("inv_gemm", "gemm"),
    });
    return teir_operation("c1", tensors, axes, primitives, schedule);
}

// ---------------------------------------------------------------------------
// c2: bqsc, qct -> bst
// Iteration: b, q  |  Primitive GEMM: M=[t], N=[s], K=[c]
// ---------------------------------------------------------------------------
static teir_operation build_c2() {
    std::vector<teir_tensor> tensors = {
        teir_tensor("Y1", teir_dtype_t::dtype_fp32),
        teir_tensor("G1b", teir_dtype_t::dtype_fp32),
        teir_tensor("H", teir_dtype_t::dtype_fp32),
    };
    std::vector<teir_axis> axes = {
        teir_axis("b", TILE, { 131072,     0, 1024 }, { 0, 0, 0 }),
        teir_axis("q", 16,   {   8192, 32768,    0 }, { 0, 0, 0 }),
        teir_axis("s", 16,   {    512,     0,   64 }, { 0, 0, 0 }),
        teir_axis("t", 64,   {      0,     1,    1 }, { 0, 0, 0 }),
        teir_axis("c", 512,  {      1,    64,    0 }, { 0, 0, 0 }),
    };
    std::vector<teir_primitive> primitives = {
        teir_primitive("zero", teir_ptype_t::ptype_zero, {"H"}, {{"M", {"t"}}, {"N", {"s"}}}, {{"data_type", "f32"}}),
        teir_primitive("gemm", teir_ptype_t::ptype_contract, {"Y1", "G1b", "H"}, {{"M", {"t"}}, {"N", {"s"}}, {"K", {"c"}}}, {{"data_type", "f32"}}),
    };
    teir_schedule schedule({"iter_b"}, {
        teir_iter_node("iter_b", "b", teir_policy_t::policy_parallel, {"iter_q"}),
        // q iteriert über K-Achse -> summiert auf. Nur bei first(q) die Matrix nullen.
        teir_iter_node("iter_q", "q", teir_policy_t::policy_sequential, {"inv_zero", "inv_gemm"}),
    }, {
        teir_inv_node("inv_zero", "zero", { teir_guard(teir_guard_kind::first, "q") }),
        teir_inv_node("inv_gemm", "gemm"),
    });
    return teir_operation("c2", tensors, axes, primitives, schedule);
}

// ---------------------------------------------------------------------------
// c3: bhp, pc -> bhc (Optimized: collapsed 'b' & 'h' dimensions into 'B')
// Iteration: None  |  Primitive GEMM: M=[c], N=[B], K=[p]
// ---------------------------------------------------------------------------
static teir_operation build_c3() {
    std::vector<teir_tensor> tensors = {
        teir_tensor("Ht", teir_dtype_t::dtype_fp32),
        teir_tensor("G2a", teir_dtype_t::dtype_fp32),
        teir_tensor("Y2", teir_dtype_t::dtype_fp32),
    };
    std::vector<teir_axis> axes = {
        teir_axis("B", TILE * 32, { 32,  0, 32 }, { 0, 0, 0 }), // B = b * h
        teir_axis("c", 32,        {  0,  1,  1 }, { 0, 0, 0 }),
        teir_axis("p", 32,        {  1, 32,  0 }, { 0, 0, 0 }),
    };
    std::vector<teir_primitive> primitives = {
        teir_primitive("zero", teir_ptype_t::ptype_zero, {"Y2"}, {{"M", {"c"}}, {"N", {"B"}}}, {{"data_type", "f32"}}),
        teir_primitive("gemm", teir_ptype_t::ptype_contract, {"Ht", "G2a", "Y2"}, {{"M", {"c"}}, {"N", {"B"}}, {"K", {"p"}}}, {{"data_type", "f32"}}),
    };
    teir_schedule schedule({"inv_zero", "inv_gemm"}, {}, {
        teir_inv_node("inv_zero", "zero"),
        teir_inv_node("inv_gemm", "gemm"),
    });
    return teir_operation("c3", tensors, axes, primitives, schedule);
}

// ---------------------------------------------------------------------------
// c4: bqc, qco -> bo (Optimized: collapsed 'q' & 'c' dimensions into 'k')
// Iteration: None  |  Primitive GEMM: M=[o], N=[b], K=[k]
// ---------------------------------------------------------------------------
static teir_operation build_c4() {
    std::vector<teir_tensor> tensors = {
        teir_tensor("Y2", teir_dtype_t::dtype_fp32),
        teir_tensor("G2b", teir_dtype_t::dtype_fp32),
        teir_tensor("logits", teir_dtype_t::dtype_fp32),
    };
    std::vector<teir_axis> axes = {
        teir_axis("b", TILE, { 1024,  0, 16 }, { 0, 0, 0 }),
        teir_axis("o", 16,   {    0,  1,  1 }, { 0, 0, 0 }),
        teir_axis("k", 1024, {    1, 16,  0 }, { 0, 0, 0 }), // k = q * c
    };
    std::vector<teir_primitive> primitives = {
        teir_primitive("zero", teir_ptype_t::ptype_zero, {"logits"}, {{"M", {"o"}}, {"N", {"b"}}}, {{"data_type", "f32"}}),
        teir_primitive("gemm", teir_ptype_t::ptype_contract, {"Y2", "G2b", "logits"}, {{"M", {"o"}}, {"N", {"b"}}, {"K", {"k"}}}, {{"data_type", "f32"}}),
    };
    teir_schedule schedule({"inv_zero", "inv_gemm"}, {}, {
        teir_inv_node("inv_zero", "zero"),
        teir_inv_node("inv_gemm", "gemm"),
    });
    return teir_operation("c4", tensors, axes, primitives, schedule);
}


int main(int argc, char* argv[]) {
    std::cout << "=== TEIR Tensorized Neural Network (MPO) Benchmark ===\n\n";

    // 1. Buffer Allokation
    std::vector<float> X(TOTAL_SAMPLES * 512);
    std::vector<int32_t> y(TOTAL_SAMPLES);
    std::vector<float> G1a(32 * 16 * 512);
    std::vector<float> G1b(16 * 512 * 64);
    std::vector<float> G2a(32 * 32);
    std::vector<float> G2b(32 * 32 * 16);
    std::vector<float> b1(1024);
    std::vector<float> b2(16);

    std::vector<float> X_tile(TILE * 512, 0.0f);
    std::vector<float> Y1(TILE * 16 * 16 * 512, 0.0f); // 256 MB
    std::vector<float> H(TILE * 16 * 64, 0.0f);        // 2 MB
    std::vector<float> Ht(TILE * 32 * 32, 0.0f);       // 2 MB
    std::vector<float> Y2(TILE * 32 * 32, 0.0f);       // 2 MB
    std::vector<float> logits(TILE * 16, 0.0f);        // 32 KB

    // 2. Data Input laden
    std::cout << "Lade tensorized_net/data/*.bin ...\n";
    load_bin("tensorized_net/data/X.bin", X);
    load_bin_int("tensorized_net/data/y.bin", y);
    load_bin("tensorized_net/data/G1a.bin", G1a);
    load_bin("tensorized_net/data/G1b.bin", G1b);
    load_bin("tensorized_net/data/G2a.bin", G2a);
    load_bin("tensorized_net/data/G2b.bin", G2b);
    load_bin("tensorized_net/data/b1.bin", b1);
    load_bin("tensorized_net/data/b2.bin", b2);

    // 3. Operationen & Interpreter Aufbau
    teir_operation op1 = build_c1();
    teir_operation op2 = build_c2();
    teir_operation op3 = build_c3();
    teir_operation op4 = build_c4();

    std::vector<void*> args1 = { X_tile.data(), G1a.data(), Y1.data() };
    std::vector<void*> args2 = { Y1.data(), G1b.data(), H.data() };
    std::vector<void*> args3 = { Ht.data(), G2a.data(), Y2.data() };
    std::vector<void*> args4 = { Y2.data(), G2b.data(), logits.data() };

    teir_interpreter interp1(op1, args1);
    teir_interpreter interp2(op2, args2);
    teir_interpreter interp3(op3, args3);
    teir_interpreter interp4(op4, args4);

    // 4. Batch Execution Pipeline
    std::cout << "\nStarte Tile-Verarbeitung (" << (TOTAL_SAMPLES + TILE - 1) / TILE << " Tiles)...\n\n";
    std::fesetenv(FE_DFL_ENV); // FPU Resetting

    int correct_predictions = 0;
    double total_teir_seconds = 0.0;

    int num_tiles = (TOTAL_SAMPLES + TILE - 1) / TILE;
    for (int tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        
        // --- A: Prepare Tile Data ---
        int valid_items = std::min(TILE, TOTAL_SAMPLES - tile_idx * TILE);
        std::copy(X.begin() + tile_idx * TILE * 512, 
                  X.begin() + tile_idx * TILE * 512 + valid_items * 512, 
                  X_tile.begin());

        // --- B: Run TEIR Contractions c1 & c2 ---
        auto t0 = std::chrono::high_resolution_clock::now();
        interp1.run();
        interp2.run();
        auto t1 = std::chrono::high_resolution_clock::now();
        
        // --- C: Bias 1 + ReLU + Transpose (H -> Ht) ---
        for (int b = 0; b < valid_items; ++b) {
            for (int s = 0; s < 16; ++s) {
                for (int t = 0; t < 64; ++t) {
                    int o = s * 64 + t; 
                    int idx = b * 1024 + o;
                    
                    // ReLU & Bias
                    float val = std::max(0.0f, H[idx] + b1[o]);
                    
                    // Transpose H[b, h1, h2] -> Ht[b, h2, h1] (h1=o/32, h2=o%32)
                    int h1 = o / 32;
                    int h2 = o % 32;
                    int t_idx = b * 1024 + h2 * 32 + h1;
                    Ht[t_idx] = val;
                }
            }
        }

        // --- D: Run TEIR Contractions c3 & c4 ---
        auto t2 = std::chrono::high_resolution_clock::now();
        interp3.run();
        interp4.run();
        auto t3 = std::chrono::high_resolution_clock::now();

        total_teir_seconds += std::chrono::duration<double>((t1 - t0) + (t3 - t2)).count();

        // --- E: Bias 2 + Argmax ---
        for (int b = 0; b < valid_items; ++b) {
            float max_val = -1e9f;
            int best_o = -1;
            
            for (int o = 0; o < 16; ++o) {
                float val = logits[b * 16 + o] + b2[o];
                if (val > max_val) {
                    max_val = val;
                    best_o = o;
                }
            }

            // Accuracy & Predict Reporting
            if (best_o == y[tile_idx * TILE + b]) {
                correct_predictions++;
            }

            if (tile_idx == 0 && b < 10) {
                std::cout << " Tile 0, Sample " << std::setw(2) << b 
                          << " | Predicted Logit: " << std::setw(2) << best_o 
                          << " | True Label: " << std::setw(2) << y[b] << "\n";
            }
        }
    }

    // 5. Metriken und Performance berechnen
    // FLOPs pro Tile für jede Kontraktion: (2 * MACs)
    double flops_c1 = 2.0 * TILE * 16 * 16 * 512 * 32; // 2.147 GFLOPs
    double flops_c2 = 2.0 * TILE * 16 * 64 * 16 * 512; // 4.294 GFLOPs
    double flops_c3 = 2.0 * TILE * 32 * 32 * 32;       // 0.016 GFLOPs
    double flops_c4 = 2.0 * TILE * 16 * 32 * 32;       // 0.008 GFLOPs
    double flops_per_tile = flops_c1 + flops_c2 + flops_c3 + flops_c4;
    double total_flops = num_tiles * flops_per_tile;

    double accuracy = (double)correct_predictions / TOTAL_SAMPLES * 100.0;

    std::cout << "\n=== Performance Report ===" << std::fixed << std::setprecision(3) << "\n";
    std::cout << " TEIR Time:  " << total_teir_seconds << " s\n";
    std::cout << " Throughput: " << (total_flops / total_teir_seconds / 1e9) << " GFLOP/s\n";
    std::cout << " Accuracy:   " << accuracy << "% (" << correct_predictions << "/" << TOTAL_SAMPLES << ")\n";
    std::cout << "==========================\n";

    return 0;
}
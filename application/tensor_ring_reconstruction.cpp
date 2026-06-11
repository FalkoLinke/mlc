/**
 * 
 *
 * Benchmark für die 5D Tensor-Ring Rekonstruktion mittels TEIR.
 * Führt drei nacheinander abhängige Kontraktionen (S1, S2, S3) aus.
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

#include "teir.h"
#include "teir_interpreter.h"

// ---------------------------------------------------------------------------
// Extents (Tensor Dimensionen)
// ---------------------------------------------------------------------------
static constexpr uint64_t P = 64;
static constexpr uint64_t Q = 64;
static constexpr uint64_t R = 64;
static constexpr uint64_t S = 64;
static constexpr uint64_t C = 16;
static constexpr uint64_t A = 16;
static constexpr uint64_t B = 16;
static constexpr uint64_t X = 512;
static constexpr uint64_t Y = 512;

// ---------------------------------------------------------------------------
// Helper für File I/O
// ---------------------------------------------------------------------------
void load_bin(const std::string& path, std::vector<float>& buf) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        std::cerr << "Warning: could not open " << path << "\n";
        return;
    }
    ifs.read(reinterpret_cast<char*>(buf.data()), buf.size() * sizeof(float));
}

void save_bin(const std::string& path, const std::vector<float>& buf) {
    std::ofstream ofs(path, std::ios::binary);
    if (ofs) {
        ofs.write(reinterpret_cast<const char*>(buf.data()), buf.size() * sizeof(float));
    }
}

// ---------------------------------------------------------------------------
// S1: qcr, ras -> qcas
// Iteration: q, a  |  Primitive GEMM: M=[c], N=[s], K=[r]
// ---------------------------------------------------------------------------
static teir_operation build_S1() {
    std::vector<teir_tensor> tensors = {
        teir_tensor("Gc", teir_dtype_t::dtype_fp32),
        teir_tensor("Ga", teir_dtype_t::dtype_fp32),
        teir_tensor("qcas", teir_dtype_t::dtype_fp32),
    };
    std::vector<teir_axis> axes = {
        teir_axis("q", Q, { 1024,    0, 16384 }, { 0, 0, 0 }),
        teir_axis("c", C, {   64,    0,  1024 }, { 0, 0, 0 }),
        teir_axis("r", R, {    1, 1024,     0 }, { 0, 0, 0 }),
        teir_axis("a", A, {    0,   64,    64 }, { 0, 0, 0 }),
        teir_axis("s", S, {    0,    1,     1 }, { 0, 0, 0 }),
    };
    std::vector<teir_primitive> primitives = {
        teir_primitive("zero", teir_ptype_t::ptype_zero, {"qcas"}, {{"M", {"c"}}, {"N", {"s"}}}, {{"data_type", "f32"}}),
        teir_primitive("gemm", teir_ptype_t::ptype_contract, {"Gc", "Ga", "qcas"}, {{"M", {"c"}}, {"N", {"s"}}, {"K", {"r"}}}, {{"data_type", "f32"}}),
    };
    teir_schedule schedule({"iter_q"}, {
        teir_iter_node("iter_q", "q", teir_policy_t::policy_parallel, {"iter_a"}),
        teir_iter_node("iter_a", "a", teir_policy_t::policy_parallel, {"inv_zero", "inv_gemm"}),
    }, {
        teir_inv_node("inv_zero", "zero"),
        teir_inv_node("inv_gemm", "gemm"),
    });
    return teir_operation("S1", tensors, axes, primitives, schedule);
}

// ---------------------------------------------------------------------------
// S2: pxq, qcas -> pxacs
// Iteration: p, a, c  |  Primitive GEMM: M=[x], N=[s], K=[q]
// ---------------------------------------------------------------------------
static teir_operation build_S2() {
    std::vector<teir_tensor> tensors = {
        teir_tensor("Gx", teir_dtype_t::dtype_fp32),
        teir_tensor("qcas", teir_dtype_t::dtype_fp32),
        teir_tensor("pxacs", teir_dtype_t::dtype_fp32),
    };
    std::vector<teir_axis> axes = {
        teir_axis("p", P, { 32768,     0, 8388608 }, { 0, 0, 0 }),
        teir_axis("x", X, {    64,     0,   16384 }, { 0, 0, 0 }),
        teir_axis("q", Q, {     1, 16384,       0 }, { 0, 0, 0 }),
        teir_axis("c", C, {     0,  1024,      64 }, { 0, 0, 0 }),
        teir_axis("a", A, {     0,    64,    1024 }, { 0, 0, 0 }),
        teir_axis("s", S, {     0,     1,       1 }, { 0, 0, 0 }),
    };
    std::vector<teir_primitive> primitives = {
        teir_primitive("zero", teir_ptype_t::ptype_zero, {"pxacs"}, {{"M", {"x"}}, {"N", {"s"}}}, {{"data_type", "f32"}}),
        teir_primitive("gemm", teir_ptype_t::ptype_contract, {"Gx", "qcas", "pxacs"}, {{"M", {"x"}}, {"N", {"s"}}, {"K", {"q"}}}, {{"data_type", "f32"}}),
    };
    teir_schedule schedule({"iter_p"}, {
        teir_iter_node("iter_p", "p", teir_policy_t::policy_parallel, {"iter_a"}),
        teir_iter_node("iter_a", "a", teir_policy_t::policy_parallel, {"iter_c"}),
        teir_iter_node("iter_c", "c", teir_policy_t::policy_parallel, {"inv_zero", "inv_gemm"}),
    }, {
        teir_inv_node("inv_zero", "zero"),
        teir_inv_node("inv_gemm", "gemm"),
    });
    return teir_operation("S2", tensors, axes, primitives, schedule);
}

// ---------------------------------------------------------------------------
// S3: pxacs, psby -> xacby
// Iteration: a, c, b, p  |  Primitive GEMM: M=[x], N=[y], K=[s]
// ---------------------------------------------------------------------------
static teir_operation build_S3() {
    std::vector<teir_tensor> tensors = {
        teir_tensor("pxacs", teir_dtype_t::dtype_fp32),
        teir_tensor("Mby", teir_dtype_t::dtype_fp32),
        teir_tensor("xacby", teir_dtype_t::dtype_fp32),
    };
    std::vector<teir_axis> axes = {
        teir_axis("p", P, { 8388608, 524288,       0 }, { 0, 0, 0 }),
        teir_axis("x", X, {   16384,      0, 2097152 }, { 0, 0, 0 }),
        teir_axis("a", A, {    1024,      0,  131072 }, { 0, 0, 0 }),
        teir_axis("c", C, {      64,      0,    8192 }, { 0, 0, 0 }),
        teir_axis("s", S, {       1,   8192,       0 }, { 0, 0, 0 }),
        teir_axis("b", B, {       0,    512,     512 }, { 0, 0, 0 }),
        teir_axis("y", Y, {       0,      1,       1 }, { 0, 0, 0 }),
    };
    std::vector<teir_primitive> primitives = {
        teir_primitive("zero", teir_ptype_t::ptype_zero, {"xacby"}, {{"M", {"x"}}, {"N", {"y"}}}, {{"data_type", "f32"}}),
        teir_primitive("gemm", teir_ptype_t::ptype_contract, {"pxacs", "Mby", "xacby"}, {{"M", {"x"}}, {"N", {"y"}}, {"K", {"s"}}}, {{"data_type", "f32"}}),
    };
    // Wir akkumulieren über "p". Nullsetzen der Ausgabe darf nur bei der allersten "p" Iteration geschehen.
    teir_schedule schedule({"iter_a"}, {
        teir_iter_node("iter_a", "a", teir_policy_t::policy_parallel, {"iter_c"}),
        teir_iter_node("iter_c", "c", teir_policy_t::policy_parallel, {"iter_b"}),
        teir_iter_node("iter_b", "b", teir_policy_t::policy_parallel, {"iter_p"}),
        teir_iter_node("iter_p", "p", teir_policy_t::policy_sequential, {"inv_zero", "inv_gemm"}),
    }, {
        teir_inv_node("inv_zero", "zero", { teir_guard(teir_guard_kind::first, "p") }),
        teir_inv_node("inv_gemm", "gemm"),
    });
    return teir_operation("S3", tensors, axes, primitives, schedule);
}

int main(int argc, char* argv[]) {
    std::cout << "=== TEIR Tensor-Ring Reconstruction Benchmark ===\n\n";

    // 1. Buffer Allokation (Größen explizit mit "ull" um Overflow zu vermeiden)
    std::vector<float> Gc(64ull * 16ull * 64ull);
    std::vector<float> Ga(64ull * 16ull * 64ull);
    std::vector<float> Gx(64ull * 512ull * 64ull);
    std::vector<float> Mby(64ull * 64ull * 16ull * 512ull);

    std::vector<float> qcas(64ull * 16ull * 16ull * 64ull, 0.0f);            // Zwischenpuffer S1 (~4 MB)
    std::vector<float> pxacs(64ull * 512ull * 16ull * 16ull * 64ull, 0.0f);  // Zwischenpuffer S2 (~2 GB)
    std::vector<float> xacby(512ull * 16ull * 16ull * 16ull * 512ull, 0.0f); // Finaler Puffer S3 (~4 GB)

    // 2. Data Input laden
    std::cout << "Lade eng_*.bin Dateien..." << std::endl;
    load_bin("tensor_ring/data/eng_Gc.bin", Gc);
    load_bin("tensor_ring/data/eng_Ga.bin", Ga);
    load_bin("tensor_ring/data/eng_Gx.bin", Gx);
    load_bin("tensor_ring/data/eng_Mby.bin", Mby);

    // 3. Operationen & Interpreter Aufbau
    teir_operation op1 = build_S1();
    teir_operation op2 = build_S2();
    teir_operation op3 = build_S3();

    std::vector<void*> args1 = { Gc.data(), Ga.data(), qcas.data() };
    std::vector<void*> args2 = { Gx.data(), qcas.data(), pxacs.data() };
    std::vector<void*> args3 = { pxacs.data(), Mby.data(), xacby.data() };

    teir_interpreter interp1(op1, args1);
    teir_interpreter interp2(op2, args2);
    teir_interpreter interp3(op3, args3);

    // 4. Execution Pipeline (inkl. Warmup wenn erforderlich)
    std::cout << "Führe Kontraktionen aus (S1 -> S2 -> S3)...\n";
    std::fesetenv(FE_DFL_ENV); // FPU Resetting

    auto t0 = std::chrono::high_resolution_clock::now();
    interp1.run();
    
    auto t1 = std::chrono::high_resolution_clock::now();
    interp2.run();
    
    auto t2 = std::chrono::high_resolution_clock::now();
    interp3.run();
    auto t3 = std::chrono::high_resolution_clock::now();

    // 5. Metriken und GFLOPS berechnen
    double s1_s = std::chrono::duration<double>(t1 - t0).count();
    double s2_s = std::chrono::duration<double>(t2 - t1).count();
    double s3_s = std::chrono::duration<double>(t3 - t2).count();
    double total_s = std::chrono::duration<double>(t3 - t0).count();

    // FLOPs = 2 * Multiplikationen/Additionen per K-Achse
    double flops_S1 = 2.0 * Q * C * R * A * S; // ~0.13 GFLOPs
    double flops_S2 = 2.0 * P * X * Q * C * A * S; // ~68.7 GFLOPs
    double flops_S3 = 2.0 * X * A * C * B * Y * P * S; // ~8796.0 GFLOPs

    std::cout << "\n=== Performance Report ===" << std::fixed << std::setprecision(3) << "\n";
    std::cout << " S1: " << s1_s << " s -> " << (flops_S1 / s1_s / 1e9) << " GFLOP/s\n";
    std::cout << " S2: " << s2_s << " s -> " << (flops_S2 / s2_s / 1e9) << " GFLOP/s\n";
    std::cout << " S3: " << s3_s << " s -> " << (flops_S3 / s3_s / 1e9) << " GFLOP/s\n";
    std::cout << " Total Time: " << total_s << " s\n";
    std::cout << " Total Perf: " << ((flops_S1 + flops_S2 + flops_S3) / total_s / 1e9) << " GFLOP/s\n\n";

    // 6. Referenz überprüfen: T[x=256, a=4, c=0, b=4, y=256]
    uint64_t ref_x = 256, ref_a = 4, ref_c = 0, ref_b = 4, ref_y = 256;
    uint64_t ref_idx = ref_x * 2097152ull + ref_a * 131072ull + ref_c * 8192ull + ref_b * 512ull + ref_y;
    
    std::cout << "Referenzelement Check:\n";
    std::cout << " T[x=256, a=4, c=0, b=4, y=256] = " 
              << xacby[ref_idx] << " (Erwartet: ~0.91)\n\n";

    // 7. Finale Binärdatei abspeichern (Für das Python Skript)
    std::cout << "Speichere rekonstruiertes Tensorfeld als eng_xacby.bin...\n";
    save_bin("eng_xacby.bin", xacby);
    std::cout << "Fertig!" << std::endl;

    return 0;
}
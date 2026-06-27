// mithilfe von KI erstellt

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <iomanip>
#include <algorithm> // für std::max
#include "data/Gemm.h"

// --- Universelle C++ Referenz-Implementierung für beliebige Layouts ---
// trans_x: 0 = Column-Major (Spaltenweise), 1 = Row-Major (Zeilenweise)
void gemm_flexible_reference(
    float const* a, 
    float const* b, 
    float*       c, 
    uint32_t     M, 
    uint32_t     N, 
    uint32_t     K, 
    uint32_t     trans_a, 
    uint32_t     trans_b, 
    uint32_t     trans_c, 
    int64_t      ld_a, 
    int64_t      ld_b, 
    int64_t      ld_c
) {
    // m läuft über die Zeilen von C und A
    for (uint32_t m = 0; m < M; m++) {         
        // n läuft über die Spalten von C und B
        for (uint32_t n = 0; n < N; n++) {
            float sum = 0.0f;
            
            // k läuft über die mathematische Innen-Dimension
            for (uint32_t k = 0; k < K; k++) {
                // A: Row-Major benötigt (m * ld + k), Column-Major benötigt (k * ld + m)
                int64_t idx_a = (trans_a == 1) ? (m * ld_a + k) : (k * ld_a + m);
                
                // B: Row-Major benötigt (k * ld + n), Column-Major benötigt (n * ld + k)
                int64_t idx_b = (trans_b == 1) ? (k * ld_b + n) : (n * ld_b + k);
                
                sum += a[idx_a] * b[idx_b];
            }
            
            // C: Row-Major benötigt (m * ld + n), Column-Major benötigt (n * ld + m)
            int64_t idx_c = (trans_c == 1) ? (m * ld_c + n) : (n * ld_c + m);
            
            // Ergebnis akkumulieren
            c[idx_c] += sum;
        }
    }
}

// --- Hilfsfunktionen zum Füllen ---
void fill_random(std::vector<float>& vec) {
    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] = static_cast<float>(rand() % 100) / 10.0f; 
    }
}

void fill_sequential(std::vector<float>& vec) {
    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] = static_cast<float>(i);
    }
}

// --- Ausgelagerte Test- und Benchmark-Funktion ---
bool run_test_and_benchmark(int M, int N, int K) {
    // Leading Dimensions: Um Speicherverletzungen zu vermeiden, müssen sie den 
    // mathematischen Layout-Regeln folgen.
    int const ld_a = M; // A ist M x K (trans_A=0, Col-Major) -> ld_a >= M
    int const ld_b = N; // B ist K x N (trans_B=1, Row-Major) -> ld_b >= N
    int const ld_c = M; // C ist M x N (trans_C=0, Col-Major) -> ld_c >= M

    int trans_A = 0;
    int trans_B = 1;
    int trans_C = 0;

    std::vector<float> a(ld_a * K);
    std::vector<float> b(K * ld_b);
    std::vector<float> c_asm(ld_c * N); 
    std::vector<float> c_ref(ld_c * N); 

    fill_random(a);
    fill_random(b);
    fill_sequential(c_asm); 
    c_ref = c_asm;


    std::cout << "------------------------------------------------------------\n";
    std::cout << " Teste M=" << M << ", N=" << N << ", K=" << K << "\n";

    // 1. KERNEL GENERIEREN & LADEN

    mini_jit::Gemm gemm;
    auto status = gemm.generate(
        M, N, K, 
        trans_A, trans_B, trans_C,
        mini_jit::Gemm::dtype_t::fp32
    );

    std::cout << "generated";

    if (status != mini_jit::Gemm::error_t::success) {
        std::cerr << " [!] Fehler bei der Kernel-Generierung! Code: " 
                  << static_cast<int>(status) << std::endl;
        return false;
    }

    auto gemm_kernel = gemm.get_kernel();
    if (!gemm_kernel) {
        std::cerr << " [!] Fehler: Kernel-Zeiger ist null!" << std::endl;
        return false;
    }

    // 2. KORREKTHEIT VERIFIZIEREN
    gemm_kernel(a.data(), b.data(), c_asm.data(), ld_a, ld_b, ld_c);
    gemm_flexible_reference(a.data(), b.data(), c_ref.data(), M, N, K, trans_A, trans_B, trans_C, ld_a, ld_b, ld_c);

    bool passed = true;
    float max_diff = 0.0f;
    int error_index = -1;

    for (int i = 0; i < ld_c * N; i++) {
        float expected = c_ref[i];
        float actual = c_asm[i];
        float diff = std::abs(actual - expected);
        
        if (diff > max_diff) max_diff = diff;
        float tol = std::max(1e-4f, 5e-4f * std::abs(expected));

        if (diff > tol && passed) {
            passed = false;
            error_index = i;
        }
    }

    if (!passed) {
        std::cout << " [!] FEHLGESCHLAGEN! Ergebnisse stimmen nicht überein.\n";
        std::cout << "     Erster Fehler bei Index: " << error_index 
                  << " (Zeile " << (error_index % ld_c) << ", Spalte " << (error_index / ld_c) << ")\n";
        std::cout << std::fixed << std::setprecision(5);
        std::cout << "     Erwartet:  " << c_ref[error_index] << "\n";
        std::cout << "     Bekommen:  " << c_asm[error_index] << "\n";
        return false; 
    }
    
    std::cout << " [✓] Verifizierung OK (Max Abweichung: " << std::fixed << std::setprecision(5) << max_diff << ")\n";

    // 3. PERFORMANCE BENCHMARKING
    
    int const num_iterations = 200000; 
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; i++) {
        
        gemm_kernel(a.data(), b.data(), c_asm.data(), ld_a, ld_b, ld_c);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    double duration = elapsed_seconds.count();

    double flops_per_call = 2.0 * M * N * K;
    double total_flops = flops_per_call * num_iterations;
    double gflops = (total_flops / 1e9) / duration;

    // --- DIAGNOSTIK-PRINT (Zeigt uns die nackte Wahrheit, falls cout spinnt) ---
    //std::cout << "  [DEBUG RAW] total_flops: " << total_flops << " | duration: " << duration << " | gflops_raw: " << gflops << "  " << flops_per_call <<"\n";

    // Schöne Ausgabe
    std::cout << " [~] Benchmark: " << num_iterations << " Durchläufe | " 
              << std::fixed << std::setprecision(3) << duration << " Sekunden | "
              << std::fixed << std::setprecision(1) << gflops << " GFLOPS\n";

    return true;
}


int main() {
    srand(time(nullptr)); 

    // Die zu testenden Dimensionen
    std::vector<int> sizes = {16, 528};
    
    std::cout << "============================================================\n";
    std::cout << " STARTE JIT GEMM TESTS FÜR ALLE KOMBINATIONEN\n";
    std::cout << "============================================================\n";

    // Alle Kombinationen durchiterieren
    for (int m : sizes) {
        for (int n : sizes) {
            for (int k : sizes) {
                // Führe Test aus. Wenn er fehlschlägt, brich das Programm ab.
                if (!run_test_and_benchmark(m, n, k)) {
                    std::cerr << "\n[!] Testreihe abgebrochen wegen Fehler bei M=" 
                              << m << ", N=" << n << ", K=" << k << "!\n";
                    return 1;
                }
            }
        }
    }
    
    std::cout << "============================================================\n";
    std::cout << " ALLE TESTS ERFOLGREICH ABGESCHLOSSEN!\n";
    std::cout << "============================================================\n";

    return 0;
}
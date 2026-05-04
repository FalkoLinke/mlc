#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono> // <-- Wichtig für die Zeitmessung

// --- Deklarationen ---
extern "C" {
    void gemm_32_32_1( float const * a,
                       float const * b,
                       float       * c,
                       int64_t       ld_a,
                       int64_t       ld_b,
                       int64_t       ld_c );
}

// --- C++ Referenz-Implementierung für die Verifizierung ---
void gemm_32_32_1_reference(float const* a, float const* b, float* c, int64_t ld_c) {
    for (int j = 0; j < 32; j++) {         
        for (int i = 0; i < 32; i++) {     
            c[j * ld_c + i] += a[i] * b[j];
        }
    }
}

// Hilfsfunktion zum Füllen mit zufälligen Float-Werten
void fill_random(std::vector<float>& vec) {
    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] = static_cast<float>(rand() % 100) / 10.0f; 
    }
}

int main() {
    srand(time(nullptr)); 

    int const M = 32;
    int const N = 32;
    int const K = 1;
    int const ld_c = 32;

    std::vector<float> a(M * K);
    std::vector<float> b(K * N);
    std::vector<float> c_asm(M * N); 
    std::vector<float> c_ref(M * N); 

    // Mit Werten füllen
    fill_random(a);
    fill_random(b);
    fill_random(c_asm);

    c_ref = c_asm;

    // ==========================================
    // 1. KORREKTHEIT VERIFIZIEREN
    // ==========================================
    
    gemm_32_32_1(a.data(), b.data(), c_asm.data(), 32, 32, ld_c);
    gemm_32_32_1_reference(a.data(), b.data(), c_ref.data(), ld_c);

    bool passed = true;
    float max_diff = 0.0f;

    for (int i = 0; i < M * N; i++) {
        float diff = std::abs(c_asm[i] - c_ref[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-4f) {
            passed = false;
            break; 
        }
    }

    if (!passed) {
        std::cout << "FEHLGESCHLAGEN! Ergebnisse stimmen nicht überein.\n";
        return 1; // Programm abbrechen, wenn der Kernel falsch rechnet
    }
    
    std::cout << "============================================\n";
    std::cout << " VERIFIZIERUNG ERFOLGREICH!\n";
    std::cout << " Maximale Abweichung: " << max_diff << "\n";
    std::cout << "============================================\n\n";

    // ==========================================
    // 2. PERFORMANCE BENCHMARKING
    // ==========================================
    std::cout << "Starte Benchmark...\n";

    // Wir rufen den Kernel sehr oft auf, um eine messbare Zeit zu erhalten
    int const num_iterations = 5000000; // 5 Millionen Durchläufe

    // Startzeit erfassen
    auto start_time = std::chrono::high_resolution_clock::now();

    // Schleife für die Messung
    for (int i = 0; i < num_iterations; i++) {
        gemm_32_32_1(a.data(), b.data(), c_asm.data(), 32, 32, ld_c);
    }

    // Endzeit erfassen
    auto end_time = std::chrono::high_resolution_clock::now();

    // Dauer in Sekunden berechnen
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    double duration = elapsed_seconds.count();

    // --- GFLOPS Berechnung ---
    // Rechenoperationen = 2 * M * N * K (Multiplikation + Addition pro Element)
    double flops_per_call = 2.0 * M * N * K;
    double total_flops = flops_per_call * num_iterations;
    
    // GFLOPS = (Total FLOPs / 1 Milliarde) / Dauer in Sekunden
    double gflops = (total_flops / 1e9) / duration;

    std::cout << "============================================\n";
    std::cout << " BENCHMARK ERGEBNISSE\n";
    std::cout << "============================================\n";
    std::cout << " Durchlaeufe:   " << num_iterations << "\n";
    std::cout << " Gesamtzeit:    " << duration << " Sekunden\n";
    std::cout << " Performance:   " << gflops << " GFLOPS\n";
    std::cout << "============================================\n";

    return 0;
}
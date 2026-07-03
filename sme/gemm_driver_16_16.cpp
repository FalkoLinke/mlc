#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <iomanip> // <-- Wichtig für std::setw (schöne tabellarische Ausgabe)

// --- Deklarationen ---
extern "C" {
    void gemm_16_16( float const * a,
                       float const * b,
                       float       * c,
                       int64_t       ld_a,
                       int64_t       ld_b,
                       int64_t       ld_c );
}

// --- C++ Referenz-Implementierung für die Verifizierung ---
void gemm_16_16_512_reference(float const* a, float const* b, float* c, int64_t ld_a, int64_t ld_b, int64_t ld_c) {
    int const K = 512; // Da es gemm_16_16_512 ist, ist K=512
    
    for (int j = 0; j < 16; j++) {         
        for (int i = 0; i < 16; i++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                // A ist Column-Major: Spalte k, Zeile i -> Index [k * ld_a + i]
                // B ist Row-Major:    Zeile k, Spalte j -> Index[k * ld_b + j]
                sum += a[k * ld_a + i] * b[k * ld_b + j];
            }
            // C ist Column-Major: Spalte j, Zeile i -> Index[j * ld_c + i]
            c[j * ld_c + i] += sum;
        }
    }
}

// Hilfsfunktion zum Füllen mit zufälligen Float-Werten
void fill_random(std::vector<float>& vec) {
    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] = static_cast<float>(rand() % 100) / 10.0f; 
    }
}

// Hilfsfunktion für sequenzielle Werte (gut fürs Debugging)
void fill_sequential(std::vector<float>& vec) {
    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] = static_cast<float>(i);
    }
}


// Hilfsfunktion für 0 Wertte
void fill_zeros(std::vector<float>& vec) {
    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] = static_cast<float>(0);
    }
}

// Hilfsfunktion zum sauberen Ausgeben einer Matrix (Column-Major)
void print_matrix(const std::vector<float>& mat, int M, int N, int ld, const std::string& title) {
    std::cout << "--- " << title << " ---\n";
    for (int i = 0; i < M; i++) {          // Zeilen
        for (int j = 0; j < N; j++) {      // Spalten
            // Index-Berechnung für Column-Major: Spalte * LeadingDimension + Zeile
            std::cout << std::setw(8) << std::setprecision(4) << mat[j * ld + i] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main() {
    srand(time(nullptr)); 

    int const M = 16;
    int const N = 16;
    int const K = 512;
    int const ld_c = 16;

    std::vector<float> a(M * K);
    std::vector<float> b(K * N);
    std::vector<float> c_asm(M * N); 
    std::vector<float> c_ref(M * N); 

    // Wir nutzen hier random Werte. Falls du es einfacher im Kopf 
    // nachvollziehen willst, kannst du hier auch fill_sequential() aufrufen!
    fill_random(a);
    fill_random(b);
    fill_sequential(c_asm);

    c_ref = c_asm;

    // ==========================================
    // 1. KORREKTHEIT VERIFIZIEREN
    // ==========================================
    
    // Kernel aufrufen
    gemm_16_16(a.data(), b.data(), c_asm.data(), M, N, ld_c);
    
    // C++ Referenz aufrufen
    gemm_16_16_512_reference(a.data(), b.data(), c_ref.data(), M, N, ld_c);

    bool passed = true;
    float max_diff = 0.0f;
    int error_index = -1; // Merkt sich, wo der erste Fehler passiert ist

    for (int i = 0; i < M * N; i++) {
        float diff = std::abs(c_asm[i] - c_ref[i]);
        if (diff > max_diff) max_diff = diff;
        
        if (diff > 1e-2f && passed) {
            passed = false;
            error_index = i;
            // Wir breaken hier NICHT mehr, damit wir die max_diff des 
            // gesamten Arrays noch fertig berechnen können.
        }
    }

    if (!passed) {
        std::cout << "============================================\n";
        std::cout << " FEHLGESCHLAGEN! Ergebnisse stimmen nicht überein.\n";
        std::cout << " Erster Fehler bei Index: " << error_index 
                  << " (Zeile " << (error_index % ld_c) << ", Spalte " << (error_index / ld_c) << ")\n";
        std::cout << " Erwartet: " << c_ref[error_index] << " | Bekommen: " << c_asm[error_index] << "\n";
        std::cout << "============================================\n\n";

        // BEIDE MATRIZEN AUSGEBEN
        print_matrix(c_ref, M, N, ld_c, "C++ Referenz (Erwartet)");
        print_matrix(c_asm, M, N, ld_c, "ASM Kernel (Aktuell)");

        return 1; // Programm abbrechen
    }

    // BEIDE MATRIZEN AUSGEBEN
    print_matrix(c_ref, M, N, ld_c, "C++ Referenz (Erwartet)");
    print_matrix(c_asm, M, N, ld_c, "ASM Kernel (Aktuell)");
    
    std::cout << "============================================\n";
    std::cout << " VERIFIZIERUNG ERFOLGREICH!\n";
    std::cout << " Maximale Abweichung: " << max_diff << "\n";
    std::cout << "============================================\n\n";

    // ==========================================
    // 2. PERFORMANCE BENCHMARKING
    // ==========================================
    std::cout << "Starte Benchmark...\n";

    int const num_iterations = 5000000; 

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; i++) {
        gemm_16_16(a.data(), b.data(), c_asm.data(), M, N, ld_c);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    double duration = elapsed_seconds.count();

    double flops_per_call = 2.0 * M * N * K;
    double total_flops = flops_per_call * num_iterations;
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
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <iomanip>

// --- Deklarationen ---
extern "C" {
    void gemm_512_512_512( float const * a,
                       float const * b,
                       float       * c,
                       int64_t       ld_a,
                       int64_t       ld_b,
                       int64_t       ld_c );
}

// --- C++ Referenz-Implementierung für die Verifizierung ---
void gemm_512_512_512_reference(float const* a, float const* b, float* c, int64_t ld_a, int64_t ld_b, int64_t ld_c) {
    int const K = 512; // Da es gemm_512_512_512 ist, ist K=512
    
    for (int j = 0; j < 512; j++) {         
        for (int i = 0; i < 512; i++) {
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

void fill_zeros(std::vector<float>& vec) {
    for (size_t i = 0; i < vec.size(); i++) {
        vec[i] = 0.0f;
    }
}

// --- Hilfsfunktionen zum Ausgeben ---

// Ausgabe für Column-Major Matrizen (A und C)
void print_matrix_col_major(const std::vector<float>& mat, int M, int N, int ld, const std::string& title) {
    std::cout << "--- " << title << " ---\n";
    for (int i = 0; i < M; i++) {          // Zeilen
        for (int j = 0; j < N; j++) {      // Spalten
            std::cout << std::setw(6) << std::setprecision(4) << mat[j * ld + i] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Ausgabe für Row-Major Matrizen (B)
void print_matrix_row_major(const std::vector<float>& mat, int M, int N, int ld, const std::string& title) {
    std::cout << "--- " << title << " ---\n";
    for (int i = 0; i < M; i++) {          // Zeilen
        for (int j = 0; j < N; j++) {      // Spalten
            std::cout << std::setw(6) << std::setprecision(4) << mat[i * ld + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}


int main() {
    srand(time(nullptr)); 

    int const M = 512;
    int const N = 512;
    int const K = 512;
    
    // Führende Dimensionen (Leading Dimensions)
    int const ld_a = 512;
    int const ld_b = 512;
    int const ld_c = 512;

    // Größe der Arrays basierend auf den Leading Dimensions berechnen
    std::vector<float> a(ld_a * K);
    std::vector<float> b(K * ld_b);
    std::vector<float> c_asm(ld_c * N); 
    std::vector<float> c_ref(ld_c * N); 

    // Mit Werten füllen
    fill_random(a);
    fill_random(b);
    fill_sequential(c_asm); 

    c_ref = c_asm;

    // ==========================================
    // 1. KORREKTHEIT VERIFIZIEREN
    // ==========================================
    
    // Kernel aufrufen 
    gemm_512_512_512(a.data(), b.data(), c_asm.data(), ld_a, ld_b, ld_c);
    
    // C++ Referenz aufrufen
    gemm_512_512_512_reference(a.data(), b.data(), c_ref.data(), ld_a, ld_b, ld_c);

    bool passed = true;
    float max_diff = 0.0f;
    int error_index = -1;

    for (int i = 0; i < M * N; i++) {
        float expected = c_ref[i];
        float actual = c_asm[i];
        float diff = std::abs(actual - expected);
        
        if (diff > max_diff) max_diff = diff;
        
        // DYNAMISCHE TOLERANZ:
        // Wir erlauben einen relativen Fehler von 0.05% (5e-4) oder 
        // einen absoluten Fehler von 1e-4 bei sehr kleinen Zahlen.
        float tol = std::max(1e-4f, 5e-4f * std::abs(expected));

        if (diff > tol && passed) {
            passed = false;
            error_index = i;
        }
    }

    if (!passed) {
        std::cout << "============================================\n";
        std::cout << " FEHLGESCHLAGEN! Ergebnisse stimmen nicht überein.\n";
        std::cout << " Erster Fehler bei Index: " << error_index 
                  << " (Zeile " << (error_index % ld_c) << ", Spalte " << (error_index / ld_c) << ")\n";
        
        std::cout << std::fixed << std::setprecision(5);
        std::cout << " Erwartet:  " << c_ref[error_index] << "\n";
        std::cout << " Bekommen:  " << c_asm[error_index] << "\n";
        std::cout << " Differenz: " << std::abs(c_asm[error_index] - c_ref[error_index]) << "\n";
        std::cout << " Erlaubte Toleranz hier: " << std::max(1e-4f, 5e-4f * std::abs(c_ref[error_index])) << "\n";
        std::cout << "============================================\n\n";

        return 1; // Programm abbrechen
    }
    
    std::cout << "============================================\n";
    std::cout << " VERIFIZIERUNG ERFOLGREICH!\n";
    std::cout << std::fixed << std::setprecision(5);
    std::cout << " Maximale Abweichung: " << max_diff << "\n";
    std::cout << "============================================\n\n";



    // ==========================================
    // 2. PERFORMANCE BENCHMARKING
    // ==========================================
    std::cout << "Starte Benchmark...\n";

    int const num_iterations = 50000; 

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; i++) {
        gemm_512_512_512(a.data(), b.data(), c_asm.data(), ld_a, ld_b, ld_c);
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
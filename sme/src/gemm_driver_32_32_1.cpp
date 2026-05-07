#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <iomanip>

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
void gemm_32_32_1_reference(float const* a, float const* b, float* c, int64_t ld_a, int64_t ld_b, int64_t ld_c) {
    int const K = 1; // Da es gemm_32_32_1 ist, ist K=1
    
    for (int j = 0; j < 32; j++) {         
        for (int i = 0; i < 32; i++) {
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

    int const M = 32;
    int const N = 32;
    int const K = 1;
    
    // Führende Dimensionen (Leading Dimensions) korrekt definieren
    int const ld_a = 32;
    int const ld_b = 32;
    int const ld_c = 32;

    // Größe der Arrays basierend auf den Leading Dimensions berechnen
    std::vector<float> a(ld_a * K);
    std::vector<float> b(K * ld_b);
    std::vector<float> c_asm(ld_c * N); 
    std::vector<float> c_ref(ld_c * N); 

    // Mit Werten füllen
    fill_sequential(a);
    fill_sequential(b);
    fill_zeros(c_asm); // C testweise mit Nullen füllen, dann sieht man das A*B Ergebnis direkt!

    c_ref = c_asm;

    // ==========================================
    // 1. KORREKTHEIT VERIFIZIEREN
    // ==========================================
    
    // Kernel aufrufen (jetzt korrekt mit ld_a und ld_b anstatt M und N!)
    gemm_32_32_1(a.data(), b.data(), c_asm.data(), ld_a, ld_b, ld_c);
    
    // C++ Referenz aufrufen
    gemm_32_32_1_reference(a.data(), b.data(), c_ref.data(), ld_a, ld_b, ld_c);

    bool passed = true;
    float max_diff = 0.0f;
    int error_index = -1;

    for (int i = 0; i < M * N; i++) {
        float diff = std::abs(c_asm[i] - c_ref[i]);
        if (diff > max_diff) max_diff = diff;
        
        if (diff > 1e-4f && passed) {
            passed = false;
            error_index = i;
        }
    }

    if (!passed) {
        std::cout << "============================================\n";
        std::cout << " FEHLGESCHLAGEN! Ergebnisse stimmen nicht überein.\n";
        std::cout << " Erster Fehler bei Index: " << error_index 
                  << " (Zeile " << (error_index % ld_c) << ", Spalte " << (error_index / ld_c) << ")\n";
        std::cout << " Erwartet: " << c_ref[error_index] << " | Bekommen: " << c_asm[error_index] << "\n";
        std::cout << "============================================\n\n";

        // BEIDE MATRIZEN UND DIE INPUTS AUSGEBEN
        print_matrix_col_major(a, M, K, ld_a, "Matrix A (Column-Major, 32x1)");
        print_matrix_row_major(b, K, N, ld_b, "Matrix B (Row-Major, 1x32)");
        print_matrix_col_major(c_ref, M, N, ld_c, "C++ Referenz (Erwartet)");
        print_matrix_col_major(c_asm, M, N, ld_c, "ASM Kernel (Aktuell)");

        return 1; // Programm abbrechen
    }
    
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
        gemm_32_32_1(a.data(), b.data(), c_asm.data(), ld_a, ld_b, ld_c);
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
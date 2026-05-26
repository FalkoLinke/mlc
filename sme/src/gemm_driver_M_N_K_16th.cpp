#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <iomanip>

// --- Deklarationen ---
extern "C" {
    void gemm_M_N_K_16( float const * a,
                       float const * b,
                       float       * c,
                       int64_t       ld_a,
                       int64_t       ld_b,
                       int64_t       ld_c );
}

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
    gemm_M_N_K_16(a.data(), b.data(), c_asm.data(), ld_a, ld_b, ld_c);
    
    // C++ Referenz aufrufen
    gemm_flexible_reference(a.data(), b.data(), c_ref.data(), M, N, K, 0, 1, 0 ,ld_a, ld_b, ld_c);

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

    int const num_iterations = 100000; 

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; i++) {
        gemm_M_N_K_16(a.data(), b.data(), c_asm.data(), ld_a, ld_b, ld_c);
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
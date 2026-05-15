#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <iomanip>

// Deklaration deiner JIT-Generator Funktion 
void const * generate_gemm_kernel_512_512_512();

typedef void (*gemm_kernel_t)(float const * a,
                              float const * b,
                              float       * c,
                              int64_t       ld_a,
                              int64_t       ld_b,
                              int64_t       ld_c);


// --- C++ Referenz-Implementierung für die Verifizierung ---
void gemm_512_512_512_reference(float const* a, float const* b, float* c, int64_t ld_a, int64_t ld_b, int64_t ld_c) {
    int const K = 512; 
    
    for (int j = 0; j < 512; j++) {         
        for (int i = 0; i < 512; i++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[k * ld_a + i] * b[k * ld_b + j];
            }
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


int main() {
    srand(time(nullptr)); 

    int const M = 512;
    int const N = 512;
    int const K = 512;
    
    int const ld_a = 512;
    int const ld_b = 512;
    int const ld_c = 512;

    std::vector<float> a(ld_a * K);
    std::vector<float> b(K * ld_b);
    std::vector<float> c_asm(ld_c * N); 
    std::vector<float> c_ref(ld_c * N); 

    fill_random(a);
    fill_random(b);
    fill_sequential(c_asm); 
    c_ref = c_asm;

    // ==========================================
    // 2. KERNEL GENERIEREN & LADEN
    // ==========================================
    std::cout << "Generiere JIT Kernel...\n";
    void const* raw_kernel_ptr = generate_gemm_kernel_512_512_512();
    
    if (raw_kernel_ptr == nullptr) {
        std::cerr << "Fehler: Kernel konnte nicht generiert werden!\n";
        return 1;
    }

    // Cast von (void const*) zu unserem ausführbaren Function Pointer Typen
    gemm_kernel_t jit_gemm = reinterpret_cast<gemm_kernel_t>(const_cast<void*>(raw_kernel_ptr));


    // ==========================================
    // 3. KORREKTHEIT VERIFIZIEREN
    // ==========================================
    std::cout << "Führe JIT Kernel aus...\n";
    
    // NEU: Rufe den Kernel über den Pointer auf!
    jit_gemm(a.data(), b.data(), c_asm.data(), ld_a, ld_b, ld_c);
    
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
        std::cout << "============================================\n\n";

        return 1; 
    }
    
    std::cout << "============================================\n";
    std::cout << " VERIFIZIERUNG ERFOLGREICH!\n";
    std::cout << std::fixed << std::setprecision(5);
    std::cout << " Maximale Abweichung: " << max_diff << "\n";
    std::cout << "============================================\n\n";

    // ==========================================
    // 4. PERFORMANCE BENCHMARKING
    // ==========================================
    std::cout << "Starte Benchmark...\n";

    int const num_iterations = 100000; 
    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_iterations; i++) {
        // NEU: Kernel hier ebenfalls über den Pointer aufrufen
        jit_gemm(a.data(), b.data(), c_asm.data(), ld_a, ld_b, ld_c);
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
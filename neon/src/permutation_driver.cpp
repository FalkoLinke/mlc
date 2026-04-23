#include <iostream>
#include <chrono>

/* 
Permutation
===========

This section develops a kernel that performs a permutation operation on the dimensions of a tensor $abc$ to obtain a tensor $cba$: $abc \rightarrow cba$.
The two tensors are stored in row-major order. Dimension $a$ has size $|a|=8$ and $b$ has size $|b|=4$. The size of dimension $c$ is a function parameter.  
*/
extern "C" {
  /**
   * @brief Permutation operation abc->cba
   * @param size_c Size of dimension c.
   * @param abc    Pointer to row-major tensor abc.
   * @param cba    Pointer to row-major tensor cba.
   **/
  void perm_neon_abc_cba(int64_t       size_c,
                         float const * abc,
                         float       * cba);

}


int main() {

    const uint64_t size_a = 8;
    const uint64_t size_b = 4;

    // only works for factors of 4
    const uint64_t size_c = 8;
    const uint64_t total_size = size_a * size_b * size_c;

    float abc[size_a][size_b][size_c];
    float cba[size_c][size_b][size_a];
    float cba_test[size_c][size_b][size_a];

    float counter = 1.0f;
    for (int a = 0; a < size_a; ++a) {
        for (int b = 0; b < size_b; ++b) {
            for (int c = 0; c < size_c; ++c) {
                abc[a][b][c] = counter++;
            }
        }
    }

    // float counter = 1.0f;
    // for (uint a = 0; a < size_a; ++a) {
    //     for (uint b = 0; b < size_b; ++b) {
    //         for (uint c = 0; c < size_c; ++c) {
    //             cba_test[a][b][c] = counter++;
    //         }
    //     }
    // }

    // Warm-up
    perm_neon_abc_cba(size_c, (const float*)abc, (float*)cba);

    // Zeitmessung über mehrere Iterationen für Genauigkeit
    const uint iterations = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (uint i = 0; i < iterations; ++i) {
        perm_neon_abc_cba(size_c, (const float*)abc, (float*)cba);  
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;



    std::cout << "Erste 8 Werte von cba (entsprechen c=0, b=0, a=0..7):" << std::endl;
    for (uint a = 0; a < size_a; ++a) {
        std::cout << cba[0][0][a] << (a < size_a - 1 ? ", " : "");
    }
    std::cout << std::endl;

    double bytes = static_cast<double>(sizeof(float)) * total_size * iterations;
    double gib = bytes / (1024.0 * 1024.0 * 1024.0);
    double gibs = gib / diff.count();

    std::cout << "GiB/s: " << gibs << std::endl;

    return 0;
}

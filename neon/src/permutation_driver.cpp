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
  void perm_neon_abc_cba(int64_t       size_c
                         float const * abc,
                         float       * cba);

}


int main() {

    const int64_t size_a = 8;
    const int64_t size_b = 4;
    const int64_t size_c = 4;

    float abc[size_a][size_b][size_c];
    float cba[size_c][size_b][size_a];

    float counter = 1.0f;
    for (int a = 0; a < size_a; ++a) {
        for (int b = 0; b < size_b; ++b) {
            for (int c = 0; c < size_c; ++c) {
                abc[a][b][c] = counter++;
            }
        }
    }


    // Warm-up
    perm_neon_abc_cba(size_c, abc, cba);

    // Zeitmessung über mehrere Iterationen für Genauigkeit
    const int iterations = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < iterations; ++i) {
        perm_neon_abc_cba(size_c, abc, cba);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;



    std::cout << "Erste 8 Werte von cba (entsprechen c=0, b=0, a=0..7):" << std::endl;
    for (int a = 0; a < size_a; ++a) {
        std::cout << cba[0][0][a] << (a < size_a - 1 ? ", " : "");
    }
    std::cout << std::endl;


    return 0;
}

#include <iostream>
#include "benchmark_kernel.h"



int main() {
    int insts_count = fmadd_kernel();
    std::cout << "Instructions count: " << insts_count << std::endl;
    return 0;
}
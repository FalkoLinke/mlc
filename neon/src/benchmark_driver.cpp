#include <iostream>
#include <chrono>

#include "benchmark_kernel.h"

typedef int(kernel_func_t)();

struct kernel_t {
public:
    kernel_func_t* const func;
    std::string const name;

    kernel_t(kernel_func_t* const func, std::string const name) : func(func), name(name) {

    }

    int call() const {
        return func();
    }
};

#define MAKE_KERNEL(FUNC) ( kernel_t( (FUNC), #FUNC ) )










void handle_kernel(kernel_t const kernel, int const repetitions) {
    std::cout << "========================" << std::endl;
    std::cout << "Benchmarking \"" << kernel.name "\"" << std::endl;

    
    
}






int main() {
    kernel_t kernels[] = {
        MAKE_KERNEL(fmadd_kernel),
    };
    size_t kernels_count = sizeof(kernels) / sizeof(kernel_t);
    int repetitions = 20;

    for (int i = 0; i < kernels_count; i++) {
        handle_kernel(kernels[i], repetitions);
    }

    return 0;
}
#ifndef BENCHMARK_KERNEL_H
#define BENCHMARK_KERNEL_H

extern "C" {

    int fmadd_kernel(int const repetitions);
    int fmla_4s_kernel(int const repetitions);
    int fmla_2s_kernel(int const repetitions);
}

#endif /*BENCHMARK_KERNEL_H*/
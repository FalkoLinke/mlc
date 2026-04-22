#ifndef BENCHMARK_KERNEL_H
#define BENCHMARK_KERNEL_H

extern "C" {

    /* Benchmark kernel for `fmadd`. */
    int fmadd_kernel(int const repetitions);
    /* Benchmark kernel for the `4s` variant of `fmla`. */
    int fmla_4s_kernel(int const repetitions);
    /* Benchmark kernel for the `2s` variant of `fmla`. */
    int fmla_2s_kernel(int const repetitions);

    /* Benchmark kernel for `fmadd`. Optimized by avoiding certain register dependencies. */
    int fmadd_kernel_v2(int const repetitions);
    /* Benchmark kernel for the `4s` variant of `fmla`. Optimized by avoiding certain register dependencies. */
    int fmla_4s_kernel_v2(int const repetitions);
    /* Benchmark kernel for the `2s` variant of `fmla`. Optimized by avoiding certain register dependencies. */
    int fmla_2s_kernel_v2(int const repetitions);
}

#endif /*BENCHMARK_KERNEL_H*/
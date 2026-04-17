#ifndef BENCHMARK_KERNEL_H
#define BENCHMARK_KERNEL_H

extern "C" {

    int fmadd_kernel();
    int fmla_4s_kernel();
    int fmla_2s_kernel();
}

#endif /*BENCHMARK_KERNEL_H*/
#include <iostream>
#include <chrono>

#include "Unary.h"


using mini_jit::Unary;

int counter = 0;








std::string op_descr(Unary::ptype_t const op) {
    switch (op) {
    case Unary::ptype_t::identity:
        return "identity";
    case Unary::ptype_t::zero:
        return "zero";
    case Unary::ptype_t::relu:
        return "relu";
    default:
        return "unknown";
    }
}









void benchmark_jit_kernel(uint64_t m, uint64_t n, Unary::ptype_t op, bool trans_b, long const reps) {
    uint64_t ma = m;
    uint64_t na = n;
    uint64_t mb = trans_b ? n : m;
    uint64_t nb = trans_b ? m : n;

    int64_t lda = ma;
    int64_t ldb = mb;

    std::vector<float> a(na * lda, 0.0f);
    std::vector<float> b(nb * ldb, 0.0f);

    Unary unary;
    unary.generate(m, n, trans_b, Unary::dtype_t::fp32, op);
    Unary::kernel_t kernel_func = unary.get_kernel();

    kernel_func(a.data(), b.data(), lda, ldb);

    auto start = std::chrono::high_resolution_clock::now();

    for (long i = 0; i < reps; i++) {
        kernel_func(a.data(), b.data(), lda, ldb);
        counter += 1;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;


    double gib_transferred = (reps / 1073741824.0f) * (m * n * sizeof(float));
    double time_taken = duration.count();
    double gibs = gib_transferred / time_taken;


    std::cout << "=========================" << std::endl;
    std::cout << "\tOp: " << op_descr(op) << std::endl;
    std::cout << "\t(m, n): (" << m << ", " << n << ")" << std::endl;
    std::cout << "\ttrans_b: " << trans_b << std::endl;
    std::cout << "\tDuration [s]: " << time_taken << std::endl;
    std::cout << "\tBytes transferred [GiB]: " << gib_transferred << std::endl;
    std::cout << "\tGiBs: " << gibs << std::endl;
}














void benchmark_jit_kernels(long const reps) {
    std::vector<uint64_t> ms = {64, 128, 512};
    std::vector<uint64_t> ns = {64, 128, 512};
    std::vector<bool> trans_bs = {false, true};
    std::vector<Unary::ptype_t> ops = {Unary::ptype_t::identity, Unary::ptype_t::zero, Unary::ptype_t::relu};

    for (Unary::ptype_t op : ops) {
        for (bool trans_b : trans_bs) {
            for (uint64_t m : ms) {
                for (uint64_t n : ns) {
                    benchmark_jit_kernel(m, n, op, trans_b, reps);
                }
            }
        }
    }
}

int main() {
    long const reps = 100000000;

    benchmark_jit_kernels(reps);

    std::cerr << counter << std::endl;
    return 0;
}

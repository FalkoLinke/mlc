#include <iostream>

#include "Kernel.h"
#include "KernelFactory.h"
#include "Unary.h"
#include "code_gen_common.hpp"

using mini_jit::Kernel;
using mini_jit::KernelFactory;
using mini_jit::Unary;




int main() {
    uint64_t const m = 16;
    uint64_t const n = 32;
    int64_t const lda = 16;
    int64_t const ldb = 32;
    bool const trans_b = true;
    Unary::ptype_t const op = Unary::ptype_t::zero;

    Unary unary;
    unary.generate(m, n, trans_b, Unary::dtype_t::fp32, op);
    unary.write("test.bin");
    Unary::kernel_t kernel = unary.get_kernel();

    std::vector<float> a(n * lda, 0.0f);
    std::vector<float> b((trans_b ? m : n) * ldb, 1.0f);
    fill_indices(a.data(), m, n, lda);
    fill_const(b.data(), trans_b ? n : m, trans_b ? m : n, ldb, 1.0f);

    kernel(a.data(), b.data(), lda, ldb);

    print_mat(b.data(), trans_b ? n : m, trans_b ? m : n, ldb);

    return 0;
}
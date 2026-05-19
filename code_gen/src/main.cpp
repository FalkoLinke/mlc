#include <iostream>

#include "Kernel.h"
#include "KernelFactory.h"
#include "Unary.h"
#include "code_gen_common.hpp"

using mini_jit::Kernel;
using mini_jit::KernelFactory;
using mini_jit::Unary;




int main() {
    uint64_t const m = 64;
    uint64_t const n = 48;
    int64_t const lda = 128;
    int64_t const ldb = 128;
    bool const trans_b = false;
    Unary::ptype_t const op = Unary::ptype_t::identity;

    Unary unary;
    unary.generate(m, n, trans_b, Unary::dtype_t::fp32, op);
    unary.write("test.bin");
    Unary::kernel_t kernel = unary.get_kernel();

    std::vector<float> a(n * lda, 0.0f);
    std::vector<float> b(n * ldb, 0.0f);
    fill_indices(a.data(), m, n, lda);

    kernel(a.data(), b.data(), lda, ldb);

    print_mat(b.data(), trans_b ? n : m, trans_b ? m : n, ldb);

    return 0;
}
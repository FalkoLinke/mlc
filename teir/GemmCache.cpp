#include "GemmCache.h"

GemmCache::Args::Args(uint32_t m, uint32_t n, uint32_t k, uint32_t trans_a, uint32_t trans_b, uint32_t trans_c, mini_jit::Gemm::dtype_t dtype) : m(m), n(n), k(k), trans_a(trans_a), trans_b(trans_b), trans_c(trans_c), dtype(dtype) {

}

bool GemmCache::Args::operator<(Args const& other) const {
    // lexicographically compare the members
    if (m < other.m) {
        return true;
    }
    if (m > other.m) {
        return false;
    }

    if (n < other.n) {
        return true;
    }
    if (n > other.n) {
        return false;
    }

    if (k < other.k) {
        return true;
    }
    if (k > other.k) {
        return false;
    }

    if (trans_a < other.trans_a) {
        return true;
    }
    if (trans_a > other.trans_a) {
        return false;
    }

    if (trans_b < other.trans_b) {
        return true;
    }
    if (trans_b > other.trans_b) {
        return false;
    }

    if (trans_c < other.trans_c) {
        return true;
    }
    if (trans_c > other.trans_c) {
        return false;
    }

    if (dtype < other.dtype) {
        return true;
    }
    if (dtype > other.dtype) {
        return false;
    }

    return false;
}






mini_jit::Gemm::kernel_t GemmCache::get_kernel(uint32_t m, uint32_t n, uint32_t k, uint32_t trans_a, uint32_t trans_b, uint32_t trans_c, mini_jit::Gemm::dtype_t dtype) {
    Args args(m, n, k, trans_a, trans_b, trans_c, dtype);

    // attempt to load a precreated unary kernel
    try {
        mini_jit::Gemm* gemm = cache.at(args).get();
        mini_jit::Gemm::kernel_t kernel = gemm->get_kernel();
        return kernel;
    } catch (std::exception const& e) {
        // std::map threw an exception due to a missing key
    }

    // no precreated unary kernel exists
    // attempt to generate a kernel
    std::unique_ptr<mini_jit::Gemm> gemm = std::make_unique<mini_jit::Gemm>();
    mini_jit::Gemm::error_t err = gemm->generate(m, n, k, trans_a, trans_b, trans_c, dtype);
    if (err != mini_jit::Gemm::error_t::success) {
        // could not generate the kernel
        return nullptr;
    }

    // obtain the kernel
    // do this before moving the unique_ptr
    mini_jit::Gemm::kernel_t kernel = gemm->get_kernel();

    // store it in the cache
    cache.emplace(std::make_pair(args, std::move(gemm)));
    
    // return the kernel
    return kernel;
}





void GemmCache::clear() {
    cache.clear();
}








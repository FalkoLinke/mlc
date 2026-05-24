#include "UnaryCache.h"


UnaryCache::Args::Args(uint32_t m, uint32_t n, uint32_t trans_b, mini_jit::Unary::dtype_t dtype, mini_jit::Unary::ptype_t ptype) : m(m), n(n), trans_b(trans_b), dtype(dtype), ptype(ptype) {

}

bool UnaryCache::Args::operator<(Args const& other) const {
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

    if (trans_b < other.trans_b) {
        return true;
    }
    if (trans_b > other.trans_b) {
        return false;
    }

    if (dtype < other.dtype) {
        return true;
    }
    if (dtype > other.dtype) {
        return false;
    }

    if (ptype < other.ptype) {
        return true;
    }
    if (ptype > other.ptype) {
        return false;
    }

    return false;
}






mini_jit::Unary::kernel_t UnaryCache::get_kernel(uint32_t m, uint32_t n, uint32_t trans_b, mini_jit::Unary::dtype_t dtype, mini_jit::Unary::ptype_t  ptype) {
    Args args(m, n, trans_b, dtype, ptype);

    // attempt to load a precreated unary kernel
    try {
        mini_jit::Unary* unary = cache.at(args).get();
        mini_jit::Unary::kernel_t kernel = unary->get_kernel();
        return kernel;
    } catch (std::exception const& e) {
        // std::map threw an exception due to a missing key
    }

    // no precreated unary kernel exists
    // attempt to generate a kernel
    std::unique_ptr<mini_jit::Unary> unary = std::make_unique<mini_jit::Unary>();
    mini_jit::Unary::error_t err = unary->generate(m, n, trans_b, dtype, ptype);
    if (err != mini_jit::Unary::error_t::success) {
        // could not generate the kernel
        return nullptr;
    }

    // obtain the kernel
    // do this before moving the unique_ptr
    mini_jit::Unary::kernel_t kernel = unary->get_kernel();

    // store it in the cache
    cache.emplace(std::make_pair(args, std::move(unary)));
    
    // return the kernel
    return kernel;
}





void UnaryCache::clear() {
    cache.clear();
}








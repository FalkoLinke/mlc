#ifndef UNARY_CACHE_H
#define UNARY_CACHE_H


#include <memory>

#include "Unary.h"




struct UnaryCache {
private:
    struct Args {
    private:
    public:
        uint32_t const m;
        uint32_t const n;
        uint32_t const trans_b;
        mini_jit::Unary::dtype_t const dtype;
        mini_jit::Unary::ptype_t const ptype;

        Args(uint32_t m, uint32_t n, uint32_t trans_b, mini_jit::Unary::dtype_t dtype, mini_jit::Unary::ptype_t);

        bool operator<(Args const& other) const;
    };

    std::map<Args, std::unique_ptr<mini_jit::Unary>> cache;

public:
    UnaryCache() = default;
    ~UnaryCache() = default;

    mini_jit::Unary::kernel_t get_kernel(
        uint32_t m,
        uint32_t n,
        uint32_t trans_b,
        mini_jit::Unary::dtype_t  dtype,
        mini_jit::Unary::ptype_t  ptype);
    
    void clear();
};





#endif
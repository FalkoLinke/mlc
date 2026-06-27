#ifndef UNARY_CACHE_H
#define UNARY_CACHE_H


#include <memory>

#include "Unary.h"



/**
 * Generates and stores `Unary` kernels.
 * Previously generated kernels may be retreived without recompilation.
 * 
 * Function pointers to the generated kernels are guaranteed to remain valid for the
 * lifetime of the holding `UnaryCache` instance.
 */
struct UnaryCache {
private:
    /** Type to use as a key in the `std::map` to look up the stored kernels. */
    struct Args {
    private:
    public:
        uint32_t const m;
        uint32_t const n;
        uint32_t const trans_b;
        mini_jit::Unary::dtype_t const dtype;
        mini_jit::Unary::ptype_t const ptype;

        Args(uint32_t m, uint32_t n, uint32_t trans_b, mini_jit::Unary::dtype_t dtype, mini_jit::Unary::ptype_t);

        /** Required for use as a key in `std::map`. */
        bool operator<(Args const& other) const;
    };

    /** The stored kernels. */
    std::map<Args, std::unique_ptr<mini_jit::Unary>> cache;

public:
    UnaryCache() = default;
    ~UnaryCache() = default;

    /**
     * Returns the stored kernel for the given parameters.
     * If no kernel is stored this function attempts to generate one.
     * Returns `nullptr` if no kernel is stored and no kernel could be generated.
     */
    mini_jit::Unary::kernel_t get_kernel(
        uint32_t m,
        uint32_t n,
        uint32_t trans_b,
        mini_jit::Unary::dtype_t dtype,
        mini_jit::Unary::ptype_t ptype);
    
    /**
     * @brief Removes all generated kernels from the cache.
     */
    void clear();
};





#endif
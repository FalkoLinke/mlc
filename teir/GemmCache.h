#ifndef TEIR_GEMM_CACHE_H
#define TEIR_GEMM_CACHE_H

#include <cstdint>
#include <map>
#include <memory>
#include "Gemm.h"


/**
 * Generates and stores `Gemm` kernels.
 * Previously generated kernels may be retreived without recompilation.
 * 
 * Function pointers to the generated kernels are guaranteed to remain valid for the
 * lifetime of the holding `GemmCache` instance.
 */
struct GemmCache {
private:
    /** Type to use as a key in the `std::map` to look up the stored kernels. */
    struct Args {
    private:
    public:
        uint32_t const m;
        uint32_t const n;
        uint32_t const k;
        uint32_t const trans_a;
        uint32_t const trans_b;
        uint32_t const trans_c;
        mini_jit::Gemm::dtype_t const dtype;

        Args(uint32_t m, uint32_t n, uint32_t k, uint32_t trans_a, uint32_t trans_b, uint32_t trans_c, mini_jit::Gemm::dtype_t dtype);

        /** Required for use as a key in `std::map`. */
        bool operator<(Args const& other) const;
    };

    /** The stored kernels. */
    std::map<Args, std::unique_ptr<mini_jit::Gemm>> cache;

public:
    GemmCache() = default;
    ~GemmCache() = default;

    /**
     * Returns the stored kernel for the given parameters.
     * If no kernel is stored this function attempts to generate one.
     * Returns `nullptr` if no kernel is stored and no kernel could be generated.
     */
    mini_jit::Gemm::kernel_t get_kernel(
        uint32_t m,
        uint32_t n,
        uint32_t k,
        uint32_t trans_a,
        uint32_t trans_b,
        uint32_t trans_c,
        mini_jit::Gemm::dtype_t dtype);
    
    /**
     * @brief Removes all generated kernels from the cache.
     */
    void clear();
};


#endif
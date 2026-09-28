#ifndef TEIR_TEIR_LOWERING_H
#define TEIR_TEIR_LOWERING_H

#include <cstdint>
#include <vector>

#include "teir.h"
#include "Unary.h"
#include "Gemm.h"


/**
 * @brief Describes how a tile primitive is executed by a JIT-generated kernel.
 *
 * The plan only depends on the primitive and the axes of the operation,
 * not on the current iteration. It is therefore computed once per primitive
 * and shared by the interpreter and the compiler.
 */
struct teir_tile_plan {
    enum kind_t {
        /** The primitive can not be executed by a tile kernel. */
        none,
        /** The primitive is executed by a `mini_jit::Unary` kernel. */
        unary,
        /** The primitive is executed by a `mini_jit::Gemm` kernel. */
        gemm,
    };

    kind_t kind = none;

    /** Kernel parameters. `k`, `trans_a` and `trans_c` are only used by GEMM kernels. */
    mini_jit::Unary::ptype_t unary_ptype = mini_jit::Unary::ptype_t::identity;
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t trans_a = 0;
    uint32_t trans_b = 0;
    uint32_t trans_c = 0;

    /**
     * The indices of the tensors passed to the kernel, in the kernel's argument order.
     * Unary kernels: (in, out), where `in` is unused for `zero`. GEMM kernels: (A, B, C).
     * An index of `no_tensor` passes `nullptr`.
     */
    std::vector<uint64_t> tensors;

    /** The leading dimensions passed to the kernel in elements, in the kernel's argument order. */
    std::vector<int64_t> lds;

    static constexpr uint64_t no_tensor = ~(uint64_t)0;
};


/**
 * @brief Finds a tile kernel for a primitive with exactly one axis per dimension (M, N and, for contractions, K).
 *
 * Elementwise primitives (zero, copy, relu) are symmetric in M and N, so both assignments of the two axes
 * to the rows and columns of the kernel are tried. Contractions select the transposition flags of the
 * GEMM kernel from the unit-stride axis of each tensor.
 *
 * @return A plan of kind `none` if no kernel supports the primitive's shape or strides.
 */
teir_tile_plan teir_plan_tile(teir_operation const& operation, teir_primitive const& primitive);


#endif

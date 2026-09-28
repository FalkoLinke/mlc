#ifndef TEIR_TEIR_PASSES_H
#define TEIR_TEIR_PASSES_H

#include <cstdint>
#include <string>
#include <vector>

#include "teir.h"


/**
 * @brief The hardware parameters used by the optimization passes.
 *
 * `teir_target::host()` reads them from the machine (macOS `sysctl`, OpenMP, `rdsvl`).
 * The defaults are the values of the Apple M4 used for the benchmarks.
 */
struct teir_target {
    /** L1 data cache of a P-core. */
    uint64_t l1d_bytes = 128 * 1024;
    /** L2 cache of the P-core cluster. */
    uint64_t l2_bytes = 16 * 1024 * 1024;
    /** Number of cores sharing the L2 cache. */
    uint64_t cores_per_l2 = 4;
    /** Number of threads used for parallel iteration nodes (P- and E-cores). */
    uint64_t threads = 10;
    /** Streaming vector length. A ZA tile holds (svl_bytes / 4)^2 FP32 values. */
    uint64_t svl_bytes = 64;
    /** Minimum number of parallel iterations per thread, so that dynamic scheduling can balance P- and E-cores. */
    uint64_t iterations_per_thread = 8;

    /** Rows and columns of the GEMM microkernel: 2 x 2 ZA tiles. */
    uint64_t microkernel() const { return 2 * svl_bytes / 4; }

    static teir_target host();
};


/** The optimization passes, which may be combined as bit flags. */
enum teir_pass_t : uint32_t {
    teir_opt_none = 0,
    teir_opt_fusion = 1,
    teir_opt_operands = 2,
    teir_opt_blocking = 4,
    teir_opt_parallel = 8,
    teir_opt_all = 15,
};

/** Short name of a single pass, e.g. "fusion". */
std::string teir_pass_name(teir_pass_t pass);


/**
 * @brief Moves the enclosing loops of contraction invocations into the GEMM primitive.
 *
 * For every contraction invocation whose parent iteration node iterates over an axis that can be fused with the
 * primitive's axis of the same dimension (M, N or K), the node is promoted into the primitive and the two axes are fused.
 * Repeated until no further loop can be moved. Larger primitives give the kernel more work per call.
 */
teir_operation teir_pass_primitive_fusion(teir_operation const& operation, teir_target const& target, std::vector<std::string>* log = nullptr);

/**
 * @brief Swaps the inputs of contractions whose output would be passed to the GEMM kernel in row-major format.
 *
 * Our generator is faster for a column-major C (e.g. 483 instead of 388 GFLOPS for the 32 x 64 x 512 tiles of matmul).
 */
teir_operation teir_pass_operand_order(teir_operation const& operation, teir_target const& target, std::vector<std::string>* log = nullptr);

/**
 * @brief Splits the M, N and K axes of large GEMM primitives into blocks that fit into the caches.
 *
 * - K block: at most `l1d_bytes / 2 / (4 * microkernel)` values, so that the A strip of a 32 x K microkernel call
 *   stays in half of the L1 cache while the microkernel sweeps over N.
 * - M and N blocks: multiples of the microkernel size (or the full extent) with the largest block of C such that the
 *   blocks of A, B and C fit into the L2 cache, or into the share of one core if `parallel` is set.
 * The outer parts of the axes become loops M_o -> N_o -> K_o around the primitive.
 */
teir_operation teir_pass_cache_blocking(teir_operation const& operation, teir_target const& target, bool parallel, std::vector<std::string>* log = nullptr);

/**
 * @brief Sets the policy `parallel` on the outermost chain of non-reduction iteration nodes.
 *
 * Starting at each root, reduction nodes are skipped while no node has been selected. Directly nested non-reduction nodes
 * are selected until their combined iteration count reaches `iterations_per_thread * threads`.
 * The runtime collapses the selected nodes into one parallel region.
 */
teir_operation teir_pass_parallelization(teir_operation const& operation, teir_target const& target, std::vector<std::string>* log = nullptr);

/**
 * @brief Applies the selected passes in the order fusion, operand order, cache blocking, parallelization.
 */
teir_operation teir_optimize(teir_operation const& operation, teir_target const& target, uint32_t passes, std::vector<std::string>* log = nullptr);


#endif

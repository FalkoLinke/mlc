#ifndef TEIR_TEIR_TRANSFORM_H
#define TEIR_TEIR_TRANSFORM_H

#include <cstdint>
#include <stdexcept>
#include <string>

#include "teir.h"


/**
 * Transformations of TEIR operations.
 *
 * Every transformation returns a transformed copy of the operation and leaves the input unchanged.
 * A transformation checks the conditions under which it preserves the result of the operation and throws
 * `teir_transform_error` if they are not satisfied, so optimization passes may simply try a transformation.
 *
 * Terminology:
 * - The written tensor of a primitive is its last tensor.
 * - An axis is a reduction axis of a subtree if a tensor written in the subtree has the stride 0 along the axis.
 * - Guards refer to axes. A guard list is a conjunction.
 */


struct teir_transform_error : std::runtime_error {
    using std::runtime_error::runtime_error;
};


/**
 * @brief Splits the axis `axis_id` of extent `e` into an outer axis `<axis_id>_o` of extent `e / inner_extent`
 * and an inner axis `<axis_id>_i` of extent `inner_extent`.
 *
 * - If the axis is iterated by an iteration node, the node iterates over the outer axis and gets a new
 *   single child iterating over the inner axis, which takes over the children.
 *   Guards `first(a)` / `last(a)` become `first(a_o), first(a_i)` / `last(a_o), last(a_i)`.
 * - If the axis is used by primitives, the primitives use the inner axis. Every run of consecutive sibling invocation
 *   nodes of these primitives is wrapped into a new sequential iteration node over the outer axis.
 *
 * Requires `1 < inner_extent < e` and `e % inner_extent == 0`.
 */
teir_operation teir_split(teir_operation const& operation, std::string const& axis_id, uint64_t inner_extent);

/**
 * @brief Fuses the axes `outer_id` and `inner_id` into a single axis `<outer_id><inner_id>`.
 *
 * Requires `stride(outer, t) == extent(inner) * stride(inner, t)` for every tensor `t` and zero offsets of the inner axis.
 * - Iterated axes: the node over `outer_id` must have the node over `inner_id` as its only child, and the inner node must not have guards.
 *   Both are merged into one node. Guards `first(o), first(i)` (or `last`) become `first(oi)`; other guards on the axes are rejected.
 * - Primitive axes: every primitive using one of the axes must list `outer_id` immediately before `inner_id` in the same dimension.
 */
teir_operation teir_fuse(teir_operation const& operation, std::string const& outer_id, std::string const& inner_id);

/**
 * @brief Promotes the axis of the iteration node `node_id` into the dimension `dim` ("M", "N" or "K") of the primitives invoked below it.
 *
 * The iteration node is removed and replaced by its children, which must all be invocation nodes.
 * - Children whose primitive accesses a tensor along the axis ("users") get the axis prepended to `dim` (outermost position).
 *   For contractions the dimension has to match the strides: `K` is read by both inputs but not written,
 *   `M` is not read by the second input, `N` is not read by the first input.
 * - All other children must be guarded by `first(axis)` and precede all users, or by `last(axis)` and follow all users.
 *   They are executed once, and the guard is removed.
 * - A primitive which is also invoked elsewhere is copied before it is changed.
 * The guards of the iteration node are added to each child. Requires zero offsets of the axis.
 */
teir_operation teir_promote(teir_operation const& operation, std::string const& node_id, std::string const& dim);

/**
 * @brief Interchanges the iteration node `node_id` with its only child, which must be an iteration node without guards.
 *
 * The node identifiers stay with their axes. The interchange is rejected if both axes are reduction axes of the subtree,
 * which would change the order in which contributions are accumulated into the same output elements.
 */
teir_operation teir_reorder(teir_operation const& operation, std::string const& node_id);

/**
 * @brief Sets the policy of the iteration node `node_id`.
 *
 * `parallel` is rejected if the axis is a reduction axis of the node's subtree,
 * since different iterations would then write the same output elements.
 */
teir_operation teir_set_policy(teir_operation const& operation, std::string const& node_id, teir_policy_t policy);


/**
 * @brief Swaps the two inputs of the contraction primitive `primitive_id` together with its M and N dimensions.
 *
 * `out[m, n] += in0[m, k] * in1[k, n]` becomes `out[n, m] += in1[n, k] * in0[k, m]` with M and N exchanged,
 * which computes the same values. The storage formats seen by the GEMM kernel change: a row-major C becomes column-major.
 */
teir_operation teir_swap_operands(teir_operation const& operation, std::string const& primitive_id);


/** @brief Returns the operation in the TEIR text format. */
std::string teir_to_string(teir_operation const& operation);


#endif

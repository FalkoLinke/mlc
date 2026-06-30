#ifndef TEIR_TEIR_INTERPRETER_H
#define TEIR_TEIR_INTERPRETER_H

#include <memory>

#include "teir.h"
#include "Unary.h"
#include "UnaryCache.h"
#include "GemmCache.h"





enum teir_interpreter_error_t {
    teir_err_unresolved_axis_id = 0,
    teir_err_unresolved_primitive_id = 1,
    teir_err_unresolved_iter_node_id = 2,
    teir_err_unresolved_inv_node_id = 3,
    teir_err_unresolved_schedule_node_id = 4,
    teir_err_unresolved_tensor_id = 5,
    teir_err_missing_primitive_lowering = 6,
    teir_err_missing_strides = 7,
    teir_err_missing_offsets = 8,
    teir_err_invalid_guard = 9,
};







struct teir_interpreter {
    private:
        UnaryCache unary_cache;
        GemmCache gemm_cache;

        uint64_t resolve_tensor_id_idx(std::string const& id) const;
        teir_axis const* resolve_axis_id(std::string const& id) const;
        teir_primitive const* resolve_primitive_id(std::string const& id) const;
        teir_iter_node const* resolve_iter_node_id(std::string const& id) const;
        teir_inv_node const* resolve_inv_node_id(std::string const& id) const;

        /**
         * @brief Checks if the index and axis satisfy the guard.
         * 
         * If the ID of `axis` and `guard.axis_id` do not match returns `true`.
         * Otherwise checks if `idx` satisfies the constraint imposed by `guard`.
         * 
         */
        bool guard_matches_index(teir_axis const& axis, uint64_t idx, teir_guard const& guard);
        /**
         * @brief Checks if the index path satisfies the given guard.
         * @param axis_path: The axes currently being iterated over.
         * @param index_path: The current indices of the iteration for the axes in axis_path.
         * @param guard: The guard to check.
         * @return `true` if the guard is satisfied, `false` otherwise.
         */
        bool guard_satisfied(std::vector<teir_axis const*> const& axis_path, std::vector<uint64_t> const& index_path, teir_guard const& guard);
        /**
         * @brief Checks if the index path satisfies all of the given guards.
         * @param axis_path: The axes currently being iterated over.
         * @param index_path: The current indices of the iteration for the axes in axis_path.
         * @param guards: The guards to check.
         * @return `true` if the guards are satisfied, `false` otherwise.
         */
        bool guards_satisfied(std::vector<teir_axis const*> const& axis_path, std::vector<uint64_t> const& index_path, std::vector<teir_guard> const& guards);
        void iterate(std::string const& node, std::vector<teir_axis const*> axis_path, std::vector<uint64_t> index_path);    
        void lower(teir_inv_node const* inv_node, std::vector<teir_axis const*> axis_path, std::vector<uint64_t> index_path);

        std::vector<void*> indexed_tensors(std::vector<teir_axis const*> const& axis_path, std::vector<uint64_t> const& index_path) const;

    public:
        teir_operation const& operation;
        std::vector<void*> const& args;

        teir_interpreter(teir_operation const& operation, std::vector<void*> const& args);

        void run();
};












#endif
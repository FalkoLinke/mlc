#ifndef TEIR_INTERPRETER_H
#define TEIR_INTERPRETER_H

#include <memory>

#include "teir.h"
#include "Unary.h"





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
};







struct teir_interpreter {
    private:
        uint64_t resolve_tensor_id_idx(std::string const& id) const;
        teir_axis const* resolve_axis_id(std::string const& id) const;
        teir_primitive const* resolve_primitive_id(std::string const& id) const;
        teir_iter_node const* resolve_iter_node_id(std::string const& id) const;
        teir_inv_node const* resolve_inv_node_id(std::string const& id) const;

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
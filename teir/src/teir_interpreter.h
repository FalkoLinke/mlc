#ifndef TEIR_INTERPRETER_H
#define TEIR_INTERPRETER_H

#include <memory>

#include "teir.h"
#include "Unary.h"







struct teir_interpreter {
    private:
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
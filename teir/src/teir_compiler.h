#ifndef TEIR_COMPILER_H
#define TEIR_COMPILER_H

#include <memory>

#include "teir.h"
#include "Kernel.h"
#include "InstGen.h"
#include "Unary.h"
#include "Gemm.h"
#include "UnaryCache.h"




struct teir_compiler {
    private:
        /** 
         * The registers which may be used to hold the loop indices.
         * The compiler raises an error if no more registers are available.
         */
        std::vector<mini_jit::InstGen::gpr_t> const loop_registers {
            mini_jit::InstGen::gpr_t::x19,
            mini_jit::InstGen::gpr_t::x20,
            mini_jit::InstGen::gpr_t::x21,
            mini_jit::InstGen::gpr_t::x22,
            mini_jit::InstGen::gpr_t::x23,
            mini_jit::InstGen::gpr_t::x24,
            mini_jit::InstGen::gpr_t::x25,
            mini_jit::InstGen::gpr_t::x26,
        };

        UnaryCache unary_cache;

        std::vector<void*> kernel_functions;
        mini_jit::Kernel kernel;
        mini_jit::InstGen ig;

        std::vector<uint64_t> resolve_tensor_labels(teir_operation const& operation, teir_primitive const& primitive) const;
        bool lower_zero_scalar(teir_operation const& operation, teir_primitive const& primitive);
        bool lower_zero_tile(teir_operation const& operation, teir_primitive const& primitive);
        bool lower_identity_scalar(teir_operation const& operation, teir_primitive const& primitive);
        bool lower_identity_tile_notrans(teir_operation const& operation, teir_primitive const& primitive);
        bool lower_identity_tile_trans(teir_operation const& operation, teir_primitive const& primitive);

        void iterate(teir_operation const& operation, std::string const& node, std::vector<teir_axis const*> axis_path, std::vector<mini_jit::InstGen::gpr_t> index_path);
        void invoke(teir_operation const& operation, teir_inv_node const* inv_node, std::vector<teir_axis const*> axis_path, std::vector<mini_jit::InstGen::gpr_t> index_path);

    public:

        void compile(teir_operation const& operation);


        using teir_function_t = void(*)(void**);
        teir_function_t get_function() const;

        void write(const char* fp) const;
};

#endif
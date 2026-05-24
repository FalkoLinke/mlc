#include <iostream>

#include "teir_compiler.h"






using mini_jit::Kernel;
using mini_jit::InstGen;
using mini_jit::Unary;
using mini_jit::Gemm;






void teir_compiler::prepare_primitive(teir_operation const& operation, teir_primitive const& primitive) {
    if (primitive.ptype == teir_ptype_t::ptype_zero) {
        if (primitive.axes.at("M").size() == 1 && primitive.axes.at("N").size() == 1) {
            teir_axis const* axis_m = operation.resolve_axis_id(primitive.axes.at("M")[0]);
            teir_axis const* axis_n = operation.resolve_axis_id(primitive.axes.at("N")[0]);

            std::unique_ptr<Unary> unary = std::make_unique<Unary>();
            Unary::error_t err = unary->generate(axis_m->extent, axis_n->extent, false, Unary::dtype_t::fp32, Unary::ptype_t::zero);
            if (err != Unary::error_t::success) {
                std::cout << "Failed to generate kernel" << std::endl;
            }
            Unary::kernel_t kernel = unary->get_kernel();

            unary_kernels.emplace_back(std::move(unary));
            kernel_functions.push_back((void*)kernel);
        }
    }
}





void teir_compiler::iterate(teir_operation const& operation, std::string const& node, std::vector<teir_axis const*> axis_path, std::vector<InstGen::gpr_t> index_path) {
    teir_iter_node const* iter_node = nullptr;
    if ((iter_node = operation.schedule.resolve_iter_id(node)) != nullptr) {
        teir_axis const* axis = operation.resolve_axis_id(iter_node->axis);

        InstGen::gpr_t reg = (InstGen::gpr_t)loop_registers[axis_path.size()];
        std::string loop_start_label = iter_node->id + "_loop";
        std::string loop_end_label = iter_node->id + "_end";

        kernel.add_instr(ig.base_movz(reg, axis->extent));
        kernel.add_label(loop_start_label);
        kernel.add_branch(ig.base_cbz(reg, loop_end_label));

        for (std::string const& child_id : iter_node->children) {
            std::vector<teir_axis const*> ap = axis_path;
            std::vector<InstGen::gpr_t> ip = index_path;
            ap.push_back(axis);
            ip.push_back(reg);
            iterate(operation, child_id, ap, ip);
        }

        kernel.add_instr(ig.base_sub(reg, reg, 1));
        kernel.add_branch(ig.base_b(loop_start_label));
        kernel.add_label(loop_end_label);

        return;
    }

    teir_inv_node const* inv_node = nullptr;
    if ((inv_node = operation.schedule.resolve_inv_id(node)) != nullptr) {
        invoke(operation, inv_node, axis_path, index_path);
        return;
    }

    std::cout << "could not resolve node id" << std::endl;
}

void teir_compiler::invoke(teir_operation const& operation, teir_inv_node const* inv_node, std::vector<teir_axis const*> axis_path, std::vector<mini_jit::InstGen::gpr_t> index_path) {
    teir_primitive const* primitive = operation.resolve_primitive_id(inv_node->primitive);

    // resolve primitive arguments
    std::vector<uint64_t> primitive_tensor_idxs;
    for (std::string const& tensor_id : primitive->tensors) {
        uint64_t tensor_idx = operation.resolve_tensor_id_idx(tensor_id);
        primitive_tensor_idxs.push_back(tensor_idx);
    }

    // prepare kernel arguments
    if (primitive->ptype == teir_ptype_t::ptype_copy || primitive->ptype == teir_ptype_t::ptype_relu) {
        // unary kernel signature
        kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x0, InstGen::gpr_t::x28, primitive_tensor_idxs[0] * 8, InstGen::addr_mode_t::unsigned_offset));
        kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x1, InstGen::gpr_t::x28, primitive_tensor_idxs[1] * 8, InstGen::addr_mode_t::unsigned_offset));
        kernel.add_instr(ig.base_movz(InstGen::gpr_t::x2, 0));
        kernel.add_instr(ig.base_movz(InstGen::gpr_t::x3, 0));
        
    } else if (primitive->ptype == teir_ptype_t::ptype_zero) {
        // unary kernel signature
        kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x1, InstGen::gpr_t::x28, primitive_tensor_idxs[0] * 8, InstGen::addr_mode_t::unsigned_offset));
        kernel.add_instr(ig.base_movz(InstGen::gpr_t::x3, 0));

    } else if (primitive->ptype == teir_ptype_t::ptype_contract) {
        // gemm kernel signature
        kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x0, InstGen::gpr_t::x28, primitive_tensor_idxs[0] * 8, InstGen::addr_mode_t::unsigned_offset));
        kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x1, InstGen::gpr_t::x28, primitive_tensor_idxs[1] * 8, InstGen::addr_mode_t::unsigned_offset));
        kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x2, InstGen::gpr_t::x28, primitive_tensor_idxs[2] * 8, InstGen::addr_mode_t::unsigned_offset));
        kernel.add_instr(ig.base_movz(InstGen::gpr_t::x3, 0));
        kernel.add_instr(ig.base_movz(InstGen::gpr_t::x4, 0));
        kernel.add_instr(ig.base_movz(InstGen::gpr_t::x5, 0));

    }

    // call the kernel
    uint64_t primitive_idx = operation.resolve_primitive_id_idx(primitive->id);
    kernel.add_instr(ig.base_movz(InstGen::gpr_t::x7, primitive_idx * 8));
    kernel.add_instr(ig.base_add(InstGen::gpr_t::x7, InstGen::gpr_t::x27, InstGen::gpr_t::x7));
    kernel.add_instr(ig.base_blr(InstGen::gpr_t::x7));
}




void teir_compiler::compile(teir_operation const& operation) {
    // function prologue
    kernel.add_instr(ig.base_stp(InstGen::gpr_t::x29, InstGen::gpr_t::x30, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
    kernel.add_instr(ig.base_mov(InstGen::gpr_t::x29, InstGen::gpr_t::sp));
    kernel.add_instr(ig.base_stp(InstGen::gpr_t::x19, InstGen::gpr_t::x20, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
    kernel.add_instr(ig.base_stp(InstGen::gpr_t::x21, InstGen::gpr_t::x22, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
    kernel.add_instr(ig.base_stp(InstGen::gpr_t::x23, InstGen::gpr_t::x24, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
    kernel.add_instr(ig.base_stp(InstGen::gpr_t::x25, InstGen::gpr_t::x26, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
    kernel.add_instr(ig.base_stp(InstGen::gpr_t::x27, InstGen::gpr_t::x28, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));

    // pointer to tensors in x28
    kernel.add_instr(ig.base_mov(InstGen::gpr_t::x28, InstGen::gpr_t::x0));

    // compile kernels for the used primitives
    for (teir_primitive const& primitive : operation.primitives) {
        prepare_primitive(operation, primitive);
    }

    // pointer to kernel dispatch table in x27
    // 
    // we make sure to store this after we finished processing the primitives to avoid the data pointer changing
    // due to vector reallocations
    void** dispatch_table = kernel_functions.data();
    kernel.add_instr(ig.base_movz(InstGen::gpr_t::x27, ((uint64_t)dispatch_table) & 0xff));
    kernel.add_instr(ig.base_movz(InstGen::gpr_t::x27, ((uint64_t)dispatch_table) & 0xff00, 16));
    kernel.add_instr(ig.base_movz(InstGen::gpr_t::x27, ((uint64_t)dispatch_table) & 0xff0000, 32));
    kernel.add_instr(ig.base_movz(InstGen::gpr_t::x27, ((uint64_t)dispatch_table) & 0xff000000, 48));

    // generate the loop kernel around the primitives
    for (std::string const& root : operation.schedule.roots) {
        iterate(operation, root, {}, {});
    }

    // finish kernel
    kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x27, InstGen::gpr_t::x28, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
    kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x25, InstGen::gpr_t::x26, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
    kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x23, InstGen::gpr_t::x24, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
    kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x21, InstGen::gpr_t::x22, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
    kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x19, InstGen::gpr_t::x20, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
    kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x29, InstGen::gpr_t::x30, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
    kernel.add_instr(ig.base_ret());
    kernel.set_kernel();
}





teir_compiler::teir_function_t teir_compiler::get_function() const {
    return (teir_function_t)kernel.get_kernel();
}

void teir_compiler::write(const char* fp) const {
    kernel.write(fp);
}
#include <iostream>

#include "teir_compiler.h"






using mini_jit::Kernel;
using mini_jit::InstGen;
using mini_jit::Unary;
using mini_jit::Gemm;







void teir_compiler::iterate(teir_operation const& operation, std::string const& node, std::vector<teir_axis const*> axis_path, std::vector<InstGen::gpr_t> index_path) {
    // TODO: handle bit length constraints on offsets, strides and extents

    // check if id resolves to an iteration node
    teir_iter_node const* iter_node = nullptr;
    if ((iter_node = operation.schedule.resolve_iter_id(node)) != nullptr) {
        teir_axis const* axis = operation.resolve_axis_id(iter_node->axis);

        InstGen::gpr_t reg = (InstGen::gpr_t)loop_registers[axis_path.size()];
        std::string loop_start_label = iter_node->id + "_loop";
        std::string loop_end_label = iter_node->id + "_end";

        // apply offsets to tensors
        for (uint64_t i = 0; i < operation.tensors.size(); i++) {
            kernel.add_instr(ig.base_movz(InstGen::gpr_t::x0, axis->offsets[i]));
            kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x1, InstGen::gpr_t::x28, i * 8, InstGen::addr_mode_t::unsigned_offset));
            kernel.add_instr(ig.base_add(InstGen::gpr_t::x2, InstGen::gpr_t::x0, InstGen::gpr_t::x1));
            kernel.add_instr(ig.base_str(InstGen::gpr_t::x2, InstGen::gpr_t::x28, i * 8, InstGen::addr_mode_t::unsigned_offset));
        }

        // loop start
        kernel.add_instr(ig.base_movz(reg, axis->extent));
        kernel.add_label(loop_start_label);
        kernel.add_branch(ig.base_cbz(reg, loop_end_label));

        // loop body
        for (std::string const& child_id : iter_node->children) {
            std::vector<teir_axis const*> ap = axis_path;
            std::vector<InstGen::gpr_t> ip = index_path;
            ap.push_back(axis);
            ip.push_back(reg);
            iterate(operation, child_id, ap, ip);
        }

        // apply strides to tensors
        for (uint64_t i = 0; i < operation.tensors.size(); i++) {
            kernel.add_instr(ig.base_movz(InstGen::gpr_t::x0, axis->strides[i]));
            kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x1, InstGen::gpr_t::x28, i * 8, InstGen::addr_mode_t::unsigned_offset));
            kernel.add_instr(ig.base_add(InstGen::gpr_t::x2, InstGen::gpr_t::x0, InstGen::gpr_t::x1));
            kernel.add_instr(ig.base_str(InstGen::gpr_t::x2, InstGen::gpr_t::x28, i * 8, InstGen::addr_mode_t::unsigned_offset));
        }

        // loop end
        kernel.add_instr(ig.base_sub(reg, reg, 1));
        kernel.add_branch(ig.base_b(loop_start_label));
        kernel.add_label(loop_end_label);

        // remove total strides from tensors
        for (uint64_t i = 0; i < operation.tensors.size(); i++) {
            kernel.add_instr(ig.base_movz(InstGen::gpr_t::x0, axis->strides[i] * axis->extent));
            kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x1, InstGen::gpr_t::x28, i * 8, InstGen::addr_mode_t::unsigned_offset));
            kernel.add_instr(ig.base_sub(InstGen::gpr_t::x2, InstGen::gpr_t::x1, InstGen::gpr_t::x0));
            kernel.add_instr(ig.base_str(InstGen::gpr_t::x2, InstGen::gpr_t::x28, i * 8, InstGen::addr_mode_t::unsigned_offset));        
        }
        // remove offsets from tensors
        for (uint64_t i = 0; i < operation.tensors.size(); i++) {
            kernel.add_instr(ig.base_movz(InstGen::gpr_t::x0, axis->offsets[i]));
            kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x1, InstGen::gpr_t::x28, i * 8, InstGen::addr_mode_t::unsigned_offset));
            kernel.add_instr(ig.base_sub(InstGen::gpr_t::x2, InstGen::gpr_t::x1, InstGen::gpr_t::x0));
            kernel.add_instr(ig.base_str(InstGen::gpr_t::x2, InstGen::gpr_t::x28, i * 8, InstGen::addr_mode_t::unsigned_offset));
        }

        return;
    }

    // check if id resolves to an invocation node
    teir_inv_node const* inv_node = nullptr;
    if ((inv_node = operation.schedule.resolve_inv_id(node)) != nullptr) {
        invoke(operation, inv_node, axis_path, index_path);
        return;
    }

    // no resolution possible
    std::cout << "could not resolve node id" << std::endl;
}

void teir_compiler::invoke(teir_operation const& operation, teir_inv_node const* inv_node, std::vector<teir_axis const*> axis_path, std::vector<mini_jit::InstGen::gpr_t> index_path) {
    teir_primitive const* primitive = operation.resolve_primitive_id(inv_node->primitive);

    bool success = false;
    
    success = lower_zero_scalar(operation, *primitive);
    if (success) {
        return;
    }

    success = lower_zero_tile(operation, *primitive);
    if (success) {
        return;
    }

    std::cout << "missing lowering" << std::endl;
}





std::vector<uint64_t> teir_compiler::resolve_tensor_labels(teir_operation const& operation, teir_primitive const& primitive) const { 
    std::vector<uint64_t> primitive_tensor_idxs;
    for (std::string const& tensor_id : primitive.tensors) {
        uint64_t tensor_idx = operation.resolve_tensor_id_idx(tensor_id);
        primitive_tensor_idxs.push_back(tensor_idx);
    }
    return primitive_tensor_idxs;
}

bool teir_compiler::lower_zero_scalar(teir_operation const& operation, teir_primitive const& primitive) {
    if (primitive.ptype != teir_ptype_t::ptype_zero) {
        return false;
    }
    if (primitive.axes.at("M").size() != 0) {
        return false;
    }
    if (primitive.axes.at("N").size() != 0) {
        return false;
    }

    std::vector<uint64_t> tensor_idxs = resolve_tensor_labels(operation, primitive);

    kernel.add_instr(ig.base_movz(InstGen::gpr_t::w1, 0));
    kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x0, InstGen::gpr_t::x28, tensor_idxs[0] * 8, InstGen::addr_mode_t::unsigned_offset));
    kernel.add_instr(ig.base_str(InstGen::gpr_t::w1, InstGen::gpr_t::x0, 0, InstGen::addr_mode_t::unsigned_offset));

    return true;
}

bool teir_compiler::lower_zero_tile(teir_operation const& operation, teir_primitive const& primitive) {
    if (primitive.ptype != teir_ptype_t::ptype_zero) {
        return false;
    }   
    if (primitive.axes.at("M").size() != 1) {
        return false;
    }
    if (primitive.axes.at("N").size() != 1) {
        return false;
    }
    teir_axis const* axis_m = operation.resolve_axis_id(primitive.axes.at("M")[0]);
    teir_axis const* axis_n = operation.resolve_axis_id(primitive.axes.at("N")[0]);
    uint64_t primitive_idx = operation.resolve_primitive_id_idx(primitive.id);

    std::vector<uint64_t> tensor_idxs = resolve_tensor_labels(operation, primitive);

    if (axis_m->strides[tensor_idxs[0]] != 4) {
        return false;
    }

    Unary::kernel_t kernel_function = unary_cache.get_kernel(axis_m->extent, axis_n->extent, false, Unary::dtype_t::fp32, Unary::ptype_t::zero);
    kernel_functions[primitive_idx] = kernel_function;

    kernel.add_instr(ig.base_movz(InstGen::gpr_t::x0, 0));
    kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x1, InstGen::gpr_t::x28, tensor_idxs[0] * 8, InstGen::addr_mode_t::unsigned_offset));
    kernel.add_instr(ig.base_movz(InstGen::gpr_t::x2, 0));
    kernel.add_instr(ig.base_movz(InstGen::gpr_t::x3, axis_n->strides[tensor_idxs[0]] / 4));

    kernel.add_instr(ig.base_movz(InstGen::gpr_t::x7, primitive_idx * 8));
    kernel.add_instr(ig.base_add(InstGen::gpr_t::x7, InstGen::gpr_t::x27, InstGen::gpr_t::x7));
    kernel.add_instr(ig.base_blr(InstGen::gpr_t::x7));

    return true;
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

    /*
    // compile kernels for the used primitives
    for (teir_primitive const& primitive : operation.primitives) {
        prepare_primitive(operation, primitive);
    }
    */

    // initialize the kernel dispatch table
    for (teir_primitive const& primitive : operation.primitives) {
        kernel_functions.push_back(nullptr);
    }

    // pointer to kernel dispatch table in x27
    // 
    // we make sure to store this after we finished initializing the table to avoid the data pointer changing
    // due to vector reallocations
    void** dispatch_table = kernel_functions.data();
    kernel.add_instr(ig.base_movk(InstGen::gpr_t::x27, ((uint64_t)dispatch_table) & 0xff));
    kernel.add_instr(ig.base_movk(InstGen::gpr_t::x27, ((uint64_t)dispatch_table) & 0xff00, 16));
    kernel.add_instr(ig.base_movk(InstGen::gpr_t::x27, ((uint64_t)dispatch_table) & 0xff0000, 32));
    kernel.add_instr(ig.base_movk(InstGen::gpr_t::x27, ((uint64_t)dispatch_table) & 0xff000000, 48));

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
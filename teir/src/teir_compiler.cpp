#include <iostream>

#include "teir_compiler.h"






using mini_jit::Kernel;
using mini_jit::InstGen;
using mini_jit::Unary;
using mini_jit::Gemm;






teir_compiler::teir_function_t teir_compiler::get_function() const {
    return (teir_function_t)kernel.get_kernel();
}

void teir_compiler::write(const char* fp) const {
    kernel.write(fp);
}











void teir_compiler::compile(teir_operation const& operation) {
    /**
     * The resulting function has the following signature
     * void func(void**)
     * 
     * It accepts the following parameters:
     * x0: Pointer to an array of pointers to the tensor memory areas.
     * 
     * The resulting function makes use of the AArch64 registers as follows:
     * 
     * x28: Pointer to the tensor array.
     * x27: Pointer to the array of function pointers stored in `kernel_functions`.
     * x19 - x26: Loop index registers.
     * x0 - x7: Scratch registers for intermediate computations and parameter registers to the JIT kernels.
     */

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

    // initialize the kernel dispatch table
    kernel_functions = std::vector<void*>(operation.primitives.size(), nullptr);

    // pointer to kernel dispatch table in x27
    // 
    // we make sure to store this after we finished initializing the table to avoid the data pointer changing
    // due to vector reallocations
    void** dispatch_table = kernel_functions.data();
    kernel.add_instr(ig.base_movk(InstGen::gpr_t::x27, ((uint64_t)dispatch_table) & 0xffff));
    kernel.add_instr(ig.base_movk(InstGen::gpr_t::x27, (((uint64_t)dispatch_table) & 0xffff0000) >> 16, 16));
    kernel.add_instr(ig.base_movk(InstGen::gpr_t::x27, (((uint64_t)dispatch_table) & 0xffff00000000) >> 32, 32));
    kernel.add_instr(ig.base_movk(InstGen::gpr_t::x27, (((uint64_t)dispatch_table) & 0xffff000000000000) >> 48, 48));

    // generate the loop kernel around the primitives
    for (std::string const& root : operation.schedule.roots) {
        iterate(operation, root, {}, {});
    }

    // function epilogue
    kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x27, InstGen::gpr_t::x28, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
    kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x25, InstGen::gpr_t::x26, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
    kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x23, InstGen::gpr_t::x24, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
    kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x21, InstGen::gpr_t::x22, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
    kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x19, InstGen::gpr_t::x20, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
    kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x29, InstGen::gpr_t::x30, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
    kernel.add_instr(ig.base_ret());

    // append the axis strides, extends and offsets
    append_shape_data(operation);

    kernel.set_kernel();
}







void teir_compiler::iterate(teir_operation const& operation, std::string const& node, std::vector<teir_axis const*> axis_path, std::vector<InstGen::gpr_t> index_path) {
    // check if `node` resolves to an invocation node
    teir_inv_node const* inv_node = nullptr;
    if ((inv_node = operation.schedule.resolve_inv_id(node)) != nullptr) {
        invoke(operation, inv_node, axis_path, index_path);
        return;
    }

    // check if `node` resolves to an iteration node
    teir_iter_node const* iter_node = nullptr;
    if ((iter_node = operation.schedule.resolve_iter_id(node)) != nullptr) {
        teir_axis const* axis = operation.resolve_axis_id(iter_node->axis);

        InstGen::gpr_t loop_reg = (InstGen::gpr_t)loop_registers[axis_path.size()];
        std::string loop_start_label = iter_node->id + "_loop";
        std::string loop_end_label = iter_node->id + "_end";

        // apply offsets to tensors
        for (uint64_t i = 0; i < operation.tensors.size(); i++) {
            teir_tensor const& tensor = operation.tensors[i];
            if (axis->offsets[i] == 0) {
                continue;
            }
            kernel.add_labeled_instr(ig.base_ldr(InstGen::gpr_t::x0, "shape_data", get_offset_for_offset(operation, axis->id, tensor.id)));
            kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x1, InstGen::gpr_t::x28, i * 8, InstGen::addr_mode_t::unsigned_offset));
            kernel.add_instr(ig.base_add(InstGen::gpr_t::x2, InstGen::gpr_t::x0, InstGen::gpr_t::x1));
            kernel.add_instr(ig.base_str(InstGen::gpr_t::x2, InstGen::gpr_t::x28, i * 8, InstGen::addr_mode_t::unsigned_offset));
        }

        // loop start
        kernel.add_labeled_instr(ig.base_ldr(loop_reg, "shape_data", get_offset_for_extend(operation, axis->id)));
        kernel.add_label(loop_start_label);
        kernel.add_labeled_instr(ig.base_cbz(loop_reg, loop_end_label));

        // loop body
        for (std::string const& child_id : iter_node->children) {
            std::vector<teir_axis const*> ap = axis_path;
            std::vector<InstGen::gpr_t> ip = index_path;
            ap.push_back(axis);
            ip.push_back(loop_reg);
            iterate(operation, child_id, ap, ip);
        }

        // apply strides to tensors
        for (uint64_t i = 0; i < operation.tensors.size(); i++) {
            teir_tensor const& tensor = operation.tensors[i];
            if (axis->strides[i] == 0) {
                continue;
            }
            kernel.add_labeled_instr(ig.base_ldr(InstGen::gpr_t::x0, "shape_data", get_offset_for_stride(operation, axis->id, tensor.id)));
            kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x1, InstGen::gpr_t::x28, i * 8, InstGen::addr_mode_t::unsigned_offset));
            kernel.add_instr(ig.base_add(InstGen::gpr_t::x2, InstGen::gpr_t::x0, InstGen::gpr_t::x1));
            kernel.add_instr(ig.base_str(InstGen::gpr_t::x2, InstGen::gpr_t::x28, i * 8, InstGen::addr_mode_t::unsigned_offset));
        }

        // loop end
        kernel.add_instr(ig.base_sub(loop_reg, loop_reg, 1));
        kernel.add_labeled_instr(ig.base_b(loop_start_label));
        kernel.add_label(loop_end_label);

        // remove total strides from tensors
        for (uint64_t i = 0; i < operation.tensors.size(); i++) {
            teir_tensor const& tensor = operation.tensors[i];
            if (axis->strides[i] == 0) {
                continue;
            }
            kernel.add_labeled_instr(ig.base_ldr(InstGen::gpr_t::x0, "shape_data", get_offset_for_stride(operation, axis->id, tensor.id)));
            kernel.add_labeled_instr(ig.base_ldr(InstGen::gpr_t::x1, "shape_data", get_offset_for_extend(operation, axis->id)));
            kernel.add_instr(ig.base_mul(InstGen::gpr_t::x0, InstGen::gpr_t::x0, InstGen::gpr_t::x1));
            kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x1, InstGen::gpr_t::x28, i * 8, InstGen::addr_mode_t::unsigned_offset));
            kernel.add_instr(ig.base_sub(InstGen::gpr_t::x2, InstGen::gpr_t::x1, InstGen::gpr_t::x0));
            kernel.add_instr(ig.base_str(InstGen::gpr_t::x2, InstGen::gpr_t::x28, i * 8, InstGen::addr_mode_t::unsigned_offset));        
        }
        // remove offsets from tensors
        for (uint64_t i = 0; i < operation.tensors.size(); i++) {
            teir_tensor const& tensor = operation.tensors[i];
            if (axis->offsets[i] == 0) {
                continue;
            }
            kernel.add_labeled_instr(ig.base_ldr(InstGen::gpr_t::x0, "shape_data", get_offset_for_offset(operation, axis->id, tensor.id)));
            kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x1, InstGen::gpr_t::x28, i * 8, InstGen::addr_mode_t::unsigned_offset));
            kernel.add_instr(ig.base_sub(InstGen::gpr_t::x2, InstGen::gpr_t::x1, InstGen::gpr_t::x0));
            kernel.add_instr(ig.base_str(InstGen::gpr_t::x2, InstGen::gpr_t::x28, i * 8, InstGen::addr_mode_t::unsigned_offset));
        }

        return;
    }

    // no resolution possible for `node`
    std::cout << "could not resolve node id" << std::endl;
}

void teir_compiler::invoke(teir_operation const& operation, teir_inv_node const* inv_node, std::vector<teir_axis const*> axis_path, std::vector<mini_jit::InstGen::gpr_t> index_path) {
    teir_primitive const* primitive = operation.resolve_primitive_id(inv_node->primitive);

    // try each of the available lowerings on `primitive`
    bool success = false;

    success = lower_zero_scalar(operation, *primitive);
    if (success) {
        return;
    }

    success = lower_zero_tile(operation, *primitive);
    if (success) {
        return;
    }

    success = lower_identity_scalar(operation, *primitive);
    if (success) {
        return;
    }

    success = lower_identity_tile_notrans(operation, *primitive);
    if (success) {
        return;
    }

    success = lower_identity_tile_trans(operation, *primitive);
    if (success) {
        return;
    }

    success = lower_relu_scalar(operation, *primitive);
    if (success) {
        return;
    }

    success = lower_relu_tile_notrans(operation, *primitive);
    if (success) {
        return;
    }

    success = lower_relu_tile_trans(operation, *primitive);
    if (success) {
        return;
    }

    std::cout << "missing lowering" << std::endl;
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
    kernel_functions[primitive_idx] = (void*)kernel_function;

    kernel.add_instr(ig.base_movz(InstGen::gpr_t::x0, 0));
    kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x1, InstGen::gpr_t::x28, tensor_idxs[0] * 8, InstGen::addr_mode_t::unsigned_offset));
    kernel.add_instr(ig.base_movz(InstGen::gpr_t::x2, 0));
    kernel.add_labeled_instr(ig.base_ldr(InstGen::gpr_t::x3, "shape_data", get_offset_for_stride(operation, axis_n->id, primitive.tensors[0])));
    kernel.add_instr(ig.base_asr(InstGen::gpr_t::x3, InstGen::gpr_t::x3, 2));

    kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x7, InstGen::gpr_t::x27, primitive_idx * 8, InstGen::addr_mode_t::unsigned_offset));
    kernel.add_instr(ig.base_blr(InstGen::gpr_t::x7));

    return true;
}

bool teir_compiler::lower_identity_scalar(teir_operation const& operation, teir_primitive const& primitive) {
    if (primitive.ptype != teir_ptype_t::ptype_copy) {
        return false;
    }
    if (primitive.axes.at("M").size() != 0) {
        return false;
    }
    if (primitive.axes.at("N").size() != 0) {
        return false;
    }

    std::vector<uint64_t> tensor_idxs = resolve_tensor_labels(operation, primitive);

    // load tensor pointers
    kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x0, InstGen::gpr_t::x28, tensor_idxs[0] * 8, InstGen::addr_mode_t::unsigned_offset));
    kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x1, InstGen::gpr_t::x28, tensor_idxs[1] * 8, InstGen::addr_mode_t::unsigned_offset));

    // perform operation
    kernel.add_instr(ig.base_ldr(InstGen::gpr_t::w2, InstGen::gpr_t::x0, 0, InstGen::addr_mode_t::unsigned_offset));
    kernel.add_instr(ig.base_str(InstGen::gpr_t::w2, InstGen::gpr_t::x1, 0, InstGen::addr_mode_t::unsigned_offset));

    return true;
}

bool teir_compiler::lower_identity_tile_notrans(teir_operation const& operation, teir_primitive const& primitive) {
    if (primitive.ptype != teir_ptype_t::ptype_copy) {
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
    if (axis_m->strides[tensor_idxs[1]] != 4) {
        return false;
    }

    Unary::kernel_t kernel_function = unary_cache.get_kernel(axis_m->extent, axis_n->extent, false, Unary::dtype_t::fp32, Unary::ptype_t::identity);
    kernel_functions[primitive_idx] = (void*)kernel_function;

    kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x0, InstGen::gpr_t::x28, tensor_idxs[0] * 8, InstGen::addr_mode_t::unsigned_offset));
    kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x1, InstGen::gpr_t::x28, tensor_idxs[1] * 8, InstGen::addr_mode_t::unsigned_offset));
    kernel.add_labeled_instr(ig.base_ldr(InstGen::gpr_t::x2, "shape_data", get_offset_for_stride(operation, axis_n->id, primitive.tensors[0])));
    kernel.add_instr(ig.base_asr(InstGen::gpr_t::x2, InstGen::gpr_t::x2, 2));
    kernel.add_labeled_instr(ig.base_ldr(InstGen::gpr_t::x3, "shape_data", get_offset_for_stride(operation, axis_n->id, primitive.tensors[1])));
    kernel.add_instr(ig.base_asr(InstGen::gpr_t::x3, InstGen::gpr_t::x3, 2));

    kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x7, InstGen::gpr_t::x27, primitive_idx * 8, InstGen::addr_mode_t::unsigned_offset));
    kernel.add_instr(ig.base_blr(InstGen::gpr_t::x7));

    return true;
}

bool teir_compiler::lower_identity_tile_trans(teir_operation const& operation, teir_primitive const& primitive) {
    if (primitive.ptype != teir_ptype_t::ptype_copy) {
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
    if (axis_n->strides[tensor_idxs[1]] != 4) {
        return false;
    }

    Unary::kernel_t kernel_function = unary_cache.get_kernel(axis_m->extent, axis_n->extent, true, Unary::dtype_t::fp32, Unary::ptype_t::identity);
    kernel_functions[primitive_idx] = (void*)kernel_function;

    kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x0, InstGen::gpr_t::x28, tensor_idxs[0] * 8, InstGen::addr_mode_t::unsigned_offset));
    kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x1, InstGen::gpr_t::x28, tensor_idxs[1] * 8, InstGen::addr_mode_t::unsigned_offset));
    kernel.add_labeled_instr(ig.base_ldr(InstGen::gpr_t::x2, "shape_data", get_offset_for_stride(operation, axis_n->id, primitive.tensors[0])));
    kernel.add_instr(ig.base_asr(InstGen::gpr_t::x2, InstGen::gpr_t::x2, 2));
    kernel.add_labeled_instr(ig.base_ldr(InstGen::gpr_t::x3, "shape_data", get_offset_for_stride(operation, axis_m->id, primitive.tensors[1])));
    kernel.add_instr(ig.base_asr(InstGen::gpr_t::x3, InstGen::gpr_t::x3, 2));


    kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x7, InstGen::gpr_t::x27, primitive_idx * 8, InstGen::addr_mode_t::unsigned_offset));
    kernel.add_instr(ig.base_blr(InstGen::gpr_t::x7));

    return true;
}

bool teir_compiler::lower_relu_scalar(teir_operation const& operation, teir_primitive const& primitive) {
    if (primitive.ptype != teir_ptype_t::ptype_relu) {
        return false;
    }
    if (primitive.axes.at("M").size() != 0) {
        return false;
    }
    if (primitive.axes.at("N").size() != 0) {
        return false;
    }

    std::vector<uint64_t> tensor_idxs = resolve_tensor_labels(operation, primitive);

    // load tensor pointers
    kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x0, InstGen::gpr_t::x28, tensor_idxs[0] * 8, InstGen::addr_mode_t::unsigned_offset));
    kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x1, InstGen::gpr_t::x28, tensor_idxs[1] * 8, InstGen::addr_mode_t::unsigned_offset));

    // perform operation
    kernel.add_instr(ig.base_smstart());
    kernel.add_instr(ig.ssve_ptrue(InstGen::pr_t::p0, InstGen::sve_size_t::s, InstGen::pr_pattern_t::vl1));
    kernel.add_instr(ig.sve_ld1w(InstGen::sve_zr_t::z0, InstGen::pr_t::p0, InstGen::gpr_t::x0, InstGen::gpr_t::xzr));
    kernel.add_instr(ig.sve_fmax(InstGen::sve_zr_t::z0, InstGen::sve_size_t::s, InstGen::pr_t::p0, 0));
    kernel.add_instr(ig.sve_st1w(InstGen::sve_zr_t::z0, InstGen::sve_size_t::s, InstGen::pr_t::p0, InstGen::gpr_t::x1, InstGen::gpr_t::xzr));
    kernel.add_instr(ig.base_smstop());

    return true;
}

bool teir_compiler::lower_relu_tile_notrans(teir_operation const& operation, teir_primitive const& primitive) {
    if (primitive.ptype != teir_ptype_t::ptype_relu) {
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
    if (axis_m->strides[tensor_idxs[1]] != 4) {
        return false;
    }

    Unary::kernel_t kernel_function = unary_cache.get_kernel(axis_m->extent, axis_n->extent, false, Unary::dtype_t::fp32, Unary::ptype_t::relu);
    kernel_functions[primitive_idx] = (void*)kernel_function;

    kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x0, InstGen::gpr_t::x28, tensor_idxs[0] * 8, InstGen::addr_mode_t::unsigned_offset));
    kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x1, InstGen::gpr_t::x28, tensor_idxs[1] * 8, InstGen::addr_mode_t::unsigned_offset));
    kernel.add_labeled_instr(ig.base_ldr(InstGen::gpr_t::x2, "shape_data", get_offset_for_stride(operation, axis_n->id, primitive.tensors[0])));
    kernel.add_instr(ig.base_asr(InstGen::gpr_t::x2, InstGen::gpr_t::x2, 2));
    kernel.add_labeled_instr(ig.base_ldr(InstGen::gpr_t::x3, "shape_data", get_offset_for_stride(operation, axis_n->id, primitive.tensors[1])));
    kernel.add_instr(ig.base_asr(InstGen::gpr_t::x3, InstGen::gpr_t::x3, 2));

    kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x7, InstGen::gpr_t::x27, primitive_idx * 8, InstGen::addr_mode_t::unsigned_offset));
    kernel.add_instr(ig.base_blr(InstGen::gpr_t::x7));

    return true;
}

bool teir_compiler::lower_relu_tile_trans(teir_operation const& operation, teir_primitive const& primitive) {
    if (primitive.ptype != teir_ptype_t::ptype_relu) {
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
    if (axis_n->strides[tensor_idxs[1]] != 4) {
        return false;
    }

    Unary::kernel_t kernel_function = unary_cache.get_kernel(axis_m->extent, axis_n->extent, true, Unary::dtype_t::fp32, Unary::ptype_t::relu);
    kernel_functions[primitive_idx] = (void*)kernel_function;

    kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x0, InstGen::gpr_t::x28, tensor_idxs[0] * 8, InstGen::addr_mode_t::unsigned_offset));
    kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x1, InstGen::gpr_t::x28, tensor_idxs[1] * 8, InstGen::addr_mode_t::unsigned_offset));
    kernel.add_labeled_instr(ig.base_ldr(InstGen::gpr_t::x2, "shape_data", get_offset_for_stride(operation, axis_n->id, primitive.tensors[0])));
    kernel.add_instr(ig.base_asr(InstGen::gpr_t::x2, InstGen::gpr_t::x2, 2));
    kernel.add_labeled_instr(ig.base_ldr(InstGen::gpr_t::x3, "shape_data", get_offset_for_stride(operation, axis_m->id, primitive.tensors[1])));
    kernel.add_instr(ig.base_asr(InstGen::gpr_t::x3, InstGen::gpr_t::x3, 2));

    kernel.add_instr(ig.base_ldr(InstGen::gpr_t::x7, InstGen::gpr_t::x27, primitive_idx * 8, InstGen::addr_mode_t::unsigned_offset));
    kernel.add_instr(ig.base_blr(InstGen::gpr_t::x7));

    return true;
}











std::vector<uint64_t> teir_compiler::resolve_tensor_labels(teir_operation const& operation, teir_primitive const& primitive) const { 
    std::vector<uint64_t> primitive_tensor_idxs;
    for (std::string const& tensor_id : primitive.tensors) {
        uint64_t tensor_idx = operation.resolve_tensor_id_idx(tensor_id);
        primitive_tensor_idxs.push_back(tensor_idx);
    }
    return primitive_tensor_idxs;
}




















void teir_compiler::append_shape_data(teir_operation const& operation) {
    // append the extends of the axes
    kernel.add_label("shape_data");
    kernel.add_label("axis_extends");
    for (teir_axis const& axis : operation.axes) {
        kernel.add_data(axis.extent);
    }

    // append the strides of the axes
    kernel.add_label("axis_strides");
    for (teir_axis const& axis : operation.axes) {
        for (uint64_t stride : axis.strides) {
            kernel.add_data(stride);
        }
    }

    // append the offsets of the axes
    kernel.add_label("axis_offsets");
    for (teir_axis const& axis : operation.axes) {
        for (uint64_t offset : axis.offsets) {
            kernel.add_data(offset);
        }
    }
}

int32_t teir_compiler::get_offset_for_extend(teir_operation const& operation, std::string const& axis_id) {
    uint64_t axis_idx = operation.resolve_axis_id_idx(axis_id);
    return axis_idx * sizeof(uint64_t);
}

int32_t teir_compiler::get_offset_for_stride(teir_operation const& operation, std::string const& axis_id, std::string const& tensor_id) {
    uint64_t axis_idx = operation.resolve_axis_id_idx(axis_id);
    uint64_t tensor_idx = operation.resolve_tensor_id_idx(tensor_id);

    uint64_t base = operation.axes.size() * sizeof(uint64_t);
    return base + axis_idx * operation.tensors.size() * sizeof(uint64_t) + tensor_idx * sizeof(uint64_t);
}

int32_t teir_compiler::get_offset_for_offset(teir_operation const& operation, std::string const& axis_id, std::string const& tensor_id) {
    uint64_t axis_idx = operation.resolve_axis_id_idx(axis_id);
    uint64_t tensor_idx = operation.resolve_tensor_id_idx(tensor_id);

    uint64_t base = operation.axes.size() * sizeof(uint64_t) + operation.axes.size() * operation.tensors.size() * sizeof(uint64_t);
    return base + axis_idx * operation.tensors.size() * sizeof(uint64_t) + tensor_idx * sizeof(uint64_t);
}

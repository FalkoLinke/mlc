#include <iostream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "teir_compiler.h"






using mini_jit::Kernel;
using mini_jit::InstGen;
using mini_jit::Unary;
using mini_jit::Gemm;






teir_compiler::teir_function_t teir_compiler::get_function() const {
    return (teir_function_t)main_kernel.get_kernel();
}

void teir_compiler::write(const char* fp) const {
    main_kernel.write(fp);
}











void teir_compiler::compile(teir_operation const& operation) {
    /**
     * The resulting function has the following signature
     * void func(void**)
     * 
     * It accepts the following parameters:
     * x0: Pointer to an array of pointers to the tensor memory areas.
     * 
     * 
     * The resulting function makes use of the AArch64 registers as follows:
     * 
     * x28: Pointer to the tensor array.
     * x27: Pointer to the array of function pointers stored in `kernel_functions`.
     * x19 - x26: Loop index registers.
     * x0 - x7: Scratch registers for intermediate computations and parameter registers to the JIT kernels.
     * 
     * 
     * The compiler appends the extends, strides, offsets and other needed data
     * in the executable memory area of `kernel` after the function.
     * Use LDR (label) instructions to load such data into registers.
     */

    // generate the tile kernels and initialize the kernel dispatch table
    //
    // the table is not resized afterwards, since the generated code holds a pointer to its data
    plans.clear();
    kernel_functions = std::vector<void*>(operation.primitives.size(), nullptr);
    for (uint64_t i = 0; i < operation.primitives.size(); i++) {
        teir_tile_plan plan = teir_plan_tile(operation, operation.primitives[i]);
        if (plan.kind == teir_tile_plan::unary) {
            kernel_functions[i] = (void*)unary_cache.get_kernel(plan.m, plan.n, plan.trans_b, Unary::dtype_t::fp32, plan.unary_ptype);
        } else if (plan.kind == teir_tile_plan::gemm) {
            kernel_functions[i] = (void*)gemm_cache.get_kernel(plan.m, plan.n, plan.k, plan.trans_a, plan.trans_b, plan.trans_c, Gemm::dtype_t::fp32);
        }
        if (kernel_functions[i] == nullptr) {
            plan.kind = teir_tile_plan::none;
        }
        plans.push_back(plan);
    }

    kernel = &main_kernel;
    append_prologue();

    // pointer to tensors in x28
    kernel->add_instr(ig.base_mov(InstGen::gpr_t::x28, InstGen::gpr_t::x0));

    // pointer to kernel dispatch table in x27
    append_mov_imm64(InstGen::gpr_t::x27, (uint64_t)kernel_functions.data());

    // generate the loop kernel around the primitives
    for (std::string const& root : operation.schedule.roots) {
        iterate(operation, root, {}, {});
    }

    append_epilogue();

    // append the axis strides, extends and offsets
    append_shape_data(operation);

    kernel->set_kernel();
}




void teir_compiler::append_prologue() {
    kernel->add_instr(ig.base_stp(InstGen::gpr_t::x29, InstGen::gpr_t::x30, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
    kernel->add_instr(ig.base_mov(InstGen::gpr_t::x29, InstGen::gpr_t::sp));
    kernel->add_instr(ig.base_stp(InstGen::gpr_t::x19, InstGen::gpr_t::x20, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
    kernel->add_instr(ig.base_stp(InstGen::gpr_t::x21, InstGen::gpr_t::x22, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
    kernel->add_instr(ig.base_stp(InstGen::gpr_t::x23, InstGen::gpr_t::x24, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
    kernel->add_instr(ig.base_stp(InstGen::gpr_t::x25, InstGen::gpr_t::x26, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
    kernel->add_instr(ig.base_stp(InstGen::gpr_t::x27, InstGen::gpr_t::x28, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
    // lowered primitives may switch to streaming mode, which zeroes the callee-saved d8 - d15
    kernel->add_instr(ig.neon_stp(InstGen::simd_fp_t::v8, InstGen::simd_fp_t::v9, InstGen::simd_sz_t::simd_d, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
    kernel->add_instr(ig.neon_stp(InstGen::simd_fp_t::v10, InstGen::simd_fp_t::v11, InstGen::simd_sz_t::simd_d, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
    kernel->add_instr(ig.neon_stp(InstGen::simd_fp_t::v12, InstGen::simd_fp_t::v13, InstGen::simd_sz_t::simd_d, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
    kernel->add_instr(ig.neon_stp(InstGen::simd_fp_t::v14, InstGen::simd_fp_t::v15, InstGen::simd_sz_t::simd_d, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
}

void teir_compiler::append_epilogue() {
    kernel->add_instr(ig.neon_ldp(InstGen::simd_fp_t::v14, InstGen::simd_fp_t::v15, InstGen::simd_sz_t::simd_d, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
    kernel->add_instr(ig.neon_ldp(InstGen::simd_fp_t::v12, InstGen::simd_fp_t::v13, InstGen::simd_sz_t::simd_d, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
    kernel->add_instr(ig.neon_ldp(InstGen::simd_fp_t::v10, InstGen::simd_fp_t::v11, InstGen::simd_sz_t::simd_d, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
    kernel->add_instr(ig.neon_ldp(InstGen::simd_fp_t::v8, InstGen::simd_fp_t::v9, InstGen::simd_sz_t::simd_d, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
    kernel->add_instr(ig.base_ldp(InstGen::gpr_t::x27, InstGen::gpr_t::x28, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
    kernel->add_instr(ig.base_ldp(InstGen::gpr_t::x25, InstGen::gpr_t::x26, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
    kernel->add_instr(ig.base_ldp(InstGen::gpr_t::x23, InstGen::gpr_t::x24, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
    kernel->add_instr(ig.base_ldp(InstGen::gpr_t::x21, InstGen::gpr_t::x22, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
    kernel->add_instr(ig.base_ldp(InstGen::gpr_t::x19, InstGen::gpr_t::x20, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
    kernel->add_instr(ig.base_ldp(InstGen::gpr_t::x29, InstGen::gpr_t::x30, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
    kernel->add_instr(ig.base_ret());
}

void teir_compiler::append_mov_imm64(InstGen::gpr_t reg, uint64_t value) {
    kernel->add_instr(ig.base_movz(reg, value & 0xffff));
    for (uint32_t shift = 16; shift < 64; shift += 16) {
        uint32_t part = (value >> shift) & 0xffff;
        if (part != 0) {
            kernel->add_instr(ig.base_movk(reg, part, shift));
        }
    }
}







void teir_compiler::iterate(teir_operation const& operation, std::string const& node, std::vector<teir_axis const*> axis_path, std::vector<InstGen::gpr_t> index_path) {
    // check if `node` resolves to an invocation node
    teir_inv_node const* inv_node = nullptr;
    if ((inv_node = operation.schedule.resolve_inv_id(node)) != nullptr) {
        std::string skip_label = inv_node->id + "_skip";
        append_branch_if_not_guards(operation, axis_path, index_path, inv_node->guards, skip_label);
        invoke(operation, inv_node, axis_path, index_path);
        kernel->add_label(skip_label);
        return;
    }

    // check if `node` resolves to an iteration node
    teir_iter_node const* iter_node = nullptr;
    if ((iter_node = operation.schedule.resolve_iter_id(node)) != nullptr) {
        if (iter_node->policy == teir_policy_t::policy_parallel) {
            iterate_parallel(operation, iter_node, axis_path, index_path);
            return;
        }

        teir_axis const* axis = operation.resolve_axis_id(iter_node->axis);
        if (axis == nullptr) {
            throw std::runtime_error("unresolved axis id: " + iter_node->axis);
        }
        if (axis_path.size() >= loop_registers.size()) {
            throw std::runtime_error("loop nest too deep at node " + iter_node->id);
        }

        InstGen::gpr_t loop_reg = (InstGen::gpr_t)loop_registers[axis_path.size()];
        std::string loop_start_label = iter_node->id + "_loop";
        std::string loop_end_label = iter_node->id + "_end";
        std::string skip_label = iter_node->id + "_skip";

        append_branch_if_not_guards(operation, axis_path, index_path, iter_node->guards, skip_label);

        // apply offsets to tensors
        for (uint64_t i = 0; i < operation.tensors.size(); i++) {
            teir_tensor const& tensor = operation.tensors[i];
            if (axis->offsets[i] == 0) {
                continue;
            }
            kernel->add_labeled_instr(ig.base_ldr(InstGen::gpr_t::x0, shape_data_label, get_offset_for_offset(operation, axis->id, tensor.id)));
            kernel->add_instr(ig.base_ldr(InstGen::gpr_t::x1, InstGen::gpr_t::x28, i * 8, InstGen::addr_mode_t::unsigned_offset));
            kernel->add_instr(ig.base_add(InstGen::gpr_t::x2, InstGen::gpr_t::x0, InstGen::gpr_t::x1));
            kernel->add_instr(ig.base_str(InstGen::gpr_t::x2, InstGen::gpr_t::x28, i * 8, InstGen::addr_mode_t::unsigned_offset));
        }

        // loop start
        kernel->add_labeled_instr(ig.base_ldr(loop_reg, shape_data_label, get_offset_for_extend(operation, axis->id)));
        kernel->add_label(loop_start_label);
        kernel->add_labeled_instr(ig.base_cbz(loop_reg, loop_end_label));

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
            kernel->add_labeled_instr(ig.base_ldr(InstGen::gpr_t::x0, shape_data_label, get_offset_for_stride(operation, axis->id, tensor.id)));
            kernel->add_instr(ig.base_ldr(InstGen::gpr_t::x1, InstGen::gpr_t::x28, i * 8, InstGen::addr_mode_t::unsigned_offset));
            kernel->add_instr(ig.base_add(InstGen::gpr_t::x2, InstGen::gpr_t::x0, InstGen::gpr_t::x1));
            kernel->add_instr(ig.base_str(InstGen::gpr_t::x2, InstGen::gpr_t::x28, i * 8, InstGen::addr_mode_t::unsigned_offset));
        }

        // loop end
        kernel->add_instr(ig.base_sub(loop_reg, loop_reg, 1));
        kernel->add_labeled_instr(ig.base_b(loop_start_label));
        kernel->add_label(loop_end_label);

        // remove total strides from tensors
        for (uint64_t i = 0; i < operation.tensors.size(); i++) {
            teir_tensor const& tensor = operation.tensors[i];
            if (axis->strides[i] == 0) {
                continue;
            }
            kernel->add_labeled_instr(ig.base_ldr(InstGen::gpr_t::x0, shape_data_label, get_offset_for_stride(operation, axis->id, tensor.id)));
            kernel->add_labeled_instr(ig.base_ldr(InstGen::gpr_t::x1, shape_data_label, get_offset_for_extend(operation, axis->id)));
            kernel->add_instr(ig.base_mul(InstGen::gpr_t::x0, InstGen::gpr_t::x0, InstGen::gpr_t::x1));
            kernel->add_instr(ig.base_ldr(InstGen::gpr_t::x1, InstGen::gpr_t::x28, i * 8, InstGen::addr_mode_t::unsigned_offset));
            kernel->add_instr(ig.base_sub(InstGen::gpr_t::x2, InstGen::gpr_t::x1, InstGen::gpr_t::x0));
            kernel->add_instr(ig.base_str(InstGen::gpr_t::x2, InstGen::gpr_t::x28, i * 8, InstGen::addr_mode_t::unsigned_offset));        
        }
        // remove offsets from tensors
        for (uint64_t i = 0; i < operation.tensors.size(); i++) {
            teir_tensor const& tensor = operation.tensors[i];
            if (axis->offsets[i] == 0) {
                continue;
            }
            kernel->add_labeled_instr(ig.base_ldr(InstGen::gpr_t::x0, shape_data_label, get_offset_for_offset(operation, axis->id, tensor.id)));
            kernel->add_instr(ig.base_ldr(InstGen::gpr_t::x1, InstGen::gpr_t::x28, i * 8, InstGen::addr_mode_t::unsigned_offset));
            kernel->add_instr(ig.base_sub(InstGen::gpr_t::x2, InstGen::gpr_t::x1, InstGen::gpr_t::x0));
            kernel->add_instr(ig.base_str(InstGen::gpr_t::x2, InstGen::gpr_t::x28, i * 8, InstGen::addr_mode_t::unsigned_offset));
        }

        kernel->add_label(skip_label);

        return;
    }

    // no resolution possible for `node`
    throw std::runtime_error("could not resolve node id: " + node);
}




void teir_compiler::iterate_parallel(teir_operation const& operation, teir_iter_node const* iter_node, std::vector<teir_axis const*> const& axis_path, std::vector<InstGen::gpr_t> const& index_path) {
    if (operation.tensors.size() > max_parallel_tensors) {
        throw std::runtime_error("too many tensors for a parallel iteration node");
    }

    // collapse directly nested parallel iteration nodes into a single iteration space
    std::vector<teir_iter_node const*> nodes = {iter_node};
    while (nodes.back()->children.size() == 1) {
        teir_iter_node const* child = operation.schedule.resolve_iter_id(nodes.back()->children[0]);
        if (child == nullptr || child->policy != teir_policy_t::policy_parallel || !child->guards.empty()) {
            break;
        }
        nodes.push_back(child);
    }
    if (axis_path.size() + nodes.size() > loop_registers.size()) {
        throw std::runtime_error("loop nest too deep at node " + iter_node->id);
    }

    std::unique_ptr<parallel_region> region = std::make_unique<parallel_region>();
    region->depth = axis_path.size();
    region->num_tensors = operation.tensors.size();

    std::vector<teir_axis const*> body_axis_path = axis_path;
    std::vector<InstGen::gpr_t> body_index_path = index_path;
    for (teir_iter_node const* node : nodes) {
        teir_axis const* axis = operation.resolve_axis_id(node->axis);
        if (axis == nullptr) {
            throw std::runtime_error("unresolved axis id: " + node->axis);
        }
        region->extents.push_back(axis->extent);
        for (uint64_t t = 0; t < operation.tensors.size(); t++) {
            region->strides.push_back(axis->strides[t]);
            region->offsets.push_back(axis->offsets[t]);
        }
        body_axis_path.push_back(axis);
        body_index_path.push_back(loop_registers[body_axis_path.size() - 1]);
    }

    // compile the loop body into a separate function
    //   x0: tensor pointers of the iteration (applied offsets and strides of the region's axes)
    //   x1: values of the loop index registers
    std::unique_ptr<Kernel> body = std::make_unique<Kernel>();
    Kernel* parent = kernel;
    kernel = body.get();

    append_prologue();
    kernel->add_instr(ig.base_mov(InstGen::gpr_t::x28, InstGen::gpr_t::x0));
    for (uint64_t r = 0; r < body_index_path.size(); r++) {
        kernel->add_instr(ig.base_ldr(body_index_path[r], InstGen::gpr_t::x1, r * 8, InstGen::addr_mode_t::unsigned_offset));
    }
    append_mov_imm64(InstGen::gpr_t::x27, (uint64_t)kernel_functions.data());

    for (std::string const& child_id : nodes.back()->children) {
        iterate(operation, child_id, body_axis_path, body_index_path);
    }

    append_epilogue();
    append_shape_data(operation);
    kernel->set_kernel();

    kernel = parent;
    region->body = (parallel_region::body_t)body->get_kernel();

    // call parallel_for(region, tensors, loop_indices) from the enclosing code
    std::string skip_label = iter_node->id + "_skip";
    append_branch_if_not_guards(operation, axis_path, index_path, iter_node->guards, skip_label);

    append_mov_imm64(InstGen::gpr_t::x0, (uint64_t)region.get());
    kernel->add_instr(ig.base_mov(InstGen::gpr_t::x1, InstGen::gpr_t::x28));

    // the loop index registers are passed on the stack, x19 at the lowest address
    kernel->add_instr(ig.base_stp(InstGen::gpr_t::x25, InstGen::gpr_t::x26, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
    kernel->add_instr(ig.base_stp(InstGen::gpr_t::x23, InstGen::gpr_t::x24, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
    kernel->add_instr(ig.base_stp(InstGen::gpr_t::x21, InstGen::gpr_t::x22, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
    kernel->add_instr(ig.base_stp(InstGen::gpr_t::x19, InstGen::gpr_t::x20, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
    kernel->add_instr(ig.base_mov(InstGen::gpr_t::x2, InstGen::gpr_t::sp));

    append_mov_imm64(InstGen::gpr_t::x7, (uint64_t)&teir_compiler::parallel_for);
    kernel->add_instr(ig.base_blr(InstGen::gpr_t::x7));

    kernel->add_instr(ig.base_ldp(InstGen::gpr_t::x19, InstGen::gpr_t::x20, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
    kernel->add_instr(ig.base_ldp(InstGen::gpr_t::x21, InstGen::gpr_t::x22, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
    kernel->add_instr(ig.base_ldp(InstGen::gpr_t::x23, InstGen::gpr_t::x24, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
    kernel->add_instr(ig.base_ldp(InstGen::gpr_t::x25, InstGen::gpr_t::x26, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));

    kernel->add_label(skip_label);

    regions.push_back(std::move(region));
    body_kernels.push_back(std::move(body));
}


static bool in_parallel_region() {
#ifdef _OPENMP
    return omp_in_parallel();
#else
    return false;
#endif
}

void teir_compiler::parallel_for(parallel_region const* region, void** tensors, uint64_t const* loop_indices) {
    uint64_t const num_axes = region->extents.size();
    uint64_t const num_tensors = region->num_tensors;
    uint64_t const depth = region->depth;

    int64_t total = 1;
    for (uint64_t extent : region->extents) {
        total *= extent;
    }

    // dynamic scheduling balances the work between the P- and E-cores
    #pragma omp parallel for schedule(dynamic) if(!in_parallel_region())
    for (int64_t it = 0; it < total; it++) {
        void* iter_tensors[max_parallel_tensors];
        uint64_t iter_indices[8];

        for (uint64_t t = 0; t < num_tensors; t++) {
            iter_tensors[t] = tensors[t];
        }
        for (uint64_t d = 0; d < depth; d++) {
            iter_indices[d] = loop_indices[d];
        }

        // split the flat iteration index into the indices of the collapsed axes
        uint64_t rest = it;
        for (int64_t a = num_axes - 1; a >= 0; a--) {
            uint64_t extent = region->extents[a];
            uint64_t i = rest % extent;
            rest /= extent;

            // the loop index registers of the compiled loops count down from the extent
            iter_indices[depth + a] = extent - i;
            for (uint64_t t = 0; t < num_tensors; t++) {
                int64_t shift = region->offsets[a * num_tensors + t] + (int64_t)i * region->strides[a * num_tensors + t];
                iter_tensors[t] = (uint8_t*)iter_tensors[t] + shift;
            }
        }

        region->body(iter_tensors, iter_indices);
    }
}




void teir_compiler::invoke(teir_operation const& operation, teir_inv_node const* inv_node, std::vector<teir_axis const*> axis_path, std::vector<mini_jit::InstGen::gpr_t> index_path) {
    teir_primitive const* primitive = operation.resolve_primitive_id(inv_node->primitive);
    if (primitive == nullptr) {
        throw std::runtime_error("unresolved primitive id: " + inv_node->primitive);
    }

    // try each of the available lowerings on `primitive`
    if (lower_tile(operation, *primitive)) {
        return;
    }
    if (lower_zero_scalar(operation, *primitive)) {
        return;
    }
    if (lower_identity_scalar(operation, *primitive)) {
        return;
    }
    if (lower_relu_scalar(operation, *primitive)) {
        return;
    }

    throw std::runtime_error("missing lowering for primitive " + primitive->id);
}




bool teir_compiler::lower_tile(teir_operation const& operation, teir_primitive const& primitive) {
    uint64_t primitive_idx = operation.resolve_primitive_id_idx(primitive.id);
    teir_tile_plan const& plan = plans[primitive_idx];
    if (plan.kind == teir_tile_plan::none) {
        return false;
    }

    // kernel arguments: tensor pointers in x0, x1(, x2), followed by the leading dimensions
    uint32_t reg = 0;
    for (uint64_t tensor_idx : plan.tensors) {
        InstGen::gpr_t arg = (InstGen::gpr_t)(InstGen::gpr_t::x0 + reg);
        if (tensor_idx == teir_tile_plan::no_tensor) {
            kernel->add_instr(ig.base_movz(arg, 0));
        } else {
            kernel->add_instr(ig.base_ldr(arg, InstGen::gpr_t::x28, tensor_idx * 8, InstGen::addr_mode_t::unsigned_offset));
        }
        reg++;
    }
    for (int64_t ld : plan.lds) {
        append_mov_imm64((InstGen::gpr_t)(InstGen::gpr_t::x0 + reg), (uint64_t)ld);
        reg++;
    }

    kernel->add_instr(ig.base_ldr(InstGen::gpr_t::x7, InstGen::gpr_t::x27, primitive_idx * 8, InstGen::addr_mode_t::unsigned_offset));
    kernel->add_instr(ig.base_blr(InstGen::gpr_t::x7));

    return true;
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

    kernel->add_instr(ig.base_movz(InstGen::gpr_t::w1, 0));
    kernel->add_instr(ig.base_ldr(InstGen::gpr_t::x0, InstGen::gpr_t::x28, tensor_idxs[0] * 8, InstGen::addr_mode_t::unsigned_offset));
    kernel->add_instr(ig.base_str(InstGen::gpr_t::w1, InstGen::gpr_t::x0, 0, InstGen::addr_mode_t::unsigned_offset));

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
    kernel->add_instr(ig.base_ldr(InstGen::gpr_t::x0, InstGen::gpr_t::x28, tensor_idxs[0] * 8, InstGen::addr_mode_t::unsigned_offset));
    kernel->add_instr(ig.base_ldr(InstGen::gpr_t::x1, InstGen::gpr_t::x28, tensor_idxs[1] * 8, InstGen::addr_mode_t::unsigned_offset));

    // perform operation
    kernel->add_instr(ig.base_ldr(InstGen::gpr_t::w2, InstGen::gpr_t::x0, 0, InstGen::addr_mode_t::unsigned_offset));
    kernel->add_instr(ig.base_str(InstGen::gpr_t::w2, InstGen::gpr_t::x1, 0, InstGen::addr_mode_t::unsigned_offset));

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
    kernel->add_instr(ig.base_ldr(InstGen::gpr_t::x0, InstGen::gpr_t::x28, tensor_idxs[0] * 8, InstGen::addr_mode_t::unsigned_offset));
    kernel->add_instr(ig.base_ldr(InstGen::gpr_t::x1, InstGen::gpr_t::x28, tensor_idxs[1] * 8, InstGen::addr_mode_t::unsigned_offset));

    // perform operation
    kernel->add_instr(ig.base_smstart());
    kernel->add_instr(ig.ssve_ptrue(InstGen::pr_t::p0, InstGen::sve_size_t::s, InstGen::pr_pattern_t::vl1));
    kernel->add_instr(ig.sve_ld1w(InstGen::sve_zr_t::z0, InstGen::pr_t::p0, InstGen::gpr_t::x0, InstGen::gpr_t::xzr));
    kernel->add_instr(ig.sve_fmax(InstGen::sve_zr_t::z0, InstGen::sve_size_t::s, InstGen::pr_t::p0, 0));
    kernel->add_instr(ig.sve_st1w(InstGen::sve_zr_t::z0, InstGen::sve_size_t::s, InstGen::pr_t::p0, InstGen::gpr_t::x1, InstGen::gpr_t::xzr));
    kernel->add_instr(ig.base_smstop());

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
    kernel->add_label(shape_data_label);
    for (teir_axis const& axis : operation.axes) {
        kernel->add_data(axis.extent);
    }

    // append the strides of the axes
    for (teir_axis const& axis : operation.axes) {
        for (uint64_t stride : axis.strides) {
            kernel->add_data(stride);
        }
    }

    // append the offsets of the axes
    for (teir_axis const& axis : operation.axes) {
        for (uint64_t offset : axis.offsets) {
            kernel->add_data(offset);
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











void teir_compiler::append_branch_if_not_guard(teir_operation const& operation, teir_axis const& axis, InstGen::gpr_t axis_reg, teir_guard const& guard, std::string label) {
    if (axis.id != guard.axis_id) {
        return;
    }

    if (guard.kind == teir_guard_kind::first) {

        // the first iteration index is axis.extend
        kernel->add_labeled_instr(ig.base_ldr(InstGen::gpr_t::x0, shape_data_label, get_offset_for_extend(operation, guard.axis_id)));
        kernel->add_instr(ig.base_sub(InstGen::gpr_t::x0, InstGen::gpr_t::x0, axis_reg, InstGen::shift_kind_t::lsl, 0, 1));
        kernel->add_labeled_instr(ig.base_b_cond(label, InstGen::br_cond_t::ne));

    } else if (guard.kind == teir_guard_kind::last) {

        // the last iteration index is 1
        kernel->add_instr(ig.base_sub(InstGen::gpr_t::x0, axis_reg, 1, 1));
        kernel->add_labeled_instr(ig.base_b_cond(label, InstGen::br_cond_t::ne));
        
    } else {

    }
}

void teir_compiler::append_branch_if_not_guard(teir_operation const& operation, std::vector<teir_axis const*> const& axis_path, std::vector<InstGen::gpr_t> const& index_path, teir_guard const& guard, std::string const& label) {
    std::vector<uint64_t> path_indices;
    for (uint64_t i = 0; i < axis_path.size(); i++) {
        if (axis_path[i]->id == guard.axis_id) {
            path_indices.push_back(i);
        }
    }

    for (uint64_t i : path_indices) {
        teir_axis const& axis = *axis_path[i];
        InstGen::gpr_t axis_reg = index_path[i];
        append_branch_if_not_guard(operation, axis, axis_reg, guard, label);
    }
}

void teir_compiler::append_branch_if_not_guards(teir_operation const& operation, std::vector<teir_axis const*> const& axis_path, std::vector<InstGen::gpr_t> const& index_path, std::vector<teir_guard> const& guards, std::string const& label) {
    for (teir_guard const& guard : guards) {
        append_branch_if_not_guard(operation, axis_path, index_path, guard, label);
    }
}
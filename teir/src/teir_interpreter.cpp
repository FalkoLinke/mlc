#include "teir_interpreter.h"
#include "teir_common.hpp"


using mini_jit::Unary;



teir_interpreter::teir_interpreter(teir_operation const& operation, std::vector<void*> const& args) : operation(operation), args(args) {
    
}





void teir_interpreter::run() {
    // Some sanity checks for better error reporting
    for (teir_axis const& axis : operation.axes) {
        if (!(axis.offsets.size() == operation.tensors.size())) {
            throw teir_err_missing_offsets;
        }
        if (!(axis.strides.size() == operation.tensors.size())) {
            throw teir_err_missing_strides;
        }
    }

    // Perform the iteration for each root node
    for (std::string const& root : operation.schedule.roots) {
        iterate(root, {}, {});
    }
}



std::vector<void*> teir_interpreter::indexed_tensors(std::vector<teir_axis const*> const& axis_path, std::vector<uint64_t> const& index_path) const {
    std::vector<void*> tensors;
    for (uint64_t arg_idx = 0; arg_idx < args.size(); arg_idx += 1) {
        uint8_t* tensor = (uint8_t*)args[arg_idx];
        
        for (uint64_t k = 0; k < axis_path.size(); k++) {
            teir_axis const* axis = axis_path[k];
            uint64_t const ik = index_path[k];

            tensor += axis->offsets[arg_idx];
            tensor += axis->strides[arg_idx] * ik;
        }

        tensors.push_back((void*)tensor);
    }
    return tensors;
}





void teir_interpreter::iterate(std::string const& node, std::vector<teir_axis const*> axis_path, std::vector<uint64_t> index_path) {
    teir_iter_node const* iter_node = nullptr;
    if ((iter_node = operation.schedule.resolve_iter_id(node)) != nullptr) {
        teir_axis const* axis = resolve_axis_id(iter_node->axis);

        for (uint64_t i = 0; i < axis->extent; i++) {
            for (std::string const& child_id : iter_node->children) {
                std::vector<teir_axis const*> ap = axis_path;
                std::vector<uint64_t> ip = index_path;
                ap.push_back(axis);
                ip.push_back(i);
                iterate(child_id, ap, ip);
            }
        }
        return;
    }

    teir_inv_node const* inv_node = nullptr;
    if ((inv_node = operation.schedule.resolve_inv_id(node)) != nullptr) {
        lower(inv_node, axis_path, index_path);
        return;
    }

    throw teir_err_unresolved_schedule_node_id;
}


void teir_interpreter::lower(teir_inv_node const* inv_node, std::vector<teir_axis const*> axis_path, std::vector<uint64_t> index_path) {
    teir_primitive const* primitive = resolve_primitive_id(inv_node->primitive);

    // compute tensor addresses to pass into the kernel
    std::vector<void*> tensors = indexed_tensors(axis_path, index_path);

    // resolve primitive arguments
    std::vector<uint64_t> primitive_tensor_idxs;
    for (std::string const& tensor_id : primitive->tensors) {
        uint64_t tensor_idx = resolve_tensor_id_idx(tensor_id);
        primitive_tensor_idxs.push_back(tensor_idx);
    }

    // dispatch kernel
    if (primitive->ptype == teir_ptype_t::ptype_zero) {
        if (primitive->axes.at("M").size() == 0 && primitive->axes.at("N").size() == 0) {
            zero<float>((float*)tensors[primitive_tensor_idxs[0]], 1, 1, 0);
            return;
        }
    }

    if (primitive->ptype == teir_ptype_t::ptype_zero) {
        if (primitive->axes.at("M").size() == 1 && primitive->axes.at("N").size() == 1) {
            teir_axis const* axis_m = resolve_axis_id(primitive->axes.at("M")[0]);
            teir_axis const* axis_n = resolve_axis_id(primitive->axes.at("N")[0]);
            if (axis_m->strides[primitive_tensor_idxs[0]] == 4) {
                Unary unary;
                Unary::error_t err = unary.generate(axis_m->extent, axis_n->extent, false, Unary::dtype_t::fp32, Unary::ptype_t::zero);
                if (err == Unary::error_t::success) {
                    Unary::kernel_t kernel = unary.get_kernel();
                    kernel(nullptr, (float*)tensors[primitive_tensor_idxs[0]], 0, axis_n->strides[primitive_tensor_idxs[0]] / 4);
                    return;
                }
            }
        }
    }

    if (primitive->ptype == teir_ptype_t::ptype_copy) {
        if (primitive->axes.at("M").size() == 0 && primitive->axes.at("N").size() == 0) {
            identity<float>(
                (float const*)tensors[primitive_tensor_idxs[0]],
                (float*)tensors[primitive_tensor_idxs[1]],
                1,
                1,
                0,
                0,
                false
            );
            return;
        }
    }

    if (primitive->ptype == teir_ptype_t::ptype_copy) {
        if (primitive->axes.at("M").size() == 1 && primitive->axes.at("N").size() == 1) {
            teir_axis const* axis_m = resolve_axis_id(primitive->axes.at("M")[0]);
            teir_axis const* axis_n = resolve_axis_id(primitive->axes.at("N")[0]);
            if (axis_m->strides[primitive_tensor_idxs[0]] == 4 && axis_m->strides[primitive_tensor_idxs[1]] == 4) {
                Unary unary;
                Unary::error_t err = unary.generate(axis_m->extent, axis_n->extent, false, Unary::dtype_t::fp32, Unary::ptype_t::identity);
                if (err == Unary::error_t::success) {
                    Unary::kernel_t kernel = unary.get_kernel();

                    kernel(
                        (float const*)tensors[primitive_tensor_idxs[0]],
                        (float*)tensors[primitive_tensor_idxs[1]],
                        axis_n->strides[primitive_tensor_idxs[0]] / 4,
                        axis_n->strides[primitive_tensor_idxs[1]] / 4
                    );

                    return;
                }
            }
        }
    }

    if (primitive->ptype == teir_ptype_t::ptype_copy) {
        if (primitive->axes.at("M").size() == 1 && primitive->axes.at("N").size() == 1) {
            teir_axis const* axis_m = resolve_axis_id(primitive->axes.at("M")[0]);
            teir_axis const* axis_n = resolve_axis_id(primitive->axes.at("N")[0]);
            if (axis_m->strides[primitive_tensor_idxs[0]] == 4 && axis_n->strides[primitive_tensor_idxs[1]] == 4) {
                Unary unary;
                Unary::error_t err = unary.generate(axis_m->extent, axis_n->extent, true, Unary::dtype_t::fp32, Unary::ptype_t::identity);
                if (err == Unary::error_t::success) {
                    Unary::kernel_t kernel = unary.get_kernel();

                    kernel(
                        (float const*)tensors[primitive_tensor_idxs[0]],
                        (float*)tensors[primitive_tensor_idxs[1]],
                        axis_n->strides[primitive_tensor_idxs[0]] / 4,
                        axis_m->strides[primitive_tensor_idxs[1]] / 4
                    );

                    return;
                }
            }
        }
    }
    

    if (primitive->ptype == teir_ptype_t::ptype_relu) {
        if (primitive->axes.at("M").size() == 0 && primitive->axes.at("N").size() == 0) {
            relu<float>(
                (float const*)tensors[primitive_tensor_idxs[0]],
                (float*)tensors[primitive_tensor_idxs[1]],
                1,
                1,
                0,
                0,
                false
            );
            return;
        }
    }

    if (primitive->ptype == teir_ptype_t::ptype_relu) {
        if (primitive->axes.at("M").size() == 1 && primitive->axes.at("N").size() == 1) {
            teir_axis const* axis_m = resolve_axis_id(primitive->axes.at("M")[0]);
            teir_axis const* axis_n = resolve_axis_id(primitive->axes.at("N")[0]);
            if (axis_m->strides[primitive_tensor_idxs[0]] == 4 && axis_m->strides[primitive_tensor_idxs[1]] == 4) {
                Unary unary;
                Unary::error_t err = unary.generate(axis_m->extent, axis_n->extent, false, Unary::dtype_t::fp32, Unary::ptype_t::relu);
                if (err == Unary::error_t::success) {
                    Unary::kernel_t kernel = unary.get_kernel();

                    kernel(
                        (float const*)tensors[primitive_tensor_idxs[0]],
                        (float*)tensors[primitive_tensor_idxs[1]],
                        axis_n->strides[primitive_tensor_idxs[0]] / 4,
                        axis_n->strides[primitive_tensor_idxs[1]] / 4
                    );

                    return;
                }
            }
        }
    }

    if (primitive->ptype == teir_ptype_t::ptype_relu) {
        if (primitive->axes.at("M").size() == 1 && primitive->axes.at("N").size() == 1) {
            teir_axis const* axis_m = resolve_axis_id(primitive->axes.at("M")[0]);
            teir_axis const* axis_n = resolve_axis_id(primitive->axes.at("N")[0]);
            if (axis_m->strides[primitive_tensor_idxs[0]] == 4 && axis_n->strides[primitive_tensor_idxs[1]] == 4) {
                Unary unary;
                Unary::error_t err = unary.generate(axis_m->extent, axis_n->extent, true, Unary::dtype_t::fp32, Unary::ptype_t::relu);
                if (err == Unary::error_t::success) {
                    Unary::kernel_t kernel = unary.get_kernel();

                    kernel(
                        (float const*)tensors[primitive_tensor_idxs[0]],
                        (float*)tensors[primitive_tensor_idxs[1]],
                        axis_n->strides[primitive_tensor_idxs[0]] / 4,
                        axis_m->strides[primitive_tensor_idxs[1]] / 4
                    );

                    return;
                }
            }
        }
    }
    
    throw teir_err_missing_primitive_lowering;
}















uint64_t teir_interpreter::resolve_tensor_id_idx(std::string const& id) const {
    uint64_t result = operation.resolve_tensor_id_idx(id);
    if (result == ~((uint64_t)0)) {
        throw teir_err_unresolved_tensor_id;
    }
    return result;
}

teir_axis const* teir_interpreter::resolve_axis_id(std::string const& id) const {
    teir_axis const* result = operation.resolve_axis_id(id);
    if (result == nullptr) {
        throw teir_err_unresolved_axis_id;
    }
    return result;
}

teir_primitive const* teir_interpreter::resolve_primitive_id(std::string const& id) const {
    teir_primitive const* result = operation.resolve_primitive_id(id);
    if (result == nullptr) {
        throw teir_err_unresolved_primitive_id;
    }
    return result;
}

teir_iter_node const* teir_interpreter::resolve_iter_node_id(std::string const& id) const {
    teir_iter_node const* result = operation.schedule.resolve_iter_id(id);
    if (result == nullptr) {
        throw teir_err_unresolved_iter_node_id;
    }
    return result;
}

teir_inv_node const* teir_interpreter::resolve_inv_node_id(std::string const& id) const {
    teir_inv_node const* result = operation.schedule.resolve_inv_id(id);
    if (result == nullptr) {
        throw teir_err_unresolved_inv_node_id;
    }
    return result;
}
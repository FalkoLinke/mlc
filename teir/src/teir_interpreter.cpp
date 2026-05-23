#include "teir_interpreter.h"
#include "teir_common.hpp"


using mini_jit::Unary;



teir_interpreter::teir_interpreter(teir_operation const& operation, std::vector<void*> const& args) : operation(operation), args(args) {
    
}





void teir_interpreter::run() {
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
        teir_axis const* axis = operation.resolve_axis_id(iter_node->axis);

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
}


void teir_interpreter::lower(teir_inv_node const* inv_node, std::vector<teir_axis const*> axis_path, std::vector<uint64_t> index_path) {
    teir_primitive const* primitive = operation.resolve_primitive_id(inv_node->primitive);

    // compute tensor addresses to pass into the kernel
    std::vector<void*> tensors = indexed_tensors(axis_path, index_path);

    // resolve primitive arguments
    std::vector<uint64_t> primitive_tensor_idxs;
    for (std::string const& tensor_id : primitive->tensors) {
        uint64_t tensor_idx = operation.resolve_tensor_id(tensor_id);
        primitive_tensor_idxs.push_back(tensor_idx);
    }

    // dispatch kernel
    if (primitive->ptype == teir_ptype_t::ptype_zero) {
        if (primitive->axes.at("M").size() == 0 && primitive->axes.at("N").size() == 0) {
            zero<float>((float*)tensors[primitive_tensor_idxs[0]], 1, 1, 0);

        } else if (primitive->axes.at("M").size() == 1 && primitive->axes.at("N").size() == 1) {
            teir_axis const* axis_m = operation.resolve_axis_id(primitive->axes.at("M")[0]);
            teir_axis const* axis_n = operation.resolve_axis_id(primitive->axes.at("N")[0]);
            if (axis_m->strides[primitive_tensor_idxs[0]] == 4) {
                Unary unary;
                unary.generate(axis_m->extent, axis_n->extent, false, Unary::dtype_t::fp32, Unary::ptype_t::zero);
                Unary::kernel_t kernel = unary.get_kernel();

                kernel(nullptr, (float*)tensors[primitive_tensor_idxs[0]], 0, axis_n->strides[primitive_tensor_idxs[0]] / 4);
            }
        }

    } else if (primitive->ptype == teir_ptype_t::ptype_copy) {
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

        } else if (primitive->axes.at("M").size() == 1 && primitive->axes.at("N").size() == 1) {
            teir_axis const* axis_m = operation.resolve_axis_id(primitive->axes.at("M")[0]);
            teir_axis const* axis_n = operation.resolve_axis_id(primitive->axes.at("N")[0]);
            if (axis_m->strides[primitive_tensor_idxs[0]] == 4 && axis_m->strides[primitive_tensor_idxs[1]] == 4) {
                Unary unary;
                unary.generate(axis_m->extent, axis_n->extent, false, Unary::dtype_t::fp32, Unary::ptype_t::identity);
                Unary::kernel_t kernel = unary.get_kernel();

                kernel(
                    (float const*)tensors[primitive_tensor_idxs[0]],
                    (float*)tensors[primitive_tensor_idxs[1]],
                    axis_n->strides[primitive_tensor_idxs[0]] / 4,
                    axis_n->strides[primitive_tensor_idxs[1]] / 4
                );

            } else if (axis_m->strides[primitive_tensor_idxs[0]] == 4 && axis_n->strides[primitive_tensor_idxs[1]] == 4) {
                Unary unary;
                unary.generate(axis_m->extent, axis_n->extent, true, Unary::dtype_t::fp32, Unary::ptype_t::identity);
                Unary::kernel_t kernel = unary.get_kernel();

                kernel(
                    (float const*)tensors[primitive_tensor_idxs[0]],
                    (float*)tensors[primitive_tensor_idxs[1]],
                    axis_n->strides[primitive_tensor_idxs[0]] / 4,
                    axis_m->strides[primitive_tensor_idxs[1]] / 4
                );
            }
        }

    } else if (primitive->ptype == teir_ptype_t::ptype_relu) {
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

        } else if (primitive->axes.at("M").size() == 1 && primitive->axes.at("N").size() == 1) {
            teir_axis const* axis_m = operation.resolve_axis_id(primitive->axes.at("M")[0]);
            teir_axis const* axis_n = operation.resolve_axis_id(primitive->axes.at("N")[0]);
            if (axis_m->strides[primitive_tensor_idxs[0]] == 4 && axis_m->strides[primitive_tensor_idxs[1]] == 4) {
                Unary unary;
                unary.generate(axis_m->extent, axis_n->extent, false, Unary::dtype_t::fp32, Unary::ptype_t::relu);
                Unary::kernel_t kernel = unary.get_kernel();

                kernel(
                    (float const*)tensors[primitive_tensor_idxs[0]],
                    (float*)tensors[primitive_tensor_idxs[1]],
                    axis_n->strides[primitive_tensor_idxs[0]] / 4,
                    axis_n->strides[primitive_tensor_idxs[1]] / 4
                );

            } else if (axis_m->strides[primitive_tensor_idxs[0]] == 4 && axis_n->strides[primitive_tensor_idxs[1]] == 4) {
                Unary unary;
                unary.generate(axis_m->extent, axis_n->extent, true, Unary::dtype_t::fp32, Unary::ptype_t::relu);
                Unary::kernel_t kernel = unary.get_kernel();

                kernel(
                    (float const*)tensors[primitive_tensor_idxs[0]],
                    (float*)tensors[primitive_tensor_idxs[1]],
                    axis_n->strides[primitive_tensor_idxs[0]] / 4,
                    axis_m->strides[primitive_tensor_idxs[1]] / 4
                );
            }
        }
    }
}
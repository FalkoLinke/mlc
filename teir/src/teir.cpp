#include "teir.h"







teir_tensor::teir_tensor(std::string id, teir_dtype_t dtype) : id(std::move(id)), dtype(std::move(dtype)) {

}

teir_axis::teir_axis(std::string id, uint64_t extent, std::vector<uint64_t> strides, std::vector<uint64_t> offsets) : id(std::move(id)), extent(std::move(extent)), strides(std::move(strides)), offsets(std::move(offsets)) {

}

teir_primitive::teir_primitive(std::string id, teir_ptype_t ptype, std::vector<std::string> tensors, std::map<std::string, std::vector<std::string>> axes, std::map<std::string, std::string> metadata) : id(std::move(id)), ptype(std::move(ptype)), tensors(std::move(tensors)), axes(std::move(axes)), metadata(std::move(metadata)) {

}

teir_iter_node::teir_iter_node(std::string id, std::string axis, teir_policy_t policy, std::vector<std::string> children) : id(std::move(id)), axis(std::move(axis)), policy(std::move(policy)), children(std::move(children)) {

}

teir_inv_node::teir_inv_node(std::string id, std::string primitive) : id(std::move(id)), primitive(std::move(primitive)) {

}











teir_schedule::teir_schedule(std::vector<std::string> roots, std::vector<teir_iter_node> iteration_nodes, std::vector<teir_inv_node> invocation_nodes) : roots(std::move(roots)), iteration_nodes(std::move(iteration_nodes)), invocation_nodes(std::move(invocation_nodes)) {

}

teir_iter_node const* teir_schedule::resolve_iter_id(std::string const& id) const {
    for (auto it = iteration_nodes.begin(); it != iteration_nodes.end(); it++) {
        if (it->id == id) {
            return &(*it);
        }
    }
    return nullptr;
}

teir_inv_node const* teir_schedule::resolve_inv_id(std::string const& id) const {
    for (auto it = invocation_nodes.begin(); it != invocation_nodes.end(); it++) {
        if (it->id == id) {
            return &(*it);
        }
    }
    return nullptr;
}










teir_operation::teir_operation(std::string id, std::vector<teir_tensor> tensors, std::vector<teir_axis> axes, std::vector<teir_primitive> primitives, teir_schedule schedule) : id(std::move(id)), tensors(std::move(tensors)), axes(std::move(axes)), primitives(std::move(primitives)), schedule(std::move(schedule)) {

}

teir_axis const* teir_operation::resolve_axis_id(std::string const& id) const {
    for (auto it = axes.begin(); it != axes.end(); it++) {
        if (it->id == id) {
            return &(*it);
        }
    }
    return nullptr;
}

teir_primitive const* teir_operation::resolve_primitive_id(std::string const& id) const {
    for (auto it = primitives.begin(); it != primitives.end(); it++) {
        if (it->id == id) {
            return &(*it);
        }
    }
    return nullptr;
}

uint64_t teir_operation::resolve_primitive_id_idx(std::string const& id) const {
    for (uint64_t i = 0; i < primitives.size(); i++) {
        if (primitives[i].id == id) {
            return i;
        }
    }
    return ~0;
}

uint64_t teir_operation::resolve_tensor_id_idx(std::string const& id) const {
    for (uint64_t i = 0; i < tensors.size(); i++) {
        if (tensors[i].id == id) {
            return i;
        }
    }
    return ~0;
}
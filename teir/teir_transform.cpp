#include "teir_transform.h"

#include <algorithm>
#include <functional>
#include <set>
#include <sstream>


namespace {

[[noreturn]] void fail(std::string const& message) {
    throw teir_transform_error(message);
}


teir_axis& axis_ref(teir_operation& operation, std::string const& id) {
    for (teir_axis& axis : operation.axes) {
        if (axis.id == id) {
            return axis;
        }
    }
    fail("unknown axis @" + id);
}

teir_iter_node* find_iter(teir_operation& operation, std::string const& id) {
    for (teir_iter_node& node : operation.schedule.iteration_nodes) {
        if (node.id == id) {
            return &node;
        }
    }
    return nullptr;
}

teir_inv_node* find_inv(teir_operation& operation, std::string const& id) {
    for (teir_inv_node& node : operation.schedule.invocation_nodes) {
        if (node.id == id) {
            return &node;
        }
    }
    return nullptr;
}

teir_primitive* find_primitive(teir_operation& operation, std::string const& id) {
    for (teir_primitive& primitive : operation.primitives) {
        if (primitive.id == id) {
            return &primitive;
        }
    }
    fail("unknown primitive @" + id);
}

uint64_t tensor_index(teir_operation const& operation, std::string const& id) {
    uint64_t idx = operation.resolve_tensor_id_idx(id);
    if (idx == ~(uint64_t)0) {
        fail("unknown tensor %" + id);
    }
    return idx;
}

/** Returns the list of children (or the roots) which contains `node_id`. */
std::vector<std::string>& parent_list(teir_operation& operation, std::string const& node_id) {
    std::vector<std::string>& roots = operation.schedule.roots;
    if (std::find(roots.begin(), roots.end(), node_id) != roots.end()) {
        return roots;
    }
    for (teir_iter_node& node : operation.schedule.iteration_nodes) {
        if (std::find(node.children.begin(), node.children.end(), node_id) != node.children.end()) {
            return node.children;
        }
    }
    fail("node @" + node_id + " is not part of the schedule");
}

bool id_used(teir_operation const& operation, std::string const& id) {
    for (teir_axis const& axis : operation.axes) {
        if (axis.id == id) {
            return true;
        }
    }
    for (teir_primitive const& primitive : operation.primitives) {
        if (primitive.id == id) {
            return true;
        }
    }
    for (teir_iter_node const& node : operation.schedule.iteration_nodes) {
        if (node.id == id) {
            return true;
        }
    }
    for (teir_inv_node const& node : operation.schedule.invocation_nodes) {
        if (node.id == id) {
            return true;
        }
    }
    return false;
}

std::string unique_id(teir_operation const& operation, std::string const& base) {
    if (!id_used(operation, base)) {
        return base;
    }
    for (int i = 2;; i++) {
        std::string id = base + "_" + std::to_string(i);
        if (!id_used(operation, id)) {
            return id;
        }
    }
}

void collect_invocations(teir_operation& operation, std::string const& node_id, std::vector<teir_inv_node*>& result) {
    if (teir_inv_node* inv = find_inv(operation, node_id)) {
        result.push_back(inv);
        return;
    }
    if (teir_iter_node* iter = find_iter(operation, node_id)) {
        std::vector<std::string> children = iter->children;
        for (std::string const& child : children) {
            collect_invocations(operation, child, result);
        }
        return;
    }
    fail("unknown node @" + node_id);
}

/** Indices of the tensors written by the invocations in the subtree of `node_id`. */
std::set<uint64_t> written_tensors(teir_operation& operation, std::string const& node_id) {
    std::vector<teir_inv_node*> invocations;
    collect_invocations(operation, node_id, invocations);
    std::set<uint64_t> written;
    for (teir_inv_node* inv : invocations) {
        teir_primitive* primitive = find_primitive(operation, inv->primitive);
        written.insert(tensor_index(operation, primitive->tensors.back()));
    }
    return written;
}

bool is_reduction_axis(teir_operation& operation, std::string const& axis_id, std::string const& node_id) {
    teir_axis const& axis = axis_ref(operation, axis_id);
    for (uint64_t t : written_tensors(operation, node_id)) {
        if (axis.strides[t] == 0) {
            return true;
        }
    }
    return false;
}

bool primitive_uses_axis(teir_primitive const& primitive, std::string const& axis_id) {
    for (auto const& dim : primitive.axes) {
        if (std::find(dim.second.begin(), dim.second.end(), axis_id) != dim.second.end()) {
            return true;
        }
    }
    return false;
}

/** `true` if any tensor of the primitive has a non-zero stride along the axis. */
bool primitive_accesses_axis(teir_operation const& operation, teir_primitive const& primitive, teir_axis const& axis) {
    for (std::string const& tensor : primitive.tensors) {
        if (axis.strides[tensor_index(operation, tensor)] != 0) {
            return true;
        }
    }
    return false;
}

void for_each_guard_list(teir_operation& operation, std::function<void(std::vector<teir_guard>&)> const& func) {
    for (teir_iter_node& node : operation.schedule.iteration_nodes) {
        func(node.guards);
    }
    for (teir_inv_node& node : operation.schedule.invocation_nodes) {
        func(node.guards);
    }
}

bool guards_reference(teir_operation& operation, std::string const& axis_id) {
    bool found = false;
    for_each_guard_list(operation, [&](std::vector<teir_guard>& guards) {
        for (teir_guard const& guard : guards) {
            found = found || guard.axis_id == axis_id;
        }
    });
    return found;
}

std::vector<teir_iter_node*> nodes_over_axis(teir_operation& operation, std::string const& axis_id) {
    std::vector<teir_iter_node*> nodes;
    for (teir_iter_node& node : operation.schedule.iteration_nodes) {
        if (node.axis == axis_id) {
            nodes.push_back(&node);
        }
    }
    return nodes;
}

void replace_axis(teir_operation& operation, std::vector<std::string> const& removed, std::vector<teir_axis> const& added) {
    std::vector<teir_axis>& axes = operation.axes;
    axes.erase(std::remove_if(axes.begin(), axes.end(), [&](teir_axis const& axis) {
        return std::find(removed.begin(), removed.end(), axis.id) != removed.end();
    }), axes.end());
    axes.insert(axes.end(), added.begin(), added.end());
}

} // namespace




teir_operation teir_split(teir_operation const& input, std::string const& axis_id, uint64_t inner_extent) {
    teir_operation operation = input;
    teir_axis const axis = axis_ref(operation, axis_id);

    if (inner_extent <= 1 || inner_extent >= axis.extent || axis.extent % inner_extent != 0) {
        fail("split: " + std::to_string(inner_extent) + " does not divide the extent of @" + axis_id + " into two axes");
    }

    std::vector<teir_iter_node*> nodes = nodes_over_axis(operation, axis_id);
    std::set<std::string> users;
    for (teir_primitive const& primitive : operation.primitives) {
        if (primitive_uses_axis(primitive, axis_id)) {
            users.insert(primitive.id);
        }
    }
    if (nodes.empty() == users.empty()) {
        fail("split: @" + axis_id + " must either be iterated or be used by primitives");
    }
    if (!users.empty()) {
        for (uint64_t offset : axis.offsets) {
            if (offset != 0) {
                fail("split: primitive axis @" + axis_id + " has offsets");
            }
        }
    }

    std::string outer_id = unique_id(operation, axis_id + "_o");
    std::string inner_id = unique_id(operation, axis_id + "_i");
    std::vector<uint64_t> outer_strides;
    for (uint64_t stride : axis.strides) {
        outer_strides.push_back(stride * inner_extent);
    }
    teir_axis outer(outer_id, axis.extent / inner_extent, outer_strides, axis.offsets);
    teir_axis inner(inner_id, inner_extent, axis.strides, std::vector<uint64_t>(axis.strides.size(), 0));

    // first(a) holds iff first(a_o) and first(a_i) hold, the same holds for last
    for_each_guard_list(operation, [&](std::vector<teir_guard>& guards) {
        std::vector<teir_guard> result;
        for (teir_guard const& guard : guards) {
            if (guard.axis_id == axis_id) {
                result.emplace_back(guard.kind, outer_id);
                result.emplace_back(guard.kind, inner_id);
            } else {
                result.push_back(guard);
            }
        }
        guards = result;
    });

    std::vector<teir_iter_node> new_nodes;

    if (!nodes.empty()) {
        for (teir_iter_node* node : nodes) {
            std::string child_id = unique_id(operation, node->id + "_i");
            new_nodes.emplace_back(child_id, inner_id, node->policy, node->children);
            node->axis = outer_id;
            node->children = {child_id};
        }
    } else {
        for (teir_primitive& primitive : operation.primitives) {
            for (auto& dim : primitive.axes) {
                std::replace(dim.second.begin(), dim.second.end(), axis_id, inner_id);
            }
        }

        // wrap every run of consecutive invocations of the changed primitives into a loop over the outer axis
        auto is_user = [&](std::string const& node_id) {
            teir_inv_node* inv = find_inv(operation, node_id);
            return inv != nullptr && users.count(inv->primitive) > 0;
        };
        auto wrap = [&](std::vector<std::string>& children) {
            std::vector<std::string> result;
            for (uint64_t i = 0; i < children.size();) {
                if (!is_user(children[i])) {
                    result.push_back(children[i]);
                    i++;
                    continue;
                }
                uint64_t j = i;
                while (j < children.size() && is_user(children[j])) {
                    j++;
                }
                std::string loop_id = "iter_" + outer_id;
                while (id_used(operation, loop_id) || std::any_of(new_nodes.begin(), new_nodes.end(), [&](teir_iter_node const& n) { return n.id == loop_id; })) {
                    loop_id += "_";
                }
                new_nodes.emplace_back(loop_id, outer_id, teir_policy_t::policy_sequential,
                                       std::vector<std::string>(children.begin() + i, children.begin() + j));
                result.push_back(loop_id);
                i = j;
            }
            children = result;
        };
        wrap(operation.schedule.roots);
        for (teir_iter_node& node : operation.schedule.iteration_nodes) {
            wrap(node.children);
        }
    }

    operation.schedule.iteration_nodes.insert(operation.schedule.iteration_nodes.end(), new_nodes.begin(), new_nodes.end());
    replace_axis(operation, {axis_id}, {outer, inner});
    return operation;
}




teir_operation teir_fuse(teir_operation const& input, std::string const& outer_id, std::string const& inner_id) {
    teir_operation operation = input;
    teir_axis const outer = axis_ref(operation, outer_id);
    teir_axis const inner = axis_ref(operation, inner_id);

    for (uint64_t t = 0; t < operation.tensors.size(); t++) {
        if (outer.strides[t] != inner.extent * inner.strides[t]) {
            fail("fuse: the strides of @" + outer_id + " and @" + inner_id + " do not match for %" + operation.tensors[t].id);
        }
        if (inner.offsets[t] != 0) {
            fail("fuse: @" + inner_id + " has offsets");
        }
    }

    std::string fused_id = unique_id(operation, outer_id + inner_id);
    teir_axis fused(fused_id, outer.extent * inner.extent, inner.strides, outer.offsets);

    std::vector<teir_iter_node*> outer_nodes = nodes_over_axis(operation, outer_id);
    std::vector<teir_iter_node*> inner_nodes = nodes_over_axis(operation, inner_id);
    bool used_by_primitives = false;
    for (teir_primitive const& primitive : operation.primitives) {
        used_by_primitives = used_by_primitives || primitive_uses_axis(primitive, outer_id) || primitive_uses_axis(primitive, inner_id);
    }

    if (!outer_nodes.empty() || !inner_nodes.empty()) {
        if (used_by_primitives || outer_nodes.size() != 1 || inner_nodes.size() != 1) {
            fail("fuse: @" + outer_id + " and @" + inner_id + " must each be iterated by exactly one node");
        }
        teir_iter_node* outer_node = outer_nodes[0];
        teir_iter_node* inner_node = inner_nodes[0];
        if (outer_node->children != std::vector<std::string>{inner_node->id} || !inner_node->guards.empty()) {
            fail("fuse: the node over @" + inner_id + " must be the only child of the node over @" + outer_id + " and must not have guards");
        }
        outer_node->axis = fused_id;
        outer_node->children = inner_node->children;
        if (inner_node->policy != outer_node->policy) {
            outer_node->policy = teir_policy_t::policy_sequential;
        }
        std::string removed = inner_node->id;
        std::vector<teir_iter_node>& nodes = operation.schedule.iteration_nodes;
        nodes.erase(std::remove_if(nodes.begin(), nodes.end(), [&](teir_iter_node const& n) { return n.id == removed; }), nodes.end());
    } else if (used_by_primitives) {
        for (teir_primitive& primitive : operation.primitives) {
            for (auto& dim : primitive.axes) {
                std::vector<std::string>& ids = dim.second;
                std::vector<std::string> result;
                for (uint64_t i = 0; i < ids.size(); i++) {
                    if (ids[i] == outer_id && i + 1 < ids.size() && ids[i + 1] == inner_id) {
                        result.push_back(fused_id);
                        i++;
                    } else if (ids[i] == outer_id || ids[i] == inner_id) {
                        fail("fuse: primitive @" + primitive.id + " does not list @" + outer_id + " directly before @" + inner_id);
                    } else {
                        result.push_back(ids[i]);
                    }
                }
                ids = result;
            }
        }
    } else {
        fail("fuse: @" + outer_id + " and @" + inner_id + " are not used");
    }

    // first(o), first(i) -> first(oi); the same for last
    for_each_guard_list(operation, [&](std::vector<teir_guard>& guards) {
        std::vector<teir_guard> result;
        for (teir_guard_kind kind : {teir_guard_kind::first, teir_guard_kind::last}) {
            bool has_outer = false;
            bool has_inner = false;
            for (teir_guard const& guard : guards) {
                has_outer = has_outer || (guard.kind == kind && guard.axis_id == outer_id);
                has_inner = has_inner || (guard.kind == kind && guard.axis_id == inner_id);
            }
            if (has_outer != has_inner) {
                fail("fuse: a guard refers to only one of @" + outer_id + " and @" + inner_id);
            }
            if (has_outer) {
                result.emplace_back(kind, fused_id);
            }
        }
        for (teir_guard const& guard : guards) {
            if (guard.axis_id != outer_id && guard.axis_id != inner_id) {
                result.push_back(guard);
            }
        }
        guards = result;
    });

    replace_axis(operation, {outer_id, inner_id}, {fused});
    return operation;
}




teir_operation teir_promote(teir_operation const& input, std::string const& node_id, std::string const& dim) {
    teir_operation operation = input;
    teir_iter_node* node_ptr = find_iter(operation, node_id);
    if (node_ptr == nullptr) {
        fail("promote: unknown iteration node @" + node_id);
    }
    teir_iter_node const node = *node_ptr;
    teir_axis const axis = axis_ref(operation, node.axis);
    if (dim != "M" && dim != "N" && dim != "K") {
        fail("promote: unknown dimension " + dim);
    }
    for (uint64_t offset : axis.offsets) {
        if (offset != 0) {
            fail("promote: @" + axis.id + " has offsets");
        }
    }
    if (nodes_over_axis(operation, axis.id).size() != 1) {
        fail("promote: @" + axis.id + " is iterated by more than one node");
    }

    // classify the children
    std::vector<bool> is_user;
    for (std::string const& child : node.children) {
        teir_inv_node* inv = find_inv(operation, child);
        if (inv == nullptr) {
            fail("promote: the children of @" + node_id + " must be invocation nodes");
        }
        is_user.push_back(primitive_accesses_axis(operation, *find_primitive(operation, inv->primitive), axis));
    }
    auto first_user = std::find(is_user.begin(), is_user.end(), true);
    if (first_user == is_user.end()) {
        fail("promote: no primitive below @" + node_id + " accesses @" + axis.id);
    }
    uint64_t first_pos = first_user - is_user.begin();
    uint64_t last_pos = is_user.size() - 1 - (std::find(is_user.rbegin(), is_user.rend(), true) - is_user.rbegin());

    for (uint64_t pos = 0; pos < node.children.size(); pos++) {
        teir_inv_node* inv = find_inv(operation, node.children[pos]);
        std::vector<teir_guard> guards;
        for (teir_guard const& guard : inv->guards) {
            if (guard.axis_id != axis.id) {
                guards.push_back(guard);
                continue;
            }
            // a non-user executed once before (first) or after (last) all users
            bool once_before = !is_user[pos] && guard.kind == teir_guard_kind::first && pos < first_pos;
            bool once_after = !is_user[pos] && guard.kind == teir_guard_kind::last && pos > last_pos;
            if (!once_before && !once_after) {
                fail("promote: the guard of @" + inv->id + " on @" + axis.id + " can not be preserved");
            }
        }
        if (!is_user[pos] && guards.size() == inv->guards.size()) {
            fail("promote: @" + inv->id + " does not access @" + axis.id + " but is executed in every iteration");
        }
        // the guards of the removed node apply to all children
        guards.insert(guards.begin(), node.guards.begin(), node.guards.end());
        inv->guards = guards;

        if (!is_user[pos]) {
            continue;
        }

        // copy the primitive if it is also invoked elsewhere
        uint64_t invocations = 0;
        for (teir_inv_node const& other : operation.schedule.invocation_nodes) {
            invocations += other.primitive == inv->primitive ? 1 : 0;
        }
        teir_primitive* primitive = find_primitive(operation, inv->primitive);
        if (invocations > 1) {
            teir_primitive copy = *primitive;
            copy.id = unique_id(operation, primitive->id + "_" + axis.id);
            inv->primitive = copy.id;
            operation.primitives.push_back(copy);
            primitive = &operation.primitives.back();
        }

        if (primitive->ptype == teir_ptype_t::ptype_contract) {
            bool in0 = axis.strides[tensor_index(operation, primitive->tensors[0])] != 0;
            bool in1 = axis.strides[tensor_index(operation, primitive->tensors[1])] != 0;
            bool out = axis.strides[tensor_index(operation, primitive->tensors[2])] != 0;
            bool valid = (dim == "K" && in0 && in1 && !out) || (dim == "M" && in0 && !in1 && out) || (dim == "N" && !in0 && in1 && out);
            if (!valid) {
                fail("promote: @" + axis.id + " is not a " + dim + " axis of @" + primitive->id);
            }
        } else if (dim == "K") {
            fail("promote: @" + primitive->id + " has no K dimension");
        }
        std::vector<std::string>& ids = primitive->axes[dim];
        ids.insert(ids.begin(), axis.id);
    }

    // replace the node by its children
    std::vector<std::string>& siblings = parent_list(operation, node_id);
    auto it = std::find(siblings.begin(), siblings.end(), node_id);
    it = siblings.erase(it);
    siblings.insert(it, node.children.begin(), node.children.end());
    std::vector<teir_iter_node>& nodes = operation.schedule.iteration_nodes;
    nodes.erase(std::remove_if(nodes.begin(), nodes.end(), [&](teir_iter_node const& n) { return n.id == node_id; }), nodes.end());

    if (guards_reference(operation, axis.id)) {
        fail("promote: guards outside of @" + node_id + " refer to @" + axis.id);
    }
    return operation;
}




teir_operation teir_reorder(teir_operation const& input, std::string const& node_id) {
    teir_operation operation = input;
    teir_iter_node* outer = find_iter(operation, node_id);
    if (outer == nullptr || outer->children.size() != 1) {
        fail("reorder: @" + node_id + " must be an iteration node with a single child");
    }
    std::string child_id = outer->children[0];
    teir_iter_node* inner = find_iter(operation, child_id);
    if (inner == nullptr || !inner->guards.empty()) {
        fail("reorder: the child of @" + node_id + " must be an iteration node without guards");
    }
    if (is_reduction_axis(operation, outer->axis, child_id) && is_reduction_axis(operation, inner->axis, child_id)) {
        fail("reorder: @" + outer->axis + " and @" + inner->axis + " are both reduction axes");
    }

    std::vector<std::string>& siblings = parent_list(operation, node_id);
    std::replace(siblings.begin(), siblings.end(), node_id, child_id);

    std::swap(outer->axis, inner->axis);
    std::swap(outer->policy, inner->policy);
    outer->id = child_id;
    inner->id = node_id;
    outer->children = {node_id};
    return operation;
}




teir_operation teir_set_policy(teir_operation const& input, std::string const& node_id, teir_policy_t policy) {
    teir_operation operation = input;
    teir_iter_node* node = find_iter(operation, node_id);
    if (node == nullptr) {
        fail("set_policy: unknown iteration node @" + node_id);
    }
    if (policy == teir_policy_t::policy_parallel && is_reduction_axis(operation, node->axis, node_id)) {
        fail("set_policy: @" + node->axis + " is a reduction axis");
    }
    node->policy = policy;
    return operation;
}




teir_operation teir_swap_operands(teir_operation const& input, std::string const& primitive_id) {
    teir_operation operation = input;
    teir_primitive* primitive = find_primitive(operation, primitive_id);
    if (primitive->ptype != teir_ptype_t::ptype_contract) {
        fail("swap_operands: @" + primitive_id + " is not a contraction");
    }
    std::swap(primitive->tensors[0], primitive->tensors[1]);
    std::swap(primitive->axes["M"], primitive->axes["N"]);
    return operation;
}




std::string teir_to_string(teir_operation const& operation) {
    std::ostringstream out;
    auto guards_to_string = [](std::vector<teir_guard> const& guards) {
        std::string result;
        for (teir_guard const& guard : guards) {
            result += result.empty() ? "  guard " : ", ";
            result += (guard.kind == teir_guard_kind::first ? "first(@" : "last(@") + guard.axis_id + ")";
        }
        return result;
    };
    auto per_tensor = [&](std::vector<uint64_t> const& values) {
        std::string result;
        for (uint64_t t = 0; t < values.size(); t++) {
            if (values[t] != 0) {
                result += (result.empty() ? "" : ", ") + operation.tensors[t].id + ": " + std::to_string(values[t]);
            }
        }
        return "{ " + result + " }";
    };

    out << "teir @" << operation.id << " {\n";
    for (teir_tensor const& tensor : operation.tensors) {
        out << "  tensor %" << tensor.id << " : " << (tensor.dtype == teir_dtype_t::dtype_fp32 ? "f32" : "f64") << "\n";
    }
    out << "\n";
    for (teir_axis const& axis : operation.axes) {
        out << "  axis @" << axis.id << " extent " << axis.extent << " strides " << per_tensor(axis.strides);
        if (std::any_of(axis.offsets.begin(), axis.offsets.end(), [](uint64_t o) { return o != 0; })) {
            out << " offsets " << per_tensor(axis.offsets);
        }
        out << "\n";
    }
    out << "\n";
    for (teir_primitive const& primitive : operation.primitives) {
        static char const* const names[] = {"Zero", "Copy", "ReLU", "Contraction"};
        out << "  primitive @" << primitive.id << " : " << names[primitive.ptype] << " axes {";
        bool first_dim = true;
        for (char const* dim : {"M", "N", "K"}) {
            auto it = primitive.axes.find(dim);
            if (it == primitive.axes.end()) {
                continue;
            }
            out << (first_dim ? " " : ", ") << dim << ": [";
            for (uint64_t i = 0; i < it->second.size(); i++) {
                out << (i ? ", @" : "@") << it->second[i];
            }
            out << "]";
            first_dim = false;
        }
        out << " }\n";
    }

    out << "\n  schedule {\n    roots [";
    for (uint64_t i = 0; i < operation.schedule.roots.size(); i++) {
        out << (i ? ", @" : "@") << operation.schedule.roots[i];
    }
    out << "]\n";

    std::function<void(std::string const&, int)> print = [&](std::string const& id, int depth) {
        std::string indent(4 + 2 * depth, ' ');
        if (teir_iter_node const* node = operation.schedule.resolve_iter_id(id)) {
            out << indent << "iter @" << node->id << " axis @" << node->axis << " policy "
                << (node->policy == teir_policy_t::policy_parallel ? "parallel  " : "sequential") << " children [";
            for (uint64_t i = 0; i < node->children.size(); i++) {
                out << (i ? ", @" : "@") << node->children[i];
            }
            out << "]" << guards_to_string(node->guards) << "\n";
            for (std::string const& child : node->children) {
                print(child, depth + 1);
            }
        } else if (teir_inv_node const* inv = operation.schedule.resolve_inv_id(id)) {
            out << indent << "invoke @" << inv->id << " primitive @" << inv->primitive << guards_to_string(inv->guards) << "\n";
        }
    };
    for (std::string const& root : operation.schedule.roots) {
        print(root, 0);
    }
    out << "  }\n}\n";
    return out.str();
}

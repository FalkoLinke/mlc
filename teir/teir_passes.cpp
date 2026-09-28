#include "teir_passes.h"

#include <algorithm>
#include <set>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#include "teir_lowering.h"
#include "teir_transform.h"


namespace {

teir_primitive const& primitive_of(teir_operation const& operation, std::string const& inv_id) {
    return *operation.resolve_primitive_id(operation.schedule.resolve_inv_id(inv_id)->primitive);
}

void collect_invocations(teir_operation const& operation, std::string const& node_id, std::vector<std::string>& result) {
    if (operation.schedule.resolve_inv_id(node_id) != nullptr) {
        result.push_back(node_id);
    } else if (teir_iter_node const* node = operation.schedule.resolve_iter_id(node_id)) {
        for (std::string const& child : node->children) {
            collect_invocations(operation, child, result);
        }
    }
}

/** `true` if a tensor written in the subtree of the iteration node has the stride 0 along the node's axis. */
bool is_reduction_node(teir_operation const& operation, teir_iter_node const& node) {
    teir_axis const& axis = *operation.resolve_axis_id(node.axis);
    std::vector<std::string> invocations;
    collect_invocations(operation, node.id, invocations);
    for (std::string const& inv : invocations) {
        uint64_t written = operation.resolve_tensor_id_idx(primitive_of(operation, inv).tensors.back());
        if (axis.strides[written] == 0) {
            return true;
        }
    }
    return false;
}

teir_iter_node const* parent_node(teir_operation const& operation, std::string const& node_id) {
    for (teir_iter_node const& node : operation.schedule.iteration_nodes) {
        if (std::find(node.children.begin(), node.children.end(), node_id) != node.children.end()) {
            return &node;
        }
    }
    return nullptr;
}

/** The dimension of a contraction the axis belongs to, determined by the tensors accessing it. */
std::string contraction_dim(teir_operation const& operation, teir_primitive const& primitive, teir_axis const& axis) {
    bool in0 = axis.strides[operation.resolve_tensor_id_idx(primitive.tensors[0])] != 0;
    bool in1 = axis.strides[operation.resolve_tensor_id_idx(primitive.tensors[1])] != 0;
    bool out = axis.strides[operation.resolve_tensor_id_idx(primitive.tensors[2])] != 0;
    if (in0 && in1 && !out) {
        return "K";
    }
    if (in0 && !in1 && out) {
        return "M";
    }
    if (!in0 && in1 && out) {
        return "N";
    }
    return "";
}

/** Divisors of `n` which are `n` itself or a multiple of `step`, in increasing order. */
std::vector<uint64_t> block_sizes(uint64_t n, uint64_t step) {
    std::vector<uint64_t> result;
    for (uint64_t d = 1; d <= n; d++) {
        if (n % d == 0 && (d == n || d % step == 0)) {
            result.push_back(d);
        }
    }
    return result;
}

} // namespace




teir_target teir_target::host() {
    teir_target target;
#ifdef __APPLE__
    auto read = [](char const* name, uint64_t& value) {
        uint64_t result = 0;
        size_t size = sizeof(result);
        if (sysctlbyname(name, &result, &size, nullptr, 0) == 0 && result != 0) {
            value = result;
        }
    };
    read("hw.perflevel0.l1dcachesize", target.l1d_bytes);
    read("hw.perflevel0.l2cachesize", target.l2_bytes);
    read("hw.perflevel0.cpusperl2", target.cores_per_l2);
#endif
#ifdef _OPENMP
    target.threads = omp_get_max_threads();
#else
    target.threads = 1;
#endif
#ifdef __APPLE__
    // rdsvl is only available on processors supporting SME
    uint64_t has_sme = 0;
    read("hw.optional.arm.FEAT_SME", has_sme);
    if (has_sme) {
        uint64_t svl = 0;
        asm volatile("rdsvl %0, #1" : "=r"(svl));
        target.svl_bytes = svl;
    }
#endif
    return target;
}


std::string teir_pass_name(teir_pass_t pass) {
    switch (pass) {
    case teir_opt_fusion:
        return "fusion";
    case teir_opt_operands:
        return "operands";
    case teir_opt_blocking:
        return "blocking";
    case teir_opt_parallel:
        return "parallel";
    default:
        return "?";
    }
}




teir_operation teir_pass_primitive_fusion(teir_operation const& input, teir_target const&, std::vector<std::string>* log) {
    teir_operation operation = input;
    bool changed = true;
    while (changed) {
        changed = false;
        for (teir_inv_node const& inv : std::vector<teir_inv_node>(operation.schedule.invocation_nodes)) {
            teir_primitive const& primitive = *operation.resolve_primitive_id(inv.primitive);
            if (primitive.ptype != teir_ptype_t::ptype_contract) {
                continue;
            }
            teir_iter_node const* parent = parent_node(operation, inv.id);
            if (parent == nullptr) {
                continue;
            }
            teir_axis const& axis = *operation.resolve_axis_id(parent->axis);
            std::string dim = contraction_dim(operation, primitive, axis);
            if (dim.empty() || primitive.axes.at(dim).size() != 1) {
                continue;
            }
            std::string parent_id = parent->id;
            std::string outer = axis.id;
            std::string inner = primitive.axes.at(dim)[0];
            try {
                teir_operation result = teir_fuse(teir_promote(operation, parent_id, dim), outer, inner);
                if (log) {
                    log->push_back("fusion: promoted @" + parent_id + " into " + dim + " of @" + primitive.id + ", fused @" + outer + " and @" + inner);
                }
                operation = result;
                changed = true;
                break;
            } catch (teir_transform_error const&) {
                // the loop can not be moved into the primitive
            }
        }
    }
    return operation;
}




teir_operation teir_pass_operand_order(teir_operation const& input, teir_target const&, std::vector<std::string>* log) {
    teir_operation operation = input;
    for (teir_primitive const& primitive : std::vector<teir_primitive>(operation.primitives)) {
        if (primitive.ptype != teir_ptype_t::ptype_contract) {
            continue;
        }
        teir_tile_plan plan = teir_plan_tile(operation, primitive);
        if (plan.kind != teir_tile_plan::gemm || plan.trans_c == 0) {
            continue;
        }
        teir_operation swapped = teir_swap_operands(operation, primitive.id);
        teir_tile_plan swapped_plan = teir_plan_tile(swapped, *swapped.resolve_primitive_id(primitive.id));
        if (swapped_plan.kind == teir_tile_plan::gemm && swapped_plan.trans_c == 0) {
            if (log) {
                log->push_back("operands: swapped the inputs of @" + primitive.id);
            }
            operation = swapped;
        }
    }
    return operation;
}




teir_operation teir_pass_cache_blocking(teir_operation const& input, teir_target const& target, bool parallel, std::vector<std::string>* log) {
    teir_operation operation = input;
    uint64_t const micro = target.microkernel();
    uint64_t const budget = target.l2_bytes / (parallel ? target.cores_per_l2 : 1) / sizeof(float);
    uint64_t const k_max = target.l1d_bytes / 2 / (micro * sizeof(float));

    for (teir_primitive const& primitive : std::vector<teir_primitive>(operation.primitives)) {
        if (primitive.ptype != teir_ptype_t::ptype_contract) {
            continue;
        }
        if (primitive.axes.at("M").size() != 1 || primitive.axes.at("N").size() != 1 || primitive.axes.at("K").size() != 1) {
            continue;
        }
        std::string m_id = primitive.axes.at("M")[0];
        std::string n_id = primitive.axes.at("N")[0];
        std::string k_id = primitive.axes.at("K")[0];
        uint64_t m = operation.resolve_axis_id(m_id)->extent;
        uint64_t n = operation.resolve_axis_id(n_id)->extent;
        uint64_t k = operation.resolve_axis_id(k_id)->extent;

        // K blocks: the largest divisor of K not exceeding k_max
        std::vector<uint64_t> k_blocks;
        for (uint64_t d = 1; d <= k; d++) {
            if (k % d == 0 && d <= k_max) {
                k_blocks.push_back(d);
            }
        }
        std::reverse(k_blocks.begin(), k_blocks.end());

        // M and N blocks: the largest C block such that the A, B and C blocks fit into the budget
        uint64_t mb = 0;
        uint64_t nb = 0;
        uint64_t kb = 0;
        for (uint64_t kc : k_blocks) {
            uint64_t best_area = 0;
            uint64_t best_ws = 0;
            for (uint64_t mc : block_sizes(m, micro)) {
                for (uint64_t nc : block_sizes(n, micro)) {
                    uint64_t ws = mc * kc + kc * nc + mc * nc;
                    uint64_t area = mc * nc;
                    if (ws <= budget && (area > best_area || (area == best_area && ws < best_ws))) {
                        best_area = area;
                        best_ws = ws;
                        mb = mc;
                        nb = nc;
                    }
                }
            }
            if (best_area > 0) {
                kb = kc;
                break;
            }
        }
        if (kb == 0 || (mb == m && nb == n && kb == k)) {
            continue;
        }

        try {
            teir_operation result = operation;
            if (mb < m) {
                result = teir_split(result, m_id, mb);
            }
            if (nb < n) {
                result = teir_split(result, n_id, nb);
            }
            if (kb < k) {
                result = teir_split(result, k_id, kb);
            }
            if (log) {
                log->push_back("blocking: @" + primitive.id + " " + std::to_string(m) + " x " + std::to_string(n) + " x " + std::to_string(k)
                               + " -> blocks " + std::to_string(mb) + " x " + std::to_string(nb) + " x " + std::to_string(kb));
            }
            operation = result;
        } catch (teir_transform_error const&) {
            // an axis is shared in a way the split does not support
        }
    }
    return operation;
}




teir_operation teir_pass_parallelization(teir_operation const& input, teir_target const& target, std::vector<std::string>* log) {
    teir_operation operation = input;
    uint64_t const min_iterations = target.iterations_per_thread * target.threads;

    for (std::string const& root : std::vector<std::string>(operation.schedule.roots)) {
        std::vector<std::string> selected;
        uint64_t iterations = 1;
        std::string id = root;
        while (teir_iter_node const* node = operation.schedule.resolve_iter_id(id)) {
            if (!is_reduction_node(operation, *node)) {
                // the runtime only collapses nested nodes without guards
                if (!selected.empty() && !node->guards.empty()) {
                    break;
                }
                selected.push_back(id);
                iterations *= operation.resolve_axis_id(node->axis)->extent;
                if (iterations >= min_iterations) {
                    break;
                }
            } else if (!selected.empty()) {
                break;
            }
            if (node->children.size() != 1) {
                break;
            }
            id = node->children[0];
        }

        if (iterations < 2) {
            continue;
        }
        for (std::string const& node_id : selected) {
            operation = teir_set_policy(operation, node_id, teir_policy_t::policy_parallel);
        }
        if (log) {
            std::string nodes;
            for (std::string const& node_id : selected) {
                nodes += (nodes.empty() ? "@" : ", @") + node_id;
            }
            log->push_back("parallel: " + nodes + " (" + std::to_string(iterations) + " iterations)");
        }
    }
    return operation;
}




teir_operation teir_optimize(teir_operation const& input, teir_target const& target, uint32_t passes, std::vector<std::string>* log) {
    teir_operation operation = input;
    if (passes & teir_opt_fusion) {
        operation = teir_pass_primitive_fusion(operation, target, log);
    }
    if (passes & teir_opt_operands) {
        operation = teir_pass_operand_order(operation, target, log);
    }
    if (passes & teir_opt_blocking) {
        operation = teir_pass_cache_blocking(operation, target, (passes & teir_opt_parallel) != 0, log);
    }
    if (passes & teir_opt_parallel) {
        operation = teir_pass_parallelization(operation, target, log);
    }
    return operation;
}

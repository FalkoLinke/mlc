#include "teir_lowering.h"

using mini_jit::Unary;


static constexpr uint64_t fp32_size = 4;


/** Plans an elementwise primitive whose kernel rows run along `axis_r` and columns along `axis_c`. */
static teir_tile_plan plan_unary(teir_primitive const& primitive, std::vector<uint64_t> const& tensor_idxs, teir_axis const& axis_r, teir_axis const& axis_c) {
    teir_tile_plan plan;
    plan.m = axis_r.extent;
    plan.n = axis_c.extent;

    if (primitive.ptype == teir_ptype_t::ptype_zero) {
        uint64_t out = tensor_idxs[0];
        if (axis_r.strides[out] != fp32_size) {
            return plan;
        }
        plan.kind = teir_tile_plan::unary;
        plan.unary_ptype = Unary::ptype_t::zero;
        plan.tensors = {teir_tile_plan::no_tensor, out};
        plan.lds = {0, (int64_t)(axis_c.strides[out] / fp32_size)};
        return plan;
    }

    uint64_t in = tensor_idxs[0];
    uint64_t out = tensor_idxs[1];
    plan.unary_ptype = primitive.ptype == teir_ptype_t::ptype_relu ? Unary::ptype_t::relu : Unary::ptype_t::identity;
    plan.tensors = {in, out};

    if (axis_r.strides[in] == fp32_size && axis_r.strides[out] == fp32_size) {
        // both tensors are stored column-major
        plan.kind = teir_tile_plan::unary;
        plan.trans_b = 0;
        plan.lds = {(int64_t)(axis_c.strides[in] / fp32_size), (int64_t)(axis_c.strides[out] / fp32_size)};
    } else if (axis_r.strides[in] == fp32_size && axis_c.strides[out] == fp32_size) {
        // the output is the transpose of the input
        plan.kind = teir_tile_plan::unary;
        plan.trans_b = 1;
        plan.lds = {(int64_t)(axis_c.strides[in] / fp32_size), (int64_t)(axis_r.strides[out] / fp32_size)};
    }
    return plan;
}


static teir_tile_plan plan_gemm(std::vector<uint64_t> const& tensor_idxs, teir_axis const& axis_m, teir_axis const& axis_n, teir_axis const& axis_k) {
    teir_tile_plan plan;
    plan.m = axis_m.extent;
    plan.n = axis_n.extent;
    plan.k = axis_k.extent;
    plan.tensors = tensor_idxs;

    uint64_t a = tensor_idxs[0];
    uint64_t b = tensor_idxs[1];
    uint64_t c = tensor_idxs[2];

    // A (m x k): column-major if m has unit stride, row-major if k has unit stride
    int64_t ld_a = 0;
    if (axis_m.strides[a] == fp32_size) {
        plan.trans_a = 0;
        ld_a = axis_k.strides[a] / fp32_size;
    } else if (axis_k.strides[a] == fp32_size) {
        plan.trans_a = 1;
        ld_a = axis_m.strides[a] / fp32_size;
    } else {
        return plan;
    }

    // B (k x n): column-major if k has unit stride, row-major if n has unit stride
    int64_t ld_b = 0;
    if (axis_k.strides[b] == fp32_size) {
        plan.trans_b = 0;
        ld_b = axis_n.strides[b] / fp32_size;
    } else if (axis_n.strides[b] == fp32_size) {
        plan.trans_b = 1;
        ld_b = axis_k.strides[b] / fp32_size;
    } else {
        return plan;
    }

    // C (m x n): column-major if m has unit stride, row-major if n has unit stride
    int64_t ld_c = 0;
    if (axis_m.strides[c] == fp32_size) {
        plan.trans_c = 0;
        ld_c = axis_n.strides[c] / fp32_size;
    } else if (axis_n.strides[c] == fp32_size) {
        plan.trans_c = 1;
        ld_c = axis_m.strides[c] / fp32_size;
    } else {
        return plan;
    }

    plan.kind = teir_tile_plan::gemm;
    plan.lds = {ld_a, ld_b, ld_c};
    return plan;
}


teir_tile_plan teir_plan_tile(teir_operation const& operation, teir_primitive const& primitive) {
    std::vector<uint64_t> tensor_idxs;
    for (std::string const& tensor_id : primitive.tensors) {
        tensor_idxs.push_back(operation.resolve_tensor_id_idx(tensor_id));
    }

    std::vector<std::string> const& ms = primitive.axes.at("M");
    std::vector<std::string> const& ns = primitive.axes.at("N");
    if (ms.size() != 1 || ns.size() != 1) {
        return teir_tile_plan();
    }
    teir_axis const& axis_m = *operation.resolve_axis_id(ms[0]);
    teir_axis const& axis_n = *operation.resolve_axis_id(ns[0]);

    if (primitive.ptype == teir_ptype_t::ptype_contract) {
        std::vector<std::string> const& ks = primitive.axes.at("K");
        if (ks.size() != 1) {
            return teir_tile_plan();
        }
        teir_axis const& axis_k = *operation.resolve_axis_id(ks[0]);
        return plan_gemm(tensor_idxs, axis_m, axis_n, axis_k);
    }

    // elementwise primitives: try both assignments of M and N to the kernel's rows
    teir_tile_plan plan = plan_unary(primitive, tensor_idxs, axis_m, axis_n);
    if (plan.kind == teir_tile_plan::none) {
        plan = plan_unary(primitive, tensor_idxs, axis_n, axis_m);
    }
    return plan;
}

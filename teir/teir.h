#ifndef TEIR_TEIR_H
#define TEIR_TEIR_H


#include <string>
#include <vector>
#include <map>


enum teir_dtype_t {
    dtype_fp32,
    dtype_fp64,
};

enum teir_ptype_t {
    ptype_zero,
    ptype_copy,
    ptype_relu,
    ptype_contract,
};

enum teir_policy_t {
    policy_sequential,
    policy_parallel,
};









struct teir_tensor {
    private:

    public:
        std::string id;
        teir_dtype_t dtype;

        teir_tensor(std::string id, teir_dtype_t dtype);
};






struct teir_axis {
    private:

    public:
        std::string id;
        uint64_t extent;
        std::vector<uint64_t> strides;
        std::vector<uint64_t> offsets;

        teir_axis(std::string id, uint64_t extent, std::vector<uint64_t> strides, std::vector<uint64_t> offsets);
};




struct teir_primitive {
    private:

    public:
        std::string id;
        teir_ptype_t ptype;
        std::vector<std::string> tensors;
        std::map<std::string, std::vector<std::string>> axes;
        std::map<std::string, std::string> metadata;

        teir_primitive(std::string id, teir_ptype_t ptype, std::vector<std::string> tensors, std::map<std::string, std::vector<std::string>> axes, std::map<std::string, std::string> metadata);
};






enum teir_guard_kind {
    first,
    last,
};

struct teir_guard {
    private:
    public:
        teir_guard_kind kind;
        std::string axis_id;

        teir_guard(teir_guard_kind kind, std::string axis_id);
};



struct teir_iter_node {
    private:
    public:
        std::string id;
        std::string axis;
        teir_policy_t policy;
        std::vector<std::string> children;
        std::vector<teir_guard> guards;

        teir_iter_node(std::string id, std::string axis, teir_policy_t policy, std::vector<std::string> children, std::vector<teir_guard> guards = {});
};

struct teir_inv_node {
    private:
    public:
        std::string id;
        std::string primitive;
        std::vector<teir_guard> guards;

        teir_inv_node(std::string id, std::string primitive, std::vector<teir_guard> guards = {});
};


struct teir_schedule {
    private:
    public:
        std::vector<std::string> roots;
        std::vector<teir_iter_node> iteration_nodes;
        std::vector<teir_inv_node> invocation_nodes;
        
        teir_schedule(std::vector<std::string> roots, std::vector<teir_iter_node> iteration_nodes, std::vector<teir_inv_node> invocation_nodes);

        teir_iter_node const* resolve_iter_id(std::string const& id) const;
        teir_inv_node const* resolve_inv_id(std::string const& id) const;
};










struct teir_operation {
    private:
    public:
        std::string id;

        std::vector<teir_tensor> tensors;
        std::vector<teir_axis> axes;
        std::vector<teir_primitive> primitives;
        teir_schedule schedule;

        teir_operation(std::string id, std::vector<teir_tensor> tensors, std::vector<teir_axis> axes, std::vector<teir_primitive> primitives, teir_schedule schedule);

        teir_axis const* resolve_axis_id(std::string const& id) const;
        uint64_t resolve_axis_id_idx(std::string const& id) const;

        teir_primitive const* resolve_primitive_id(std::string const& id) const;
        uint64_t resolve_primitive_id_idx(std::string const& id) const;

        uint64_t resolve_tensor_id_idx(std::string const& id) const;
};















#endif /*TEIR_H*/
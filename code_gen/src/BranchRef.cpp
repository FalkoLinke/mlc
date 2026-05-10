#include "BranchRef.h"

mini_jit::BranchRef::BranchRef(uint32_t idx, LabeledBranch branch) : idx(idx), label(branch.label), offs_bits(branch.offs_bits), offs_shift(branch.offs_shift) {
    
}

mini_jit::BranchRef::BranchRef(uint32_t const idx, std::string const label, uint32_t offs_bits, uint32_t offs_shift) : idx(idx), label(label), offs_bits(offs_bits), offs_shift(offs_shift) {

}

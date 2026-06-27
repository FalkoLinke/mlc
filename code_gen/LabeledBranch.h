#ifndef LABELED_BRANCH_H
#define LABELED_BRANCH_H

#include <cstdint>
#include <string>

namespace mini_jit {
    struct LabeledBranch;
}

struct mini_jit::LabeledBranch {
private:

public:
    //! the instruction
    uint32_t const ins;

    //! the label to be resolved
    std::string const label;

    //! the number of bits allowed for the offset
    uint32_t const offs_bits;

    //! the place in the instruction to insert the offset at
    uint32_t const offs_shift;

    LabeledBranch(uint32_t const ins, std::string const label, uint32_t offs_bits, uint32_t offs_shift);

    ~LabeledBranch() = default;

    LabeledBranch( LabeledBranch const & ) = default;
    LabeledBranch & operator=( LabeledBranch const & ) = delete;
    LabeledBranch( LabeledBranch && ) noexcept = default;
    LabeledBranch & operator=( LabeledBranch && ) noexcept = delete;
};


#endif /*LABELED_BRANCH_H*/
#ifndef BRANCH_REF_H
#define BRANCH_REF_H

#include <string>
#include <cstdint>

#include "LabeledBranch.h"

namespace mini_jit {
    struct BranchRef;
}


struct mini_jit::BranchRef {
  private:

  public:
    //! the index of the instruction in the kernel
    uint32_t const idx;

    //! the label to be resolved
    std::string const label;

    //! the number of bits allowed for the offset
    uint32_t const offs_bits;

    //! the place in the instruction to insert the offset at
    uint32_t const offs_shift;

    BranchRef(uint32_t const idx, LabeledBranch branch);
    BranchRef(uint32_t const idx, std::string const label, uint32_t offs_bits, uint32_t offs_shift);

    ~BranchRef() = default;

    BranchRef( BranchRef const & ) = default;
    BranchRef & operator=( BranchRef const & ) = delete;
    BranchRef( BranchRef && ) noexcept = default;
    BranchRef & operator=( BranchRef && ) noexcept = delete;
};


#endif /*BRANCH_REF_H*/
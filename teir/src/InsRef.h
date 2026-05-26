#ifndef INS_REF_H
#define INS_REF_H

#include <string>
#include <cstdint>

#include "LabeledInstruction.h"

namespace mini_jit {
    struct InsRef;
}


struct mini_jit::InsRef {
  private:

  public:
    //! the index of the instruction in the kernel
    uint32_t const idx;

    LabeledInstruction const labeled_ins;

    InsRef(uint32_t const idx, LabeledInstruction ins);

    ~InsRef() = default;

    InsRef( InsRef const & ) = default;
    InsRef & operator=( InsRef const & ) = delete;
    InsRef( InsRef && ) noexcept = default;
    InsRef & operator=( InsRef && ) noexcept = delete;
};


#endif /*INS_REF_H*/
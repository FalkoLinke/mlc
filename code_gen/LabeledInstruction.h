#ifndef CODEGEN_LABELED_INSTRUCTION_H
#define CODEGEN_LABELED_INSTRUCTION_H

#include <cstdint>
#include <string>

namespace mini_jit {
    struct LabeledInstruction;
}

struct mini_jit::LabeledInstruction {
private:

public:
    //! the instruction
    uint32_t const ins;

    //! the label to be resolved
    std::string const label;

    //! Value to add onto the offset before writing it to the instruction.
    int32_t const bias;

    //! the number of bits allowed for the offset
    uint32_t const offs_bits;

    //! the place in the instruction to insert the offset at
    uint32_t const offs_shift;

    LabeledInstruction(uint32_t const ins, std::string const label, int32_t bias, uint32_t offs_bits, uint32_t offs_shift);

    ~LabeledInstruction() = default;

    LabeledInstruction( LabeledInstruction const & ) = default;
    LabeledInstruction & operator=( LabeledInstruction const & ) = delete;
    LabeledInstruction( LabeledInstruction && ) noexcept = default;
    LabeledInstruction & operator=( LabeledInstruction && ) noexcept = delete;
};


#endif /*CODEGEN_LABELED_INSTRUCTION_H*/
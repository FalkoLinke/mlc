#include "LabeledBranch.h"


mini_jit::LabeledBranch::LabeledBranch(uint32_t const ins, std::string const label, uint32_t offs_bits, uint32_t offs_shift) : ins(ins), label(label), offs_bits(offs_bits), offs_shift(offs_shift) {

}

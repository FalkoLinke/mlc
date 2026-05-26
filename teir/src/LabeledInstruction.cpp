#include "LabeledInstruction.h"


mini_jit::LabeledInstruction::LabeledInstruction(uint32_t const ins, std::string const label, uint32_t bias, uint32_t offs_bits, uint32_t offs_shift) : ins(ins), label(label), bias(bias), offs_bits(offs_bits), offs_shift(offs_shift) {

}

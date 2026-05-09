#include "InstGen.h"
#include <sstream>
#include <iomanip>
#include <bitset>

std::string mini_jit::InstGen::to_string_hex( uint32_t inst ) {
  std::stringstream l_ss;
  l_ss << "0x" << std::hex
               << std::setfill('0')
               << std::setw(8)
               << inst;

  return l_ss.str();
}

std::string mini_jit::InstGen::to_string_bin( uint32_t inst ) {
  std::string l_res = "0b";
  l_res += std::bitset<32>(inst).to_string();

  return l_res;
}

uint32_t mini_jit::InstGen::base_add( gpr_t rd, gpr_t rn, uint32_t imm12, uint32_t flags1) {
  uint32_t ins = 0x11000000;

  ins |= (rd & 0x1f);
  ins |= (rn & 0x1f) << 5;
  ins |= (imm12 & 0xfff) << 10;
  ins |= (flags1 & 0x1) << 29;
  ins |= (rd & 0x20) << (32-6);

  return ins;
}

uint32_t mini_jit::InstGen::base_add( gpr_t rd, gpr_t rn, gpr_t rm, shift_kind_t sk = shift_kind_t::lsl, uint32_t shift6 = 0x0, uint32_t flags1) {
  uint32_t ins = 0x0b000000;

  ins |= (rd & 0x1f);
  ins |= (rn & 0x1f) << 5;
  ins |= (shift6 & 0x3f) << 10;
  ins |= (rm & 0x1f) << 16;
  ins |= (sk & 0x3) << 22;
  ins |= (flags1 & 0x1) << 29;
  ins |= (rd & 0x20) << (32-6);

  return ins;
}

uint32_t mini_jit::InstGen::base_b( int32_t imm26 ) {
  uint32_t ins = 0x14000000;

  ins |= (imm26 & 0x3ffffff);

  return ins;
}

uint32_t mini_jit::InstGen::base_b_cond( int32_t imm19, br_cond_t cond ) {
  uint32_t ins = 0x54000000;

  ins |= (cond & 0xf);
  ins |= (imm19 & 0x3ffff) << 5;

  return ins;
}

uint32_t mini_jit::InstGen::base_br_cbnz( gpr_t   reg,
                                          int32_t imm19 ) {
  uint32_t l_ins = 0x35000000;

  // set register id
  uint32_t l_reg_id = reg & 0x1f;
  l_ins |= l_reg_id;

  // set size of the register
  uint32_t l_reg_size = reg & 0x20;
  l_ins |= l_reg_size << (32-6);

  // set immediate
  uint32_t l_imm = imm19 & 0x7ffff;
  l_ins |= l_imm << 5;

  return l_ins;
}

uint32_t mini_jit::InstGen::base_cbz( gpr_t rt, int32_t imm19) {
  uint32_t ins = 0x34000000;

  ins |= (rt & 0x1f);
  ins |= (imm19 & 0x3ffffff) << 5;
  ins |= (rt & 0x20) << (32-6);

  return ins;
}

uint32_t mini_jit::InstGen::base_movz( gpr_t rd, uint32_t imm16, uint32_t shift2 ) {
  uint32_t ins = 0x52800000;

  uint32_t rd_id = rd & 0x1f;
  ins |= rd_id;

  uint32_t sf = rd & 0x20;
  ins |= sf << (32-6);

  uint32_t hw = shift2 & 0x3;
  ins |= hw << 21;

  return ins;
}

uint32_t mini_jit::InstGen::base_ret( gpr_t reg ) {
  uint32_t ins = 0xd65f0000;

  uint32_t reg_id = reg & 0x1f;
  ins |= reg_id << 5;

  return ins;
}

uint32_t mini_jit::InstGen::base_sub( gpr_t rd, gpr_t rn, uint32_t imm12, uint32_t flags1) {
  uint32_t ins = 0x51000000;

  ins |= (rd & 0x1f);
  ins |= (rn & 0x1f) << 5;
  ins |= (imm12 & 0xfff) << 10;
  ins |= (flags1 & 0x1) << 29;
  ins |= (rd & 0x20) << (32-6);

  return ins;
}

uint32_t mini_jit::InstGen::base_sub( gpr_t rd, gpr_t rn, gpr_t rm, shift_kind_t sk = shift_kind_t::lsl, uint32_t shift6 = 0x0, uint32_t flags1) {
  uint32_t ins = 0x4b000000;

  ins |= (rd & 0x1f);
  ins |= (rn & 0x1f) << 5;
  ins |= (shift6 & 0x3f) << 10;
  ins |= (rm & 0x1f) << 16;
  ins |= (sk & 0x3) << 22;
  ins |= (flags1 & 0x1) << 29;
  ins |= (rd & 0x20) << (32-6);

  return ins;
}

uint32_t mini_jit::InstGen::base_smstart( ssve_spec_t spec ) {
  uint32_t ins = 0xD503417F;

  uint32_t spec_id = spec & 0x3;
  ins |= spec_id << 9;

  return ins;
}

uint32_t mini_jit::InstGen::base_smstop( ssve_spec_t spec ) {
  uint32_t ins = 0xD503407F;

  uint32_t spec_id = spec & 0x3;
  ins |= spec_id << 9;

  return ins;
}

uint32_t mini_jit::InstGen::neon_dp_fmla_vector( simd_fp_t  reg_dest,
                                                 simd_fp_t  reg_src1,
                                                 simd_fp_t  reg_src2,
                                                 arr_spec_t arr_spec ) {
  uint32_t l_ins = 0x0e20cc00;

  // set destination register id
  uint32_t l_reg_id = reg_dest & 0x1f;
  l_ins |= l_reg_id;

  // set first source register id
  l_reg_id = reg_src1 & 0x1f;
  l_ins |= l_reg_id << 5;

  // set second source register id
  l_reg_id = reg_src2 & 0x1f;
  l_ins |= l_reg_id << 16;

  // set arrangement specifier
  uint32_t l_arr_spec = arr_spec & 0x40400000;
  l_ins |= l_arr_spec;

  return l_ins;
}

uint32_t mini_jit::InstGen::ssve_pfalse( pr_t pd ) {
  uint32_t ins = 0x2518e400;

  ins |= (pd & 0xf);

  return ins;
}

uint32_t mini_jit::InstGen::ssve_ptrue( pr_t pd, sve_size_t sz, pr_pattern_t pattern, uint32_t flags1) {
  uint32_t ins = 0x2518e000;

  ins |= (pd & 0xf);
  ins |= (pattern & 0x1f) << 5;
  ins |= (flags1 & 0x1) << 16;
  ins |= (sz & 0x3) << 22;

  return ins;
}
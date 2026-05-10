#ifndef MINI_JIT_INSTGEN_H
#define MINI_JIT_INSTGEN_H

#include <cstdint>
#include <string>

#include "LabeledBranch.h"

namespace mini_jit {
  class InstGen;
}

class mini_jit::InstGen {
  public:
    //! general-purpose registers
    typedef enum : uint32_t {
      w0  =  0,
      w1  =  1,
      w2  =  2,
      w3  =  3,
      w4  =  4,
      w5  =  5,
      w6  =  6,
      w7  =  7,
      w8  =  8,
      w9  =  9,
      w10 = 10,
      w11 = 11,
      w12 = 12,
      w13 = 13,
      w14 = 14,
      w15 = 15,
      w16 = 16,
      w17 = 17,
      w18 = 18,
      w19 = 19,
      w20 = 20,
      w21 = 21,
      w22 = 22,
      w23 = 23,
      w24 = 24,
      w25 = 25,
      w26 = 26,
      w27 = 27,
      w28 = 28,
      w29 = 29,
      w30 = 30,

      x0  = 32+0,
      x1  = 32+1,
      x2  = 32+2,
      x3  = 32+3,
      x4  = 32+4,
      x5  = 32+5,
      x6  = 32+6,
      x7  = 32+7,
      x8  = 32+8,
      x9  = 32+9,
      x10 = 32+10,
      x11 = 32+11,
      x12 = 32+12,
      x13 = 32+13,
      x14 = 32+14,
      x15 = 32+15,
      x16 = 32+16,
      x17 = 32+17,
      x18 = 32+18,
      x19 = 32+19,
      x20 = 32+20,
      x21 = 32+21,
      x22 = 32+22,
      x23 = 32+23,
      x24 = 32+24,
      x25 = 32+25,
      x26 = 32+26,
      x27 = 32+27,
      x28 = 32+28,
      x29 = 32+29,
      x30 = 32+30,

      wzr =       31,
      xzr =    32+31,
      sp  = 64+32+31
    } gpr_t;

    typedef enum : uint32_t {
      eq = 0,
      ne = 1,
      cs = 2,
      cc = 3,
      mi = 4,
      pl = 5,
      vs = 6,
      vc = 7,
      hi = 8,
      ls = 9,
      ge = 10,
      lt = 11,
      gt = 12,
      le = 13,
      al = 14,
      nv = 15,
    } br_cond_t;

    typedef enum : uint32_t {
      post_index = 1,
      pre_index = 3,
      signed_offset = 2,
    } addr_mode_t;

    //! simd&fp registers
    typedef enum : uint32_t {
      v0  =  0,
      v1  =  1,
      v2  =  2,
      v3  =  3,
      v4  =  4,
      v5  =  5,
      v6  =  6,
      v7  =  7,
      v8  =  8,
      v9  =  9,
      v10 = 10,
      v11 = 11,
      v12 = 12,
      v13 = 13,
      v14 = 14,
      v15 = 15,
      v16 = 16,
      v17 = 17,
      v18 = 18,
      v19 = 19,
      v20 = 20,
      v21 = 21,
      v22 = 22,
      v23 = 23,
      v24 = 24,
      v25 = 25,
      v26 = 26,
      v27 = 27,
      v28 = 28,
      v29 = 29,
      v30 = 30,
      v31 = 31
    } simd_fp_t;

    typedef enum : uint32_t {
      z0 = 0,
      z1 = 1,
      z2 = 2,
      z3 = 3,
      z4 = 4,
      z5 = 5,
      z6 = 6,
      z7 = 7,
      z8 = 8,
      z9 = 9,
      z10 = 10,
      z11 = 11,
      z12 = 12,
      z13 = 13,
      z14 = 14,
      z15 = 15,
      z16 = 16,
      z17 = 17,
      z18 = 18,
      z19 = 19,
      z20 = 20,
      z21 = 21,
      z22 = 22,
      z23 = 23,
      z24 = 24,
      z25 = 25,
      z26 = 26,
      z27 = 27,
      z28 = 28,
      z29 = 29,
      z30 = 30,
      z31 = 31,
    } sve_zr_t;

    typedef enum : uint32_t {
      p0 = 0,
      p1 = 1,
      p2 = 2,
      p3 = 3,
      p4 = 4,
      p5 = 5,
      p6 = 6,
      p7 = 7,
      p8 = 8,
      p9 = 9,
      p10 = 10,
      p11 = 11,
      p12 = 12,
      p13 = 13,
      p14 = 14,
      p15 = 15,
    } pr_t;

    typedef enum : uint32_t {
      pow2 = 0,
      vl1 = 1,
      vl2 = 2,
      vl3 = 3,
      vl4 = 4,
      vl5 = 5,
      vl6 = 6,
      vl7 = 7,
      vl8 = 8,
      vl16 = 9,
      vl32 = 10,
      vl64 = 11,
      vl128 = 12,
      vl256 = 13,
      mul4 = 29,
      mul3 = 30,
      all = 31,
    } pr_pattern_t;

    typedef enum : uint32_t {
      b = 0b00,
      h = 0b01,
      s = 0b10,
      d = 0b11,
    } sve_size_t;

    //! arrangement specifiers
    typedef enum : uint32_t {
      s2 = 0x0,
      s4 = 0x40000000,
      d2 = 0x40400000
    } arr_spec_t;

    //! streaming SVE mode options
    typedef enum : uint32_t {
      sm = 0x1,
      za = 0x2,
      smza = 0x3,
    } ssve_spec_t;

    typedef enum : uint32_t {
      lsl = 0x0,
      lsr = 0x1,
      asr = 0x2,
    } shift_kind_t;

    typedef enum : uint32_t {
      horz = 0,
      vert = 1,
    } sme_hv_kind_t;

    static uint32_t base_add( gpr_t rd, gpr_t rn, uint32_t imm12, uint32_t flags1 = 0x0);

    static uint32_t base_add( gpr_t rd, gpr_t rn, gpr_t rm, shift_kind_t sk = shift_kind_t::lsl, uint32_t shift6 = 0x0, uint32_t flags1 = 0x0);

    static uint32_t base_b( int32_t imm26 );

    static LabeledBranch base_b( std::string label );

    static uint32_t base_b_cond( int32_t imm19, br_cond_t cond );

    static LabeledBranch base_b_cond( std::string label, br_cond_t cond );

    /**
     * @brief Generates a CBNZ instruction.
     *
     * @param reg general-purpose register.
     * @param imm19 immediate value (not the offset bytes!).
     *
     * @return instruction.
     **/
    static uint32_t base_br_cbnz( gpr_t   reg,
                                  int32_t imm19 );

    static LabeledBranch base_br_cbnz( gpr_t reg, std::string label );

    static uint32_t base_cbz( gpr_t rt, int32_t imm19);

    static LabeledBranch base_cbz( gpr_t rt, std::string label );

    static uint32_t base_ldp( gpr_t rt1, gpr_t rt2, gpr_t rn, uint32_t imm, addr_mode_t addr_mode);

    static uint32_t base_mov( gpr_t rd, gpr_t rm );

    static uint32_t base_movz( gpr_t rd, uint32_t imm16, uint32_t shift2 = 0x0);

    static uint32_t base_orr( gpr_t rd, gpr_t rn, gpr_t rm, shift_kind_t sk = shift_kind_t::lsl, uint32_t imm6 = 0x0);

    static uint32_t base_ret( gpr_t reg = gpr_t::x30 );

    static uint32_t base_stp( gpr_t rt1, gpr_t rt2, gpr_t rn, uint32_t imm, addr_mode_t addr_mode);

    static uint32_t base_sub( gpr_t rd, gpr_t rn, uint32_t imm12, uint32_t flags1 = 0x0);

    static uint32_t base_sub( gpr_t rd, gpr_t rn, gpr_t rm, shift_kind_t sk = shift_kind_t::lsl, uint32_t shift6 = 0x0, uint32_t flags1 = 0x0);

    static uint32_t base_smstart( ssve_spec_t spec = ssve_spec_t::smza );

    static uint32_t base_smstop( ssve_spec_t spec = ssve_spec_t::smza );

    /**
     * @brief Generates an FMLA (vector) instruction.
     *
     * @param reg_dest destination register.
     * @param reg_src1 first source register.
     * @param reg_src2 second source register.
     * @param arr_spec arrangement specifier.
     *
     * @return instruction.
     **/
    static uint32_t neon_dp_fmla_vector( simd_fp_t   reg_dest,
                                         simd_fp_t   reg_src1,
                                         simd_fp_t   reg_src2,
                                         arr_spec_t  arr_spec );




    static uint32_t sve_ld1w( sve_zr_t zt, pr_t pg, gpr_t rn, uint32_t imm4);

    static uint32_t sve_ld1w( sve_zr_t zt, pr_t pg, gpr_t rn, gpr_t rm);

    static uint32_t ssve_pfalse( pr_t pd );

    static uint32_t ssve_ptrue( pr_t pd, sve_size_t sz, pr_pattern_t pattern = pr_pattern_t::all, uint32_t flags1 = 0x0);

    static uint32_t sve_st1w( sve_zr_t zt, sve_size_t sz, pr_t pg, gpr_t rn, uint32_t imm4);

    static uint32_t sve_st1w( sve_zr_t zt, sve_size_t sz, pr_t pg, gpr_t rn, gpr_t rm);

    static uint32_t sme_ld1w(uint32_t za_tile2, sme_hv_kind_t hv, gpr_t rs, uint32_t offs2, pr_t pg3, gpr_t rn, gpr_t rm);

    static uint32_t sme_st1w(uint32_t za_tile2, sme_hv_kind_t hv, gpr_t rs, uint32_t offs2, pr_t pg3, gpr_t rn, gpr_t rm);

    static uint32_t sme_zero( uint32_t mask8 );

    /**
     * @brief Converts the given instruction to a hex string.
     *
     * @param inst instruction.
     *
     * @return hex string.
     **/
    static std::string to_string_hex( uint32_t inst );

    /**
     * @brief Converts the given instruction to a binary string.
     *
     * @param inst instruction.
     *
     * @return binary string.
     **/
    static std::string to_string_bin( uint32_t inst );
};

#endif
#include <cstdint>
#include <iostream>
#include <stdbool.h>
#include <cmath>

#include "Gemm.h"
#include "Kernel.h"
#include "InstGen.h"

using mini_jit::InstGen;
using gpr_t = InstGen::gpr_t;
using pr_t = InstGen::pr_t;
using sve_zr_t = InstGen::sve_zr_t;
using sve_size_t = InstGen::sve_size_t;
using addr_mode_t = InstGen::addr_mode_t;
using shift_kind_t = InstGen::shift_kind_t;
using sme_hv_kind_t = InstGen::sme_hv_kind_t;

// LDR ZA[Wv, #imm], [Xn, #imm, MUL VL]
uint32_t ldr_za(int wv, int off4, int xn) {
    uint32_t inst = 0xE1000000;
    uint32_t rv = wv - 12;

    inst |= ((rv & 0x3) << 13);
    inst |= ((xn & 0x1f) << 5);
    inst |= (off4 & 0xf);
    
    return inst;
}

// STR ZA[Wv, #imm], [Xn, #imm, MUL VL]
uint32_t str_za(int wv, int off4, int xn) {
    uint32_t inst = 0xE1200000; 
    
    uint32_t rv = (wv - 12) & 0x3;
    inst |= (rv << 13);              
    inst |= ((xn & 0x1F) << 5);
    inst |= (off4 & 0xF);

    return inst;
}

// LDR Zt, [Xn, #imm, MUL VL]
uint32_t ldr_z(int zt, int xn, int imm) {

    uint32_t inst = 0x85804000;

    inst |= (zt & 0x1F);                     
    inst |= ((xn & 0x1F) << 5);             
    inst |= ((imm & 0x3) << 10);            
    inst |= ((imm & 0x1F8) << 16);   
    return inst;
}

// FMOPA ZAda.S, Pn/M, Pm/M, Zn.S, Zm.S (FP32 non-widening)
uint32_t fmopa(int zada, int pn, int pm, int zn, int zm) {

    uint32_t inst = 0x80800000;
    inst |= ((zm & 0x1f) << 16);
    inst |= ((pm & 0x7) << 13);
    inst |= ((pn & 0x7) << 10);
    inst |= ((zn & 0x1f) << 5);
    inst |= (zada & 0x3);

    return inst;
};

auto reg_x = [](uint32_t id) { return static_cast<gpr_t>(id | 0x20); };
auto reg_w = [](uint32_t id) { return static_cast<gpr_t>(id); };












/**
 * The signature of a function which generates a GEMM microkernel of a certain fixed size (m x n).
 * 
 * @param kernel: The kernel into which to generate the microkernel.
 * @param label_prefix: A label prefix for unique labels.
 * @param k: The k dimension.
 * @param trans_a: 1 if A is row-major, 0 if A is col-major.
 * @param trans_b: 1 if B is row-major, 0 if B is col-major.
 * @param trans_c: 1 if C is row-major, 0 if C is col-major.
 * @param dtype: The data type of the matrices.
 */
typedef void(gemm_microkernel_generator)(mini_jit::Kernel&, std::string const&, uint32_t k, uint32_t trans_a, uint32_t trans_b, uint32_t trans_c, mini_jit::Gemm::dtype_t);

/**
 * The signature of a function which generates a predicated GEMM microkernel of a certain fixed size (m x n).
 * 
 * @param kernel: The kernel into which to generate the microkernel.
 * @param label_prefix: A label prefix for unique labels.
 * @param ms_count: The number of true indices in the m dimension. Must be less or equal to the fixed m size.
 * @param ns_count: The number of true indices in the n dimension. Must be less or equal to the fixed n size.
 * @param k: The k dimension.
 * @param trans_a: 1 if A is row-major, 0 if A is col-major.
 * @param trans_b: 1 if B is row-major, 0 if B is col-major.
 * @param trans_c: 1 if C is row-major, 0 if C is col-major.
 * @param dtype: The data type of the matrices.
 */
typedef void(gemm_microkernel_predicated_generator)(mini_jit::Kernel&, std::string const&, uint32_t ms_count, uint32_t ns_count, uint32_t k, uint32_t trans_a, uint32_t trans_b, uint32_t trans_c, mini_jit::Gemm::dtype_t);

/**
 * A descriptor for a single GEMM microkernel generation function.
 * 
 * The generated microkernel makes use of the aarch64 registers as follows:
 * - x0: Pointer to A.
 * - x1: Pointer to B.
 * - x2: Pointer to C.
 * - x3: Leading dimension of A.
 * - x4: Leading dimension of B.
 * - x5: Leading dimension of C.
 * 
 * x0 - x7 and x9 - x14 may be overwritten by the microkernel.
 * x15, x19 - x30 are retained.
 * All of the scalable vector registers z0 - z31 may be overwritten.
 * All of the scalable predicate registers may be overwritten.
 * The ZA matrix may be overwritten.
 */
struct gemm_microkernel_desc_t {
  /** The size of m intrinsic to the generated microkernel. */
  uint32_t m;
  /** The size of n intrinsic to the generated microkernel. */
  uint32_t n;
  /** A pointer to an unpredicated generation function. Either this or `predicated_generator` is `NULL`. */
  gemm_microkernel_generator* generator;
  /** A pointer to an predicated generation function. Either this or `generator` is `NULL`. */
  gemm_microkernel_predicated_generator* predicated_generator;
};











void generate_matrix_predicated_load_za_m16_n16(mini_jit::Kernel& kernel, InstGen::gpr_t ptr_reg, InstGen::gpr_t ld_reg, InstGen::pr_t pred_reg, uint64_t rows_count, uint64_t za_tile) {
  InstGen ig;

  for (uint64_t i = 0; i < std::min(16ull, rows_count); i++) {
    kernel.add_instr(ig.base_movz(InstGen::gpr_t::w12, 0));
    kernel.add_instr(ig.sme_ld1w(za_tile, InstGen::sme_hv_kind_t::horz, InstGen::gpr_t::w12, 0, pred_reg, ptr_reg, InstGen::gpr_t::xzr));
    kernel.add_instr(ig.base_add(ptr_reg, ptr_reg, ld_reg, InstGen::shift_kind_t::lsl, 2));
    kernel.add_instr(ig.base_add(InstGen::gpr_t::w12, InstGen::gpr_t::w12, 1));
  }
}

void generate_matrix_predicated_store_za_m16_n16(mini_jit::Kernel& kernel, InstGen::gpr_t ptr_reg, InstGen::gpr_t ld_reg, InstGen::pr_t pred_reg, uint64_t rows_count, uint64_t za_tile) {
  InstGen ig;

  for (uint64_t i = 0; i < std::min(16ull, rows_count); i++) {
    kernel.add_instr(ig.base_movz(InstGen::gpr_t::w12, 0));
    kernel.add_instr(ig.sme_st1w(za_tile, InstGen::sme_hv_kind_t::horz, InstGen::gpr_t::w12, 0, pred_reg, ptr_reg, InstGen::gpr_t::xzr));
    kernel.add_instr(ig.base_add(ptr_reg, ptr_reg, ld_reg, InstGen::shift_kind_t::lsl, 2));
    kernel.add_instr(ig.base_add(InstGen::gpr_t::w12, InstGen::gpr_t::w12, 1));
  }
}





void generate_gemm_microkernel_m32_n32_ta0_tb1_tc0(mini_jit::Kernel& kernel, std::string const& label_prefix, uint32_t k, mini_jit::Gemm::dtype_t dtype) {
  InstGen ig;

  InstGen::pr_t p0 = InstGen::pr_t::p0;

  InstGen::gpr_t gpr_a = InstGen::gpr_t::x0;
  InstGen::gpr_t gpr_b = InstGen::gpr_t::x1;
  InstGen::gpr_t gpr_c = InstGen::gpr_t::x2;
  InstGen::gpr_t gpr_lda = InstGen::gpr_t::x3;
  InstGen::gpr_t gpr_ldb = InstGen::gpr_t::x4;
  InstGen::gpr_t gpr_ldc = InstGen::gpr_t::x5;


  kernel.add_instr(ig.ssve_ptrue(p0, InstGen::sve_size_t::s));

  // load C into ZA tiles
  kernel.add_instr(ig.base_mov(InstGen::gpr_t::x6, gpr_c));
  generate_matrix_predicated_load_za_m16_n16(kernel, gpr_c, gpr_ldc, p0, 16, 0);
  generate_matrix_predicated_load_za_m16_n16(kernel, gpr_c, gpr_ldc, p0, 16, 1);
  kernel.add_instr(ig.base_mov(gpr_c, InstGen::gpr_t::x6));
  kernel.add_instr(ig.base_add(gpr_c, gpr_c, 4 * 16));
  generate_matrix_predicated_load_za_m16_n16(kernel, gpr_c, gpr_ldc, p0, 16, 0);
  generate_matrix_predicated_load_za_m16_n16(kernel, gpr_c, gpr_ldc, p0, 16, 1);
  kernel.add_instr(ig.base_mov(gpr_c, InstGen::gpr_t::x6));

  // perform gemm
  std::string const loop_start_label = label_prefix + "_loop01";
  std::string const loop_end_label = label_prefix + "_end01";
  InstGen::gpr_t loop_reg = InstGen::gpr_t::x6;

  InstGen::sve_zr_t zr_a0 = InstGen::sve_zr_t::z0;
  InstGen::sve_zr_t zr_a1 = InstGen::sve_zr_t::z1;
  InstGen::sve_zr_t zr_b0 = InstGen::sve_zr_t::z2;
  InstGen::sve_zr_t zr_b1 = InstGen::sve_zr_t::z3;

  kernel.add_instr(ig.base_movz(loop_reg, k));
  kernel.add_label(loop_start_label);
  kernel.add_labeled_instr(ig.base_cbz(loop_reg, loop_end_label));

  kernel.add_instr(ig.sve_ld1w(zr_a0, p0, gpr_a, 0));
  kernel.add_instr(ig.sve_ld1w(zr_a1, p0, gpr_a, 1));
  kernel.add_instr(ig.sve_ld1w(zr_b0, p0, gpr_b, 0));
  kernel.add_instr(ig.sve_ld1w(zr_b1, p0, gpr_b, 1));

  kernel.add_instr(fmopa(0, p0, p0, zr_a0, zr_b0));
  kernel.add_instr(fmopa(1, p0, p0, zr_a0, zr_b1));
  kernel.add_instr(fmopa(2, p0, p0, zr_a1, zr_b0));
  kernel.add_instr(fmopa(3, p0, p0, zr_a1, zr_b1));

  kernel.add_instr(ig.base_add(gpr_a, gpr_a, gpr_lda, InstGen::shift_kind_t::lsl, 2));
  kernel.add_instr(ig.base_add(gpr_b, gpr_b, gpr_ldb, InstGen::shift_kind_t::lsl, 2));
  kernel.add_instr(ig.base_sub(loop_reg, loop_reg, 1));
  kernel.add_labeled_instr(ig.base_b(loop_start_label));
  kernel.add_label(loop_end_label);

  // store C from ZA tiles
  kernel.add_instr(ig.base_mov(InstGen::gpr_t::x6, gpr_c));
  generate_matrix_predicated_store_za_m16_n16(kernel, gpr_c, gpr_ldc, p0, 16, 0);
  generate_matrix_predicated_store_za_m16_n16(kernel, gpr_c, gpr_ldc, p0, 16, 1);
  kernel.add_instr(ig.base_mov(gpr_c, InstGen::gpr_t::x6));
  kernel.add_instr(ig.base_add(gpr_c, gpr_c, 4 * 16));
  generate_matrix_predicated_store_za_m16_n16(kernel, gpr_c, gpr_ldc, p0, 16, 0);
  generate_matrix_predicated_store_za_m16_n16(kernel, gpr_c, gpr_ldc, p0, 16, 1);
  kernel.add_instr(ig.base_mov(gpr_c, InstGen::gpr_t::x6));
}



void generate_gemm_microkernel_predicated_m16_n16_ta0_tb1_tc0(mini_jit::Kernel& kernel, std::string const& label_prefix, uint32_t k, mini_jit::InstGen::pr_t pra, mini_jit::InstGen::pr_t prb, uint32_t rows_count, uint32_t cols_count, mini_jit::Gemm::dtype_t dtype) {
  rows_count = std::min(16u, rows_count);
  cols_count = std::min(16u, cols_count);

  InstGen ig;

  InstGen::gpr_t gpr_a = InstGen::gpr_t::x0;
  InstGen::gpr_t gpr_b = InstGen::gpr_t::x1;
  InstGen::gpr_t gpr_c = InstGen::gpr_t::x2;
  InstGen::gpr_t gpr_lda = InstGen::gpr_t::x3;
  InstGen::gpr_t gpr_ldb = InstGen::gpr_t::x4;
  InstGen::gpr_t gpr_ldc = InstGen::gpr_t::x5;

  generate_matrix_predicated_load_za_m16_n16(kernel, gpr_c, gpr_ldc, pra, rows_count, 0);

  std::string const loop_start_label = label_prefix + "_loop01";
  std::string const loop_end_label = label_prefix + "_end01";
  InstGen::gpr_t loop_reg = InstGen::gpr_t::x6;

  InstGen::sve_zr_t za = InstGen::sve_zr_t::z0;
  InstGen::sve_zr_t zb = InstGen::sve_zr_t::z1;

  kernel.add_instr(ig.base_movz(loop_reg, k));
  kernel.add_label(loop_start_label);
  kernel.add_labeled_instr(ig.base_cbz(loop_reg, loop_end_label));

  kernel.add_instr(ig.sve_ld1w(za, pra, gpr_a, InstGen::gpr_t::xzr));
  kernel.add_instr(ig.sve_ld1w(zb, prb, gpr_b, InstGen::gpr_t::xzr));
  kernel.add_instr(fmopa(0, prb, pra, zb, za));

  kernel.add_instr(ig.base_add(gpr_a, gpr_a, gpr_lda, InstGen::shift_kind_t::lsl, 2));
  kernel.add_instr(ig.base_add(gpr_b, gpr_b, gpr_ldb, InstGen::shift_kind_t::lsl, 2));
  kernel.add_instr(ig.base_sub(loop_reg, loop_reg, 1));
  kernel.add_labeled_instr(ig.base_b(loop_start_label));
  kernel.add_label(loop_end_label);


  generate_matrix_predicated_store_za_m16_n16(kernel, gpr_c, gpr_ldc, pra, rows_count, 0);
}






void generate_gemm_microkernel_loop(mini_jit::Kernel& kernel, std::string const& label_prefix, gemm_microkernel_desc_t desc, uint32_t mbegin, uint32_t mend, uint32_t nbegin, uint32_t nend, uint32_t m, uint32_t n, uint32_t k, uint32_t trans_a, uint32_t trans_b, uint32_t trans_c, mini_jit::Gemm::dtype_t dtype) {
  InstGen ig;
  InstGen::gpr_t gpr_tmp1 = InstGen::gpr_t::x6;

  InstGen::gpr_t gpr_a = InstGen::gpr_t::x0;
  InstGen::gpr_t gpr_b = InstGen::gpr_t::x1;
  InstGen::gpr_t gpr_c = InstGen::gpr_t::x2;
  InstGen::gpr_t gpr_lda = InstGen::gpr_t::x3;
  InstGen::gpr_t gpr_ldb = InstGen::gpr_t::x4;
  InstGen::gpr_t gpr_ldc = InstGen::gpr_t::x5;

  uint32_t dtype_size = 4;

  // initialize loop stride registers
  InstGen::gpr_t gpr_a_sm = InstGen::gpr_t::x22;
  InstGen::gpr_t gpr_b_sn = InstGen::gpr_t::x27;
  InstGen::gpr_t gpr_c_sm = InstGen::gpr_t::x28;
  InstGen::gpr_t gpr_c_sn = InstGen::gpr_t::x15;

  if (trans_a) {
    kernel.add_instr(ig.base_movz(gpr_a_sm, desc.m));
    kernel.add_instr(ig.base_mul(gpr_a_sm, gpr_lda, gpr_a_sm));
  } else {
    kernel.add_instr(ig.base_movz(gpr_a_sm, desc.m));
  }

  if (trans_b) {
    kernel.add_instr(ig.base_movz(gpr_b_sn, desc.n));
  } else {
    kernel.add_instr(ig.base_movz(gpr_b_sn, desc.n));
    kernel.add_instr(ig.base_mul(gpr_b_sn, gpr_ldb, gpr_b_sn));
  }

  if (trans_c) {
    kernel.add_instr(ig.base_movz(gpr_c_sn, desc.n));
    kernel.add_instr(ig.base_movz(gpr_c_sm, desc.m));
    kernel.add_instr(ig.base_mul(gpr_c_sm, gpr_ldc, gpr_c_sm));
  } else {
    kernel.add_instr(ig.base_movz(gpr_c_sm, desc.m));
    kernel.add_instr(ig.base_movz(InstGen::gpr_t::x6, desc.n));
    kernel.add_instr(ig.base_mul(gpr_c_sn, gpr_ldc, InstGen::gpr_t::x6));
  }

  // initialize data pointers to point to the first segment
  if (trans_a) {
    kernel.add_instr(ig.base_movz(gpr_tmp1, mbegin));
    kernel.add_instr(ig.base_mul(gpr_tmp1, gpr_tmp1, gpr_lda));
    kernel.add_instr(ig.base_add(gpr_a, gpr_a, gpr_tmp1, InstGen::shift_kind_t::lsl, 2));
  } else {
    kernel.add_instr(ig.base_add(gpr_a, gpr_a, mbegin * dtype_size));
  }

  if (trans_b) {
    kernel.add_instr(ig.base_add(gpr_b, gpr_b, nbegin * dtype_size));
  } else {
    kernel.add_instr(ig.base_movz(gpr_tmp1, nbegin));
    kernel.add_instr(ig.base_mul(gpr_tmp1, gpr_tmp1, gpr_ldb));
    kernel.add_instr(ig.base_add(gpr_b, gpr_b, gpr_tmp1, InstGen::shift_kind_t::lsl, 2));
  }

  if (trans_c) {
    kernel.add_instr(ig.base_add(gpr_c, gpr_c, nbegin * dtype_size));
    kernel.add_instr(ig.base_movz(gpr_tmp1, mbegin));
    kernel.add_instr(ig.base_mul(gpr_tmp1, gpr_tmp1, gpr_ldc));
    kernel.add_instr(ig.base_add(gpr_c, gpr_c, gpr_tmp1, InstGen::shift_kind_t::lsl, 2));
  } else {
    kernel.add_instr(ig.base_add(gpr_c, gpr_c, mbegin * dtype_size));
    kernel.add_instr(ig.base_movz(gpr_tmp1, nbegin));
    kernel.add_instr(ig.base_mul(gpr_tmp1, gpr_tmp1, gpr_ldc));
    kernel.add_instr(ig.base_add(gpr_c, gpr_c, gpr_tmp1, InstGen::shift_kind_t::lsl, 2));
  }

  // generate loops
  std::string const m_loop_start_label = label_prefix + "_mloop_start";
  std::string const m_loop_end_label = label_prefix + "_mloop_end";
  std::string const n_loop_start_label = label_prefix + "_nloop_start";
  std::string const n_loop_end_label = label_prefix + "_nloop_end";
  uint32_t n_loop_iters_count = (nend - nbegin) / desc.n;
  uint32_t m_loop_iters_count = (mend - mbegin) / desc.m;
  InstGen::gpr_t m_loop_reg = InstGen::gpr_t::x26;
  InstGen::gpr_t n_loop_reg = InstGen::gpr_t::x25;

  InstGen::gpr_t gpr_abckp1 = InstGen::gpr_t::x19;
  InstGen::gpr_t gpr_abckp2 = InstGen::gpr_t::x20;
  InstGen::gpr_t gpr_bbckp1 = InstGen::gpr_t::x21;
  InstGen::gpr_t gpr_cbckp1 = InstGen::gpr_t::x23;
  InstGen::gpr_t gpr_cbckp2 = InstGen::gpr_t::x24;

  kernel.add_instr(ig.base_mov(gpr_abckp1, gpr_a));
  kernel.add_instr(ig.base_mov(gpr_bbckp1, gpr_b));
  kernel.add_instr(ig.base_mov(gpr_cbckp1, gpr_c));
  kernel.add_instr(ig.base_movz(n_loop_reg, n_loop_iters_count));
  kernel.add_label(n_loop_start_label);
  kernel.add_labeled_instr(ig.base_cbz(n_loop_reg, n_loop_end_label));

  kernel.add_instr(ig.base_mov(gpr_abckp2, gpr_abckp1));
  kernel.add_instr(ig.base_mov(gpr_cbckp2, gpr_cbckp1));
  kernel.add_instr(ig.base_movz(m_loop_reg, m_loop_iters_count));
  kernel.add_label(m_loop_start_label);
  kernel.add_labeled_instr(ig.base_cbz(m_loop_reg, m_loop_end_label));

  kernel.add_instr(ig.base_mov(gpr_a, gpr_abckp2));
  kernel.add_instr(ig.base_mov(gpr_b, gpr_bbckp1));
  kernel.add_instr(ig.base_mov(gpr_c, gpr_cbckp2));
  desc.generator(kernel, label_prefix + "_micro", k, trans_a, trans_b, trans_c, dtype);

  kernel.add_instr(ig.base_add(gpr_cbckp2, gpr_cbckp2, gpr_c_sm, InstGen::shift_kind_t::lsl, 2));
  kernel.add_instr(ig.base_add(gpr_abckp2, gpr_abckp2, gpr_a_sm, InstGen::shift_kind_t::lsl, 2));
  kernel.add_instr(ig.base_sub(m_loop_reg, m_loop_reg, 1));
  kernel.add_labeled_instr(ig.base_b(m_loop_start_label));
  kernel.add_label(m_loop_end_label);

  kernel.add_instr(ig.base_add(gpr_cbckp1, gpr_cbckp1, gpr_c_sn, InstGen::shift_kind_t::lsl, 2));
  kernel.add_instr(ig.base_add(gpr_bbckp1, gpr_bbckp1, gpr_b_sn, InstGen::shift_kind_t::lsl, 2));
  kernel.add_instr(ig.base_sub(n_loop_reg, n_loop_reg, 1));
  kernel.add_labeled_instr(ig.base_b(n_loop_start_label));
  kernel.add_label(n_loop_end_label);
}









// new version

mini_jit::Gemm::error_t mini_jit::Gemm::generate( uint32_t m, uint32_t n, uint32_t k, uint32_t trans_a, uint32_t trans_b, uint32_t trans_c, dtype_t  dtype) {
  InstGen ig;

  // function prologue
  kernel.add_instr(ig.base_stp(InstGen::gpr_t::x29, InstGen::gpr_t::x30, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
  kernel.add_instr(ig.base_stp(InstGen::gpr_t::x19, InstGen::gpr_t::x20, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
  kernel.add_instr(ig.base_stp(InstGen::gpr_t::x21, InstGen::gpr_t::x22, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
  kernel.add_instr(ig.base_stp(InstGen::gpr_t::x23, InstGen::gpr_t::x24, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
  kernel.add_instr(ig.base_stp(InstGen::gpr_t::x25, InstGen::gpr_t::x26, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
  kernel.add_instr(ig.base_stp(InstGen::gpr_t::x27, InstGen::gpr_t::x28, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
  kernel.add_instr(ig.neon_stp(InstGen::simd_fp_t::v8, InstGen::simd_fp_t::v9, InstGen::simd_sz_t::simd_d, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
  kernel.add_instr(ig.neon_stp(InstGen::simd_fp_t::v10, InstGen::simd_fp_t::v11, InstGen::simd_sz_t::simd_d, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
  kernel.add_instr(ig.neon_stp(InstGen::simd_fp_t::v12, InstGen::simd_fp_t::v13, InstGen::simd_sz_t::simd_d, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
  kernel.add_instr(ig.neon_stp(InstGen::simd_fp_t::v14, InstGen::simd_fp_t::v15, InstGen::simd_sz_t::simd_d, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
  kernel.add_instr(ig.base_mov(InstGen::gpr_t::x29, InstGen::gpr_t::sp));
  kernel.add_instr(ig.base_smstart());

  
  // function epilogue
  kernel.add_instr(ig.base_smstop());
  kernel.add_instr(ig.neon_ldp(InstGen::simd_fp_t::v14, InstGen::simd_fp_t::v15, InstGen::simd_sz_t::simd_d, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
  kernel.add_instr(ig.neon_ldp(InstGen::simd_fp_t::v12, InstGen::simd_fp_t::v13, InstGen::simd_sz_t::simd_d, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
  kernel.add_instr(ig.neon_ldp(InstGen::simd_fp_t::v10, InstGen::simd_fp_t::v11, InstGen::simd_sz_t::simd_d, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
  kernel.add_instr(ig.neon_ldp(InstGen::simd_fp_t::v8, InstGen::simd_fp_t::v9, InstGen::simd_sz_t::simd_d, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
  kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x27, InstGen::gpr_t::x28, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
  kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x25, InstGen::gpr_t::x26, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
  kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x23, InstGen::gpr_t::x24, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
  kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x21, InstGen::gpr_t::x22, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
  kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x19, InstGen::gpr_t::x20, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
  kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x29, InstGen::gpr_t::x30, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
  kernel.add_instr(ig.base_ret());

  kernel.set_kernel();
}































// Old version
namespace mini_jit {

    /**
     * @brief Generate a kernel for matrix multiplication.
     * @param m       Number of rows in A and C.
     * @param n       Number of columns in B and C.
     * @param k       Number of columns in A and rows in B.
     * @param trans_a 0 if A is stored in column-major order, 1 if A is stored in row-major order.
     * @param trans_b 0 if B is stored in column-major order, 1 if B is stored in row-major order.
     * @param trans_c 0 if C is stored in column-major order, 1 if C is stored in row-major order.
     * @param dtype   Data type of the matrices.
     * @return error_t::success on success, another error_t value otherwise.
     **/
    Gemm::error_t Gemm::generate_v2( uint32_t m,
                      uint32_t n,
                      uint32_t k,
                      uint32_t trans_a,
                      uint32_t trans_b,
                      uint32_t trans_c,
                      dtype_t  dtype ) 
    {
      mini_jit::InstGen gen;

      if (trans_a != 0  || trans_c != 0) {
        return error_t::onlyACColumnMajor;
      }
      if (trans_b != 1) {
        return error_t::onlyBRowMajor;
      }
      if (dtype != dtype_t::fp32) {
        return error_t::onlyfp32;
      }
      if (k == 0|| m == 0 || n == 0) {
        return error_t::mnk_zero;
      }
      if (m % 16 != 0 || n % 16 != 0 || k % 16 != 0) {
        return error_t::only16;
      }

      uint32_t m_32 = m / 32;
      uint32_t n_32 = n / 32;

      //kernel.add_instr(gen.base_smstart());
      kernel.add_instr(0xd503477f);
      
      kernel.add_instr(gen.ssve_ptrue(pr_t::p0, sve_size_t::s)); // ptrue p0.s
      kernel.add_instr(gen.ssve_ptrue(pr_t::p1, sve_size_t::s)); // ptrue p1.s
      
      kernel.add_instr(0x04BF5829); // rdsvl x9, #1
      kernel.add_instr(0xD342FD29); // lsr x9, x9, #2  (UBFM)
      
      kernel.add_instr(0xD37EF4AA); // lsl x10, x5, #2 (UBFM)
      kernel.add_instr(0xD37EF46B); // lsl x11, x3, #2 (UBFM)
      kernel.add_instr(0xD37EF491); // lsl x17, x4, #2 (UBFM)
      
      // N Variables loop counter (Elements processed) in x16
      kernel.add_instr(gen.base_movz(reg_x(16), 0, 0)); // mov x16, #0
      // ---------------------- N_loop ----------------------
      for (u_int32_t i_n = 0; i_n < n_32; ++i_n) {
        
        // M Variables loop counter (Elements processed) in x15
        kernel.add_instr(gen.base_movz(reg_x(15), 0, 0)); // mov x15, #0
        // ---------------------- M_loop ----------------------
        for (u_int32_t i_m = 0; i_m < m_32; ++i_m) {
          
          kernel.add_instr(gen.base_mov(reg_x(6), reg_x(2))); // mov x6, x2
          kernel.add_instr(gen.base_add(reg_x(6), reg_x(6), reg_x(15), shift_kind_t::lsl, 2, 0)); // add x6, x6, x15, lsl #2
          
          kernel.add_instr(0x9B057E0E); // mul x14, x16, x5 (MADD X14, X16, X5, XZR)
          
          kernel.add_instr(gen.base_add(reg_x(6), reg_x(6), reg_x(14), shift_kind_t::lsl, 2, 0)); // add x6, x6, x14, lsl #2
          kernel.add_instr(gen.base_movz(reg_w(12), 0, 0)); // mov w12, #0
          kernel.add_instr(gen.base_movz(reg_w(13), 2, 0)); // mov w13, #2
          
          // Load C (Teil 1) - .rept 16
          for (u_int32_t i = 0; i < 16; ++i) {
            kernel.add_instr(ldr_za(12, 0, 6));
            //kernel.add_instr(ld1w_za(0, 0, 14, 0, 0, 6, 31));
            kernel.add_instr(ldr_za(12, 1, 6));
            //kernel.add_instr(ld1w_za(1, 0, 14, 0, 0, 6, 31));
            kernel.add_instr(gen.base_add(reg_w(12), reg_w(12), 4, 0));
            kernel.add_instr(gen.base_add(reg_x(6), reg_x(6), reg_x(10), shift_kind_t::lsl, 0, 0));
            
          }
          
          // Load C (Teil 2) - .rept 16
          for (u_int32_t i = 0; i < 16; ++i) {
            kernel.add_instr(ldr_za(13, 0, 6));
            kernel.add_instr(ldr_za(13, 1, 6));
            kernel.add_instr(gen.base_add(reg_w(13), reg_w(13), 4, 0));
            kernel.add_instr(gen.base_add(reg_x(6), reg_x(6), reg_x(10), shift_kind_t::lsl, 0, 0));
          }
          
          kernel.add_instr(gen.base_mov(reg_x(7), reg_x(0))); // mov x7, x0
          kernel.add_instr(gen.base_mov(reg_x(8), reg_x(1))); // mov x8, x1
          kernel.add_instr(gen.base_add(reg_x(7), reg_x(7), reg_x(15), shift_kind_t::lsl, 2, 0)); // add x7, x7, x15, lsl #2
          kernel.add_instr(gen.base_add(reg_x(8), reg_x(8), reg_x(16), shift_kind_t::lsl, 2, 0)); // add x8, x8, x16, lsl #2
          
          
          // ---------------------- K_loop ----------------------
          for (u_int32_t i_k=0; i_k < k; ++i_k) {
            // load A and B 32 floats and perform the outer product
            // tile 0
            kernel.add_instr(ldr_z(0, 7, 0)); // ldr z0, [x7, #0, mul vl]
            kernel.add_instr(ldr_z(2, 8, 0)); // ldr z2, [x8, #0, mul vl]
            kernel.add_instr(fmopa(0, 0, 0, 2, 0)); // fmopa za0.s, p0/m, p0/m, z2.s, z0.s
            
            // tile 1
            kernel.add_instr(ldr_z(1, 7, 1)); // ldr z1, [x7, #1, mul vl]
            kernel.add_instr(fmopa(1, 0, 0, 2, 1)); // fmopa za1.s, p0/m, p0/m, z2.s, z1.s
            
            // tile 2
            kernel.add_instr(ldr_z(3, 8, 1)); // ldr z3, [x8, #1, mul vl]
            kernel.add_instr(fmopa(2, 0, 0, 3, 0)); // fmopa za2.s, p0/m, p0/m, z3.s, z0.s
            
            // tile 3
            kernel.add_instr(fmopa(3, 0, 0, 3, 1)); // fmopa za3.s, p0/m, p0/m, z3.s, z1.s
            
            // move to the next K tile
            // A is column-major, so we move in K dimension by adding ld_a (x3) to the pointer
            kernel.add_instr(gen.base_add(reg_x(7), reg_x(7), reg_x(11), shift_kind_t::lsl, 0, 0)); // add x7, x7, x11
            kernel.add_instr(gen.base_add(reg_x(8), reg_x(8), reg_x(17), shift_kind_t::lsl, 0, 0)); // add x8, x8, x17
          }
          
          // store the results back to C
          kernel.add_instr(gen.base_mov(reg_x(6), reg_x(2))); // mov x6, x2
          kernel.add_instr(gen.base_add(reg_x(6), reg_x(6), reg_x(15), shift_kind_t::lsl, 2, 0)); // add x6, x6, x15, lsl #2
          
          kernel.add_instr(0x9B057E0E); // mul x14, x16, x5
          
          kernel.add_instr(gen.base_add(reg_x(6), reg_x(6), reg_x(14), shift_kind_t::lsl, 2, 0)); // add x6, x6, x14, lsl #2
          kernel.add_instr(gen.base_movz(reg_w(12), 0, 0)); // mov w12, #0
          kernel.add_instr(gen.base_movz(reg_w(13), 2, 0)); // mov w13, #2
          
          // Store C (Teil 1) - .rept 16
          for (u_int32_t i = 0; i < 16; ++i) {
            kernel.add_instr(str_za(12, 0, 6));
            kernel.add_instr(str_za(12, 1, 6));
            kernel.add_instr(gen.base_add(reg_w(12), reg_w(12), 4, 0));
            kernel.add_instr(gen.base_add(reg_x(6), reg_x(6), reg_x(10), shift_kind_t::lsl, 0, 0));
          }
          
          // Store C (Teil 2) - .rept 16
          for (u_int32_t i = 0; i < 16; ++i) {
            kernel.add_instr(str_za(13, 0, 6));
            kernel.add_instr(str_za(13, 1, 6));
            kernel.add_instr(gen.base_add(reg_w(13), reg_w(13), 4, 0));
            kernel.add_instr(gen.base_add(reg_x(6), reg_x(6), reg_x(10), shift_kind_t::lsl, 0, 0));
          }
          
          // M tiles check
          // processed elements in M dimension, x9 floats per tile, we process 2 tiles (32*2=64 floats) at a time
          kernel.add_instr(gen.base_add(reg_x(15), reg_x(15), reg_x(9), shift_kind_t::lsl, 1, 0)); // add x15, x15, x9, lsl #1
        }
        
        // N tiles check
        // processed elements in N dimension, x9 floats per tile, we process 2 tiles (32*2=64 floats) at a time
        kernel.add_instr(gen.base_add(reg_x(16), reg_x(16), reg_x(9), shift_kind_t::lsl, 1, 0)); // add x16, x16, x9, lsl #1
      }
      
      
      // 16er Rest Betrachtung
      // ===================================================================================================

      uint32_t n_rem = (n - n_32 * 32) / 16;  // 0 or 1: is there a 16-wide N remainder column?
      uint32_t m_rem = (m - m_32 * 32) / 16;  // 0 or 1: is there a 16-wide M remainder row?

      // Helper lambda: emit a single 16x16 tile (load C, K-loop, store C)
      // x15 = m_row (element index), x16 = n_col (element index)
      auto emit_16x16_tile = [&]() {
          // C pointer: base + m_row*4 + (n_col * ld_c)*4
          kernel.add_instr(gen.base_mov(reg_x(6), reg_x(2)));
          kernel.add_instr(gen.base_add(reg_x(6), reg_x(6), reg_x(15), shift_kind_t::lsl, 2, 0));
          kernel.add_instr(0x9B057E0E); // mul x14, x16, x5
          kernel.add_instr(gen.base_add(reg_x(6), reg_x(6), reg_x(14), shift_kind_t::lsl, 2, 0));
          kernel.add_instr(gen.base_movz(reg_w(12), 0, 0));

          // Load C
          for (uint32_t i = 0; i < 16; ++i) {
              kernel.add_instr(gen.sme_ld1w(0, sme_hv_kind_t::horz, reg_w(12), 0, pr_t::p0, reg_x(6), InstGen::gpr_t::xzr));
              kernel.add_instr(gen.base_add(reg_w(12), reg_w(12), 1, 0));
              kernel.add_instr(gen.base_add(reg_x(6), reg_x(6), reg_x(10), shift_kind_t::lsl, 0, 0));
          }

          // A pointer: A + m_row*4 (column-major: rows are contiguous)
          kernel.add_instr(gen.base_mov(reg_x(7), reg_x(0)));
          kernel.add_instr(gen.base_add(reg_x(7), reg_x(7), reg_x(15), shift_kind_t::lsl, 2, 0));
          // B pointer: B + n_col*4 (row-major: columns are contiguous within a row)
          kernel.add_instr(gen.base_mov(reg_x(8), reg_x(1)));
          kernel.add_instr(gen.base_add(reg_x(8), reg_x(8), reg_x(16), shift_kind_t::lsl, 2, 0));

          // K loop
          for (uint32_t i_k = 0; i_k < k; ++i_k) {
              kernel.add_instr(gen.sve_ld1w(sve_zr_t::z0, pr_t::p0, reg_x(7), 0));
              kernel.add_instr(gen.sve_ld1w(sve_zr_t::z2, pr_t::p0, reg_x(8), 0));
              kernel.add_instr(fmopa(0, 0, 0, 2, 0));
              kernel.add_instr(gen.base_add(reg_x(7), reg_x(7), reg_x(11), shift_kind_t::lsl, 0, 0));
              kernel.add_instr(gen.base_add(reg_x(8), reg_x(8), reg_x(17), shift_kind_t::lsl, 0, 0));
          }

          // Store C (recompute pointer cleanly - x14 was clobbered by K loop advances)
          kernel.add_instr(gen.base_mov(reg_x(6), reg_x(2)));
          kernel.add_instr(gen.base_add(reg_x(6), reg_x(6), reg_x(15), shift_kind_t::lsl, 2, 0));
          kernel.add_instr(0x9B057E0E); // mul x14, x16, x5
          kernel.add_instr(gen.base_add(reg_x(6), reg_x(6), reg_x(14), shift_kind_t::lsl, 2, 0));
          kernel.add_instr(gen.base_movz(reg_w(12), 0, 0));

          for (uint32_t i = 0; i < 16; ++i) {
              kernel.add_instr(gen.sme_st1w(0, sme_hv_kind_t::horz, reg_w(12), 0, pr_t::p0, reg_x(6), InstGen::gpr_t::xzr));
              kernel.add_instr(gen.base_add(reg_w(12), reg_w(12), 1, 0));
              kernel.add_instr(gen.base_add(reg_x(6), reg_x(6), reg_x(10), shift_kind_t::lsl, 0, 0));
          }
      };

      // N-remainder strip: the rightmost 16-wide column block, covering ALL m rows
      // Covers m_row in [0, m) in steps of 16
      if (n_rem > 0) {
          uint32_t n_col = n_32 * 32;              // element-index of the remainder column
          uint32_t total_m_tiles = m / 16;         // every 16-row block across full M height

          for (uint32_t i_m = 0; i_m < total_m_tiles; ++i_m) {
              uint32_t m_row = i_m * 16;
              kernel.add_instr(gen.base_movz(reg_x(15), m_row, 0));
              kernel.add_instr(gen.base_movz(reg_x(16), n_col, 0));
              emit_16x16_tile();
          }
      }

      // M-remainder strip: the bottom 16-wide row block, covering only the 32-wide N columns
      // (the corner with n_rem is already handled above)
      // Covers n_col in [0, n_32*32) in steps of 16
      if (m_rem > 0) {
          uint32_t m_row = m_32 * 32;              // element-index of the remainder row
          uint32_t total_n_tiles = n_32 * 2;       // each 32-wide N tile = 2 half-tiles of 16

          for (uint32_t i_n = 0; i_n < total_n_tiles; ++i_n) {
              uint32_t n_col = i_n * 16;
              kernel.add_instr(gen.base_movz(reg_x(15), m_row, 0));
              kernel.add_instr(gen.base_movz(reg_x(16), n_col, 0));
              emit_16x16_tile();
          }
      }

      // std::cout << "halooo end" << std::endl;



      //kernel.add_instr(gen.base_smstop());
      kernel.add_instr(0xd503467f);
      
      // kernel.add_instr(0xd65f03c0);
      kernel.add_instr(gen.base_ret());
        
        
      this->kernel.set_kernel();
      return error_t::success;
    }

    /*
     * A kernel is a function that takes the following parameters:
     * - a:           Pointer to matrix A.
     * - b:           Pointer to matrix B.
     * - c:           Pointer to C matrix.
     * - ld_a:        Leading dimension of A.
     * - ld_b:        Leading dimension of B.
     * - ld_c:        Leading dimension of C.
     */
    using kernel_t = void (*)( void    const * a,
                               void    const * b,
                               void          * c,
                               int64_t         ld_a,
                               int64_t         ld_b,
                               int64_t         ld_c);

    /**
     * @brief Get the generated kernel: C += A * B.
     * @return pointer to the generated kernel.
     **/
    Gemm::kernel_t Gemm::get_kernel() const {
        return (kernel_t) this->kernel.get_kernel();
    }
};

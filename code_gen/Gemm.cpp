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







struct mat_rect_t {
  uint32_t mbegin;
  uint32_t mend;
  uint32_t nbegin;
  uint32_t nend;
};

bool mat_rect_is_empty(mat_rect_t rect) {
  return rect.mend <= rect.mbegin || rect.nend <= rect.nbegin;
}

uint32_t mat_rect_area(mat_rect_t rect) {
  if (mat_rect_is_empty(rect)) {
    return 0;
  }
  return (rect.mend - rect.mbegin) * (rect.nend - rect.nbegin);
}

mat_rect_t mat_rect_union(mat_rect_t r1, mat_rect_t r2) {
  if (mat_rect_is_empty(r1) || mat_rect_is_empty(r2)) {
    return mat_rect_t {0, 0, 0, 0};
  }
  return {
    .mbegin = std::min(r1.mbegin, r2.mbegin),
    .mend = std::max(r1.mend, r2.mend),
    .nbegin = std::min(r1.nbegin, r2.nbegin),
    .nend = std::max(r1.nend, r2.nend),
  };
}

std::vector<mat_rect_t> split_mat_rect(mat_rect_t root, mat_rect_t window) {
  if (mat_rect_is_empty(root)) {
    return {};
  }
  if (mat_rect_is_empty(window)) {
    return {root};
  }

  mat_rect_t top_left  = {.mbegin = root.mbegin, .mend = window.mbegin, .nbegin = root.nbegin, .nend = window.mbegin};
  mat_rect_t top       = {.mbegin = window.mbegin, .mend = window.mend, .nbegin = root.nbegin, .nend = window.nbegin};
  mat_rect_t top_right = {.mbegin = window.mend, .mend = root.mend, .nbegin = root.nbegin, .nend = window.nbegin};
  mat_rect_t left      = {.mbegin = root.mbegin, .mend = window.mbegin, .nbegin = window.nbegin, .nend = window.nend};
  mat_rect_t right     = {.mbegin = window.mend, .mend = root.mend, .nbegin = window.nbegin, .nend = window.nend};
  mat_rect_t bot_left  = {.mbegin = root.mbegin, .mend = window.mbegin, .nbegin = window.nend, .nend = root.nend};
  mat_rect_t bot       = {.mbegin = window.mbegin, .mend = window.mend, .nbegin = window.nend, .nend = root.nend};
  mat_rect_t bot_right = {.mbegin = window.mend, .mend = root.mend, .nbegin = window.nend, .nend = root.nend};

  return {
    top_left,
    top,
    top_right,
    left,
    right,
    bot_left,
    bot,
    bot_right
  };
}

mat_rect_t tiled_subrect(mat_rect_t window, uint32_t mt, uint32_t nt) {
  if (mat_rect_is_empty(window)) {
    return window;
  }
  uint32_t mw = window.mend - window.mbegin;
  uint32_t nw = window.nend - window.nbegin;
  uint32_t consumed_ms = (mw / mt) * mt;
  uint32_t consumed_ns = (nw / nt) * nt;

  mat_rect_t act_window = {.mbegin = window.mbegin, .mend = window.mbegin + consumed_ms, .nbegin = window.nbegin, .nend = window.nbegin + consumed_ns};

  return act_window;
}



















struct mat_tiled_rect_t {
  mat_rect_t rect;
  uint32_t mt;
  uint32_t nt;
};

bool mat_tiled_rect_has_uncovered_area(mat_tiled_rect_t rect) {
  if (mat_rect_is_empty(rect.rect)) {
    return false;
  }
  return ((rect.rect.mend - rect.rect.mbegin) % rect.mt) == 0 &&
        ((rect.rect.nend - rect.rect.nbegin) % rect.nt) == 0;
}

std::vector<mat_tiled_rect_t> tile_matrix_v1(mat_rect_t root) {
  std::vector<mat_tiled_rect_t> result;

  mat_rect_t covered_area_32_32 = tiled_subrect(root, 32, 32);
  if (!mat_rect_is_empty(covered_area_32_32)) {
    result.push_back({.rect = covered_area_32_32, .mt = 32, .nt = 32});
  }
  
  std::vector<mat_rect_t> remaining_areas_32_32 = split_mat_rect(root, covered_area_32_32);
  for (mat_rect_t area : remaining_areas_32_32) {
    if (mat_rect_is_empty(area)) {
      continue;
    }

    mat_rect_t covered_area_16_16 = tiled_subrect(area, 16, 16);
    if (!mat_rect_is_empty(covered_area_16_16)) {
      result.push_back({.rect = covered_area_16_16, .mt = 16, .nt = 16});
    }

    std::vector<mat_rect_t> remaining_areas_16_16 = split_mat_rect(area, covered_area_16_16);
    for (mat_rect_t area : remaining_areas_16_16) {
      if (mat_rect_is_empty(area)) {
        continue;
      }

      uint32_t mt = std::min(16u, area.mend - area.mbegin);
      uint32_t nt = std::min(16u, area.nend - area.nbegin);
      mat_rect_t covered_area_16_16_p = tiled_subrect(area, mt, nt);
      if (!mat_rect_is_empty(covered_area_16_16_p)) {
        result.push_back({.rect = covered_area_16_16_p, .mt = mt, .nt = nt});
      }

      std::vector<mat_rect_t> remaining_areas_16_16_p = split_mat_rect(area, covered_area_16_16_p);
      for (mat_rect_t area : remaining_areas_16_16_p) {
        if (mat_rect_is_empty(area)) {
          continue;
        }
        result.push_back({.rect = area, .mt = area.mend - area.mbegin, .nt = area.nend - area.nbegin});
      }
    }
  }

  return result;
}

std::vector<mat_tiled_rect_t> tile_matrix_v2(mat_rect_t root) {
  std::vector<mat_tiled_rect_t> result;
  
  mat_rect_t covered_area_16_16 = tiled_subrect(root, 16, 16);
  if (!mat_rect_is_empty(covered_area_16_16)) {
    result.push_back({.rect = covered_area_16_16, .mt = 16, .nt = 16});
  }

  std::vector<mat_rect_t> remaining_areas_16_16 = split_mat_rect(root, covered_area_16_16);
  for (mat_rect_t area : remaining_areas_16_16) {
    if (mat_rect_is_empty(area)) {
      continue;
    }

    uint32_t mt = std::min(16u, area.mend - area.mbegin);
    uint32_t nt = std::min(16u, area.nend - area.nbegin);
    mat_rect_t covered_area_16_16_p = tiled_subrect(area, mt, nt);
    if (!mat_rect_is_empty(covered_area_16_16_p)) {
      result.push_back({.rect = covered_area_16_16_p, .mt = mt, .nt = nt});
    }

    std::vector<mat_rect_t> remaining_areas_16_16_p = split_mat_rect(area, covered_area_16_16_p);
    for (mat_rect_t area : remaining_areas_16_16_p) {
      if (mat_rect_is_empty(area)) {
        continue;
      }
      result.push_back({.rect = area, .mt = area.mend - area.mbegin, .nt = area.nend - area.nbegin});
    }
  }

  return result;
}

void print_tiling(std::vector<mat_tiled_rect_t> tiling) {
  for (mat_tiled_rect_t tiled_rect : tiling) {
    std::cout << "(" << tiled_rect.rect.mbegin << "-" << tiled_rect.rect.mend << ", " << tiled_rect.rect.nbegin << "-" << tiled_rect.rect.nend << ") (" << tiled_rect.mt << ", " << tiled_rect.nt << ")" << std::endl;
  }
}
















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









typedef void(gemm_microkernel_generator_t)(mini_jit::Kernel& kernel, std::string const& label_prefix, mat_tiled_rect_t tiled_rect, uint32_t k, uint32_t trans_a, uint32_t trans_b, uint32_t trans_c, mini_jit::Gemm::dtype_t dtype);

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
 * - x6: Pointer to stack memory.
 * 
 * x0 - x7 and x9 - x14 may be overwritten by the microkernel.
 * x15, x19 - x30 are retained.
 * All of the scalable vector registers z0 - z31 may be overwritten.
 * All of the scalable predicate registers may be overwritten.
 * The ZA matrix may be overwritten.
 */
struct gemm_microkernel_desc_t {
  uint32_t mt;
  uint32_t nt;

  uint32_t stack_mem_size;

  gemm_microkernel_generator_t* generator;
};











void generate_matrix_predicated_load_za_m16_n16(mini_jit::Kernel& kernel, InstGen::gpr_t ptr_reg, InstGen::gpr_t ld_reg, InstGen::pr_t pred_reg, uint64_t rows_count, uint64_t za_tile) {
  InstGen ig;

  kernel.add_instr(ig.base_movz(InstGen::gpr_t::w12, 0));
  for (uint64_t i = 0; i < std::min(16ull, rows_count); i++) {
    kernel.add_instr(ig.sme_ld1w(za_tile, InstGen::sme_hv_kind_t::horz, InstGen::gpr_t::w12, 0, pred_reg, ptr_reg, InstGen::gpr_t::xzr));
    kernel.add_instr(ig.base_add(ptr_reg, ptr_reg, ld_reg, InstGen::shift_kind_t::lsl, 2));
    kernel.add_instr(ig.base_add(InstGen::gpr_t::w12, InstGen::gpr_t::w12, 1));
  }
}

void generate_matrix_predicated_store_za_m16_n16(mini_jit::Kernel& kernel, InstGen::gpr_t ptr_reg, InstGen::gpr_t ld_reg, InstGen::pr_t pred_reg, uint64_t rows_count, uint64_t za_tile) {
  InstGen ig;

  kernel.add_instr(ig.base_movz(InstGen::gpr_t::w12, 0));
  for (uint64_t i = 0; i < std::min(16ull, rows_count); i++) {
    kernel.add_instr(ig.sme_st1w(za_tile, InstGen::sme_hv_kind_t::horz, InstGen::gpr_t::w12, 0, pred_reg, ptr_reg, InstGen::gpr_t::xzr));
    kernel.add_instr(ig.base_add(ptr_reg, ptr_reg, ld_reg, InstGen::shift_kind_t::lsl, 2));
    kernel.add_instr(ig.base_add(InstGen::gpr_t::w12, InstGen::gpr_t::w12, 1));
  }
}

void generate_predicate_init(mini_jit::Kernel& kernel, InstGen::pr_t pr, InstGen::sve_size_t sz, InstGen::gpr_t r1, InstGen::gpr_t r2, uint32_t begin, uint32_t end) {
  InstGen ig;
  kernel.add_instr(ig.base_movz(r1, begin));
  kernel.add_instr(ig.base_movz(r2, end));
  kernel.add_instr(ig.sve_whilelt(pr, sz, r1, r2));
}

void generate_matrix_load_vec_m16_n16(mini_jit::Kernel& kernel, uint32_t transpose, InstGen::gpr_t ptr_reg, InstGen::gpr_t ld_reg, InstGen::pr_t pred_reg, uint32_t vecs_count) {
  InstGen ig;

  vecs_count = std::min(16u, vecs_count);

  // load
  for (uint32_t i = 0; i < vecs_count; i++) {
    InstGen::sve_zr_t zr = static_cast<InstGen::sve_zr_t>(InstGen::sve_zr_t::z0 + i);
    kernel.add_instr(ig.sve_ld1w(zr, pred_reg, ptr_reg, 0));
    kernel.add_instr(ig.base_add(ptr_reg, ptr_reg, ld_reg, InstGen::shift_kind_t::lsl, 2));
  }

  if (!transpose) {
    return;
  }

  // transpose
  kernel.add_instr(ig.sme_zip(InstGen::sve_zr_t::z0, InstGen::sve_zr_t::z0, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sme_zip(InstGen::sve_zr_t::z4, InstGen::sve_zr_t::z4, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sme_zip(InstGen::sve_zr_t::z8, InstGen::sve_zr_t::z8, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sme_zip(InstGen::sve_zr_t::z12, InstGen::sve_zr_t::z12, InstGen::sve_size_t::d));

  kernel.add_instr(ig.sme_uzp(InstGen::sve_zr_t::z16, InstGen::sve_zr_t::z0, InstGen::sve_zr_t::z4, InstGen::sve_size_t::s));
  kernel.add_instr(ig.sme_uzp(InstGen::sve_zr_t::z18, InstGen::sve_zr_t::z8, InstGen::sve_zr_t::z12, InstGen::sve_size_t::s));
  kernel.add_instr(ig.sme_uzp(InstGen::sve_zr_t::z20, InstGen::sve_zr_t::z1, InstGen::sve_zr_t::z5, InstGen::sve_size_t::s));
  kernel.add_instr(ig.sme_uzp(InstGen::sve_zr_t::z22, InstGen::sve_zr_t::z9, InstGen::sve_zr_t::z13, InstGen::sve_size_t::s));
  kernel.add_instr(ig.sme_uzp(InstGen::sve_zr_t::z24, InstGen::sve_zr_t::z2, InstGen::sve_zr_t::z6, InstGen::sve_size_t::s));
  kernel.add_instr(ig.sme_uzp(InstGen::sve_zr_t::z26, InstGen::sve_zr_t::z10, InstGen::sve_zr_t::z14, InstGen::sve_size_t::s));
  kernel.add_instr(ig.sme_uzp(InstGen::sve_zr_t::z28, InstGen::sve_zr_t::z3, InstGen::sve_zr_t::z7, InstGen::sve_size_t::s));
  kernel.add_instr(ig.sme_uzp(InstGen::sve_zr_t::z30, InstGen::sve_zr_t::z11, InstGen::sve_zr_t::z15, InstGen::sve_size_t::s));

  kernel.add_instr(ig.sve_uzp1(InstGen::sve_zr_t::z0, InstGen::sve_zr_t::z16, InstGen::sve_zr_t::z18, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_uzp2(InstGen::sve_zr_t::z1, InstGen::sve_zr_t::z16, InstGen::sve_zr_t::z18, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_uzp1(InstGen::sve_zr_t::z2, InstGen::sve_zr_t::z17, InstGen::sve_zr_t::z19, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_uzp2(InstGen::sve_zr_t::z3, InstGen::sve_zr_t::z17, InstGen::sve_zr_t::z19, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_uzp1(InstGen::sve_zr_t::z4, InstGen::sve_zr_t::z20, InstGen::sve_zr_t::z22, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_uzp2(InstGen::sve_zr_t::z5, InstGen::sve_zr_t::z20, InstGen::sve_zr_t::z22, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_uzp1(InstGen::sve_zr_t::z6, InstGen::sve_zr_t::z21, InstGen::sve_zr_t::z23, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_uzp2(InstGen::sve_zr_t::z7, InstGen::sve_zr_t::z21, InstGen::sve_zr_t::z23, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_uzp1(InstGen::sve_zr_t::z8, InstGen::sve_zr_t::z24, InstGen::sve_zr_t::z26, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_uzp2(InstGen::sve_zr_t::z9, InstGen::sve_zr_t::z24, InstGen::sve_zr_t::z26, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_uzp1(InstGen::sve_zr_t::z10, InstGen::sve_zr_t::z25, InstGen::sve_zr_t::z27, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_uzp2(InstGen::sve_zr_t::z11, InstGen::sve_zr_t::z25, InstGen::sve_zr_t::z27, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_uzp1(InstGen::sve_zr_t::z12, InstGen::sve_zr_t::z28, InstGen::sve_zr_t::z30, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_uzp2(InstGen::sve_zr_t::z13, InstGen::sve_zr_t::z28, InstGen::sve_zr_t::z30, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_uzp1(InstGen::sve_zr_t::z14, InstGen::sve_zr_t::z29, InstGen::sve_zr_t::z31, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_uzp2(InstGen::sve_zr_t::z15, InstGen::sve_zr_t::z29, InstGen::sve_zr_t::z31, InstGen::sve_size_t::d));

  kernel.add_instr(ig.sve_trn1(InstGen::sve_zr_t::z16, InstGen::sve_zr_t::z0, InstGen::sve_zr_t::z1, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_trn2(InstGen::sve_zr_t::z18, InstGen::sve_zr_t::z0, InstGen::sve_zr_t::z1, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_trn1(InstGen::sve_zr_t::z17, InstGen::sve_zr_t::z2, InstGen::sve_zr_t::z3, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_trn2(InstGen::sve_zr_t::z19, InstGen::sve_zr_t::z2, InstGen::sve_zr_t::z3, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_trn1(InstGen::sve_zr_t::z20, InstGen::sve_zr_t::z4, InstGen::sve_zr_t::z5, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_trn2(InstGen::sve_zr_t::z22, InstGen::sve_zr_t::z4, InstGen::sve_zr_t::z5, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_trn1(InstGen::sve_zr_t::z21, InstGen::sve_zr_t::z6, InstGen::sve_zr_t::z7, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_trn2(InstGen::sve_zr_t::z23, InstGen::sve_zr_t::z6, InstGen::sve_zr_t::z7, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_trn1(InstGen::sve_zr_t::z24, InstGen::sve_zr_t::z8, InstGen::sve_zr_t::z9, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_trn2(InstGen::sve_zr_t::z26, InstGen::sve_zr_t::z8, InstGen::sve_zr_t::z9, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_trn1(InstGen::sve_zr_t::z25, InstGen::sve_zr_t::z10, InstGen::sve_zr_t::z11, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_trn2(InstGen::sve_zr_t::z27, InstGen::sve_zr_t::z10, InstGen::sve_zr_t::z11, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_trn1(InstGen::sve_zr_t::z28, InstGen::sve_zr_t::z12, InstGen::sve_zr_t::z13, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_trn2(InstGen::sve_zr_t::z30, InstGen::sve_zr_t::z12, InstGen::sve_zr_t::z13, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_trn1(InstGen::sve_zr_t::z29, InstGen::sve_zr_t::z14, InstGen::sve_zr_t::z15, InstGen::sve_size_t::d));
  kernel.add_instr(ig.sve_trn2(InstGen::sve_zr_t::z31, InstGen::sve_zr_t::z14, InstGen::sve_zr_t::z15, InstGen::sve_size_t::d));

  for (uint32_t i = 0; i < 16; i++) {
    InstGen::sve_zr_t zd = static_cast<InstGen::sve_zr_t>(InstGen::sve_zr_t::z0 + i);
    InstGen::sve_zr_t zn = static_cast<InstGen::sve_zr_t>(InstGen::sve_zr_t::z16 + i);
    kernel.add_instr(ig.sve_mov(zd, zn));
  }
}

void generate_matrix_temp_store(mini_jit::Kernel& kernel, InstGen::gpr_t gpr_memory_ptr, InstGen::gpr_t gpr_mat_ptr, InstGen::sve_zr_t zn) {
  InstGen ig;

  kernel.add_instr(ig.base_add(gpr_mat_ptr, gpr_memory_ptr, 8 * 16 * 4));
  for (int32_t i = 0; i < 16; i++) {
    InstGen::sve_zr_t zr = static_cast<InstGen::sve_zr_t>((zn + i) % 32);
    kernel.add_instr(ig.sve_st1w(zr, InstGen::sve_size_t::s, InstGen::pr_t::p0, gpr_mat_ptr, i - 8));
  }
}

void generate_matrix_temp_load(mini_jit::Kernel& kernel, InstGen::gpr_t gpr_mat_ptr, InstGen::sve_zr_t zn) {
  InstGen ig;

  for (int32_t i = 0; i < 16; i++) {
    InstGen::sve_zr_t zr = static_cast<InstGen::sve_zr_t>((zn + i) % 32);
    kernel.add_instr(ig.sve_ld1w(zr, InstGen::pr_t::p0, gpr_mat_ptr, i - 8));
  }
}










void generate_gemm_microkernel_m32_n32(mini_jit::Kernel& kernel, std::string const& label_prefix, mat_tiled_rect_t tiled_rect, uint32_t k, uint32_t trans_a, uint32_t trans_b, uint32_t trans_c, mini_jit::Gemm::dtype_t dtype) {
  InstGen ig;

  InstGen::pr_t p0 = InstGen::pr_t::p0;

  InstGen::gpr_t gpr_a = InstGen::gpr_t::x0;
  InstGen::gpr_t gpr_b = InstGen::gpr_t::x1;
  InstGen::gpr_t gpr_c = InstGen::gpr_t::x2;
  InstGen::gpr_t gpr_lda = InstGen::gpr_t::x3;
  InstGen::gpr_t gpr_ldb = InstGen::gpr_t::x4;
  InstGen::gpr_t gpr_ldc =  InstGen::gpr_t::x5;

  kernel.add_instr(ig.ssve_ptrue(p0, InstGen::sve_size_t::s));

  // load C into ZA tiles
  kernel.add_instr(ig.base_mov(InstGen::gpr_t::x6, gpr_c));
  generate_matrix_predicated_load_za_m16_n16(kernel, gpr_c, gpr_ldc, p0, 16, 0);
  generate_matrix_predicated_load_za_m16_n16(kernel, gpr_c, gpr_ldc, p0, 16, 1);
  kernel.add_instr(ig.base_mov(gpr_c, InstGen::gpr_t::x6));
  kernel.add_instr(ig.base_add(gpr_c, gpr_c, 4 * 16));
  generate_matrix_predicated_load_za_m16_n16(kernel, gpr_c, gpr_ldc, p0, 16, 2);
  generate_matrix_predicated_load_za_m16_n16(kernel, gpr_c, gpr_ldc, p0, 16, 3);
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

  if (trans_c) {
    kernel.add_instr(fmopa(0, p0, p0, zr_a0, zr_b0));
    kernel.add_instr(fmopa(2, p0, p0, zr_a0, zr_b1));
    kernel.add_instr(fmopa(1, p0, p0, zr_a1, zr_b0));
    kernel.add_instr(fmopa(3, p0, p0, zr_a1, zr_b1));
  } else {
    kernel.add_instr(fmopa(0, p0, p0, zr_b0, zr_a0));
    kernel.add_instr(fmopa(1, p0, p0, zr_b1, zr_a0));
    kernel.add_instr(fmopa(2, p0, p0, zr_b0, zr_a1));
    kernel.add_instr(fmopa(3, p0, p0, zr_b1, zr_a1));
  }

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
  generate_matrix_predicated_store_za_m16_n16(kernel, gpr_c, gpr_ldc, p0, 16, 2);
  generate_matrix_predicated_store_za_m16_n16(kernel, gpr_c, gpr_ldc, p0, 16, 3);
  kernel.add_instr(ig.base_mov(gpr_c, InstGen::gpr_t::x6));
}









void generate_gemm_microkernel_predicated_m16_n16(mini_jit::Kernel& kernel, std::string const& label_prefix, mat_tiled_rect_t tiled_rect, uint32_t k, uint32_t trans_a, uint32_t trans_b, uint32_t trans_c, mini_jit::Gemm::dtype_t dtype) {
  uint32_t ms_count = std::min(16u, tiled_rect.mt);
  uint32_t ns_count = std::min(16u, tiled_rect.nt);

  uint32_t dtype_size = 4;

  InstGen ig;

  InstGen::gpr_t gpr_a = InstGen::gpr_t::x0;
  InstGen::gpr_t gpr_b = InstGen::gpr_t::x1;
  InstGen::gpr_t gpr_c = InstGen::gpr_t::x2;
  InstGen::gpr_t gpr_lda = InstGen::gpr_t::x3;
  InstGen::gpr_t gpr_ldb = InstGen::gpr_t::x4;
  InstGen::gpr_t gpr_ldc = InstGen::gpr_t::x5;
  InstGen::gpr_t gpr_mem = InstGen::gpr_t::x6;
  InstGen::gpr_t gpr_mat = InstGen::gpr_t::x14;

  InstGen::pr_t p0 = InstGen::pr_t::p0;
  InstGen::pr_t prm = InstGen::pr_t::p1;
  InstGen::pr_t prn = InstGen::pr_t::p2;
  InstGen::pr_t prk = InstGen::pr_t::p3;

  kernel.add_instr(ig.ssve_ptrue(p0, InstGen::sve_size_t::s));
  generate_predicate_init(kernel, prm, InstGen::sve_size_t::s, InstGen::gpr_t::x10, InstGen::gpr_t::x11, 0, ms_count);
  generate_predicate_init(kernel, prn, InstGen::sve_size_t::s, InstGen::gpr_t::x10, InstGen::gpr_t::x11, 0, ns_count);

  kernel.add_instr(ig.base_mov(InstGen::gpr_t::x7, gpr_c));
  if (trans_c) {
    generate_matrix_predicated_load_za_m16_n16(kernel, gpr_c, gpr_ldc, prn, ms_count, 0);
  } else {
    generate_matrix_predicated_load_za_m16_n16(kernel, gpr_c, gpr_ldc, prm, ns_count, 0);
  }
  kernel.add_instr(ig.base_mov(gpr_c, InstGen::gpr_t::x7));


  if (!trans_a && trans_b) {
    std::string const loop_start_label = label_prefix + "_loop01";
    std::string const loop_end_label = label_prefix + "_end01";
    InstGen::gpr_t loop_reg = InstGen::gpr_t::x7;

    InstGen::sve_zr_t za = InstGen::sve_zr_t::z0;
    InstGen::sve_zr_t zb = InstGen::sve_zr_t::z1;

    kernel.add_instr(ig.base_movz(loop_reg, k));
    kernel.add_label(loop_start_label);
    kernel.add_labeled_instr(ig.base_cbz(loop_reg, loop_end_label));

    kernel.add_instr(ig.sve_ld1w(za, prm, gpr_a, 0));
    kernel.add_instr(ig.sve_ld1w(zb, prn, gpr_b, 0));
    if (trans_c) {
      kernel.add_instr(fmopa(0, prm, prn, za, zb));
    } else {
      kernel.add_instr(fmopa(0, prn, prm, zb, za));
    }

    kernel.add_instr(ig.base_add(gpr_a, gpr_a, gpr_lda, InstGen::shift_kind_t::lsl, 2));
    kernel.add_instr(ig.base_add(gpr_b, gpr_b, gpr_ldb, InstGen::shift_kind_t::lsl, 2));
    kernel.add_instr(ig.base_sub(loop_reg, loop_reg, 1));
    kernel.add_labeled_instr(ig.base_b(loop_start_label));
    kernel.add_label(loop_end_label);


  } else {
    uint32_t kloop_iters_count = k / 16;
    uint32_t kloop_remainder = k % 16;

    std::string const loop_start_label = label_prefix + "_loop01";
    std::string const loop_end_label = label_prefix + "_end01";
    InstGen::gpr_t loop_reg = InstGen::gpr_t::x7;
    InstGen::gpr_t gpr_abckp = InstGen::gpr_t::x10;
    InstGen::gpr_t gpr_bbckp = InstGen::gpr_t::x11;
    InstGen::gpr_t gpr_sa = InstGen::gpr_t::x12;
    InstGen::gpr_t gpr_sb = InstGen::gpr_t::x13;

    kernel.add_instr(ig.base_mov(gpr_abckp, gpr_a));
    kernel.add_instr(ig.base_mov(gpr_bbckp, gpr_b));
    if (trans_a) {
      kernel.add_instr(ig.base_movz(gpr_sa, 1));
    } else {
      kernel.add_instr(ig.base_mov(gpr_sa, gpr_lda));
    }
    if (trans_b) {
      kernel.add_instr(ig.base_mov(gpr_sb, gpr_ldb));
    } else {
      kernel.add_instr(ig.base_movz(gpr_sb, 1));
    }

    if (kloop_iters_count >= 1) {
      kernel.add_instr(ig.base_movz(loop_reg, kloop_iters_count));
      kernel.add_label(loop_start_label);
      kernel.add_labeled_instr(ig.base_cbz(loop_reg, loop_end_label));

      kernel.add_instr(ig.base_mov(gpr_a, gpr_abckp));
      kernel.add_instr(ig.base_mov(gpr_b, gpr_bbckp));

      generate_matrix_load_vec_m16_n16(kernel, trans_a, gpr_a, gpr_lda, trans_a ? p0 : prm, trans_a ? ms_count : 16);
      generate_matrix_temp_store(kernel, gpr_mem, gpr_mat, InstGen::sve_zr_t::z0);
      generate_matrix_load_vec_m16_n16(kernel, !trans_b, gpr_b, gpr_ldb, !trans_b ? p0 : prn, !trans_b ? ns_count : 16);
      generate_matrix_temp_load(kernel, gpr_mat, InstGen::sve_zr_t::z16);
      
      for (uint32_t i = 0; i < 16; i++) {
        InstGen::sve_zr_t za = static_cast<InstGen::sve_zr_t>(InstGen::sve_zr_t::z16 + i);
        InstGen::sve_zr_t zb = static_cast<InstGen::sve_zr_t>(InstGen::sve_zr_t::z0 + i);
        if (trans_c) {
          kernel.add_instr(fmopa(0, prm, prn, za, zb));
        } else {
          kernel.add_instr(fmopa(0, prn, prm, zb, za));
        }
      }

      kernel.add_instr(ig.base_add(gpr_abckp, gpr_abckp, gpr_sa, InstGen::shift_kind_t::lsl, 6));
      kernel.add_instr(ig.base_add(gpr_bbckp, gpr_bbckp, gpr_sb, InstGen::shift_kind_t::lsl, 6));
      kernel.add_instr(ig.base_sub(loop_reg, loop_reg, 1));
      kernel.add_labeled_instr(ig.base_b(loop_start_label));
      kernel.add_label(loop_end_label);
    }

    if (kloop_remainder != 0) {
      kernel.add_instr(ig.base_mov(gpr_a, gpr_abckp));
      kernel.add_instr(ig.base_mov(gpr_b, gpr_bbckp));
      generate_predicate_init(kernel, prk, InstGen::sve_size_t::s, InstGen::gpr_t::x10, InstGen::gpr_t::x11, 0, kloop_remainder);
      generate_matrix_load_vec_m16_n16(kernel, trans_a, gpr_a, gpr_lda, trans_a ? prk : prm, trans_a ? ms_count : kloop_remainder);
      generate_matrix_temp_store(kernel, gpr_mem, gpr_mat, InstGen::sve_zr_t::z0);
      generate_matrix_load_vec_m16_n16(kernel, !trans_b, gpr_b, gpr_ldb, !trans_b ? prk : prn, !trans_b ? ns_count : kloop_remainder);
      generate_matrix_temp_load(kernel, gpr_mat, InstGen::sve_zr_t::z16);

      for (uint32_t i = 0; i < kloop_remainder; i++) {
        InstGen::sve_zr_t za = static_cast<InstGen::sve_zr_t>(InstGen::sve_zr_t::z16 + i);
        InstGen::sve_zr_t zb = static_cast<InstGen::sve_zr_t>(InstGen::sve_zr_t::z0 + i);
        if (trans_c) {
          kernel.add_instr(fmopa(0, prm, prn, za, zb));
        } else {
          kernel.add_instr(fmopa(0, prn, prm, zb, za));
        }
      }
    }
  }



  if (trans_c) {
    generate_matrix_predicated_store_za_m16_n16(kernel, gpr_c, gpr_ldc, prn, ms_count, 0);
  } else {
    generate_matrix_predicated_store_za_m16_n16(kernel, gpr_c, gpr_ldc, prm, ns_count, 0);
  }
}





















gemm_microkernel_desc_t select_gemm_microkernel(mini_jit::Kernel& kernel, std::string const& label_prefix, mat_tiled_rect_t tiled_rect, uint32_t k, uint32_t trans_a, uint32_t trans_b, uint32_t trans_c, mini_jit::Gemm::dtype_t dtype) {
  uint32_t dtype_size = 4;

  if (tiled_rect.mt == 32 && tiled_rect.nt == 32) {
    gemm_microkernel_desc_t result;
    result.mt = 32;
    result.nt = 32;
    result.stack_mem_size = 0;
    result.generator = generate_gemm_microkernel_m32_n32;
    return result;
  }

  if (tiled_rect.mt <= 16 && tiled_rect.nt <= 16) {
    gemm_microkernel_desc_t result;
    result.mt = tiled_rect.mt;
    result.nt = tiled_rect.nt;
    result.stack_mem_size = 16 * 16 * dtype_size;
    result.generator = generate_gemm_microkernel_predicated_m16_n16;
    return result;
  }

  throw mini_jit::Gemm::error_t::error;
}

void generate_gemm_microkernel(mini_jit::Kernel& kernel, std::string const& label_prefix, mat_tiled_rect_t tiled_rect, uint32_t k, uint32_t trans_a, uint32_t trans_b, uint32_t trans_c, mini_jit::Gemm::dtype_t dtype) {
  gemm_microkernel_desc_t desc = select_gemm_microkernel(kernel, label_prefix, tiled_rect, k, trans_a, trans_b, trans_c, dtype);
  desc.generator(kernel, label_prefix, tiled_rect, k, trans_a, trans_b, trans_c, dtype);
}

void generate_gemm_microkernel_loop(mini_jit::Kernel& kernel, std::string const& label_prefix, mat_tiled_rect_t tiled_rect, uint32_t k, uint32_t trans_a, uint32_t trans_b, uint32_t trans_c, mini_jit::Gemm::dtype_t dtype) {
  if (mat_rect_is_empty(tiled_rect.rect)) {
    return;
  }

  uint32_t mbegin = tiled_rect.rect.mbegin;
  uint32_t nbegin = tiled_rect.rect.nbegin;
  uint32_t mend = tiled_rect.rect.mend;
  uint32_t nend = tiled_rect.rect.nend;

  uint32_t mt = tiled_rect.mt;
  uint32_t nt = tiled_rect.nt;

  uint32_t n_loop_iters_count = (nend - nbegin) / nt;
  uint32_t m_loop_iters_count = (mend - mbegin) / mt;
  if (n_loop_iters_count <= 0 || m_loop_iters_count <= 0) {
    return;
  }

  InstGen ig;
  InstGen::gpr_t gpr_tmp1 = InstGen::gpr_t::x7;

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
    kernel.add_instr(ig.base_movz(gpr_a_sm, mt));
    kernel.add_instr(ig.base_mul(gpr_a_sm, gpr_lda, gpr_a_sm));
  } else {
    kernel.add_instr(ig.base_movz(gpr_a_sm, mt));
  }

  if (trans_b) {
    kernel.add_instr(ig.base_movz(gpr_b_sn, nt));
  } else {
    kernel.add_instr(ig.base_movz(gpr_b_sn, nt));
    kernel.add_instr(ig.base_mul(gpr_b_sn, gpr_ldb, gpr_b_sn));
  }

  if (trans_c) {
    kernel.add_instr(ig.base_movz(gpr_c_sn, nt));
    kernel.add_instr(ig.base_movz(gpr_c_sm, mt));
    kernel.add_instr(ig.base_mul(gpr_c_sm, gpr_ldc, gpr_c_sm));
  } else {
    kernel.add_instr(ig.base_movz(gpr_c_sm, mt));
    kernel.add_instr(ig.base_movz(gpr_tmp1, nt));
    kernel.add_instr(ig.base_mul(gpr_c_sn, gpr_ldc, gpr_tmp1));
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
  kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x7, InstGen::gpr_t::x3, InstGen::gpr_t::sp, 32, InstGen::addr_mode_t::signed_offset));
  kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x4, InstGen::gpr_t::x5, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::signed_offset));
  kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x6, InstGen::gpr_t::x7, InstGen::gpr_t::sp, 0, InstGen::addr_mode_t::signed_offset));
  generate_gemm_microkernel(kernel, label_prefix, tiled_rect, k, trans_a, trans_b, trans_c, dtype);

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






void generate_gemm_loops(mini_jit::Kernel& kernel, std::string const& label_prefix, std::vector<mat_tiled_rect_t> const& tiling, uint32_t k, uint32_t trans_a, uint32_t trans_b, uint32_t trans_c, mini_jit::Gemm::dtype_t dtype) {
  InstGen ig;

  kernel.add_instr(ig.base_stp(InstGen::gpr_t::x0, InstGen::gpr_t::x1, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
  kernel.add_instr(ig.base_stp(InstGen::gpr_t::x2, InstGen::gpr_t::x3, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
  kernel.add_instr(ig.base_stp(InstGen::gpr_t::x4, InstGen::gpr_t::x5, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));
  kernel.add_instr(ig.base_stp(InstGen::gpr_t::x6, InstGen::gpr_t::x7, InstGen::gpr_t::sp, -16, InstGen::addr_mode_t::pre_index));

  for (uint32_t i = 0; i < tiling.size(); i++) {
    mat_tiled_rect_t tiled_rect = tiling[i];
    if (mat_rect_is_empty(tiled_rect.rect)) {
      continue;
    }

    kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x6, InstGen::gpr_t::x7, InstGen::gpr_t::sp, 0, InstGen::addr_mode_t::signed_offset));
    kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x4, InstGen::gpr_t::x5, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::signed_offset));
    kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x2, InstGen::gpr_t::x3, InstGen::gpr_t::sp, 32, InstGen::addr_mode_t::signed_offset));
    kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x0, InstGen::gpr_t::x1, InstGen::gpr_t::sp, 48, InstGen::addr_mode_t::signed_offset));
    generate_gemm_microkernel_loop(kernel, label_prefix + "_" + std::to_string(i), tiled_rect, k, trans_a, trans_b, trans_c, dtype);
  }

  kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x6, InstGen::gpr_t::x7, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
  kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x4, InstGen::gpr_t::x5, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
  kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x2, InstGen::gpr_t::x3, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
  kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x0, InstGen::gpr_t::x1, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
}






uint32_t max_stack_mem_size_for_tiling(mini_jit::Kernel& kernel, std::string const& label_prefix, std::vector<mat_tiled_rect_t> const& tiling, uint32_t k, uint32_t trans_a, uint32_t trans_b, uint32_t trans_c, mini_jit::Gemm::dtype_t dtype) {
  uint32_t result = 0;
  for (mat_tiled_rect_t tiled_rect : tiling) {
    gemm_microkernel_desc_t desc = select_gemm_microkernel(kernel, label_prefix, tiled_rect, k, trans_a, trans_b, trans_c, dtype);
    result = std::max(result, desc.stack_mem_size);
  }
  return result;
}

void append_words(mini_jit::Kernel& kernel, std::vector<uint32_t> const& words, std::string const& label) {
  kernel.add_label(label);
  for (uint32_t word : words) {
    kernel.add_data(word);
  }
}


void generate_gemm(mini_jit::Kernel& kernel, std::string const& label_prefix, std::vector<mat_tiled_rect_t> const& tiling, uint32_t k, uint32_t trans_a, uint32_t trans_b, uint32_t trans_c, mini_jit::Gemm::dtype_t dtype) {
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
  // allocate requested stack memory
  uint32_t stack_mem_size = max_stack_mem_size_for_tiling(kernel, label_prefix, tiling, k, trans_a, trans_b, trans_c, dtype);
  if (stack_mem_size != 0) {
    kernel.add_instr(ig.base_sub(InstGen::gpr_t::sp, InstGen::gpr_t::sp, stack_mem_size));
    kernel.add_instr(ig.base_mov(InstGen::gpr_t::x6, InstGen::gpr_t::sp));
  }
  kernel.add_instr(ig.base_smstart());

  // setup expected register values
  kernel.add_instr(ig.ssve_ptrue(InstGen::pr_t::p0, sve_size_t::b));

  // main GEMM operation
  generate_gemm_loops(kernel, label_prefix, tiling, k, trans_a, trans_b, trans_c, dtype);

  // function epilogue
  kernel.add_instr(ig.base_smstop());
  // deallocate requested stack memory
  if (stack_mem_size != 0) {
    kernel.add_instr(ig.base_add(InstGen::gpr_t::sp, InstGen::gpr_t::sp, stack_mem_size));
  }
  // pop callee saved registers
  kernel.add_instr(ig.neon_ldp(InstGen::simd_fp_t::v14, InstGen::simd_fp_t::v15, InstGen::simd_sz_t::simd_d, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
  kernel.add_instr(ig.neon_ldp(InstGen::simd_fp_t::v12, InstGen::simd_fp_t::v13, InstGen::simd_sz_t::simd_d, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
  kernel.add_instr(ig.neon_ldp(InstGen::simd_fp_t::v10, InstGen::simd_fp_t::v11, InstGen::simd_sz_t::simd_d, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
  kernel.add_instr(ig.neon_ldp(InstGen::simd_fp_t::v8, InstGen::simd_fp_t::v9, InstGen::simd_sz_t::simd_d, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
  kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x27, InstGen::gpr_t::x28, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
  kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x25, InstGen::gpr_t::x26, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
  kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x23, InstGen::gpr_t::x24, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
  kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x21, InstGen::gpr_t::x22, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
  kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x19, InstGen::gpr_t::x20, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
  kernel.add_instr(ig.base_ldp(InstGen::gpr_t::x29, InstGen::gpr_t::x30, InstGen::gpr_t::sp, 16, InstGen::addr_mode_t::post_index));
  kernel.add_instr(ig.base_ret());

  // data for transpose algorithms
  append_words(kernel, {0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30}, "_trn1_2x2");
  append_words(kernel, {1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31}, "_trn2_2x2");
  append_words(kernel, {0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29}, "_trn1_4x4");
  append_words(kernel, {2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31}, "_trn2_4x4");
  append_words(kernel, {0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27}, "_trn1_8x8");
  append_words(kernel, {4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31}, "_trn2_8x8");
  append_words(kernel, {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23}, "_trn1_16x16");
  append_words(kernel, {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31}, "_trn2_16x16");
}









// new version
mini_jit::Gemm::error_t mini_jit::Gemm::generate( uint32_t m, uint32_t n, uint32_t k, uint32_t trans_a, uint32_t trans_b, uint32_t trans_c, dtype_t  dtype) {
  // compute tiling
  mat_rect_t root = {.mbegin = 0, .mend = m, .nbegin = 0, .nend = n};
  std::vector<mat_tiled_rect_t> tiling = tile_matrix_v2(root);

  // generate kernel for tiling
  try {
    generate_gemm(kernel, "", tiling, k, trans_a, trans_b, trans_c, dtype);
  } catch (mini_jit::Gemm::error_t err) {
    return err;
  }

  kernel.set_kernel();
  return error_t::success;
}































// Old version
namespace mini_jit {

    /**
     * @ief Generate a kernel for matrix multiplication.
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


    void Gemm::write(const char* fp) const {
      this->kernel.write(fp);
    }
};

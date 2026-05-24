#include "Unary.h"
#include "Kernel.h"
#include "InstGen.h"



using mini_jit::Unary;
using mini_jit::InstGen;
using sve_zr_t = mini_jit::InstGen::sve_zr_t;
using pr_t = mini_jit::InstGen::pr_t;
using sve_size_t = mini_jit::InstGen::sve_size_t;
using gpr_t = mini_jit::InstGen::gpr_t;
using shift_kind_t = mini_jit::InstGen::shift_kind_t;
using sme_hv_kind_t = mini_jit::InstGen::sme_hv_kind_t;





Unary::error_t Unary::generate_16x16_microkernel(Unary::ptype_t op, uint32_t trans_b) {
    // params:
    // x0 - ptr to a
    // x1 - ptr to b
    // x2 - lda
    // x3 - ldb

    // also used
    // p0 - row filter predicate
    // x6 - byte offset into A
    // x7 - byte offset into B
    // w12, w13 - ZA tile selector
    // z0 - z7 - temporary storage for data of A
    // za0.s - temporary storage for data of A and B

    InstGen ig;

    // load 16x16 matrix
    if (op == ptype_t::identity) {
        kernel.add_instr(ig.base_movz(gpr_t::x6, 0));
        kernel.add_instr(ig.base_movz(gpr_t::w12, 0));
        for (uint32_t c = 0; c < 16; c++) {
            kernel.add_instr(ig.sme_ld1w(0, sme_hv_kind_t::horz, gpr_t::w12, 0, pr_t::p0, gpr_t::x0, gpr_t::x6));
            kernel.add_instr(ig.base_add(gpr_t::x6, gpr_t::x6, gpr_t::x2));
            kernel.add_instr(ig.base_add(gpr_t::w12, gpr_t::w12, 1));
        }
    } else if (op == ptype_t::relu) {
        kernel.add_instr(ig.base_movz(gpr_t::x6, 0));
        kernel.add_instr(ig.base_movz(gpr_t::w12, 0));
        for (uint32_t c = 0; c < 16; c++) {
            sve_zr_t zr = (sve_zr_t)((sve_zr_t::z0 + c) % 8);
            kernel.add_instr(ig.sve_ld1w(zr, pr_t::p0, gpr_t::x0, gpr_t::x6));
            kernel.add_instr(ig.base_add(gpr_t::x6, gpr_t::x6, gpr_t::x2));
            kernel.add_instr(ig.sve_fmax(zr, sve_size_t::s, pr_t::p0, 0));
            kernel.add_instr(ig.sme_mov_s(0, sme_hv_kind_t::horz, gpr_t::w12, 0, pr_t::p0, zr));
            kernel.add_instr(ig.base_add(gpr_t::w12, gpr_t::w12, 1));
        }
    }

    // store 16x16 matrix
    kernel.add_instr(ig.base_movz(gpr_t::x7, 0));
    kernel.add_instr(ig.base_movz(gpr_t::w13, 0));
    for (uint32_t i = 0; i < 16; i++) {
        kernel.add_instr(ig.sme_st1w(
            0,
            trans_b ? sme_hv_kind_t::vert : sme_hv_kind_t::horz,
            gpr_t::w13, 0,
            pr_t::p0, 
            gpr_t::x1,
            gpr_t::x7
        ));
        kernel.add_instr(ig.base_add(gpr_t::x7, gpr_t::x7, gpr_t::x3));
        kernel.add_instr(ig.base_add(gpr_t::w13, gpr_t::w13, 1));
    }

    return Unary::error_t::success;
}

Unary::error_t Unary::generate( uint32_t m, uint32_t n, uint32_t trans_b, dtype_t  dtype, ptype_t  ptype ) {
    // x0 - ptr to a (passed to microkernel)
    // x1 - ptr to b (passed to microkernel)
    // x2 - lda
    // x3 - ldb
    // x4 - ptr to a (used in loop)
    // x5 - ptr to b (used in loop)
    // x9 - n loop counter
    // x10 - m loop counter
    
    // pr0 - set all to true

    InstGen ig;
    error_t err = error_t::success;

    // Make sure the input dimensions are a multiple of the streaming scalable vector length.
    if (!(m % 16 == 0 && n % 16 == 0)) {
        err = error_t::unsupported_parameters;
        return err;
    }

    // Make sure the input dimensions fit into the 16 bits provided by movz.
    if (!( ((m / 16) & ~0xffff) == 0 )) {
        err = error_t::unsupported_parameters;
        return err;
    }
    if (!( ((n / 16) & ~0xffff) == 0 )) {
        err = error_t::unsupported_parameters;
        return err;
    }

    kernel.add_instr(ig.base_smstart());
    kernel.add_instr(ig.ssve_ptrue(pr_t::p0, sve_size_t::s));
    kernel.add_instr(ig.base_mov(gpr_t::x4, gpr_t::x0));
    kernel.add_instr(ig.base_mov(gpr_t::x5, gpr_t::x1));
    if (ptype == ptype_t::zero) {
        kernel.add_instr(ig.sme_zero(~0));
    }

    kernel.add_instr(ig.base_movz(gpr_t::x9, n / 16));          // This instruction only allows 16 bits in its immediate.
    kernel.add_label("loop01_start");
    kernel.add_branch(ig.base_cbz(gpr_t::x9, "loop01_end"));
    kernel.add_instr(ig.base_mov(gpr_t::x0, gpr_t::x4));
    kernel.add_instr(ig.base_mov(gpr_t::x1, gpr_t::x5));

    kernel.add_instr(ig.base_movz(gpr_t::x10, m / 16));          // This instruction only allows 16 bits in its immediate.
    kernel.add_label("loop02_start");
    kernel.add_branch(ig.base_cbz(gpr_t::x10, "loop02_end"));

    err = generate_16x16_microkernel(ptype, trans_b);
    if (err != error_t::success) {
        return err;
    }
    
    if (trans_b) {
        kernel.add_instr(ig.base_add(gpr_t::x1, gpr_t::x1, gpr_t::x3, shift_kind_t::lsl, 6));
    } else {
        kernel.add_instr(ig.base_add(gpr_t::x1, gpr_t::x1, 64));
    }
    kernel.add_instr(ig.base_add(gpr_t::x0, gpr_t::x0, 64));
    kernel.add_instr(ig.base_sub(gpr_t::x10, gpr_t::x10, 1));
    kernel.add_branch(ig.base_b("loop02_start"));
    kernel.add_label("loop02_end");

    if (trans_b) {
        kernel.add_instr(ig.base_add(gpr_t::x5, gpr_t::x5, 64));
    } else {
        kernel.add_instr(ig.base_add(gpr_t::x5, gpr_t::x5, gpr_t::x3, shift_kind_t::lsl, 6));
    }
    kernel.add_instr(ig.base_add(gpr_t::x4, gpr_t::x4, gpr_t::x2, shift_kind_t::lsl, 6));
    kernel.add_instr(ig.base_sub(gpr_t::x9, gpr_t::x9, 1));
    kernel.add_branch(ig.base_b("loop01_start"));
    kernel.add_label("loop01_end");

    kernel.add_instr(ig.base_smstop());
    kernel.add_instr(ig.base_ret());
    kernel.set_kernel();
    return Unary::error_t::success;
}



Unary::kernel_t Unary::get_kernel() const {
    return (kernel_t)kernel.get_kernel();
}

void Unary::write(const char* fp) const {
    kernel.write(fp);
}


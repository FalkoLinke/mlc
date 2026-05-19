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





Unary::error_t Unary::generate_identity_notrans_fp32(uint32_t m, uint32_t n) {
    InstGen ig;

    // The register stack to keep track of the registers we load values into.
    sve_zr_t regs[] = { 
        sve_zr_t::z0, 
        sve_zr_t::z1, 
        sve_zr_t::z2, 
        sve_zr_t::z3, 
        sve_zr_t::z4, 
        sve_zr_t::z5, 
        sve_zr_t::z6, 
        sve_zr_t::z7,  
    };
    uint32_t regs_count = sizeof(regs) / sizeof(sve_zr_t);
    uint32_t regs_top = 0;

    kernel.add_instr(ig.base_smstart());

    // setup predicate register
    kernel.add_instr(ig.ssve_ptrue(pr_t::p0, sve_size_t::s));

    uint32_t lrow = 0;
    uint32_t lcol = 0;
    uint32_t srow = 0;
    uint32_t scol = 0;
    uint32_t remaining_values = m * n;
    while (remaining_values > 0) {
        // fetch incoming values and store them in the registers on the JIT-stack
        while (remaining_values > 0 && regs_top < regs_count) {
            sve_zr_t reg = regs[regs_top];
            regs_top += 1;

            kernel.add_instr(ig.sve_ld1w(reg, pr_t::p0, gpr_t::x0, lrow / 16));
            remaining_values -= 16;

            lrow += 16;
            if (lrow >= m) {
                lcol += 1;
                lrow = 0;
                kernel.add_instr(ig.base_add(gpr_t::x0, gpr_t::x0, gpr_t::x2, shift_kind_t::lsl, 2));
            }
        }

        // store the filled registers on the JIT-stack in the proper order
        for (uint32_t i = 0; i < regs_top; i++) {
            sve_zr_t reg = regs[i];

            kernel.add_instr(ig.sve_st1w(reg, sve_size_t::s, pr_t::p0, gpr_t::x1, srow / 16));

            srow += 16;
            if (srow >= m) {
                scol += 1;
                srow = 0;
                kernel.add_instr(ig.base_add(gpr_t::x1, gpr_t::x1, gpr_t::x3, shift_kind_t::lsl, 2));
            }
        }
        regs_top = 0;
    }
    
    kernel.add_instr(ig.base_smstop());
    kernel.add_instr(ig.base_ret());
    
    kernel.set_kernel();

    return Unary::error_t::success;
}

Unary::error_t Unary::generate_16x16_microkernel(Unary::ptype_t op, uint32_t trans_b) {
    // params:
    // x0 - ptr to a
    // x1 - ptr to b
    // x2 - lda
    // x3 - ldb
    // p0 - row filter predicate

    // also used
    // w12, w13, x6, x7

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

Unary::error_t Unary::generate_identity_fp32(uint32_t m, uint32_t n, uint32_t trans_b) {
    // x0 - ptr to a
    // x1 - ptr to b
    // x2 - lda
    // x3 - ldb
    // x4 - ptr to a
    // x5 - ptr to b
    // x9 - n loop counter
    // x10 - m loop counter
    
    // pr0 - set all to true

    InstGen ig;
    kernel.add_instr(ig.base_smstart());
    kernel.add_instr(ig.ssve_ptrue(pr_t::p0, sve_size_t::s));
    kernel.add_instr(ig.base_mov(gpr_t::x4, gpr_t::x0));
    kernel.add_instr(ig.base_mov(gpr_t::x5, gpr_t::x1));

    kernel.add_instr(ig.base_movz(gpr_t::x9, n / 16));
    kernel.add_label("loop01_start");
    kernel.add_branch(ig.base_cbz(gpr_t::x9, "loop01_end"));
    kernel.add_instr(ig.base_mov(gpr_t::x0, gpr_t::x4));
    kernel.add_instr(ig.base_mov(gpr_t::x1, gpr_t::x5));

    kernel.add_instr(ig.base_movz(gpr_t::x10, m / 16));
    kernel.add_label("loop02_start");
    kernel.add_branch(ig.base_cbz(gpr_t::x10, "loop02_end"));

    generate_16x16_microkernel(ptype_t::identity, trans_b);
    
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

Unary::error_t Unary::generate( uint32_t m, uint32_t n, uint32_t trans_b, dtype_t  dtype, ptype_t  ptype ) {
    // x0 - ptr to a
    // x1 - ptr to b
    // x2 - lda
    // x3 - ldb
    // x4 - ptr to a
    // x5 - ptr to b
    // x9 - n loop counter
    // x10 - m loop counter
    
    // pr0 - set all to true

    InstGen ig;
    kernel.add_instr(ig.base_smstart());
    kernel.add_instr(ig.ssve_ptrue(pr_t::p0, sve_size_t::s));
    kernel.add_instr(ig.base_mov(gpr_t::x4, gpr_t::x0));
    kernel.add_instr(ig.base_mov(gpr_t::x5, gpr_t::x1));
    if (ptype == ptype_t::zero) {
        kernel.add_instr(ig.sme_zero(~0));
    }

    kernel.add_instr(ig.base_movz(gpr_t::x9, n / 16));
    kernel.add_label("loop01_start");
    kernel.add_branch(ig.base_cbz(gpr_t::x9, "loop01_end"));
    kernel.add_instr(ig.base_mov(gpr_t::x0, gpr_t::x4));
    kernel.add_instr(ig.base_mov(gpr_t::x1, gpr_t::x5));

    kernel.add_instr(ig.base_movz(gpr_t::x10, m / 16));
    kernel.add_label("loop02_start");
    kernel.add_branch(ig.base_cbz(gpr_t::x10, "loop02_end"));

    generate_16x16_microkernel(ptype, trans_b);
    
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


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

Unary::error_t Unary::generate( uint32_t m, uint32_t n, uint32_t trans_b, dtype_t  dtype, ptype_t  ptype ) {
    if (ptype == Unary::ptype_t::identity) {
        if (trans_b) {
            return Unary::error_t::unsupported_args;
        } else {
            return generate_identity_notrans_fp32(m, n);
        }
    }

    InstGen ig;
    kernel.add_instr(ig.base_ret());
    kernel.set_kernel();

    return Unary::error_t::unsupported_args;
}



Unary::kernel_t Unary::get_kernel() const {
    return (kernel_t)kernel.get_kernel();
}

void Unary::write(const char* fp) const {
    kernel.write(fp);
}


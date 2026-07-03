#include "KernelFactory.h"
#include "InstGen.h"

using mini_jit::KernelFactory;
using mini_jit::Kernel;
using mini_jit::InstGen;

using gpr_t = InstGen::gpr_t;
using pr_t = InstGen::pr_t;
using sve_zr_t = InstGen::sve_zr_t;
using sve_size_t = InstGen::sve_size_t;
using addr_mode_t = InstGen::addr_mode_t;
using shift_kind_t = InstGen::shift_kind_t;
using sme_hv_kind_t = InstGen::sme_hv_kind_t;

KernelFactory::IdentityKernel KernelFactory::generate_identity_16_16(Kernel& kernel) {
    InstGen ig;
    
    kernel.add_instr(ig.base_stp(gpr_t::x29, gpr_t::x30, gpr_t::sp, -16, addr_mode_t::pre_index));
    kernel.add_instr(ig.base_mov(gpr_t::x29, gpr_t::sp));
    kernel.add_instr(ig.base_smstart());

    kernel.add_labeled_instr(ig.base_cbz(gpr_t::x4, "notrans"));
    kernel.add_labeled_instr(ig.base_b("trans"));

    kernel.add_label("notrans");

    kernel.add_instr(ig.ssve_ptrue(pr_t::p0, sve_size_t::s));
    kernel.add_instr(ig.base_movz(gpr_t::x5, 16));
    kernel.add_label("loop01");
    kernel.add_labeled_instr(ig.base_cbz(gpr_t::x5, "end01"));

    kernel.add_instr(ig.sve_ld1w(sve_zr_t::z0, pr_t::p0, gpr_t::x0, 0));
    kernel.add_instr(ig.sve_st1w(sve_zr_t::z0, sve_size_t::s, pr_t::p0, gpr_t::x1, 0));
    
    kernel.add_instr(ig.base_add(gpr_t::x0, gpr_t::x0, gpr_t::x2, shift_kind_t::lsl, 2));
    kernel.add_instr(ig.base_add(gpr_t::x1, gpr_t::x1, gpr_t::x3, shift_kind_t::lsl, 2));
    kernel.add_instr(ig.base_sub(gpr_t::x5, gpr_t::x5, 1, 1));
    kernel.add_labeled_instr(ig.base_b("loop01"));
    kernel.add_label("end01");
    kernel.add_labeled_instr(ig.base_b("ret"));


    kernel.add_label("trans");

    kernel.add_instr(ig.ssve_ptrue(pr_t::p0, sve_size_t::s));
    kernel.add_instr(ig.base_movz(gpr_t::w12, 0));
    kernel.add_instr(ig.base_movz(gpr_t::x5, 4));
    kernel.add_label("loop02");
    kernel.add_labeled_instr(ig.base_cbz(gpr_t::x5, "end02"));

    kernel.add_instr(ig.sme_ld1w(0, sme_hv_kind_t::horz, gpr_t::w12, 0, pr_t::p0, gpr_t::x0, gpr_t::xzr));
    kernel.add_instr(ig.base_add(gpr_t::x0, gpr_t::x0, gpr_t::x2, shift_kind_t::lsl, 2));
    kernel.add_instr(ig.sme_ld1w(0, sme_hv_kind_t::horz, gpr_t::w12, 1, pr_t::p0, gpr_t::x0, gpr_t::xzr));
    kernel.add_instr(ig.base_add(gpr_t::x0, gpr_t::x0, gpr_t::x2, shift_kind_t::lsl, 2));
    kernel.add_instr(ig.sme_ld1w(0, sme_hv_kind_t::horz, gpr_t::w12, 2, pr_t::p0, gpr_t::x0, gpr_t::xzr));
    kernel.add_instr(ig.base_add(gpr_t::x0, gpr_t::x0, gpr_t::x2, shift_kind_t::lsl, 2));
    kernel.add_instr(ig.sme_ld1w(0, sme_hv_kind_t::horz, gpr_t::w12, 3, pr_t::p0, gpr_t::x0, gpr_t::xzr));
    kernel.add_instr(ig.base_add(gpr_t::x0, gpr_t::x0, gpr_t::x2, shift_kind_t::lsl, 2));

    kernel.add_instr(ig.base_add(gpr_t::w12, gpr_t::w12, 4));
    kernel.add_instr(ig.base_sub(gpr_t::x5, gpr_t::x5, 1));
    kernel.add_labeled_instr(ig.base_b("loop02"));
    kernel.add_label("end02");

    kernel.add_instr(ig.base_movz(gpr_t::w12, 0));
    kernel.add_instr(ig.base_movz(gpr_t::x5, 4));
    kernel.add_label("loop03");
    kernel.add_labeled_instr(ig.base_cbz(gpr_t::x5, "end03"));

    kernel.add_instr(ig.sme_st1w(0, sme_hv_kind_t::vert, gpr_t::w12, 0, pr_t::p0, gpr_t::x1, gpr_t::xzr));
    kernel.add_instr(ig.base_add(gpr_t::x1, gpr_t::x1, gpr_t::x3, shift_kind_t::lsl, 2));
    kernel.add_instr(ig.sme_st1w(0, sme_hv_kind_t::vert, gpr_t::w12, 1, pr_t::p0, gpr_t::x1, gpr_t::xzr));
    kernel.add_instr(ig.base_add(gpr_t::x1, gpr_t::x1, gpr_t::x3, shift_kind_t::lsl, 2));
    kernel.add_instr(ig.sme_st1w(0, sme_hv_kind_t::vert, gpr_t::w12, 2, pr_t::p0, gpr_t::x1, gpr_t::xzr));
    kernel.add_instr(ig.base_add(gpr_t::x1, gpr_t::x1, gpr_t::x3, shift_kind_t::lsl, 2));
    kernel.add_instr(ig.sme_st1w(0, sme_hv_kind_t::vert, gpr_t::w12, 3, pr_t::p0, gpr_t::x1, gpr_t::xzr));
    kernel.add_instr(ig.base_add(gpr_t::x1, gpr_t::x1, gpr_t::x3, shift_kind_t::lsl, 2));

    kernel.add_instr(ig.base_add(gpr_t::w12, gpr_t::w12, 4));
    kernel.add_instr(ig.base_sub(gpr_t::x5, gpr_t::x5, 1));
    kernel.add_labeled_instr(ig.base_b("loop03"));
    kernel.add_label("end03");

    kernel.add_label("ret");
    kernel.add_instr(ig.base_smstop());
    kernel.add_instr(ig.base_ldp(gpr_t::x29, gpr_t::x30, gpr_t::sp, 16, addr_mode_t::post_index));
    kernel.add_instr(ig.base_ret());

    kernel.set_kernel();

    IdentityKernel result = (IdentityKernel)kernel.get_kernel();
    return result;
}




KernelFactory::ZeroKernel KernelFactory::generate_zero_16_16(Kernel& kernel) {
    InstGen ig;

    kernel.add_instr(ig.base_smstart());
    kernel.add_instr(ig.sme_zero(~0));
    kernel.add_instr(ig.base_movz(gpr_t::w12, 0));
    kernel.add_instr(ig.ssve_ptrue(pr_t::p0, sve_size_t::s));

    kernel.add_instr(ig.base_movz(gpr_t::x2, 16));
    kernel.add_label("loop01");
    kernel.add_labeled_instr(ig.base_cbz(gpr_t::x2, "end01"));

    kernel.add_instr(ig.sme_st1w(0, InstGen::sme_hv_kind_t::horz, gpr_t::w12, 0, pr_t::p0, gpr_t::x0, gpr_t::xzr));

    kernel.add_instr(ig.base_add(gpr_t::x0, gpr_t::x0, gpr_t::x1, InstGen::shift_kind_t::lsl, 2));
    kernel.add_instr(ig.base_sub(gpr_t::x2, gpr_t::x2, 1));
    kernel.add_labeled_instr(ig.base_b("loop01"));
    kernel.add_label("end01");

    kernel.add_instr(ig.base_smstop());
    kernel.add_instr(ig.base_ret());

    kernel.set_kernel();
    ZeroKernel result = (ZeroKernel)kernel.get_kernel();
    return result;
}


KernelFactory::ReluKernel KernelFactory::generate_relu_16_16(Kernel& kernel) {
    InstGen ig;
    
    kernel.add_instr(ig.base_stp(gpr_t::x29, gpr_t::x30, gpr_t::sp, -16, addr_mode_t::pre_index));
    kernel.add_instr(ig.base_mov(gpr_t::x29, gpr_t::sp));
    kernel.add_instr(ig.base_smstart());

    kernel.add_labeled_instr(ig.base_cbz(gpr_t::x4, "notrans"));
    kernel.add_labeled_instr(ig.base_b("trans"));

    kernel.add_label("notrans");

    kernel.add_instr(ig.ssve_ptrue(pr_t::p0, sve_size_t::s));
    kernel.add_instr(ig.base_movz(gpr_t::x5, 16));
    kernel.add_label("loop01");
    kernel.add_labeled_instr(ig.base_cbz(gpr_t::x5, "end01"));

    kernel.add_instr(ig.sve_ld1w(sve_zr_t::z0, pr_t::p0, gpr_t::x0, 0));
    kernel.add_instr(ig.sve_fmax(sve_zr_t::z0, sve_size_t::s, pr_t::p0, 0));
    kernel.add_instr(ig.sve_st1w(sve_zr_t::z0, sve_size_t::s, pr_t::p0, gpr_t::x1, 0));
    
    kernel.add_instr(ig.base_add(gpr_t::x0, gpr_t::x0, gpr_t::x2, shift_kind_t::lsl, 2));
    kernel.add_instr(ig.base_add(gpr_t::x1, gpr_t::x1, gpr_t::x3, shift_kind_t::lsl, 2));
    kernel.add_instr(ig.base_sub(gpr_t::x5, gpr_t::x5, 1, 1));
    kernel.add_labeled_instr(ig.base_b("loop01"));
    kernel.add_label("end01");
    kernel.add_labeled_instr(ig.base_b("ret"));


    kernel.add_label("trans");

    kernel.add_instr(ig.ssve_ptrue(pr_t::p0, sve_size_t::s));
    kernel.add_instr(ig.base_movz(gpr_t::w12, 0));
    kernel.add_instr(ig.base_movz(gpr_t::x5, 4));
    kernel.add_label("loop02");
    kernel.add_labeled_instr(ig.base_cbz(gpr_t::x5, "end02"));

    kernel.add_instr(ig.sve_ld1w(sve_zr_t::z0, pr_t::p0, gpr_t::x0, 0));
    kernel.add_instr(ig.base_add(gpr_t::x0, gpr_t::x0, gpr_t::x2, shift_kind_t::lsl, 2));
    kernel.add_instr(ig.sve_fmax(sve_zr_t::z0, sve_size_t::s, pr_t::p0, 0));
    kernel.add_instr(ig.sme_mov_s(0, sme_hv_kind_t::horz, gpr_t::w12, 0, pr_t::p0, sve_zr_t::z0));

    kernel.add_instr(ig.sve_ld1w(sve_zr_t::z1, pr_t::p0, gpr_t::x0, 0));
    kernel.add_instr(ig.base_add(gpr_t::x0, gpr_t::x0, gpr_t::x2, shift_kind_t::lsl, 2));
    kernel.add_instr(ig.sve_fmax(sve_zr_t::z1, sve_size_t::s, pr_t::p0, 0));
    kernel.add_instr(ig.sme_mov_s(0, sme_hv_kind_t::horz, gpr_t::w12, 1, pr_t::p0, sve_zr_t::z1));

    kernel.add_instr(ig.sve_ld1w(sve_zr_t::z2, pr_t::p0, gpr_t::x0, 0));
    kernel.add_instr(ig.base_add(gpr_t::x0, gpr_t::x0, gpr_t::x2, shift_kind_t::lsl, 2));
    kernel.add_instr(ig.sve_fmax(sve_zr_t::z2, sve_size_t::s, pr_t::p0, 0));
    kernel.add_instr(ig.sme_mov_s(0, sme_hv_kind_t::horz, gpr_t::w12, 2, pr_t::p0, sve_zr_t::z2));

    kernel.add_instr(ig.sve_ld1w(sve_zr_t::z3, pr_t::p0, gpr_t::x0, 0));
    kernel.add_instr(ig.base_add(gpr_t::x0, gpr_t::x0, gpr_t::x2, shift_kind_t::lsl, 2));
    kernel.add_instr(ig.sve_fmax(sve_zr_t::z3, sve_size_t::s, pr_t::p0, 0));
    kernel.add_instr(ig.sme_mov_s(0, sme_hv_kind_t::horz, gpr_t::w12, 3, pr_t::p0, sve_zr_t::z3));

    kernel.add_instr(ig.base_add(gpr_t::w12, gpr_t::w12, 4));
    kernel.add_instr(ig.base_sub(gpr_t::x5, gpr_t::x5, 1));
    kernel.add_labeled_instr(ig.base_b("loop02"));
    kernel.add_label("end02");

    kernel.add_instr(ig.base_movz(gpr_t::w12, 0));
    kernel.add_instr(ig.base_movz(gpr_t::x5, 4));
    kernel.add_label("loop03");
    kernel.add_labeled_instr(ig.base_cbz(gpr_t::x5, "end03"));

    kernel.add_instr(ig.sme_st1w(0, sme_hv_kind_t::vert, gpr_t::w12, 0, pr_t::p0, gpr_t::x1, gpr_t::xzr));
    kernel.add_instr(ig.base_add(gpr_t::x1, gpr_t::x1, gpr_t::x3, shift_kind_t::lsl, 2));
    kernel.add_instr(ig.sme_st1w(0, sme_hv_kind_t::vert, gpr_t::w12, 1, pr_t::p0, gpr_t::x1, gpr_t::xzr));
    kernel.add_instr(ig.base_add(gpr_t::x1, gpr_t::x1, gpr_t::x3, shift_kind_t::lsl, 2));
    kernel.add_instr(ig.sme_st1w(0, sme_hv_kind_t::vert, gpr_t::w12, 2, pr_t::p0, gpr_t::x1, gpr_t::xzr));
    kernel.add_instr(ig.base_add(gpr_t::x1, gpr_t::x1, gpr_t::x3, shift_kind_t::lsl, 2));
    kernel.add_instr(ig.sme_st1w(0, sme_hv_kind_t::vert, gpr_t::w12, 3, pr_t::p0, gpr_t::x1, gpr_t::xzr));
    kernel.add_instr(ig.base_add(gpr_t::x1, gpr_t::x1, gpr_t::x3, shift_kind_t::lsl, 2));

    kernel.add_instr(ig.base_add(gpr_t::w12, gpr_t::w12, 4));
    kernel.add_instr(ig.base_sub(gpr_t::x5, gpr_t::x5, 1));
    kernel.add_labeled_instr(ig.base_b("loop03"));
    kernel.add_label("end03");

    kernel.add_label("ret");
    kernel.add_instr(ig.base_smstop());
    kernel.add_instr(ig.base_ldp(gpr_t::x29, gpr_t::x30, gpr_t::sp, 16, addr_mode_t::post_index));
    kernel.add_instr(ig.base_ret());



    kernel.set_kernel();
    ReluKernel result = (ReluKernel)kernel.get_kernel();
    return result;
}
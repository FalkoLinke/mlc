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
    /*
    kernel.add_instr( 0xa9bf7bfd );
    kernel.add_instr( 0x910003fd );
    kernel.add_instr( 0xd503477f );
    kernel.add_instr( 0xb4000044 );
    kernel.add_instr( 0x1400000b );
    kernel.add_instr( 0x2598e3e0 );
    kernel.add_instr( 0xd2800205 );
    kernel.add_instr( 0xb40000e5 );
    kernel.add_instr( 0xa540a000 );
    kernel.add_instr( 0xe540e020 );
    kernel.add_instr( 0x8b020800 );
    kernel.add_instr( 0x8b030821 );
    kernel.add_instr( 0xf10004a5 );
    kernel.add_instr( 0x17fffffa );
    kernel.add_instr( 0x1400001e );
    kernel.add_instr( 0x2598e3e0 );
    kernel.add_instr( 0x5280000c );
    kernel.add_instr( 0xd2800085 );
    kernel.add_instr( 0xb4000185 );
    kernel.add_instr( 0xe09f0000 );
    kernel.add_instr( 0x8b020800 );
    kernel.add_instr( 0xe09f0001 );
    kernel.add_instr( 0x8b020800 );
    kernel.add_instr( 0xe09f0002 );
    kernel.add_instr( 0x8b020800 );
    kernel.add_instr( 0xe09f0003 );
    kernel.add_instr( 0x8b020800 );
    kernel.add_instr( 0x1100118c );
    kernel.add_instr( 0xf10004a5 );
    kernel.add_instr( 0x17fffff5 );
    kernel.add_instr( 0x5280000c );
    kernel.add_instr( 0xd2800085 );
    kernel.add_instr( 0xb4000185 );
    kernel.add_instr( 0xe0bf8020 );
    kernel.add_instr( 0x8b030821 );
    kernel.add_instr( 0xe0bf8021 );
    kernel.add_instr( 0x8b030821 );
    kernel.add_instr( 0xe0bf8022 );
    kernel.add_instr( 0x8b030821 );
    kernel.add_instr( 0xe0bf8023 );
    kernel.add_instr( 0x8b030821 );
    kernel.add_instr( 0x1100118c );
    kernel.add_instr( 0xf10004a5 );
    kernel.add_instr( 0x17fffff5 );
    kernel.add_instr( 0xd503467f );
    kernel.add_instr( 0xa8c17bfd );
    kernel.add_instr( 0xd65f03c0 );
    */

    InstGen ig;
    
    kernel.add_instr(ig.base_stp(gpr_t::x29, gpr_t::x30, gpr_t::sp, -16, addr_mode_t::pre_index));
    kernel.add_instr(ig.base_mov(gpr_t::x29, gpr_t::sp));
    kernel.add_instr(ig.base_smstart());

    kernel.add_branch(ig.base_cbz(gpr_t::x4, "notrans"));
    kernel.add_branch(ig.base_b("trans"));

    kernel.add_label("notrans");

    kernel.add_instr(ig.ssve_ptrue(pr_t::p0, sve_size_t::s));
    kernel.add_instr(ig.base_movz(gpr_t::x5, 16));
    kernel.add_label("loop01");
    kernel.add_branch(ig.base_cbz(gpr_t::x5, "end01"));

    kernel.add_instr(ig.sve_ld1w(sve_zr_t::z0, pr_t::p0, gpr_t::x0, 0));
    kernel.add_instr(ig.sve_st1w(sve_zr_t::z0, sve_size_t::s, pr_t::p0, gpr_t::x1, 0));
    
    kernel.add_instr(ig.base_add(gpr_t::x0, gpr_t::x0, gpr_t::x2, shift_kind_t::lsl, 2));
    kernel.add_instr(ig.base_add(gpr_t::x1, gpr_t::x1, gpr_t::x3, shift_kind_t::lsl, 2));
    kernel.add_instr(ig.base_sub(gpr_t::x5, gpr_t::x5, 1, 1));
    kernel.add_branch(ig.base_b("loop01"));
    kernel.add_label("end01");
    kernel.add_branch(ig.base_b("ret"));


    kernel.add_label("trans");

    kernel.add_instr(ig.ssve_ptrue(pr_t::p0, sve_size_t::s));
    kernel.add_instr(ig.base_movz(gpr_t::w12, 0));
    kernel.add_instr(ig.base_movz(gpr_t::x5, 4));
    kernel.add_label("loop02");
    kernel.add_branch(ig.base_cbz(gpr_t::x5, "end02"));

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
    kernel.add_branch(ig.base_b("loop02"));
    kernel.add_label("end02");

    kernel.add_instr(ig.base_movz(gpr_t::w12, 0));
    kernel.add_instr(ig.base_movz(gpr_t::x5, 4));
    kernel.add_label("loop03");
    kernel.add_branch(ig.base_cbz(gpr_t::x5, "end03"));

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
    kernel.add_branch(ig.base_b("loop03"));
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
    /*
    kernel.add_instr( 0xd503477f );
    kernel.add_instr( 0xc00800ff );
    kernel.add_instr( 0x5280000c );
    kernel.add_instr( 0x2598e3e0 );
    kernel.add_instr( 0xd2800202 );
    kernel.add_instr( 0xb40000a2 );
    kernel.add_instr( 0xe0bf0000 );
    kernel.add_instr( 0x8b010800 );
    kernel.add_instr( 0xf1000442 );
    kernel.add_instr( 0x17fffffc );
    kernel.add_instr( 0xd503467f );
    kernel.add_instr( 0xd65f03c0 );
    */

    InstGen ig;

    kernel.add_instr(ig.base_smstart());
    kernel.add_instr(ig.sme_zero(~0));
    kernel.add_instr(ig.base_movz(gpr_t::w12, 0));
    kernel.add_instr(ig.ssve_ptrue(pr_t::p0, sve_size_t::s));

    kernel.add_instr(ig.base_movz(gpr_t::x2, 16));
    kernel.add_label("loop01");
    kernel.add_branch(ig.base_cbz(gpr_t::x2, "end01"));

    kernel.add_instr(ig.sme_st1w(0, InstGen::sme_hv_kind_t::horz, gpr_t::w12, 0, pr_t::p0, gpr_t::x0, gpr_t::xzr));

    kernel.add_instr(ig.base_add(gpr_t::x0, gpr_t::x0, gpr_t::x1, InstGen::shift_kind_t::lsl, 2));
    kernel.add_instr(ig.base_sub(gpr_t::x2, gpr_t::x2, 1));
    kernel.add_branch(ig.base_b("loop01"));
    kernel.add_label("end01");

    kernel.add_instr(ig.base_smstop());
    kernel.add_instr(ig.base_ret());

    kernel.set_kernel();
    ZeroKernel result = (ZeroKernel)kernel.get_kernel();
    return result;
}


KernelFactory::ReluKernel KernelFactory::generate_relu_16_16(Kernel& kernel) {
    /*
    kernel.add_instr( 0xa9bf7bfd );
    kernel.add_instr( 0x910003fd );
    kernel.add_instr( 0xd503477f );
    kernel.add_instr( 0xb4000044 );
    kernel.add_instr( 0x1400000c );
    kernel.add_instr( 0x2598e3e0 );
    kernel.add_instr( 0xd2800205 );
    kernel.add_instr( 0xb4000105 );
    kernel.add_instr( 0xa540a000 );
    kernel.add_instr( 0x659e8000 );
    kernel.add_instr( 0xe540e020 );
    kernel.add_instr( 0x8b020800 );
    kernel.add_instr( 0x8b030821 );
    kernel.add_instr( 0xf10004a5 );
    kernel.add_instr( 0x17fffff9 );
    kernel.add_instr( 0x14000026 );
    kernel.add_instr( 0x2598e3e0 );
    kernel.add_instr( 0x5280000c );
    kernel.add_instr( 0xd2800085 );
    kernel.add_instr( 0xb4000285 );
    kernel.add_instr( 0xa540a000 );
    kernel.add_instr( 0x8b020800 );
    kernel.add_instr( 0x659e8000 );
    kernel.add_instr( 0xc0800000 );
    kernel.add_instr( 0xa540a001 );
    kernel.add_instr( 0x8b020800 );
    kernel.add_instr( 0x659e8001 );
    kernel.add_instr( 0xc0800021 );
    kernel.add_instr( 0xa540a002 );
    kernel.add_instr( 0x8b020800 );
    kernel.add_instr( 0x659e8002 );
    kernel.add_instr( 0xc0800042 );
    kernel.add_instr( 0xa540a003 );
    kernel.add_instr( 0x8b020800 );
    kernel.add_instr( 0x659e8003 );
    kernel.add_instr( 0xc0800063 );
    kernel.add_instr( 0x1100118c );
    kernel.add_instr( 0xf10004a5 );
    kernel.add_instr( 0x17ffffed );
    kernel.add_instr( 0x5280000c );
    kernel.add_instr( 0xd2800085 );
    kernel.add_instr( 0xb4000185 );
    kernel.add_instr( 0xe0bf8020 );
    kernel.add_instr( 0x8b030821 );
    kernel.add_instr( 0xe0bf8021 );
    kernel.add_instr( 0x8b030821 );
    kernel.add_instr( 0xe0bf8022 );
    kernel.add_instr( 0x8b030821 );
    kernel.add_instr( 0xe0bf8023 );
    kernel.add_instr( 0x8b030821 );
    kernel.add_instr( 0x1100118c );
    kernel.add_instr( 0xf10004a5 );
    kernel.add_instr( 0x17fffff5 );
    kernel.add_instr( 0xd503467f );
    kernel.add_instr( 0xa8c17bfd );
    kernel.add_instr( 0xd65f03c0 );
    */

    InstGen ig;
    
    kernel.add_instr(ig.base_stp(gpr_t::x29, gpr_t::x30, gpr_t::sp, -16, addr_mode_t::pre_index));
    kernel.add_instr(ig.base_mov(gpr_t::x29, gpr_t::sp));
    kernel.add_instr(ig.base_smstart());

    kernel.add_branch(ig.base_cbz(gpr_t::x4, "notrans"));
    kernel.add_branch(ig.base_b("trans"));

    kernel.add_label("notrans");

    kernel.add_instr(ig.ssve_ptrue(pr_t::p0, sve_size_t::s));
    kernel.add_instr(ig.base_movz(gpr_t::x5, 16));
    kernel.add_label("loop01");
    kernel.add_branch(ig.base_cbz(gpr_t::x5, "end01"));

    kernel.add_instr(ig.sve_ld1w(sve_zr_t::z0, pr_t::p0, gpr_t::x0, 0));
    kernel.add_instr(ig.sve_fmax(sve_zr_t::z0, sve_size_t::s, pr_t::p0, 0));
    kernel.add_instr(ig.sve_st1w(sve_zr_t::z0, sve_size_t::s, pr_t::p0, gpr_t::x1, 0));
    
    kernel.add_instr(ig.base_add(gpr_t::x0, gpr_t::x0, gpr_t::x2, shift_kind_t::lsl, 2));
    kernel.add_instr(ig.base_add(gpr_t::x1, gpr_t::x1, gpr_t::x3, shift_kind_t::lsl, 2));
    kernel.add_instr(ig.base_sub(gpr_t::x5, gpr_t::x5, 1, 1));
    kernel.add_branch(ig.base_b("loop01"));
    kernel.add_label("end01");
    kernel.add_branch(ig.base_b("ret"));


    kernel.add_label("trans");

    kernel.add_instr(ig.ssve_ptrue(pr_t::p0, sve_size_t::s));
    kernel.add_instr(ig.base_movz(gpr_t::w12, 0));
    kernel.add_instr(ig.base_movz(gpr_t::x5, 4));
    kernel.add_label("loop02");
    kernel.add_branch(ig.base_cbz(gpr_t::x5, "end02"));

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
    kernel.add_branch(ig.base_b("loop02"));
    kernel.add_label("end02");

    kernel.add_instr(ig.base_movz(gpr_t::w12, 0));
    kernel.add_instr(ig.base_movz(gpr_t::x5, 4));
    kernel.add_label("loop03");
    kernel.add_branch(ig.base_cbz(gpr_t::x5, "end03"));

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
    kernel.add_branch(ig.base_b("loop03"));
    kernel.add_label("end03");

    kernel.add_label("ret");
    kernel.add_instr(ig.base_smstop());
    kernel.add_instr(ig.base_ldp(gpr_t::x29, gpr_t::x30, gpr_t::sp, 16, addr_mode_t::post_index));
    kernel.add_instr(ig.base_ret());



    kernel.set_kernel();
    ReluKernel result = (ReluKernel)kernel.get_kernel();
    return result;
}
#include "Kernel.h"
#include "InstGen.h"
#include <iostream>

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

/**
 * Generates a GEMM kernel for 512x512x512 matrices.
 */
void generate_gemm_kernel_512_512_512(mini_jit::Kernel& kernel) {

    mini_jit::InstGen gen;

    auto reg_x = [](uint32_t id) { return static_cast<gpr_t>(id | 0x20); };
    auto reg_w = [](uint32_t id) { return static_cast<gpr_t>(id); };


    //kernel.add_instr(gen.base_smstart());
    kernel.add_instr(0xd503477f);

    kernel.add_instr(gen.ssve_ptrue(pr_t::p0, sve_size_t::s)); // ptrue p0.s

    kernel.add_instr(0x04BF5829); // rdsvl x9, #1
    kernel.add_instr(0xD342FD29); // lsr x9, x9, #2  (UBFM)
    
    kernel.add_instr(0xD37EF4AA); // lsl x10, x5, #2 (UBFM)
    kernel.add_instr(0xD37EF46B); // lsl x11, x3, #2 (UBFM)
    kernel.add_instr(0xD37EF491); // lsl x17, x4, #2 (UBFM)

    // N Variables loop counter in x16
    kernel.add_instr(gen.base_movz(reg_x(16), 0, 0)); // mov x16, #0

    // ---------------------- N_loop ----------------------
    kernel.add_label("N_loop");

    // M Variables loop counter in x15
    kernel.add_instr(gen.base_movz(reg_x(15), 0, 0)); // mov x15, #0
    // ---------------------- M_loop ----------------------
    kernel.add_label("M_loop");
    kernel.add_instr(gen.base_mov(reg_x(6), reg_x(2))); // mov x6, x2
    kernel.add_instr(gen.base_add(reg_x(6), reg_x(6), reg_x(15), shift_kind_t::lsl, 2, 0)); // add x6, x6, x15, lsl #2
    
    kernel.add_instr(0x9B057E0E); // mul x14, x16, x5 (MADD X14, X16, X5, XZR)
    
    kernel.add_instr(gen.base_add(reg_x(6), reg_x(6), reg_x(14), shift_kind_t::lsl, 2, 0)); // add x6, x6, x14, lsl #2
    kernel.add_instr(gen.base_movz(reg_w(12), 0, 0)); // mov w12, #0
    kernel.add_instr(gen.base_movz(reg_w(13), 2, 0)); // mov w13, #2

     // Load C (Teil 1) - .rept 8 
    for (int i = 0; i < 8; ++i) {
        kernel.add_instr(ldr_za(12, 0, 6));
        kernel.add_instr(ldr_za(12, 1, 6));
        kernel.add_instr(gen.base_add(reg_w(12), reg_w(12), 4, 0));
        kernel.add_instr(gen.base_add(reg_x(6), reg_x(6), reg_x(10), shift_kind_t::lsl, 0, 0));

        kernel.add_instr(ldr_za(12, 0, 6));
        kernel.add_instr(ldr_za(12, 1, 6));
        kernel.add_instr(gen.base_add(reg_w(12), reg_w(12), 4, 0));
        kernel.add_instr(gen.base_add(reg_x(6), reg_x(6), reg_x(10), shift_kind_t::lsl, 0, 0));
    }

    // Load C (Teil 2) - .rept 8
    for (int i = 0; i < 8; ++i) {
        kernel.add_instr(ldr_za(13, 0, 6));
        kernel.add_instr(ldr_za(13, 1, 6));
        kernel.add_instr(gen.base_add(reg_w(13), reg_w(13), 4, 0));
        kernel.add_instr(gen.base_add(reg_x(6), reg_x(6), reg_x(10), shift_kind_t::lsl, 0, 0));

        kernel.add_instr(ldr_za(13, 0, 6));
        kernel.add_instr(ldr_za(13, 1, 6));
        kernel.add_instr(gen.base_add(reg_w(13), reg_w(13), 4, 0));
        kernel.add_instr(gen.base_add(reg_x(6), reg_x(6), reg_x(10), shift_kind_t::lsl, 0, 0));
    }


    kernel.add_instr(gen.base_mov(reg_x(7), reg_x(0))); // mov x7, x0
    kernel.add_instr(gen.base_mov(reg_x(8), reg_x(1))); // mov x8, x1
    kernel.add_instr(gen.base_add(reg_x(7), reg_x(7), reg_x(15), shift_kind_t::lsl, 2, 0)); // add x7, x7, x15, lsl #2
    kernel.add_instr(gen.base_add(reg_x(8), reg_x(8), reg_x(16), shift_kind_t::lsl, 2, 0)); // add x8, x8, x16, lsl #2

    // K Variables loop counter in x29
    kernel.add_instr(gen.base_movz(reg_x(14), 512, 0)); // mov x14, #512

    // ---------------------- K_loop ----------------------
    kernel.add_label("K_loop");

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

    kernel.add_instr(gen.base_sub(reg_x(14), reg_x(14), 1, 1)); // subs x14, x14, #1
    kernel.add_branch(gen.base_b_cond("K_loop", InstGen::br_cond_t(1))); // AArch64 Bedingung: NE = 1; // b.ne N_loop

    // store the results back to C
    kernel.add_instr(gen.base_mov(reg_x(6), reg_x(2))); // mov x6, x2
    kernel.add_instr(gen.base_add(reg_x(6), reg_x(6), reg_x(15), shift_kind_t::lsl, 2, 0)); // add x6, x6, x15, lsl #2
    
    kernel.add_instr(0x9B057E0E); // mul x14, x16, x5
    
    kernel.add_instr(gen.base_add(reg_x(6), reg_x(6), reg_x(14), shift_kind_t::lsl, 2, 0)); // add x6, x6, x14, lsl #2
    kernel.add_instr(gen.base_movz(reg_w(12), 0, 0)); // mov w12, #0
    kernel.add_instr(gen.base_movz(reg_w(13), 2, 0)); // mov w13, #2

    // Store C (Teil 1) - .rept 8
    for (int i = 0; i < 8; ++i) {
        kernel.add_instr(str_za(12, 0, 6));
        kernel.add_instr(str_za(12, 1, 6));
        kernel.add_instr(gen.base_add(reg_w(12), reg_w(12), 4, 0));
        kernel.add_instr(gen.base_add(reg_x(6), reg_x(6), reg_x(10), shift_kind_t::lsl, 0, 0));

        kernel.add_instr(str_za(12, 0, 6));
        kernel.add_instr(str_za(12, 1, 6));
        kernel.add_instr(gen.base_add(reg_w(12), reg_w(12), 4, 0));
        kernel.add_instr(gen.base_add(reg_x(6), reg_x(6), reg_x(10), shift_kind_t::lsl, 0, 0));
    }

    // Store C (Teil 2) - .rept 8
    for (int i = 0; i < 8; ++i) {
        kernel.add_instr(str_za(13, 0, 6));
        kernel.add_instr(str_za(13, 1, 6));
        kernel.add_instr(gen.base_add(reg_w(13), reg_w(13), 4, 0));
        kernel.add_instr(gen.base_add(reg_x(6), reg_x(6), reg_x(10), shift_kind_t::lsl, 0, 0));

        kernel.add_instr(str_za(13, 0, 6));
        kernel.add_instr(str_za(13, 1, 6));
        kernel.add_instr(gen.base_add(reg_w(13), reg_w(13), 4, 0));
        kernel.add_instr(gen.base_add(reg_x(6), reg_x(6), reg_x(10), shift_kind_t::lsl, 0, 0));
    }

    // M tiles check
    // processed elements in M dimension, x9 floats per tile, we process 2 tiles (32*2=64 floats) at a time
    kernel.add_instr(gen.base_add(reg_x(15), reg_x(15), reg_x(9), shift_kind_t::lsl, 1, 0)); // add x15, x15, x9, lsl #1
    kernel.add_instr(gen.base_sub(reg_x(31), reg_x(15), reg_x(3), shift_kind_t::lsl, 0, 1)); // cmp x15, x3 (subs xzr, x15, x3)
    kernel.add_branch(gen.base_b_cond("M_loop", InstGen::br_cond_t(1))); // b.ne M_loop

    // N tiles check
    // processed elements in N dimension, x9 floats per tile, we process 2 tiles (32*2=64 floats) at a time
    kernel.add_instr(gen.base_add(reg_x(16), reg_x(16), reg_x(9), shift_kind_t::lsl, 1, 0)); // add x16, x16, x9, lsl #1
    kernel.add_instr(gen.base_sub(reg_x(31), reg_x(16), reg_x(4), shift_kind_t::lsl, 0, 1)); // cmp x16, x4
    kernel.add_branch(gen.base_b_cond("N_loop", InstGen::br_cond_t(1))); // b.ne N_loop


    //kernel.add_instr(gen.base_smstop());
    kernel.add_instr(0xd503467f);
    
    // kernel.add_instr(0xd65f03c0);
    kernel.add_instr(gen.base_ret());

    //kernel.write( "gemm_512_512_512.bin" );
    kernel.set_kernel();
}
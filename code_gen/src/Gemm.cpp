#include <cstdint>
#include "../data/Gemm.h"
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
    Gemm::error_t Gemm::generate( uint32_t m,
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

      uint32_t m_rest = m - m_32 * 32;
      uint32_t n_rest = n - n_32 * 32;

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
              kernel.add_instr(ldr_za(12, 1, 6));
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

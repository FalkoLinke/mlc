

/*
    void gemm_mk_kn_nm_fp32_m16_n32_k512_tbl_v2(float const* in0, float const* in1, float* out, uint64_t lda, uint64_t ldb, uint64_t ldc);
*/
    .global _gemm_mk_kn_nm_fp32_m16_n32_k512_tbl_v2
_gemm_mk_kn_nm_fp32_m16_n32_k512_tbl_v2:
    stp x29, x30, [sp, #-16]!
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!
    mov x29, sp
    smstart

    ptrue p0.s

    // load C
    mov x6, x2
    mov w12, #0
    .rept 16
    ld1w za0h.s[w12, 0], p0/z, [x2]
    add x2, x2, x5, LSL #2
    add w12, w12, #1
    .endr

    mov w12, #0
    .rept 16
    ld1w za1h.s[w12, 0], p0/z, [x2]
    add x2, x2, x5, LSL #2
    add w12, w12, #1
    .endr
    mov x2, x6

    mov x9, x0
    mov x7, #512
_loop02:
    cbz x7, _end02
    mov x0, x9

    // load 16x16 tile of A
    // load input matrix
    ld1w z0.s, p0/z, [x0]
    add x0, x0, x3, LSL #2
    ld1w z1.s, p0/z, [x0]
    add x0, x0, x3, LSL #2
    ld1w z2.s, p0/z, [x0]
    add x0, x0, x3, LSL #2
    ld1w z3.s, p0/z, [x0]
    add x0, x0, x3, LSL #2
    ld1w z4.s, p0/z, [x0]
    add x0, x0, x3, LSL #2
    ld1w z5.s, p0/z, [x0]
    add x0, x0, x3, LSL #2
    ld1w z6.s, p0/z, [x0]
    add x0, x0, x3, LSL #2
    ld1w z7.s, p0/z, [x0]
    add x0, x0, x3, LSL #2
    ld1w z8.s, p0/z, [x0]
    add x0, x0, x3, LSL #2
    ld1w z9.s, p0/z, [x0]
    add x0, x0, x3, LSL #2
    ld1w z10.s, p0/z, [x0]
    add x0, x0, x3, LSL #2
    ld1w z11.s, p0/z, [x0]
    add x0, x0, x3, LSL #2
    ld1w z12.s, p0/z, [x0]
    add x0, x0, x3, LSL #2
    ld1w z13.s, p0/z, [x0]
    add x0, x0, x3, LSL #2
    ld1w z14.s, p0/z, [x0]
    add x0, x0, x3, LSL #2
    ld1w z15.s, p0/z, [x0]
    add x0, x0, x3, LSL #2

    // transpose
    // 2x2 submatrix transpose
    trn2 z23.s, z14.s, z15.s
    trn1 z21.s, z14.s, z15.s
    trn2 z22.s, z12.s, z13.s
    trn1 z20.s, z12.s, z13.s

    trn2 z19.s, z10.s, z11.s
    trn1 z17.s, z10.s, z11.s
    trn2 z18.s, z8.s, z9.s
    trn1 z16.s, z8.s, z9.s
    
    trn2 z15.s, z6.s, z7.s
    trn1 z13.s, z6.s, z7.s
    trn2 z14.s, z4.s, z5.s
    trn1 z12.s, z4.s, z5.s

    trn2 z11.s, z2.s, z3.s
    trn1 z9.s, z2.s, z3.s
    trn2 z10.s, z0.s, z1.s
    trn1 z8.s, z0.s, z1.s

    // 4x4 submatrix transpose
    trn2 z7.d, z22.d, z23.d
    trn1 z3.d, z22.d, z23.d    
    trn2 z5.d, z20.d, z21.d
    trn1 z1.d, z20.d, z21.d

    trn2 z6.d, z18.d, z19.d
    trn1 z2.d, z18.d, z19.d
    trn2 z4.d, z16.d, z17.d
    trn1 z0.d, z16.d, z17.d

    trn2 z23.d, z14.d, z15.d
    trn1 z19.d, z14.d, z15.d
    trn2 z21.d, z12.d, z13.d
    trn1 z17.d, z12.d, z13.d

    trn2 z22.d, z10.d, z11.d
    trn1 z18.d, z10.d, z11.d
    trn2 z20.d, z8.d, z9.d
    trn1 z16.d, z8.d, z9.d

    // 8x8 submatrix transpose
    adr x12, _trn1_8x8
    ld1w z14.s, p0/z, [x12]
    adr x13, _trn2_8x8
    ld1w z15.s, p0/z, [x13]

    tbl z13.s, {z6.s, z7.s}, z15.s
    tbl z12.s, {z6.s, z7.s}, z14.s
    tbl z11.s, {z4.s, z5.s}, z15.s
    tbl z10.s, {z4.s, z5.s}, z14.s
    tbl z9.s, {z2.s, z3.s}, z15.s
    tbl z8.s, {z2.s, z3.s}, z14.s
    tbl z7.s, {z0.s, z1.s}, z15.s
    tbl z6.s, {z0.s, z1.s}, z14.s
    tbl z5.s, {z22.s, z23.s}, z15.s
    tbl z4.s, {z22.s, z23.s}, z14.s
    tbl z3.s, {z20.s, z21.s}, z15.s
    tbl z2.s, {z20.s, z21.s}, z14.s
    tbl z1.s, {z18.s, z19.s}, z15.s
    tbl z0.s, {z18.s, z19.s}, z14.s
    tbl z23.s, {z16.s, z17.s}, z15.s
    tbl z22.s, {z16.s, z17.s}, z14.s

    mov z21.d, z13.d
    mov z20.d, z5.d
    mov z19.d, z11.d
    mov z18.d, z3.d
    mov z17.d, z9.d
    mov z16.d, z1.d
    mov z15.d, z7.d
    mov z14.d, z23.d

    mov z13.d, z12.d
    mov z12.d, z4.d
    mov z11.d, z10.d
    mov z10.d, z2.d
    mov z9.d, z8.d
    mov z8.d, z0.d
    mov z7.d, z6.d
    mov z6.d, z22.d

    // 16x16 submatrix transpose
    adr x12, _trn1_16x16
    ld1w z4.s, p0/z, [x12]
    adr x13, _trn2_16x16
    ld1w z5.s, p0/z, [x13]

    tbl z2.s, {z20.s, z21.s}, z5.s
    tbl z1.s, {z20.s, z21.s}, z4.s
    tbl z0.s, {z18.s, z19.s}, z5.s
    tbl z23.s, {z18.s, z19.s}, z4.s
    tbl z22.s, {z16.s, z17.s}, z5.s
    tbl z21.s, {z16.s, z17.s}, z4.s
    tbl z20.s, {z14.s, z15.s}, z5.s
    tbl z19.s, {z14.s, z15.s}, z4.s
    tbl z18.s, {z12.s, z13.s}, z5.s
    tbl z17.s, {z12.s, z13.s}, z4.s
    tbl z16.s, {z10.s, z11.s}, z5.s
    tbl z15.s, {z10.s, z11.s}, z4.s
    tbl z14.s, {z8.s, z9.s}, z5.s
    tbl z13.s, {z8.s, z9.s}, z4.s
    tbl z12.s, {z6.s, z7.s}, z5.s
    tbl z11.s, {z6.s, z7.s}, z4.s

    // compute outer products
    ld1w z3.s, p0/z, [x1]
    ld1w z4.s, p0/z, [x1, #1, MUL VL]
    fmopa za0.s, p0/m, p0/m, z3.s, z11.s
    fmopa za1.s, p0/m, p0/m, z4.s, z11.s
    add x1, x1, x4, LSL #2

    ld1w z3.s, p0/z, [x1]
    ld1w z4.s, p0/z, [x1, #1, MUL VL]
    fmopa za0.s, p0/m, p0/m, z3.s, z13.s
    fmopa za1.s, p0/m, p0/m, z4.s, z13.s
    add x1, x1, x4, LSL #2

    ld1w z3.s, p0/z, [x1]
    ld1w z4.s, p0/z, [x1, #1, MUL VL]
    fmopa za0.s, p0/m, p0/m, z3.s, z15.s
    fmopa za1.s, p0/m, p0/m, z4.s, z15.s
    add x1, x1, x4, LSL #2

    ld1w z3.s, p0/z, [x1]
    ld1w z4.s, p0/z, [x1, #1, MUL VL]
    fmopa za0.s, p0/m, p0/m, z3.s, z17.s
    fmopa za1.s, p0/m, p0/m, z4.s, z17.s
    add x1, x1, x4, LSL #2

    ld1w z3.s, p0/z, [x1]
    ld1w z4.s, p0/z, [x1, #1, MUL VL]
    fmopa za0.s, p0/m, p0/m, z3.s, z19.s
    fmopa za1.s, p0/m, p0/m, z4.s, z19.s
    add x1, x1, x4, LSL #2

    ld1w z3.s, p0/z, [x1]
    ld1w z4.s, p0/z, [x1, #1, MUL VL]
    fmopa za0.s, p0/m, p0/m, z3.s, z21.s
    fmopa za1.s, p0/m, p0/m, z4.s, z21.s
    add x1, x1, x4, LSL #2

    ld1w z3.s, p0/z, [x1]
    ld1w z4.s, p0/z, [x1, #1, MUL VL]
    fmopa za0.s, p0/m, p0/m, z3.s, z23.s
    fmopa za1.s, p0/m, p0/m, z4.s, z23.s
    add x1, x1, x4, LSL #2

    ld1w z3.s, p0/z, [x1]
    ld1w z4.s, p0/z, [x1, #1, MUL VL]
    fmopa za0.s, p0/m, p0/m, z3.s, z1.s
    fmopa za1.s, p0/m, p0/m, z4.s, z1.s
    add x1, x1, x4, LSL #2

    ld1w z3.s, p0/z, [x1]
    ld1w z4.s, p0/z, [x1, #1, MUL VL]
    fmopa za0.s, p0/m, p0/m, z3.s, z12.s
    fmopa za1.s, p0/m, p0/m, z4.s, z12.s
    add x1, x1, x4, LSL #2

    ld1w z3.s, p0/z, [x1]
    ld1w z4.s, p0/z, [x1, #1, MUL VL]
    fmopa za0.s, p0/m, p0/m, z3.s, z14.s
    fmopa za1.s, p0/m, p0/m, z4.s, z14.s
    add x1, x1, x4, LSL #2

    ld1w z3.s, p0/z, [x1]
    ld1w z4.s, p0/z, [x1, #1, MUL VL]
    fmopa za0.s, p0/m, p0/m, z3.s, z16.s
    fmopa za1.s, p0/m, p0/m, z4.s, z16.s
    add x1, x1, x4, LSL #2

    ld1w z3.s, p0/z, [x1]
    ld1w z4.s, p0/z, [x1, #1, MUL VL]
    fmopa za0.s, p0/m, p0/m, z3.s, z18.s
    fmopa za1.s, p0/m, p0/m, z4.s, z18.s
    add x1, x1, x4, LSL #2

    ld1w z3.s, p0/z, [x1]
    ld1w z4.s, p0/z, [x1, #1, MUL VL]
    fmopa za0.s, p0/m, p0/m, z3.s, z20.s
    fmopa za1.s, p0/m, p0/m, z4.s, z20.s
    add x1, x1, x4, LSL #2

    ld1w z3.s, p0/z, [x1]
    ld1w z4.s, p0/z, [x1, #1, MUL VL]
    fmopa za0.s, p0/m, p0/m, z3.s, z22.s
    fmopa za1.s, p0/m, p0/m, z4.s, z22.s
    add x1, x1, x4, LSL #2

    ld1w z3.s, p0/z, [x1]
    ld1w z4.s, p0/z, [x1, #1, MUL VL]
    fmopa za0.s, p0/m, p0/m, z3.s, z0.s
    fmopa za1.s, p0/m, p0/m, z4.s, z0.s
    add x1, x1, x4, LSL #2

    ld1w z3.s, p0/z, [x1]
    ld1w z4.s, p0/z, [x1, #1, MUL VL]
    fmopa za0.s, p0/m, p0/m, z3.s, z2.s
    fmopa za1.s, p0/m, p0/m, z4.s, z2.s
    add x1, x1, x4, LSL #2

    add x9, x9, #64
    sub x7, x7, #16
    b _loop02
_end02:

    // store C
    mov w12, #0
    .rept 16
    st1w za0h.s[w12, 0], p0, [x2]
    add x2, x2, x5, LSL #2
    add w12, w12, #1
    .endr

    mov w12, #0
    .rept 16
    st1w za1h.s[w12, 0], p0, [x2]
    add x2, x2, x5, LSL #2
    add w12, w12, #1
    .endr

    smstop
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ldp x29, x30, [sp], #16
    ret



    .p2align 2
_trn1_2x2:
    .word 0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30
_trn2_2x2:
    .word 1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31
_trn1_4x4:
    .word 0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29
_trn2_4x4:
    .word 2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31
_trn1_8x8:
    .word 0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27
_trn2_8x8:
    .word 4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31
_trn1_16x16:
    .word 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23
_trn2_16x16:
    .word 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31

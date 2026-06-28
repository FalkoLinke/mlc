

/*
    void transpose_16x16_fp32_tbl(float const* in0, float* out, uint64_t ldi, uint64_t ldo);

    Loads in0 into SVE vector registers, transposes them and stores them to out.

    This algorithm does not preserve the contents of any of the SVE vector registers.
    The algorithm modifies the x4, x5 registers and the input and output pointers.

    The transposition algorithm expects the input matrix in the following registers
    from lowest to highest column index:
    z0
    z1
    z2
    z3
    z4
    z5
    z6
    z7
    z8
    z9
    z10
    z11
    z12
    z13
    z14
    z15

    The result of the transposition algorithm is stored in the following registers 
    from lowest to highest column index:
    z0
    z1
    z2
    z3
    z4
    z5
    z6
    z7
    z8
    z9
    z10
    z11
    z12
    z13
    z14
    z15
*/
    .global _transpose_16x16_fp32_tbl
_transpose_16x16_fp32_tbl:
    stp x29, x30, [sp, #-16]!
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!
    mov x29, sp

    smstart

    ptrue p0.s

    mov x6, x0
    mov x7, #50000

k_loop:
    mov x0, x6

    // load input matrix
    ld1w z0.s, p0/z, [x0]
    add x0, x0, x2, LSL #2
    ld1w z1.s, p0/z, [x0]
    add x0, x0, x2, LSL #2
    ld1w z2.s, p0/z, [x0]
    add x0, x0, x2, LSL #2
    ld1w z3.s, p0/z, [x0]
    add x0, x0, x2, LSL #2
    ld1w z4.s, p0/z, [x0]
    add x0, x0, x2, LSL #2
    ld1w z5.s, p0/z, [x0]
    add x0, x0, x2, LSL #2
    ld1w z6.s, p0/z, [x0]
    add x0, x0, x2, LSL #2
    ld1w z7.s, p0/z, [x0]
    add x0, x0, x2, LSL #2
    ld1w z8.s, p0/z, [x0]
    add x0, x0, x2, LSL #2
    ld1w z9.s, p0/z, [x0]
    add x0, x0, x2, LSL #2
    ld1w z10.s, p0/z, [x0]
    add x0, x0, x2, LSL #2
    ld1w z11.s, p0/z, [x0]
    add x0, x0, x2, LSL #2
    ld1w z12.s, p0/z, [x0]
    add x0, x0, x2, LSL #2
    ld1w z13.s, p0/z, [x0]
    add x0, x0, x2, LSL #2
    ld1w z14.s, p0/z, [x0]
    add x0, x0, x2, LSL #2
    ld1w z15.s, p0/z, [x0]
    add x0, x0, x2, LSL #2

    // transpose

    // 2x2 submatrix transpose
    trn1 z16.s, z0.s, z1.s
    trn2 z17.s, z0.s, z1.s
    trn1 z18.s, z2.s, z3.s
    trn2 z19.s, z2.s, z3.s
    trn1 z20.s, z4.s, z5.s
    trn2 z21.s, z4.s, z5.s
    trn1 z22.s, z6.s, z7.s
    trn2 z23.s, z6.s, z7.s
    trn1 z24.s, z8.s, z9.s
    trn2 z25.s, z8.s, z9.s
    trn1 z26.s, z10.s, z11.s
    trn2 z27.s, z10.s, z11.s
    trn1 z28.s, z12.s, z13.s
    trn2 z29.s, z12.s, z13.s
    trn1 z30.s, z14.s, z15.s
    trn2 z31.s, z14.s, z15.s

    // 4x4 submatrix transpose
    trn1 z0.d, z16.d, z18.d
    trn1 z2.d, z17.d, z19.d
    trn2 z4.d, z16.d, z18.d
    trn2 z6.d, z17.d, z19.d
    trn1 z1.d, z20.d, z22.d
    trn1 z3.d, z21.d, z23.d
    trn2 z5.d, z20.d, z22.d
    trn2 z7.d, z21.d, z23.d
    trn1 z8.d, z24.d, z26.d
    trn1 z10.d, z25.d, z27.d
    trn2 z12.d, z24.d, z26.d
    trn2 z14.d, z25.d, z27.d
    trn1 z9.d, z28.d, z30.d
    trn1 z11.d, z29.d, z31.d
    trn2 z13.d, z28.d, z30.d
    trn2 z15.d, z29.d, z31.d

    // 8x8 submatrix transpose
    adr x4, _trn1_8x8               // prepare 8x8 trn instruction using tbl
    ld1w z30.s, p0/z, [x4]          // z30 is trn1
    adr x5, _trn2_8x8
    ld1w z31.s, p0/z, [x5]          // z31 is trn2

    tbl z16.s, {z0.s, z1.s}, z30.s  // arguments to tbl must be consecutively numbered
    tbl z18.s, {z2.s, z3.s}, z30.s
    tbl z20.s, {z4.s, z5.s}, z30.s
    tbl z22.s, {z6.s, z7.s}, z30.s
    mov z17.d, z30.d
    tbl z24.s, {z0.s, z1.s}, z31.s
    tbl z26.s, {z2.s, z3.s}, z31.s
    tbl z28.s, {z4.s, z5.s}, z31.s
    tbl z30.s, {z6.s, z7.s}, z31.s
    mov z0.d, z17.d
    mov z1.d, z31.d
    tbl z17.s, {z8.s, z9.s}, z0.s
    tbl z19.s, {z10.s, z11.s}, z0.s
    tbl z21.s, {z12.s, z13.s}, z0.s
    tbl z23.s, {z14.s, z15.s}, z0.s
    tbl z25.s, {z8.s, z9.s}, z1.s
    tbl z27.s, {z10.s, z11.s}, z1.s
    tbl z29.s, {z12.s, z13.s}, z1.s
    tbl z31.s, {z14.s, z15.s}, z1.s

    // 16x16 submatrix transpose
    adr x4, _trn1_16x16
    ld1w z14.s, p0/z, [x4]
    adr x5, _trn2_16x16
    ld1w z15.s, p0/z, [x5]

    tbl z0.s, {z16.s, z17.s}, z14.s
    tbl z1.s, {z18.s, z19.s}, z14.s
    tbl z2.s, {z20.s, z21.s}, z14.s
    tbl z3.s, {z22.s, z23.s}, z14.s
    tbl z4.s, {z24.s, z25.s}, z14.s
    tbl z5.s, {z26.s, z27.s}, z14.s
    tbl z6.s, {z28.s, z29.s}, z14.s
    tbl z7.s, {z30.s, z31.s}, z14.s
    tbl z8.s, {z16.s, z17.s}, z15.s
    tbl z9.s, {z18.s, z19.s}, z15.s
    tbl z10.s, {z20.s, z21.s}, z15.s
    tbl z11.s, {z22.s, z23.s}, z15.s
    tbl z12.s, {z24.s, z25.s}, z15.s
    tbl z13.s, {z26.s, z27.s}, z15.s
    tbl z14.s, {z28.s, z29.s}, z15.s
    mov z16.d, z15.d
    tbl z15.s, {z30.s, z31.s}, z16.s

    // store output matrix
/*
    mov z0.d, z16.d
    mov z1.d, z17.d
    mov z2.d, z18.d
    mov z3.d, z19.d
    mov z4.d, z20.d
    mov z5.d, z21.d
    mov z6.d, z22.d
    mov z7.d, z23.d
    mov z8.d, z24.d
    mov z9.d, z25.d
    mov z10.d, z26.d
    mov z11.d, z27.d
    mov z12.d, z28.d
    mov z13.d, z29.d
    mov z14.d, z30.d
    mov z15.d, z31.d
*/

    subs x7, x7, #1
    b.ne k_loop

    st1w z0.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z1.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z2.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z3.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z4.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z5.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z6.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z7.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z8.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z9.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z10.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z11.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z12.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z13.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z14.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z15.s, p0, [x1]
    add x1, x1, x3, LSL #2



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




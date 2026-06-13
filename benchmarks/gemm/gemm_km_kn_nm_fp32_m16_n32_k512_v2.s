

/*
    void gemm_km_kn_nm_fp32_m16_n32_k512_v2(float const* in0, float const* in1, float* out, uint64_t lda, uint64_t ldb, uint64_t ldc);
*/
    .global _gemm_km_kn_nm_fp32_m16_n32_k512_v2
_gemm_km_kn_nm_fp32_m16_n32_k512_v2:
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

    zero {za2.s, za3.s}

    mov x6, #512
_loop01:
    cbz x6, _end01

    ld1w z0.s, p0/z, [x0]
    ld1w z1.s, p0/z, [x1]
    ld1w z2.s, p0/z, [x1, #1, MUL VL]

    fmopa za0.s, p0/m, p0/m, z1.s, z0.s
    fmopa za1.s, p0/m, p0/m, z2.s, z0.s

    add x0, x0, x3, LSL #2
    add x1, x1, x4, LSL #2
    sub x6, x6, #1

    ld1w z3.s, p0/z, [x0]
    ld1w z4.s, p0/z, [x1]
    ld1w z5.s, p0/z, [x1, #1, MUL VL]

    fmopa za2.s, p0/m, p0/m, z4.s, z3.s
    fmopa za3.s, p0/m, p0/m, z5.s, z3.s

    add x0, x0, x3, LSL #2
    add x1, x1, x4, LSL #2
    sub x6, x6, #1
    b _loop01
_end01:

    // add za tiles
    mov w12, #0
    .rept 4
    mova {z0.s - z3.s}, za0h.s[w12, 0:3]
    mova {z4.s - z7.s}, za2h.s[w12, 0:3]
    fadd z0.s, z0.s, z4.s
    fadd z1.s, z1.s, z5.s
    fadd z2.s, z2.s, z6.s
    fadd z3.s, z3.s, z7.s
    mova za0h.s[w12, 0:3], {z0.s - z3.s}
    add w12, w12, #4
    .endr

    mov w12, #0
    .rept 4
    mova {z0.s - z3.s}, za1h.s[w12, 0:3]
    mova {z4.s - z7.s}, za3h.s[w12, 0:3]
    fadd z0.s, z0.s, z4.s
    fadd z1.s, z1.s, z5.s
    fadd z2.s, z2.s, z6.s
    fadd z3.s, z3.s, z7.s
    mova za1h.s[w12, 0:3], {z0.s - z3.s}
    add w12, w12, #4
    .endr

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
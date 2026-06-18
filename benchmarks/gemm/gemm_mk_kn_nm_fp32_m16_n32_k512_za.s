

/*
    void gemm_mk_kn_nm_fp32_m16_n32_k512_za(float const* in0, float const* in1, float* out, uint64_t lda, uint64_t ldb, uint64_t ldc);
*/
    .global _gemm_mk_kn_nm_fp32_m16_n32_k512_za
_gemm_mk_kn_nm_fp32_m16_n32_k512_za:
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

    // load A into ZA tile
    mov x0, x9
    mov w12, #0
    .rept 16
    ld1w za2h.s[w12, 0], p0/z, [x0]
    add x0, x0, x3, LSL #2 
    add w12, w12, #1
    .endr

    mov w12, #0
    mov x6, #16
_loop01:
    cbz x6, _end01

    mov z0.s, p0/m, za2v.s[w12, 0]
    ld1w z1.s, p0/z, [x1]
    ld1w z2.s, p0/z, [x1, #1, MUL VL]

    fmopa za0.s, p0/m, p0/m, z1.s, z0.s
    fmopa za1.s, p0/m, p0/m, z2.s, z0.s

    add x1, x1, x4, LSL #2
    add w12, w12, #1
    sub x6, x6, #1
    b _loop01
_end01:

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
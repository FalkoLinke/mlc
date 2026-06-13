


/*
    void transpose_16x16_fp32_simd(float const* in0, float* out, uint64_t ldi, uint64_t ldo);
*/
    .global _transpose_16x16_fp32_simd
_transpose_16x16_fp32_simd:
    stp x29, x30, [sp, #-16]!
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!
    mov x29, sp

    mov x6, x0
    mov x7, x1
    mov x4, #4
_loop01:
    cbz x4, _end01

    mov x9, x6
    mov x10, x7
    mov x5, #4
_loop02:
    cbz x5, _end02
    mov x0, x9
    mov x1, x10

    ld1 {v0.4s}, [x0]
    add x0, x0, x2, LSL #2
    ld1 {v1.4s}, [x0]
    add x0, x0, x2, LSL #2
    ld1 {v2.4s}, [x0]
    add x0, x0, x2, LSL #2
    ld1 {v3.4s}, [x0]
    add x0, x0, x2, LSL #2

    trn1 v4.4s, v0.4s, v1.4s
    trn2 v5.4s, v0.4s, v1.4s
    trn1 v6.4s, v2.4s, v3.4s
    trn2 v7.4s, v2.4s, v3.4s

    trn1 v0.2d, v4.2d, v6.2d
    trn1 v1.2d, v5.2d, v7.2d
    trn2 v2.2d, v4.2d, v6.2d
    trn2 v3.2d, v5.2d, v7.2d

    st1 {v0.4s}, [x1]
    add x1, x1, x3, LSL #2
    st1 {v1.4s}, [x1]
    add x1, x1, x3, LSL #2
    st1 {v2.4s}, [x1]
    add x1, x1, x3, LSL #2
    st1 {v3.4s}, [x1]
    add x1, x1, x3, LSL #2

    add x9, x9, #16
    add x10, x10, x3, LSL #4
    sub x5, x5, #1
    b _loop02
_end02:

    add x6, x6, x2, LSL #4
    add x7, x7, #16
    sub x4, x4, #1
    b _loop01
_end01:

    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ldp x29, x30, [sp], #16
    ret


    .global _transpose_16x16_fp32_sme2
_transpose_16x16_fp32_sme2:
    stp x29, x30, [sp, #-16]!
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!
    mov x29, sp

    smstart

    ptrue p0.s
    ptrue pn8.s

    ld1w {z0.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z1.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z2.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z3.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z4.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z5.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z6.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z7.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z8.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z9.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z10.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z11.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z12.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z13.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z14.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z15.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2

    zip { z0.s - z3.s }, { z0.s - z3.s }
    zip { z4.s - z7.s }, { z4.s - z7.s }
    zip { z8.s - z11.s }, { z8.s - z11.s }
    zip { z12.s - z15.s }, { z12.s - z15.s }


    st1w { z0.s - z3.s }, pn8, [x1]
    add x1, x1, x3, lsl #4
    st1w { z4.s - z7.s }, pn8, [x1]
    add x1, x1, x3, lsl #4
    st1w { z8.s - z11.s }, pn8, [x1]
    add x1, x1, x3, lsl #4
    st1w { z12.s - z15.s }, pn8, [x1]
    add x1, x1, x3, lsl #4

    smstop

    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ldp x29, x30, [sp], #16
    ret









/*
    void transpose_16x16_fp32_za(float const* in0, float* out, uint64_t ldi, uint64_t ldo);

    Loads in0 into ZA0 horizontal slices and stores the ZA0 vertical slices into out.
*/
    .global _transpose_16x16_fp32_za
_transpose_16x16_fp32_za:
    stp x29, x30, [sp, #-16]!           // we add the function prologue here to keep as many things similar between the kernels
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!
    mov x29, sp

    smstart
    ptrue p0.s

    mov x4, x0
    mov x5, #50000

k_loop:
    mov x0, x4

    mov w12, #0
    .rept 4
    ld1w { z0.S, z1.S }, pn8/Z, [x0]
    mova za0h.S[w12, #0], p0/m, z0.S
    mova za0h.S[w12, #1], p0/m, z1.S
    add x0, x0, x3, lsl #4
    ld1w { z2.S, z3.S }, pn8/Z, [x0]
    mova za0h.S[w12, #2], p0/m, z2.S
    mova za0h.S[w12, #3], p0/m, z3.S
    add x0, x0, x3, lsl #4
    add w12, w12, #4
    .endr

    mov w14, #0

    MOVA { z0.S - z3.S }, za0v.S[w14, 0:3 ]
    add w14, w14, #3
    MOVA { z4.S - z7.S }, za0v.S[w14, 0:3 ]
    add w14, w14, #3
    MOVA { z8.S - z11.S }, za0v.S[w14, 0:3 ]
    add w14, w14, #3
    MOVA { z12.S - z15.S }, za0v.S[w14, 0:3 ]

    subs x5, x5, #1
    b.ne k_loop

    st1w { z0.s - z3.s }, pn8, [x1]
    add x1, x1, x3, lsl #4
    st1w { z4.s - z7.s }, pn8, [x1]
    add x1, x1, x3, lsl #4
    st1w { z8.s - z11.s }, pn8, [x1]
    add x1, x1, x3, lsl #4
    st1w { z12.s - z15.s }, pn8, [x1]
    add x1, x1, x3, lsl #4

    // mov w12, #0
    // .rept 16
    // st1w za0v.s[w12, 0], p0, [x1]
    // add x1, x1, x3, LSL #2
    // add w12, w12, #1
    // .endr

    smstop

    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ldp x29, x30, [sp], #16
    ret


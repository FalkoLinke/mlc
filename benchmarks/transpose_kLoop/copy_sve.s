

/*
    void copy_sve(float const* in0, float* out, uint64_t ldi, uint64_t ldo);
*/
    .global _copy_sve
_copy_sve:
    stp x29, x30, [sp, #-16]!
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!
    mov x29, sp

    smstart

    ptrue p0.s

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
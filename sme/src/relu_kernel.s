    .text



/*
void relu_16_16( float const * a,
                float       * b,
                int64_t       ld_a,
                int64_t       ld_b,
                int32_t       trans_b );
*/
    .global FUNCLABEL(relu_16_16)
FUNCLABEL(relu_16_16):
    stp x29, x30, [sp, #-16]!
    mov fp, sp

    // perform transpose if necessary and copy
    stp x0, x1, [sp, #-16]!
    stp x2, x3, [sp, #-16]!
    stp x4, x5, [sp, #-16]!
    bl identity_16_16
    ldp x4, x5, [sp], #16
    ldp x2, x3, [sp], #16
    ldp x0, x1, [sp], #16

    // perform the RELU inplace on B
    ptrue p0.b
    mov x5, #16
loop01:
    cbz x5, end01

    ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1]
    fmax z0.s, p0/m, z0.s, #0.0
    fmax z1.s, p0/m, z1.s, #0.0
    fmax z2.s, p0/m, z2.s, #0.0
    fmax z3.s, p0/m, z3.s, #0.0
    st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x1]

    add x1, x1, x3
    subs x5, x5, #1
    b loop01
end01:


    ldp x29, x30, [sp], #16
    ret

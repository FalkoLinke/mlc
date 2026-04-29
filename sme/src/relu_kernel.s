#ifdef __APPLE__
#define FUNCLABEL(NAME) _##NAME
#else
#define FUNCLABEL(NAME) NAME
#endif /* __APPLE__ */


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

    ptrue p1.s, VL4

    // perform transpose if necessary and copy
    stp x0, x1, [sp, #-16]!
    stp x2, x3, [sp, #-16]!
    stp x4, x5, [sp, #-16]!
    bl FUNCLABEL(identity_16_16)
    ldp x4, x5, [sp], #16
    ldp x2, x3, [sp], #16
    ldp x0, x1, [sp], #16

    // perform the RELU inplace on B
    ptrue p0.b
    mov x5, #16
loop01:
    cbz x5, end01

    ld4w {z0.s, z1.s, z2.s, z3.s}, p1/z, [x1]
    fmax z0.s, p0/m, z0.s, #0.0
    fmax z1.s, p0/m, z1.s, #0.0
    fmax z2.s, p0/m, z2.s, #0.0
    fmax z3.s, p0/m, z3.s, #0.0
    st4w {z0.s, z1.s, z2.s, z3.s}, p1, [x1]

    add x1, x1, x3, LSL #2
    subs x5, x5, #1
    b loop01
end01:


    ldp x29, x30, [sp], #16
    ret

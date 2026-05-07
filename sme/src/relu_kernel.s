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

    // perform transpose if necessary and copy
    stp x0, x1, [sp, #-16]!
    stp x2, x3, [sp, #-16]!
    stp x4, x5, [sp, #-16]!
    bl FUNCLABEL(identity_16_16)
    ldp x4, x5, [sp], #16
    ldp x2, x3, [sp], #16
    ldp x0, x1, [sp], #16

    // perform the RELU inplace on B
    smstart

    mov x6, x1

    ptrue p0.b
    ptrue p1.s, VL4
    mov x5, #4
loop01:
    cbz x5, end01

    ld4w {z0.s, z1.s, z2.s, z3.s}, p1/z, [x1]
    add x1, x1, x3, LSL #2  
    ld4w {z4.s, z5.s, z6.s, z7.s}, p1/z, [x1]
    add x1, x1, x3, LSL #2
    ld4w {z16.s, z17.s, z18.s, z19.s}, p1/z, [x1]
    add x1, x1, x3, LSL #2
    ld4w {z20.s, z21.s, z22.s, z23.s}, p1/z, [x1]
    add x1, x1, x3, LSL #2

    fmax z0.s, p0/m, z0.s, #0.0
    fmax z1.s, p0/m, z1.s, #0.0
    fmax z2.s, p0/m, z2.s, #0.0
    fmax z3.s, p0/m, z3.s, #0.0
    fmax z4.s, p0/m, z4.s, #0.0
    fmax z5.s, p0/m, z5.s, #0.0
    fmax z6.s, p0/m, z6.s, #0.0
    fmax z7.s, p0/m, z7.s, #0.0
    fmax z16.s, p0/m, z16.s, #0.0
    fmax z17.s, p0/m, z17.s, #0.0
    fmax z18.s, p0/m, z18.s, #0.0
    fmax z19.s, p0/m, z19.s, #0.0
    fmax z20.s, p0/m, z20.s, #0.0
    fmax z21.s, p0/m, z21.s, #0.0
    fmax z22.s, p0/m, z22.s, #0.0
    fmax z23.s, p0/m, z23.s, #0.0

    st4w {z0.s, z1.s, z2.s, z3.s}, p1, [x6]
    add x6, x6, x3, LSL #2
    st4w {z4.s, z5.s, z6.s, z7.s}, p1, [x6]
    add x6, x6, x3, LSL #2
    st4w {z16.s, z17.s, z18.s, z19.s}, p1, [x6]
    add x6, x6, x3, LSL #2
    st4w {z20.s, z21.s, z22.s, z23.s}, p1, [x6]
    add x6, x6, x3, LSL #2

    subs x5, x5, #1
    b loop01
end01:

    smstop
    ldp x29, x30, [sp], #16
    ret

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
    smstart

    cbz x4, relu_16_16_notrans
    b relu_16_16_trans
relu_16_16_notrans:

    ptrue p0.s
    mov x5, #16
relu_16_16_loop01:
    cbz x5, relu_16_16_loop01_end

    ld1w z0.s, p0/z, [x0]
    fmax z0.s, p0/m, z0.s, #0.0
    st1w z0.s, p0, [x1]

    add x0, x0, x2, LSL #2
    add x1, x1, x3, LSL #2
    subs x5, x5, #1
    b relu_16_16_loop01
relu_16_16_loop01_end:

    b relu_16_16_ret
relu_16_16_trans:

    ptrue p0.s
    mov w12, #0
    mov x5, #16
relu_16_16_loop02:
    cbz x5, relu_16_16_loop02_end

    ld1w z0.s, p0/z, [x0]
    fmax z0.s, p0/m, z0.s, #0.0
    mov za0h.s[w12, 0], p0/m, z0.s

    add w12, w12, #1
    add x0, x0, x2, LSL #2
    subs x5, x5, #1
    b relu_16_16_loop02
relu_16_16_loop02_end:

    mov w12, #0
    mov x5, #16
relu_16_16_loop03:
    cbz x5, relu_16_16_loop03_end

    st1w za0v.s[w12, 0], p0, [x1]

    add w12, w12, #1
    add x1, x1, x3, LSL #2
    subs x5, x5, #1
    b relu_16_16_loop03
relu_16_16_loop03_end:

relu_16_16_ret:
    smstop
    ldp x29, x30, [sp], #16
    ret





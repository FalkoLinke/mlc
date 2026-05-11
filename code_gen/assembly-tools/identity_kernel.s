#ifdef __APPLE__
#define FUNCLABEL(NAME) _##NAME
#else
#define FUNCLABEL(NAME) NAME
#endif /* __APPLE__ */



    .text











/*
void identity_16_16( float const * a,
                   float       * b,
                   int64_t       ld_a,
                   int64_t       ld_b,
                   int32_t       trans_b );
*/
    .global FUNCLABEL(identity_16_16)
FUNCLABEL(identity_16_16):
    stp x29, x30, [sp, #-16]!
    mov fp, sp
    smstart

    cbz x4, identity_16_16_notrans
    b identity_16_16_trans
identity_16_16_notrans:

    ptrue p0.s
    mov x5, #16
identity_16_16_loop01:
    cbz x5, identity_16_16_loop01_end

    ld1w z0.s, p0/z, [x0]
    st1w z0.s, p0, [x1]

    add x0, x0, x2, LSL #2
    add x1, x1, x3, LSL #2
    subs x5, x5, #1
    b identity_16_16_loop01
identity_16_16_loop01_end:

    b identity_16_16_ret
identity_16_16_trans:

    ptrue p0.s
    mov w12, #0
    mov x5, #4
identity_16_16_loop02:
    cbz x5, identity_16_16_loop02_end

    ld1w za0h.s[w12, 0], p0/z, [x0]
    add x0, x0, x2, LSL #2
    ld1w za0h.s[w12, 1], p0/z, [x0]
    add x0, x0, x2, LSL #2
    ld1w za0h.s[w12, 2], p0/z, [x0]
    add x0, x0, x2, LSL #2
    ld1w za0h.s[w12, 3], p0/z, [x0]
    add x0, x0, x2, LSL #2

    add w12, w12, #4
    subs x5, x5, #1
    b identity_16_16_loop02
identity_16_16_loop02_end:

    mov w12, #0
    mov x5, #4
identity_16_16_loop03:
    cbz x5, identity_16_16_loop03_end

    st1w za0v.s[w12, 0], p0, [x1]
    add x1, x1, x3, LSL #2
    st1w za0v.s[w12, 1], p0, [x1]
    add x1, x1, x3, LSL #2
    st1w za0v.s[w12, 2], p0, [x1]
    add x1, x1, x3, LSL #2
    st1w za0v.s[w12, 3], p0, [x1]
    add x1, x1, x3, LSL #2

    add w12, w12, #4
    subs x5, x5, #1
    b identity_16_16_loop03
identity_16_16_loop03_end:

identity_16_16_ret:
    smstop
    ldp x29, x30, [sp], #16
    ret









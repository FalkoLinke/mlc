#ifdef __APPLE__
#define FUNCLABEL(NAME) _##NAME
#else
#define FUNCLABEL(NAME) NAME
#endif /* __APPLE__ */



    .text








/*
void identity_4_4( float const * a,
                   float       * b,
                   int64_t       ld_a,
                   int64_t       ld_b,
                   int32_t       trans_b );
*/
    .global FUNCLABEL(identity_4_4)
FUNCLABEL(identity_4_4):
    smstart

    ptrue p0.s, VL4

    // load A
    ld1w z0.s, p0/z, [x0]
    add x0, x0, x2, LSL #2
    ld1w z1.s, p0/z, [x0]
    add x0, x0, x2, LSL #2
    ld1w z2.s, p0/z, [x0]
    add x0, x0, x2, LSL #2
    ld1w z3.s, p0/z, [x0]
    add x0, x0, x2, LSL #2

    // skip transpose if necessary
    cbz x4, skip01

    // perform transpose on 2x2 submatrices
    trn1 z4.s, z0.s, z1.s
    trn2 z5.s, z0.s, z1.s
    trn1 z6.s, z2.s, z3.s
    trn2 z7.s, z2.s, z3.s

    // transpose matrix of submatrices
    trn1 z0.d, z4.d, z6.d
    trn1 z1.d, z5.d, z7.d
    trn2 z2.d, z4.d, z6.d
    trn2 z3.d, z5.d, z7.d

skip01:
    // store result
    st1w z0.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z1.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z2.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z3.s, p0, [x1]
    add x1, x1, x3, LSL #2

    smstop
    ret






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

    // select pointer offset values to apply to the B pointer
    cbz x4, skip02
    // offset values with transpose
    mov x13, #16
    mov x14, x3
    lsl x14, x14, #4
    b skip03
skip02:
    // offset values without transpose
    mov x13, x3
    lsl x13, x13, #4
    mov x14, #16
skip03:

    // perform copy
    mov x9, x0              // x9 and x10 point to A
    mov x11, x1             // x11 and x12 point to B
    mov x5, #4              // x5 and x6 are loop counters
loop01:
    cbz x5, end01

    mov x10, x9
    mov x12, x11
    mov x6, #4
loop02:
    cbz x6, end02

    // perform submatrix transpose, writing to the correct target submatrix
    mov x0, x10
    mov x1, x12
    bl FUNCLABEL(identity_4_4)

    add x12, x12, x14       // B pointer changes based on transpose flag
    add x10, x10, #16       // A pointer changes normally
    subs x6, x6, #1
    b loop02
end02:

    add x11, x11, x13       // B pointer changes based on transpose flag
    add x9, x9, x2, LSL #4  // A pointer changes normally
    subs x5, x5, #1
    b loop01
end01:

    ldp x29, x30, [sp], #16
    ret













/*

Notes:

A = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, A, B],
    [C, D, E, F]
]
A^T = [
    [0, 4, 8, C],
    [1, 5, 9, D],
    [2, 6, A, E],
    [3, 7, B, F]
]


v0 = [0, 4, 2, 6]
v1 = [1, 5, 3, 7]
v2 = [8, C, A, E]
v3 = [9, D, B, F]



*/




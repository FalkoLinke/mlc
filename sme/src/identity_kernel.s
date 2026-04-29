/*



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

ld
trn
mov


v0 = [0, 4, 2, 6]
v1 = [1, 5, 3, 7]
v2 = [8, C, A, E]
v3 = [9, D, B, F]








*/







    .text








/*
void identity_4_4( float const * a,
                   float       * b,
                   int64_t       ld_a,
                   int64_t       ld_b,
                   int32_t       trans_b );
*/
    .global identity_4_4
identity_4_4:
    // load A
    ld1 { v0.4s }, [x0], x2
    ld1 { v1.4s }, [x0], x2
    ld1 { v2.4s }, [x0], x2
    ld1 { v3.4s }, [x0], x2

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
    st1 { v0.4s }, [x1], x3
    st1 { v1.4s }, [x1], x3
    st1 { v2.4s }, [x1], x3
    st1 { v3.4s }, [x1], x3
    ret






/*
void identity_16_16( float const * a,
                   float       * b,
                   int64_t       ld_a,
                   int64_t       ld_b,
                   int32_t       trans_b );
*/
    .global identity_16_16
identity_16_16:
    stp x29, x30, [sp, #-16]!
    mov fp, sp

    mov x9, x0
    mov x11, x1
    mov x5, #4
loop01:
    cbz x5, end01

    mov x10, x9
    mov x12, x11
    mov x6, #4
loop02:
    cbz x6, end02

    mov x0, x10
    mov x1, x12
    bl identity_4_4

    add x12, x12, x3, LSL #2
    add x10, x10, #16
    subs x6, x6, #1
    b loop02
end02:

    add x11, x11, #16
    add x9, x9, x2, LSL #2
    subs x5, x5, #1
    b loop01
end01:

    ldp x29, x30, [sp], #16
    ret

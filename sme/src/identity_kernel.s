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
    // prepare predicates
    ptrue p0.b
    ptrue p1.s, VL4
    ptrue p2.s, VL2
    not p3.b, p0/z, p2.b
    and p3.b, p0/z, p3.b, p1.b

    // load A
    ld1 { v0.4s }, [x0], x2
    ld1 { v1.4s }, [x0], x2
    ld1 { v2.4s }, [x0], x2
    ld1 { v3.4s }, [x0], x2

    cbz x4, notrans01

    // perform transpose on 2x2 submatrices
    trn1 z4.s, z0.s, z1.s
    trn2 z5.s, z0.s, z1.s
    trn1 z6.s, z2.s, z3.s
    trn2 z7.s, z2.s, z3.s

    // store result
    st1w z6.s, p2, [x1, #4*2]
    st1w z4.s, p2, [x1], x3
    st1w z7.s, p2, [x1, #4*2]
    st1w z5.s, p2, [x1], x3
    st1w z6.s, p3, [x1, #4*2]
    st1w z4.s, p3, [x1], x3
    st1w z7.s, p3, [x1, #4*2]
    st1w z5.s, p3, [x1], x3
    ret

notrans01:
    // store nontransposed variant
    st1 { v0.4s }, [x1], x3
    st1 { v1.4s }, [x1], x3
    st1 { v2.4s }, [x1], x3
    st1 { v3.4s }, [x1], x3
    ret

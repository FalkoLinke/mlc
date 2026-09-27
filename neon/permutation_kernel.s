#ifdef __APPLE__
#define FUNCLABEL(NAME) _##NAME
#else
#define FUNCLABEL(NAME) NAME
#endif /* __APPLE__ */


    .text


/*
Permutation operation abc->cba for row-major tensors with |a| = 8 and |b| = 4.

void perm_neon_abc_cba( int64_t       size_c,
                        float const * abc,
                        float       * cba );

Element (a, b, c) of the input is located at  abc[ (a * 4 + b) * size_c + c ],
element (c, b, a) of the output at            cba[ (c * 4 + b) * 8 + a ].

Scalar reference version: one element is moved per iteration of the innermost loop.

x0 - size_c
x1 - pointer to abc
x2 - pointer to cba
x5 - index of the current a
x6 - index of the current b
x7 - pointer to the current input element
x8 - pointer to the current output element
x9 - number of remaining c values
x10 - size_c * 4: distance between two b values of the input in bytes
x11 - the element currently being moved
*/
    .global FUNCLABEL(perm_neon_abc_cba_scalar)
FUNCLABEL(perm_neon_abc_cba_scalar):
    cbz x0, scalar_end                      // nothing to do for size_c = 0

    lsl x10, x0, #2                         // size_c * 4 bytes

    mov x5, #0
scalar_a_loop:
    mov x6, #0
scalar_b_loop:
    lsl x7, x5, #2                          // a * 4
    add x7, x7, x6                          // a * 4 + b
    mul x7, x7, x10                         // (a * 4 + b) * size_c * 4 bytes
    add x7, x1, x7                          // input pointer of (a, b, 0)

    lsl x8, x6, #3                          // b * 8
    add x8, x8, x5                          // b * 8 + a
    lsl x8, x8, #2                          // (b * 8 + a) * 4 bytes
    add x8, x2, x8                          // output pointer of (0, b, a)

    mov x9, x0
scalar_c_loop:
    ldr w11, [x7], #4                       // load (a, b, c), advance to c + 1
    str w11, [x8]                           // store (c, b, a)
    add x8, x8, #128                        // one c value of the output: 4 * 8 * 4 bytes
    subs x9, x9, #1
    b.ne scalar_c_loop

    add x6, x6, #1
    cmp x6, #4
    b.ne scalar_b_loop

    add x5, x5, #1
    cmp x5, #8
    b.ne scalar_a_loop

scalar_end:
    ret




/*
Permutation operation abc->cba for row-major tensors with |a| = 8 and |b| = 4.

void perm_neon_abc_cba( int64_t       size_c,
                        float const * abc,
                        float       * cba );

For a fixed b the operation is a transposition of the 8 x size_c matrix formed by the
a and c dimensions. The kernel therefore transposes two 4x4 blocks per iteration, which
covers all eight a values and four c values. The 8 resulting values of one c value are
contiguous in the output and are written with a single `stp`.
Remaining c values (size_c % 4) are moved element-wise by the tail loop.

x0 - size_c
x1 - pointer to abc
x2 - pointer to cba
x3 - input pointer of the current b
x4 - output pointer of the current b
x5 - number of remaining b values
x6 - 16 * size_c: distance between two a values of the input in bytes
x7 - number of remaining c values
x8 - input pointer of the current c block
x9 - output pointer of the current c block
x10 - size_c * 4: distance between two b values of the input in bytes
x11 - input pointer of the current a
x12 - the element currently being moved by the tail loop

v0 - v7   - the 8 loaded rows, one per a value
v16 - v23 - the transposed rows, two per c value
v24 - v31 - intermediate results of the transpositions
*/
    .global FUNCLABEL(perm_neon_abc_cba)
FUNCLABEL(perm_neon_abc_cba):
    cbz x0, perm_end                        // nothing to do for size_c = 0

    lsl x10, x0, #2                         // size_c * 4 bytes
    lsl x6, x10, #2                         // 4 * size_c * 4 bytes

    mov x3, x1
    mov x4, x2
    mov x5, #4

perm_b_loop:
    mov x7, x0                              // all c values are still to be processed
    mov x8, x3
    mov x9, x4

    cmp x7, #4
    b.lt perm_c_tail                        // less than one full block

perm_c_loop:
    mov x11, x8                             // load 4 c values for each of the 8 a values
    ldr q0, [x11]
    add x11, x11, x6
    ldr q1, [x11]
    add x11, x11, x6
    ldr q2, [x11]
    add x11, x11, x6
    ldr q3, [x11]
    add x11, x11, x6
    ldr q4, [x11]
    add x11, x11, x6
    ldr q5, [x11]
    add x11, x11, x6
    ldr q6, [x11]
    add x11, x11, x6
    ldr q7, [x11]

    trn1 v24.4s, v0.4s, v1.4s               // transpose the 4x4 block of a = 0 .. 3
    trn2 v25.4s, v0.4s, v1.4s
    trn1 v26.4s, v2.4s, v3.4s
    trn2 v27.4s, v2.4s, v3.4s
    zip1 v16.2d, v24.2d, v26.2d
    zip1 v17.2d, v25.2d, v27.2d
    zip2 v18.2d, v24.2d, v26.2d
    zip2 v19.2d, v25.2d, v27.2d

    trn1 v28.4s, v4.4s, v5.4s               // transpose the 4x4 block of a = 4 .. 7
    trn2 v29.4s, v4.4s, v5.4s
    trn1 v30.4s, v6.4s, v7.4s
    trn2 v31.4s, v6.4s, v7.4s
    zip1 v20.2d, v28.2d, v30.2d
    zip1 v21.2d, v29.2d, v31.2d
    zip2 v22.2d, v28.2d, v30.2d
    zip2 v23.2d, v29.2d, v31.2d

    stp q16, q20, [x9]                      // all 8 a values of one c value are contiguous
    stp q17, q21, [x9, #128]                // one c value of the output: 4 * 8 * 4 bytes
    stp q18, q22, [x9, #256]
    stp q19, q23, [x9, #384]

    add x8, x8, #16                         // advance by 4 c values
    add x9, x9, #512
    sub x7, x7, #4
    cmp x7, #4
    b.ge perm_c_loop

perm_c_tail:
    cbz x7, perm_b_next

perm_c_tail_loop:
    mov x11, x8
    ldr w12, [x11]                          // move the 8 a values of one c value
    str w12, [x9]
    add x11, x11, x6
    ldr w12, [x11]
    str w12, [x9, #4]
    add x11, x11, x6
    ldr w12, [x11]
    str w12, [x9, #8]
    add x11, x11, x6
    ldr w12, [x11]
    str w12, [x9, #12]
    add x11, x11, x6
    ldr w12, [x11]
    str w12, [x9, #16]
    add x11, x11, x6
    ldr w12, [x11]
    str w12, [x9, #20]
    add x11, x11, x6
    ldr w12, [x11]
    str w12, [x9, #24]
    add x11, x11, x6
    ldr w12, [x11]
    str w12, [x9, #28]

    add x8, x8, #4                          // advance by one c value
    add x9, x9, #128
    subs x7, x7, #1
    b.ne perm_c_tail_loop

perm_b_next:
    add x3, x3, x10                         // advance to the next b value
    add x4, x4, #32                         // one b value of the output: 8 * 4 bytes
    subs x5, x5, #1
    b.ne perm_b_loop

perm_end:
    ret

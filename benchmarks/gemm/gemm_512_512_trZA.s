#ifdef __APPLE__
#define FUNCLABEL(NAME) _##NAME
#else
#define FUNCLABEL(NAME) NAME
#endif /* __APPLE__ */


.text

/*
    * @brief GEMM that computes: C+=AB.
    * @param a    Pointer to matrix A in mk layout: element (m,k) at a[m*ld_a + k].
    * @param b    Pointer to row-major matrix B: element (k,n) at b[k*ld_b + n].
    * @param c    Pointer to column-major matrix C.
    * @param ld_a Leading dimension of A.
    * @param ld_b Leading dimension of B.
    * @param ld_c Leading dimension of C.

   void gemm_512_512_trZA( float   const * a,
                        float   const * b,
                        float         * c,
                        int64_t         ld_a,
                        int64_t         ld_b,
                        int64_t         ld_c );
*/

// one k step of the 2x2-tile outer product:
// za0 = C(m 0:15,  n 0:15),  za1 = C(m 16:31, n 0:15)
// za2 = C(m 0:15,  n 16:31), za3 = C(m 16:31, n 16:31)
// \mb0 / \mb1 = A m-vectors of block 0 / block 1 for this k
.macro K_GROUP mb0, mb1
    ld1w { z30.S, z31.S }, pn8/Z, [x8]
    add x8, x8, x17 // x17 = ld_b*4, next k row of B
    fmopa za0.s, p0/m, p0/m, z30.s, z\mb0\().s
    fmopa za1.s, p0/m, p0/m, z30.s, z\mb1\().s
    fmopa za2.s, p0/m, p0/m, z31.s, z\mb0\().s
    fmopa za3.s, p0/m, p0/m, z31.s, z\mb1\().s
.endm

    .global FUNCLABEL(gemm_512_512_trZA)
FUNCLABEL(gemm_512_512_trZA):

    stp x29, x30, [sp, #-16]!
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!
    mov x29, sp

    smstart
    ptrue p0.s
    ptrue pn8.s

    rdsvl x9, #1   // bytes per vector
    lsr x9, x9, #2 // floats per vector

    lsl x10, x5, #2  // x10 = ld_c*4, used for pointer arithmetic when loading/storing C
    lsl x11, x3, #2  // x11 = ld_a*4, used for pointer arithmetic when loading A in K loop
    lsl x17, x4, #2  // x17 = ld_b*4, used for pointer arithmetic when loading B in K loop

    // N Variables loop counter in x16
    mov x16, #0
N_loop:
    // M Variables loop counter in x15
    mov x15, #0
M_loop:
    // load C block (32x32): za0/za1 = columns n..n+15, za2/za3 = columns n+16..n+31
    mov x6, x2
    add x6, x6, x15, lsl #2 // x15*4 bytes weiter springen, zum richtigen M streifen
    mul x14, x16, x5
    add x6, x6, x14, lsl #2 // zum richtigen N streifen
    mov w12, #0

    .rept 8
    ld1w { z0.S, z1.S }, pn8/Z, [x6]
    mova za0h.S[w12, #0], p0/m, z0.S
    mova za1h.S[w12, #0], p0/m, z1.S
    add x6, x6, x10
    ld1w { z2.S, z3.S }, pn8/Z, [x6]
    mova za0h.S[w12, #1], p0/m, z2.S
    mova za1h.S[w12, #1], p0/m, z3.S
    add x6, x6, x10
    add w12, w12, #2
    .endr

    mov w12, #0
    .rept 8
    ld1w { z0.S, z1.S }, pn8/Z, [x6]
    mova za2h.S[w12, #0], p0/m, z0.S
    mova za3h.S[w12, #0], p0/m, z1.S
    add x6, x6, x10
    ld1w { z2.S, z3.S }, pn8/Z, [x6]
    mova za2h.S[w12, #1], p0/m, z2.S
    mova za3h.S[w12, #1], p0/m, z3.S
    add x6, x6, x10
    add w12, w12, #2
    .endr

    // A/B pointers for this (m,n) block
    mov x7, x0
    mul x14, x15, x3        // m-block offset: x15 rows of A, ld_a floats each
    add x7, x7, x14, lsl #2
    mov x8, x1
    add x8, x8, x16, lsl #2 // n-block offset within the B rows

    // K loop counter in x6 (x6 is free until the store epilogue;
    // w12-w15 are all used as ZA slice selects inside the loop)
    mov x6, #512
K_loop:

    // spill accumulator tile za0 to the stack: the A transpose staging below
    // clobbers it. za0.s consists of ZA array vectors 0,4,8,...,60.
    addsvl  sp, sp, #-16
    mov     w14, #0
    mov     x13, sp
    .rept 16
    str     za[w14, #0], [x13]
    add     w14, w14, #4
    addsvl  x13, x13, #1
    .endr

    // transpose A block 0 (rows m..m+15, current 16 k values) through za0
    mov w13, #0
    .rept 16
    ld1w { z0.S }, p0/Z, [x7]
    mova za0h.S[w13, #0], p0/m, z0.S
    add x7, x7, x11
    add w13, w13, #1
    .endr

    // read the transposed block out of the vertical slices: z0-z15 = k 0..15
    mov w14, #0
    mova { z0.S - z3.S }, za0v.S[w14, 0:3 ]
    add w14, w14, #4
    mova { z4.S - z7.S }, za0v.S[w14, 0:3 ]
    add w14, w14, #4
    mova { z8.S - z11.S }, za0v.S[w14, 0:3 ]
    add w14, w14, #4
    mova { z12.S - z15.S }, za0v.S[w14, 0:3 ]

    // transpose A block 1 (rows m+16..m+31); z16 as temp, z0-z15 are live
    mov w13, #0
    .rept 16
    ld1w { z16.S }, p0/Z, [x7]
    mova za0h.S[w13, #0], p0/m, z16.S
    add x7, x7, x11
    add w13, w13, #1
    .endr

    mov w14, #0
    mova { z16.S - z19.S }, za0v.S[w14, 0:3 ]
    add w14, w14, #4
    mova { z20.S - z23.S }, za0v.S[w14, 0:3 ]
    add w14, w14, #4
    mova { z24.S - z27.S }, za0v.S[w14, 0:3 ]
    add w14, w14, #4
    mova { z28.S - z31.S }, za0v.S[w14, 0:3 ]

    // restore accumulator tile za0 from the stack
    mov     w14, #0
    mov     x13, sp
    .rept 16
    ldr     za[w14, #0], [x13]
    add     w14, w14, #4
    addsvl  x13, x13, #1
    .endr
    addsvl  sp, sp, #16

    // park z30/z31 (A block 1, k=14/15) on the stack: the B loads reuse z30/z31
    addsvl  sp, sp, #-2
    str     z30, [sp]
    str     z31, [sp, #1, mul vl]

    K_GROUP 0, 16
    K_GROUP 1, 17
    K_GROUP 2, 18
    K_GROUP 3, 19
    K_GROUP 4, 20
    K_GROUP 5, 21
    K_GROUP 6, 22
    K_GROUP 7, 23
    K_GROUP 8, 24
    K_GROUP 9, 25
    K_GROUP 10, 26
    K_GROUP 11, 27
    K_GROUP 12, 28
    K_GROUP 13, 29

    // pop A block 1 k=14/15 (z0/z1 were consumed by the first two groups)
    ldr     z0, [sp]
    ldr     z1, [sp, #1, mul vl]
    addsvl  sp, sp, #2

    K_GROUP 14, 0
    K_GROUP 15, 1

    // next k block: rewind the 32 staged A rows, advance 16 floats in k
    sub x7, x7, x11, lsl #5
    add x7, x7, #64

    subs x6, x6, #16
    b.ne K_loop


    // store the results back to C
    mov x6, x2
    add x6, x6, x15, lsl #2 // x15*4 bytes weiter springen, zum richtigen M streifen
    mul x14, x16, x5
    add x6, x6, x14, lsl #2 // zum richtigen N streifen
    mov w12, #0

    .rept 16
    mov x13, #0
    st1w { za0h.S[w12, 0] }, p0, [x6, x13, lsl #2]
    mov x13, #16
    st1w { za1h.S[w12, 0] }, p0, [x6, x13, lsl #2]
    add w12, w12, #1
    add x6, x6, x10
    .endr

    mov w12, #0

    .rept 16
    mov x13, #0
    st1w { za2h.S[w12, 0] }, p0, [x6, x13, lsl #2]
    mov x13, #16
    st1w { za3h.S[w12, 0] }, p0, [x6, x13, lsl #2]
    add w12, w12, #1
    add x6, x6, x10
    .endr


    // check if we have processed all M tiles
    add x15, x15, x9, lsl #1 // we process 2 tiles (2*16=32 floats) at a time
    cmp x15, x3
    b.ne M_loop

    // N Tiles check
    add x16, x16, x9, lsl #1 // we process 2 tiles (2*16=32 floats) at a time
    cmp x16, x4
    b.ne N_loop


    smstop
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ldp x29, x30, [sp], #16
    ret

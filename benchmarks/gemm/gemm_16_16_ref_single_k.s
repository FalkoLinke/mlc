#ifdef __APPLE__
#define FUNCLABEL(NAME) _##NAME
#else
#define FUNCLABEL(NAME) NAME
#endif /* __APPLE__ */


.text


/*
    * @brief GEMM that computes: C+=AB.
    * @param a    Pointer to column-major matrix A.
    * @param b    Pointer to row-major matrix B.
    * @param c    Pointer to column-major matrix C.
    * @param ld_a Leading dimension of A.
    * @param ld_b Leading dimension of B.
    * @param ld_c Leading dimension of C.

   void gemm_16_16( float   const * a,
                      float   const * b,
                      float         * c,
                      int64_t         ld_a,
                      int64_t         ld_b,
                      int64_t         ld_c );
*/
    .global FUNCLABEL(gemm_16_16_ref_single_k)
FUNCLABEL(gemm_16_16_ref_single_k):


    smstart
    ptrue p0.s, vl16
    ptrue pn8.s


    // K Variables loop counter in x29
    mov x9, #512
    
    mov x6, x2  // C pointer
    mov x7, x0  // A pointer
    mov x8, x1  // B pointer

    lsl x10, x5, #2  // x10 = ld_c*4, used for pointer arithmetic when loading/storing C
    lsl x11, x3, #2  // x11 = ld_a*4, used for pointer arithmetic when loading A in K loop
    lsl x17, x4, #2  // x16 = ld_b*4, used for pointer arithmetic when loading B in K loop
    
    // load C
    mov w12, #0


    .rept 4
    ld1w { z0.S, z1.S }, pn8/Z, [x6]
    mova za0h.S[w12, #0], p0/m, z0.S
    mova za0h.S[w12, #1], p0/m, z1.S
    add x6, x6, x10, lsl #1
    ld1w { z2.S, z3.S }, pn8/Z, [x6]
    mova za0h.S[w12, #2], p0/m, z2.S
    mova za0h.S[w12, #3], p0/m, z3.S
    add x6, x6, x10, lsl #1
    add w12, w12, #4
    .endr

K_loop:

    // load A z0 and B z2 16 floats at a time and perform the outer product
    ld1w {z0.S}, p0/Z, [x7]
    ld1w {z2.S}, p0/Z, [x8]
    fmopa za0.s, p0/m, p0/m, z2.s, z0.s

    add x7, x7, x11
    add x8, x8, x17

    subs x9, x9, #1
    b.ne K_loop


    // store C Results
    mov w12, #0
    mov x6, x2

    .rept 16
    st1w { za0h.S[w12, 0] }, p0, [x6]
    add w12, w12, #1
    add x6, x6, x10
    .endr 

    smstop
    
    ret

//     smstart
//     ptrue p0.s, vl16

//     // K Variables loop counter in x29
//     mov x9, #512
//     mov x7, x0
//     lsl x11, x3, #2
//     mov x8, x1
//     lsl x12, x4, #2

//     // load C
//     mov w12, #0
//     mov x6, x2
//     lsl x10, x5, #2
    
//     .rept 4
    
//     ldr za[w12, #0], [x6]
//     add x6, x6, x10
//     add w12, w12, #4
//     ldr za[w12, #0], [x6]
//     add x6, x6, x10
//     add w12, w12, #4
//     ldr za[w12, #0], [x6]
//     add x6, x6, x10
//     add w12, w12, #4
//     ldr za[w12, #0], [x6]
//     add x6, x6, x10
//     add w12, w12, #4

//     .endr

// K_loop:

//     // load A z0 and B z2 16 floats at a time and perform the outer product
//     ldr z0, [x7, #0, mul vl]
//     ldr z2, [x8, #0, mul vl]
//     fmopa za0.s, p0/m, p0/m, z2.s, z0.s

//     add x7, x7, x11
//     add x8, x8, x12

//     subs x9, x9, #1
//     cbnz x9, K_loop


//     // store Results
//     mov w12, #0
//     mov x6, x2

//     .rept 4

//     str za[w12, #0], [x6]
//     add x6, x6, x10
//     add w12, w12, #4
//     str za[w12, #0], [x6]
//     add x6, x6, x10
//     add w12, w12, #4
//     str za[w12, #0], [x6]
//     add x6, x6, x10
//     add w12, w12, #4
//     str za[w12, #0], [x6]
//     add x6, x6, x10
//     add w12, w12, #4

//     .endr

//     smstop
    
//     ret
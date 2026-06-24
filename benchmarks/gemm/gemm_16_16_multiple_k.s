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
    .global FUNCLABEL(gemm_16_16_multiple_k)
FUNCLABEL(gemm_16_16_multiple_k):


    smstart
    ptrue p0.s, vl16
    ptrue pn8.s


    // K Variables loop counter in x9
    mov x9, #512
    
    mov x7, x0  // A pointer
    mov x8, x1  // B pointer

    lsl x11, x3, #2  // x11 = ld_a*4, used for pointer arithmetic when loading A in K loop
    lsl x12, x4, #2  // x12 = ld_b*4, used for pointer arithmetic when loading B in K loop
    lsl x10, x5, #2  // x10 = ld_c*4, used for pointer arithmetic when loading/storing C
    
    // load C
    mov x6, x2  // C pointer
    mov w13, #0
    
    .rept 4
    ld1w { z0.S, z1.S }, pn8/Z, [x6]
    mova za0h.S[w13, #0], p0/m, z0.S
    mova za0h.S[w13, #1], p0/m, z1.S
    add x6, x6, x10, lsl #1
    ld1w { z2.S, z3.S }, pn8/Z, [x6]
    mova za0h.S[w13, #2], p0/m, z2.S
    mova za0h.S[w13, #3], p0/m, z3.S
    add x6, x6, x10, lsl #1
    add w13, w13, #4
    .endr

K_loop:

    // load A z0 and B z2 16 floats at a time and perform the outer product
    ld1w {z0.S}, p0/z, [x7]
    ld1w {z2.S}, p0/z, [x8]
    fmopa za0.s, p0/m, p0/m, z2.s, z0.s

    add x7, x7, x11
    add x8, x8, x12

    ld1w {z0.S}, p0/z, [x7]
    ld1w {z2.S}, p0/z, [x8]
    fmopa za1.s, p0/m, p0/m, z2.s, z0.s
       
    add x7, x7, x11
    add x8, x8, x12
        
    ld1w {z0.S}, p0/z, [x7]
    ld1w {z2.S}, p0/z, [x8]
    fmopa za2.s, p0/m, p0/m, z2.s, z0.s

    add x7, x7, x11
    add x8, x8, x12
        
    ld1w {z0.S}, p0/z, [x7]
    ld1w {z2.S}, p0/z, [x8]
    fmopa za3.s, p0/m, p0/m, z2.s, z0.s

    add x7, x7, x11
    add x8, x8, x12

    subs x9, x9, #4
    b.ne K_loop


    // store Results
    mov w13, #0
    mov x6, x2


    .rept 16
    st1w { za0h.S[w13, 0] }, p0, [x6]
    add w13, w13, #1
    add x6, x6, x10
    .endr 

    smstop
    
    ret
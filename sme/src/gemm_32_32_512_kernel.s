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

   void gemm_32_32_512( float   const * a,
                      float   const * b,
                      float         * c,
                      int64_t         ld_a,
                      int64_t         ld_b,
                      int64_t         ld_c );
*/
    .global FUNCLABEL(gemm_32_32_512)
FUNCLABEL(gemm_32_32_512):


    smstart
    ptrue p0.s
    // K Variables loop counter in x29
    mov x29, #512
    mov x7, x0
    mov x8, x1


    // load C
    mov x6, x2
    mov w12, #0
    mov w13, #2
    
    .rept 4
    ldr za[w12, #0], [x6, #0, mul vl]
    ldr za[w12, #1], [x6, #1, mul vl]
    add w12, w12, #4
    addvl x6, x6, #2

    ldr za[w12, #0], [x6, #0, mul vl]
    ldr za[w12, #1], [x6, #1, mul vl]
    add w12, w12, #4
    addvl x6, x6, #2

    ldr za[w12, #0], [x6, #0, mul vl]
    ldr za[w12, #1], [x6, #1, mul vl]
    add w12, w12, #4
    addvl x6, x6, #2

    ldr za[w12, #0], [x6, #0, mul vl]
    ldr za[w12, #1], [x6, #1, mul vl]
    add w12, w12, #4
    addvl x6, x6, #2
    .endr

    .rept 4
    ldr za[w13, #0], [x6, #0, mul vl]
    ldr za[w13, #1], [x6, #1, mul vl]
    add w13, w13, #4
    addvl x6, x6, #2

    ldr za[w13, #0], [x6, #0, mul vl]
    ldr za[w13, #1], [x6, #1, mul vl]
    add w13, w13, #4
    addvl x6, x6, #2

    ldr za[w13, #0], [x6, #0, mul vl]
    ldr za[w13, #1], [x6, #1, mul vl]
    add w13, w13, #4
    addvl x6, x6, #2

    ldr za[w13, #0], [x6, #0, mul vl]
    ldr za[w13, #1], [x6, #1, mul vl]
    add w13, w13, #4
    addvl x6, x6, #2
    .endr

K_loop:

    // load A z0 and B z2 16 floats at a time and perform the outer product for tile 0
    ldr z0, [x7, #0, mul vl]
    ldr z2, [x8, #0, mul vl]
    fmopa za0.s, p0/m, p0/m, z2.s, z0.s

    //tile 1
    ldr z1, [x7, #1, mul vl]
    fmopa za1.s, p0/m, p0/m, z2.s, z1.s

    //tile 2
    ldr z3, [x8, #1, mul vl]
    fmopa za2.s, p0/m, p0/m, z3.s, z0.s
    
    //tile 3
    fmopa za3.s, p0/m, p0/m, z3.s, z1.s

    addvl x7, x7, #2
    addvl x8, x8, #2

    subs x29, x29, #1
    cbnz x29, K_loop


    // store the results back to C
    // first za registers
    mov x6, x2
    mov w12, #0
    mov w13, #2
    
    .rept 4
    str za[w12, #0], [x6, #0, mul vl]
    str za[w12, #1], [x6, #1, mul vl]
    add w12, w12, #4
    addvl x6, x6, #2

    str za[w12, #0], [x6, #0, mul vl]
    str za[w12, #1], [x6, #1, mul vl]
    add w12, w12, #4
    addvl x6, x6, #2

    str za[w12, #0], [x6, #0, mul vl]
    str za[w12, #1], [x6, #1, mul vl]
    add w12, w12, #4
    addvl x6, x6, #2

    str za[w12, #0], [x6, #0, mul vl]
    str za[w12, #1], [x6, #1, mul vl]
    add w12, w12, #4
    addvl x6, x6, #2
    .endr

    .rept 4
    str za[w13, #0], [x6, #0, mul vl]
    str za[w13, #1], [x6, #1, mul vl]
    add w13, w13, #4
    addvl x6, x6, #2

    str za[w13, #0], [x6, #0, mul vl]
    str za[w13, #1], [x6, #1, mul vl]
    add w13, w13, #4
    addvl x6, x6, #2

    str za[w13, #0], [x6, #0, mul vl]
    str za[w13, #1], [x6, #1, mul vl]
    add w13, w13, #4
    addvl x6, x6, #2

    str za[w13, #0], [x6, #0, mul vl]
    str za[w13, #1], [x6, #1, mul vl]
    add w13, w13, #4
    addvl x6, x6, #2
    .endr

    smstop
    
    ret

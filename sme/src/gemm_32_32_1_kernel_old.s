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

   void gemm_32_32_1( float   const * a,
                      float   const * b,
                      float         * c,
                      int64_t         ld_a,
                      int64_t         ld_b,
                      int64_t         ld_c );
*/
    .global FUNCLABEL(gemm_32_32_1)
FUNCLABEL(gemm_32_32_1):


    smstart
    // C counter
    lsr x29, x5, #1
    // first za reg
    mov w12, #0
    mov w13, #2

    mov x6, x2
    addvl x7, x2, #31

    ptrue p0.s

    // load C
load_C_loop:

    //tile 0 and 1
    ldr za[w12, #0], [x6, #0, mul vl]
    ldr za[w12, #1], [x6, #1, mul vl]
    
    add w12, w12, #2
    addvl x6, x6, #2

    //tile 2 and 3
    ldr za[w13, #0], [x7, #0, mul vl]
    ldr za[w13, #1], [x7, #1, mul vl]

    add w13, w13, #2
    addvl x7, x7, #2

    subs x29, x29, #1
    cbnz x29, load_C_loop


    // load A z0 and B z2 16 floats at a time and perform the outer product for tile 0
    ldr z0, [x0, #0, mul vl]
    ldr z2, [x1, #0, mul vl]
    fmopa za0.s, p0/m, p0/m, z0.s, z2.s

    //tile 1
    ldr z1, [x0, #1, mul vl]
    fmopa za1.s, p0/m, p0/m, z1.s, z2.s

    //tile 2
    ldr z3, [x1, #1, mul vl]
    fmopa za2.s, p0/m, p0/m, z0.s, z3.s
    
    //tile 3
    fmopa za3.s, p0/m, p0/m, z1.s, z3.s



    // store the results back to C
    lsr x29, x5, #1
    // first za reg
    mov w12, #0
    mov w13, #2

    mov x6, x2
    addvl x7, x2, #31

store_C_loop:

    str za[w12, #0], [x6, #0, mul vl]
    str za[w12, #1], [x6, #1, mul vl]
    
    add w12, w12, #2
    addvl x6, x6, #2

    str za[w13, #0], [x7, #0, mul vl]
    str za[w13, #1], [x7, #1, mul vl]

    add w13, w13, #2
    addvl x7, x7, #2

    subs x29, x29, #1
    cbnz x29, store_C_loop

    smstop
    
    ret

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
    .global FUNCLABEL(gemm_16_16)
FUNCLABEL(gemm_16_16):


    smstart
    ptrue p0.s

    // K Variables loop counter in x29
    mov x29, #512
    mov x7, x0
    mov x8, x1

    // load C
    mov w12, #0
    mov x6, x2
    
    .rept 4
    
    ldr za[w12, #0], [x6]
    addvl x6, x6, #1
    add w12, w12, #4
    ldr za[w12, #0], [x6]
    addvl x6, x6, #1
    add w12, w12, #4
    ldr za[w12, #0], [x6]
    addvl x6, x6, #1
    add w12, w12, #4
    ldr za[w12, #0], [x6]
    addvl x6, x6, #1
    add w12, w12, #4

    .endr

K_loop:

    // load A z0 and B z2 16 floats at a time and perform the outer product
    ldr z0, [x7, #0, mul vl]
    ldr z2, [x8, #0, mul vl]
    fmopa za0.s, p0/m, p0/m, z2.s, z0.s

    addvl x7, x7, #1
    addvl x8, x8, #1

    subs x29, x29, #1
    cbnz x29, K_loop


    // store Results
    mov w12, #0
    mov x6, x2

    .rept 4

    str za[w12, #0], [x6]
    addvl x6, x6, #1
    add w12, w12, #4
    str za[w12, #0], [x6]
    addvl x6, x6, #1
    add w12, w12, #4
    str za[w12, #0], [x6]
    addvl x6, x6, #1
    add w12, w12, #4
    str za[w12, #0], [x6]
    addvl x6, x6, #1
    add w12, w12, #4

    .endr

    smstop
    
    ret
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

   void gemm_16_16_512( float   const * a,
                      float   const * b,
                      float         * c,
                      int64_t         ld_a,
                      int64_t         ld_b,
                      int64_t         ld_c );
*/
    .global FUNCLABEL(gemm_16_16_512)
FUNCLABEL(gemm_16_16_512):

    // C stride
    mov x30, x5
    lsr x30, x30, #1
    // first za reg
    mov w12, #0
    mov x6, x3
    lsl x7, x6, #5 // x7 = x6 * 32 (size of one column of C in bytes)

    smstart
    ptrue p0.s

    // load C
load_C_loop_16:
    subs x30, x30, #1

    ldr za[w12, #0], [x6, #0, mul vl]
    ldr za[w12, #1], [x6, #1, mul vl]

    ldr za[w12, #2], [x7, #0, mul vl]
    ldr za[w12, #3], [x7, #1, mul vl]

    add w12, w12, #4
    addvl, x6, x6, #2
    addvl, x7, x7, #2

    cbnz x30, load_C_loop_16


    // load A z0 and B z2 16 floats at a time and perform the outer product
    ld1w {z0.s}, p0/z, [x0, #0, mul vl]
    ld1w {z2.s}, p0/z, [x1, #0, mul vl]
    fmopa za0.s, p0/m, p0/m, z0.s, z2.s

    ld1w {z1.s}, p0/z, [x0, #1, mul vl]
    fmopa za1.s, p0/m, p0/m, z1.s, z2.s

    ld1w {z3.s}, p0/z, [x1, #1, mul vl]
    fmopa za2.s, p0/m, p0/m, z0.s, z3.s
    
    fmopa za3.s, p0/m, p0/m, z1.s, z3.s



    // store the results back to C
    mov x30, x5
    lsr x30, x30, #1
    mov w12, #0
    mov x6, x3
    lsl x7, x6, #5 // x7 = x6 * 32 (size of one column of C in bytes)

store_C_loop_16:
    subs x30, x30, #1

    ldr za[w12, #0], [x6, #0, mul vl]
    ldr za[w12, #1], [x6, #1, mul vl]

    ldr za[w12, #2], [x7, #0, mul vl]
    ldr za[w12, #3], [x7, #1, mul vl]

    add w12, w12, #4
    addvl x6, x6, #4
    addvl x7, x7, #4

    cbnz x30, store_C_loop_16

    smstop
    
    ret
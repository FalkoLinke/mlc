#ifdef __APPLE__
#define FUNCLABEL(NAME) _##NAME
#else
#define FUNCLABEL(NAME) NAME
#endif /* __APPLE__ */


    .text



/*
       /**
    * @brief GEMM that computes: C+=AB.
    * @param a    Pointer to column-major matrix A.
    * @param b    Pointer to row-major matrix B.
    * @param c    Pointer to column-major matrix C.
    * @param ld_a Leading dimension of A.
    * @param ld_b Leading dimension of B.
    * @param ld_c Leading dimension of C.
    **/
   void gemm_32_32_1( float   const * a,
                      float   const * b,
                      float         * c,
                      int64_t         ld_a,
                      int64_t         ld_b,
                      int64_t         ld_c );
*/
    .global FUNCLABEL(gemm_32_32_1)
FUNCLABEL(gemm_32_32_1):

    mov w12, #0
    mov x6, x3
    // C stride
    mov x30, x5

    smstart
    ptrue p0.s

    // load C
load_C_loop:
    subs x30, x30, #1

    ldr za[w12], [x6, #0, mul vl]
    ldr za[w12, #1], [x6, #1, mul vl]

    // ldr za[w12, #2], [x6, #2, mul vl]
    // ldr za[w12, #3], [x6, #3, mul vl]

    // ldr za[w12, #4], [x6, #4, mul vl]
    // ldr za[w12, #5], [x6, #5, mul vl]

    // ldr za[w12, #6], [x6, #6, mul vl]
    // ldr za[w12, #7], [x6, #7, mul vl]

    add w12, w12, #2
    addvl, x6, x6, #2

    cbnz x30, load_C_loop


    // load A
    ld1w {z0.s}, p0/z, [x0]
    ld1w {z1.s}, p0/z, [x0, #64]
    
    // load B
    ld1w {z2.s}, p0/z, [x1]
    ld1w {z3.s}, p0/z, [x1, #64]

    fmopa za0.s, p0/m, p0/m, z0.s, z2.s
    fmopa za1.s, p0/m, p0/m, z1.s, z3.s






    smstop
    
    ret

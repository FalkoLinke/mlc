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

   void gemm_512_512_512( float   const * a,
                        float   const * b,
                        float         * c,
                        int64_t         ld_a,
                        int64_t         ld_b,
                        int64_t         ld_c );
*/
    .global FUNCLABEL(gemm_512_512_512)
FUNCLABEL(gemm_512_512_512):


    smstart
    ptrue p0.s
    rdsvl x9, #1 // Vector Length 
    lsr x9, x9, #2 // floats in Vector length
    
    lsl x10, x5, #2  // x10 = ld_c*4, used for pointer arithmetic when loading/storing C
    lsl x11, x3, #2  // x11 = ld_a*4, used for pointer arithmetic when loading A in K loop
    lsl x16, x2, #2  // x16 = ld_b*4, used for pointer arithmetic when loading B in K loop

    // N Variables loop counter in x16
    mov x16, #0
N_loop:
    // M Variables loop counter in x15
    mov x15, #0
M_loop:
    // load C
    mov x6, x2
    add x6, x6, x15, lsl #2 // x15*4 bytes weiter springen, zum richtigen M streifen
    mov w12, #0
    mov w13, #2
    
    .rept 8
    ldr za[w12, #0], [x6, #0, mul vl]
    ldr za[w12, #1], [x6, #1, mul vl]
    add w12, w12, #4
    add x6, x6, x10

    ldr za[w12, #0], [x6, #0, mul vl]
    ldr za[w12, #1], [x6, #1, mul vl]
    add w12, w12, #4
    add x6, x6, x10
    .endr

    .rept 8
    ldr za[w13, #0], [x6, #0, mul vl]
    ldr za[w13, #1], [x6, #1, mul vl]
    add w13, w13, #4
    add x6, x6, x10

    ldr za[w13, #0], [x6, #0, mul vl]
    ldr za[w13, #1], [x6, #1, mul vl]
    add w13, w13, #4
    add x6, x6, x10
    .endr

    mov x7, x0
    mov x8, x1
    add x7, x7, x15, lsl #2 // x15*4 bytes weiter springen, zum richtigen M streifen

    // K Variables loop counter in x29
    mov x14, #512
K_loop:

    // load A and B 32 floats and perform the outer product
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

    // move to the next K tile
    // A is column-major, so we move in K dimension by adding ld_a (x3) to the pointer
    add x7, x7, x11 // x11 = ld_a*4, move to the next K tile for A
    addvl x8, x8, #2

    subs x14, x14, #1
    cbnz x14, K_loop


    // store the results back to C
    mov x6, x2
    add x6, x6, x15, lsl #2 // x15*4 bytes weiter springen, zum richtigen M streifen
    mov w12, #0
    mov w13, #2
    
    .rept 8
    str za[w12, #0], [x6, #0, mul vl]
    str za[w12, #1], [x6, #1, mul vl]
    add w12, w12, #4
    add x6, x6, x10

    str za[w12, #0], [x6, #0, mul vl]
    str za[w12, #1], [x6, #1, mul vl]
    add w12, w12, #4
    add x6, x6, x10
    .endr

    .rept 8
    str za[w13, #0], [x6, #0, mul vl]
    str za[w13, #1], [x6, #1, mul vl]
    add w13, w13, #4
    add x6, x6, x10

    str za[w13, #0], [x6, #0, mul vl]
    str za[w13, #1], [x6, #1, mul vl]
    add w13, w13, #4
    add x6, x6, x10
    .endr

    // check if we have processed all M tiles (ld_a(x3)/floats in vector length(x9))
    add x15, x15, x9, lsl #1 // processed elements in M dimension, x9 floats per tile, we process 2 tiles (32*2=64 floats) at a time
    cmp x15, x3
    b.ne M_loop

    // N Tiles check
    add x16, x16, x9, lsl #1 // processed elements in N dimension, x9 floats per tile, we process 2 tiles (32*2=64 floats) at a time
    cmp x15, x2
    b.ne M_loop

    smstop
    
    ret

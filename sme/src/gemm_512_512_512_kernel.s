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

    #str x19, [sp, #16]

    smstart
    ptrue p0.s
    ptrue pn8.s
    
    rdsvl x9, #1 // Vector Length 
    lsr x9, x9, #2 // floats in Vector length
    
    lsl x10, x5, #2  // x10 = ld_c*4, used for pointer arithmetic when loading/storing C
    lsl x11, x3, #2  // x11 = ld_a*4, used for pointer arithmetic when loading A in K loop
    lsl x17, x4, #2  // x16 = ld_b*4, used for pointer arithmetic when loading B in K loop

    // N Variables loop counter in x16
    mov x16, #0
N_loop:
    // M Variables loop counter in x15
    mov x15, #0
M_loop:
    // load C
    mov x6, x2
    add x6, x6, x15, lsl #2 // x15*4 bytes weiter springen, zum richtigen M streifen
    mul x14, x16, x5
    add x6, x6, x14, lsl #2 // zum richtigen N streifen
    mov w12, #0
    mov w13, #2


    .rept 8
    ld1w { z0.S, z1.S }, pn8/Z, [x6]
    mova za0h.S[w12, #0], p0/m, z0.S
    mova za1h.S[w12, #0], p0/m, z1.S
    add x6, x6, x10
    ld1w { z2.S, z3.S }, pn8/Z, [x6]
    mova za0h.S[w12, #1], p0/m, z2.S
    mova za1h.S[w12, #1], p0/m, z3.S
    add x6, x6, x10
    // ld1w { z4.S, z5.S }, pn8/Z, [x6]
    // mova za0h.S[w12, #2], p0/m, z4.S
    // mova za1h.S[w12, #2], p0/m, z5.S
    // add x6, x6, x10
    // ld1w { z6.S, z7.S }, pn8/Z, [x6]
    // mova za0h.S[w12, #3], p0/m, z6.S
    // mova za1h.S[w12, #3], p0/m, z7.S
    // add x6, x6, x10
    add w12, w12, #2
    .endr

    .rept 8
    ld1w { z0.S, z1.S }, pn8/Z, [x6]
    mova za2h.S[w12, #0], p0/m, z0.S
    mova za3h.S[w12, #0], p0/m, z1.S
    add x6, x6, x10
    ld1w { z2.S, z3.S }, pn8/Z, [x6]
    mova za2h.S[w12, #1], p0/m, z2.S
    mova za3h.S[w12, #1], p0/m, z3.S
    add x6, x6, x10
    // ld1w { z4.S, z5.S }, pn8/Z, [x6]
    // mova za2h.S[w12, #2], p0/m, z4.S
    // mova za3h.S[w12, #2], p0/m, z5.S
    // add x6, x6, x10
    // ld1w { z6.S, z7.S }, pn8/Z, [x6]
    // mova za2h.S[w12, #3], p0/m, z6.S
    // mova za3h.S[w12, #3], p0/m, z7.S
    // add x6, x6, x10
    add w12, w12, #2
    .endr


    #mov x19, #0
    #mov x13, #16
    // .rept 16
    // mov x13, #0
    // ld1w { za0h.S[w12, 0] }, p0/Z, [x6, x13, lsl #2]
    // mov x13, #16
    // ld1w { za1h.S[w12, 0] }, p0/Z, [x6, x13, lsl #2]
    // add w12, w12, #1
    // add x6, x6, x10
    // .endr 

    // mov w12, #0

    // .rept 16
    // mov x13, #0
    // ld1w { za2h.S[w12, 0] }, p0/Z, [x6, x13, lsl #2]
    // mov x13, #16
    // ld1w { za3h.S[w12, 0] }, p0/Z, [x6, x13, lsl #2]
    // add w12, w12, #1
    // add x6, x6, x10
    // .endr 

    
    // .rept 16
    // ldr za[w12, #0], [x6, #0, mul vl]
    // ldr za[w12, #1], [x6, #1, mul vl]
    // add w12, w12, #4
    // add x6, x6, x10
    // .endr

    // .rept 16
    // ldr za[w13, #0], [x6, #0, mul vl]
    // ldr za[w13, #1], [x6, #1, mul vl]
    // add w13, w13, #4
    // add x6, x6, x10
    // .endr

    mov x7, x0
    mov x8, x1
    add x7, x7, x15, lsl #2 // x15*4 bytes weiter springen, zum richtigen M streifen
    add x8, x8, x16, lsl #2 //  springen, zum richtigen N streifen



    // K Variables loop counter in x29
    mov x14, x5
K_loop:

    // load A and B 32 floats and perform the outer product
    ld1w { z0.S, z1.S }, pn8/Z, [x7]
    ld1w { z2.S, z3.S }, pn8/Z, [x8]
    #ldr z0, [x7, #0, mul vl]
    #ld1w z0.S, p0/Z, [x7]

    #ldr z2, [x8, #0, mul vl]
    #ld1w z2.S, p0/Z, [x8]

    fmopa za0.s, p0/m, p0/m, z2.s, z0.s

    //tile 1
    #ldr z1, [x7, #1, mul vl]
    #mov x13, #16
    #ld1w z1.S, p0/Z, [x7, x13, lsl #2]
    fmopa za1.s, p0/m, p0/m, z2.s, z1.s

    //tile 2
    #ldr z3, [x8, #1, mul vl]
    #ld1w z3.S, p0/Z, [x8, x13, lsl #2]
    fmopa za2.s, p0/m, p0/m, z3.s, z0.s
    
    //tile 3
    fmopa za3.s, p0/m, p0/m, z3.s, z1.s

    // move to the next K tile
    // A is column-major, so we move in K dimension by adding ld_a (x3) to the pointer
    add x7, x7, x11 // x11 = ld_a*4, move to the next K tile for A
    add x8, x8, x17 // x17 = ld_b*4, move to the next K tile for B

    subs x14, x14, #1
    b.ne K_loop


    // store the results back to C
    mov x6, x2
    add x6, x6, x15, lsl #2 // x15*4 bytes weiter springen, zum richtigen M streifen
    mul x14, x16, x5
    add x6, x6, x14, lsl #2 // zum richtigen N streifen
    mov w12, #0
    mov w13, #2
    
    // .rept 16
    // str za[w12, #0], [x6, #0, mul vl]
    // str za[w12, #1], [x6, #1, mul vl]
    // add w12, w12, #4
    // add x6, x6, x10
    // .endr

    // .rept 16
    // str za[w13, #0], [x6, #0, mul vl]
    // str za[w13, #1], [x6, #1, mul vl]
    // add w13, w13, #4
    // add x6, x6, x10
    // .endr


    #mov x19, #0
    #mov x13, #16
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


    // .rept 4
    // mova z0.S, p0/m, za0h.S[w12, #0]
    // mova z1.S, p0/m, za1h.S[w12, #0]
    // st1w { z0.S, z1.S }, pn8, [x6]
    // add x6, x6, x10
    // mova z2.S, p0/m, za0h.S[w12, #1]
    // mova z3.S, p0/m, za1h.S[w12, #1]
    // st1w { z2.S, z3.S }, pn8, [x6]
    // add x6, x6, x10
    // mova z4.S, p0/m, za0h.S[w12, #2]
    // mova z5.S, p0/m, za1h.S[w12, #2]
    // st1w { z4.S, z5.S }, pn8, [x6]
    // add x6, x6, x10
    // mova z6.S, p0/m, za0h.S[w12, #3]
    // mova z7.S, p0/m, za1h.S[w12, #3]
    // st1w { z6.S, z7.S }, pn8, [x6]
    // add x6, x6, x10
    // add w12, w12, #4
    // .endr

    // .rept 4
    // mova z0.S, p0/m, za2h.S[w12, #0]
    // mova z1.S, p0/m, za3h.S[w12, #0]
    // st1w { z0.S, z1.S }, pn8, [x6]
    // add x6, x6, x10
    // mova z2.S, p0/m, za2h.S[w12, #1]
    // mova z3.S, p0/m, za3h.S[w12, #1]
    // st1w { z2.S, z3.S }, pn8, [x6]
    // add x6, x6, x10
    // mova z4.S, p0/m, za2h.S[w12, #2]
    // mova z5.S, p0/m, za3h.S[w12, #2]
    // st1w { z4.S, z5.S }, pn8, [x6]
    // add x6, x6, x10
    // mova z6.S, p0/m, za2h.S[w12, #3]
    // mova z7.S, p0/m, za3h.S[w12, #3]
    // st1w { z6.S, z7.S }, pn8, [x6]
    // add x6, x6, x10
    // add w12, w12, #4
    // .endr

    // check if we have processed all M tiles (ld_a(x3)/floats in vector length(x9))
    add x15, x15, x9, lsl #1 // processed elements in M dimension, x9 floats per tile, we process 2 tiles (32*2=64 floats) at a time
    cmp x15, x3
    b.ne M_loop

    // N Tiles check
    add x16, x16, x9, lsl #1 // processed elements in N dimension, x9 floats per tile, we process 2 tiles (32*2=64 floats) at a time
    cmp x16, x4
    b.ne N_loop


    smstop
    #ldr x19, [sp, #16]
    
    ret

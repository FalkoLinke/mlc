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

   void gemm_512_512_trSME( float   const * a,
                        float   const * b,
                        float         * c,
                        int64_t         ld_a,
                        int64_t         ld_b,
                        int64_t         ld_c );
*/
    .global FUNCLABEL(gemm_512_512_trSME)
FUNCLABEL(gemm_512_512_trSME):

    stp x29, x30, [sp, #-16]!
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!
    mov x29, sp

    smstart
    ptrue p0.s
    ptrue p1.s, vl8 // staging loads only need 8 k values per row
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


    mov x7, x0
    mov x8, x1
    mul x14, x15, x3        // m-block offset: x15 rows of A, ld_a floats each
    add x7, x7, x14, lsl #2
    add x8, x8, x16, lsl #2 // n-block offset within the B rows

    lsl x13, x3, #4 // x13 = 16*ld_a floats, register offset to row m+16+i
    // K Variables loop counter in x14
    mov x14, #512
K_loop:

    // stage A: z_i = zip(row m+i, row m+16+i), 8 k values each — the
    // transpose network below then yields mb0/mb1 m-vectors per k
    .irp i, 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    ld1w {z24.S}, p1/z, [x7]
    ld1w {z25.S}, p1/z, [x7, x13, lsl #2]
    zip1 z\i\().s, z24.s, z25.s
    add x7, x7, x11
    .endr

    zip { z0.d - z3.d }, { z0.d - z3.d }
    zip { z4.d - z7.d }, { z4.d - z7.d }
    zip { z8.d - z11.d }, { z8.d - z11.d }
    zip { z12.d - z15.d }, { z12.d - z15.d }

    uzp { z16.s, z17.s}, z0.s, z4.s 
    uzp { z18.s, z19.s}, z8.s, z12.s

    uzp1 z26.d, z16.d, z18.d
    uzp2 z27.d, z16.d, z18.d

    trn1 z28.d, z26.d, z27.d 
    trn2 z30.d, z26.d, z27.d


    uzp1 z26.d, z17.d, z19.d
    uzp2 z27.d, z17.d, z19.d

    trn1 z29.d, z26.d, z27.d 
    trn2 z31.d, z26.d, z27.d 

    // load A and B 32 floats and perform the outer product
    ld1w { z20.S, z21.S }, pn8/Z, [x8]
    add x8, x8, x17 // x17 = ld_b*4, move to the next K tile for B

    fmopa za0.s, p0/m, p0/m, z20.s, z28.s

    //tile 1
    fmopa za1.s, p0/m, p0/m, z20.s, z29.s

    //tile 2
    fmopa za2.s, p0/m, p0/m, z21.s, z28.s
    
    //tile 3
    fmopa za3.s, p0/m, p0/m, z21.s, z29.s
    
    ld1w { z20.S, z21.S }, pn8/Z, [x8]
    add x8, x8, x17 // x17 = ld_b*4, move to the next K tile for B

    fmopa za0.s, p0/m, p0/m, z20.s, z30.s

    //tile 1
    fmopa za1.s, p0/m, p0/m, z20.s, z31.s

    //tile 2
    fmopa za2.s, p0/m, p0/m, z21.s, z30.s
    
    //tile 3
    fmopa za3.s, p0/m, p0/m, z21.s, z31.s

    uzp { z16.s, z17.s}, z1.s, z5.s 
    uzp { z18.s, z19.s}, z9.s, z13.s

    uzp1 z26.d, z16.d, z18.d
    uzp2 z27.d, z16.d, z18.d

    trn1 z28.d, z26.d, z27.d 
    trn2 z30.d, z26.d, z27.d


    uzp1 z26.d, z17.d, z19.d
    uzp2 z27.d, z17.d, z19.d

    trn1 z29.d, z26.d, z27.d 
    trn2 z31.d, z26.d, z27.d 

    // load A and B 32 floats and perform the outer product
    ld1w { z20.S, z21.S }, pn8/Z, [x8]
    add x8, x8, x17 // x17 = ld_b*4, move to the next K tile for B

    fmopa za0.s, p0/m, p0/m, z20.s, z28.s

    //tile 1
    fmopa za1.s, p0/m, p0/m, z20.s, z29.s

    //tile 2
    fmopa za2.s, p0/m, p0/m, z21.s, z28.s
    
    //tile 3
    fmopa za3.s, p0/m, p0/m, z21.s, z29.s
    
    ld1w { z20.S, z21.S }, pn8/Z, [x8]
    add x8, x8, x17 // x17 = ld_b*4, move to the next K tile for B

    fmopa za0.s, p0/m, p0/m, z20.s, z30.s

    //tile 1
    fmopa za1.s, p0/m, p0/m, z20.s, z31.s

    //tile 2
    fmopa za2.s, p0/m, p0/m, z21.s, z30.s
    
    //tile 3
    fmopa za3.s, p0/m, p0/m, z21.s, z31.s

    uzp { z16.s, z17.s}, z2.s, z6.s 
    uzp { z18.s, z19.s}, z10.s, z14.s

    uzp1 z26.d, z16.d, z18.d
    uzp2 z27.d, z16.d, z18.d

    trn1 z28.d, z26.d, z27.d 
    trn2 z30.d, z26.d, z27.d


    uzp1 z26.d, z17.d, z19.d
    uzp2 z27.d, z17.d, z19.d

    trn1 z29.d, z26.d, z27.d 
    trn2 z31.d, z26.d, z27.d 

    // load A and B 32 floats and perform the outer product
    ld1w { z20.S, z21.S }, pn8/Z, [x8]
    add x8, x8, x17 // x17 = ld_b*4, move to the next K tile for B

    fmopa za0.s, p0/m, p0/m, z20.s, z28.s

    //tile 1
    fmopa za1.s, p0/m, p0/m, z20.s, z29.s

    //tile 2
    fmopa za2.s, p0/m, p0/m, z21.s, z28.s
    
    //tile 3
    fmopa za3.s, p0/m, p0/m, z21.s, z29.s
    
    ld1w { z20.S, z21.S }, pn8/Z, [x8]
    add x8, x8, x17 // x17 = ld_b*4, move to the next K tile for B

    fmopa za0.s, p0/m, p0/m, z20.s, z30.s

    //tile 1
    fmopa za1.s, p0/m, p0/m, z20.s, z31.s

    //tile 2
    fmopa za2.s, p0/m, p0/m, z21.s, z30.s
    
    //tile 3
    fmopa za3.s, p0/m, p0/m, z21.s, z31.s

    uzp { z16.s, z17.s}, z3.s, z7.s 
    uzp { z18.s, z19.s}, z11.s, z15.s

    uzp1 z26.d, z16.d, z18.d
    uzp2 z27.d, z16.d, z18.d

    trn1 z28.d, z26.d, z27.d 
    trn2 z30.d, z26.d, z27.d


    uzp1 z26.d, z17.d, z19.d
    uzp2 z27.d, z17.d, z19.d

    trn1 z29.d, z26.d, z27.d 
    trn2 z31.d, z26.d, z27.d 

    // load A and B 32 floats and perform the outer product
    ld1w { z20.S, z21.S }, pn8/Z, [x8]
    add x8, x8, x17 // x17 = ld_b*4, move to the next K tile for B

    fmopa za0.s, p0/m, p0/m, z20.s, z28.s

    //tile 1
    fmopa za1.s, p0/m, p0/m, z20.s, z29.s

    //tile 2
    fmopa za2.s, p0/m, p0/m, z21.s, z28.s
    
    //tile 3
    fmopa za3.s, p0/m, p0/m, z21.s, z29.s
    
    ld1w { z20.S, z21.S }, pn8/Z, [x8]
    add x8, x8, x17 // x17 = ld_b*4, move to the next K tile for B

    fmopa za0.s, p0/m, p0/m, z20.s, z30.s

    //tile 1
    fmopa za1.s, p0/m, p0/m, z20.s, z31.s

    //tile 2
    fmopa za2.s, p0/m, p0/m, z21.s, z30.s
    
    //tile 3
    fmopa za3.s, p0/m, p0/m, z21.s, z31.s

    // next k block: rewind the 16 staged rows, advance 8 floats in k
    sub x7, x7, x11, lsl #4
    add x7, x7, #32

    subs x14, x14, #8
    b.ne K_loop


    // store the results back to C
    mov x6, x2
    add x6, x6, x15, lsl #2 // x15*4 bytes weiter springen, zum richtigen M streifen
    mul x14, x16, x5
    add x6, x6, x14, lsl #2 // zum richtigen N streifen
    mov w12, #0
    mov w13, #2
    

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

    // check if we have processed all M tiles (ld_a(x3)/floats in vector length(x9))
    add x15, x15, x9, lsl #1 // processed elements in M dimension, x9 floats per tile, we process 2 tiles (32*2=64 floats) at a time
    cmp x15, x3
    b.ne M_loop

    // N Tiles check
    add x16, x16, x9, lsl #1 // processed elements in N dimension, x9 floats per tile, we process 2 tiles (32*2=64 floats) at a time
    cmp x16, x4
    b.ne N_loop


    smstop

    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ldp x29, x30, [sp], #16
    
    ret

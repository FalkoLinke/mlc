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

    stp x29, x30, [sp, #-16]!
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!
    mov x29, sp


    smstart
    ptrue p0.s, vl16
    ptrue pn8.s

    zero {za1.s}
    zero {za2.s}
    zero {za3.s}


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
    ld1w {z1.S}, p0/z, [x8]
    fmopa za0.s, p0/m, p0/m, z1.s, z0.s
    add x7, x7, x11
    add x8, x8, x12


    ld1w {z0.S}, p0/z, [x7]
    ld1w {z1.S}, p0/z, [x8]
    fmopa za1.s, p0/m, p0/m, z1.s, z0.s
    add x7, x7, x11
    add x8, x8, x12


    ld1w {z0.S}, p0/z, [x7]
    ld1w {z1.S}, p0/z, [x8]
    fmopa za2.s, p0/m, p0/m, z1.s, z0.s

    add x7, x7, x11
    add x8, x8, x12
        
    ld1w {z0.S}, p0/z, [x7]
    ld1w {z1.S}, p0/z, [x8]
    fmopa za3.s, p0/m, p0/m, z1.s, z0.s

    add x7, x7, x11
    add x8, x8, x12

    subs x9, x9, #4
    b.ne K_loop

    //     // load A z0 and B z2 16 floats at a time and perform the outer product
    // ld1w {z0.S, z1.S}, pn8/Z, [x7]
    // ld1w {z2.S, z3.S}, pn8/Z, [x8]
    // fmopa za0.s, p0/m, p0/m, z2.s, z0.s

    // add x7, x7, x11, lsl #1
    // add x8, x8, x12, lsl #1

    // fmopa za1.s, p0/m, p0/m, z3.s, z1.s

    // ld1w {z0.S, z1.S}, pn8/Z, [x7]
    // ld1w {z2.S, z3.S}, pn8/Z, [x8]
    // fmopa za0.s, p0/m, p0/m, z2.s, z0.s

    // add x7, x7, x11, lsl #1
    // add x8, x8, x12, lsl #1

    // fmopa za1.s, p0/m, p0/m, z3.s, z1.s

    // add x7, x7, x11, lsl #1
    // add x8, x8, x12, lsl #1

    // subs x9, x9, #4
    // b.ne K_loop

    // store Results
    mov w13, #0
    mov w8, #0
    mov x6, x2

    // .rept 4
    // mova  {z0.s, z1.s, z2.s, z3.s}, za0h.s[W13, 0:3]
    // add za.s[w8,0,VGx4], {z0.s, z1.s, z2.s, z3.s}
    // mova  {z4.s, z5.s, z6.s, z7.s}, za1h.s[W13, 0:3]
    // add za.s[w8,0,VGx4], {z4.s, z5.s, z6.s, z7.s}
    // mova  {z8.s, z9.s, z10.s, z11.s}, za2h.s[W13, 0:3]
    // add za.s[w8,0,VGx4], {z8.s, z9.s, z10.s, z11.s}
    // add w8, w8, #4
    // add w13, w13, #4
    // //add   za0h, {z0.s, z1.s, z2.s, z3.s}, {z4.s, z5.s, z6.s, z7.s}
    // // mova  {Z10.S - Z13.S}, ZA2H.S[W13, 0:3]
    // // add   {Z2.S - Z5.S}, {Z2.S - Z5.S}, {Z10.S - Z13.S}
    // // mova  {Z14.S - Z17.S}, ZA3H.S[W13, 0:3]
    // // add   {Z2.S - Z5.S}, {Z2.S - Z5.S}, {Z14.S - Z17.S}
    // // add w13, w13, #4
    // // st1w { z2.S - z5.S }, pn8, [x6]
    // // add x6, x6, x10, lsl #2
    // .endr

    .rept 4
    mova { z0.s - z3.s }, za0h.s[w13, 0:3]
    mova { z4.s - z7.s }, za1h.s[w13, 0:3]
    mova { z8.s - z11.s }, za2h.s[w13, 0:3]
    mova { z12.s - z15.s }, za3h.s[w13, 0:3]
    fadd z0.s, z0.s, z4.s
    fadd z1.s, z1.s, z5.s
    fadd z2.s, z2.s, z6.s
    fadd z3.s, z3.s, z7.s
    fadd z0.s, z0.s, z8.s
    fadd z1.s, z1.s, z9.s
    fadd z2.s, z2.s, z10.s
    fadd z3.s, z3.s, z11.s
    fadd z0.s, z0.s, z12.s
    fadd z1.s, z1.s, z13.s
    fadd z2.s, z2.s, z14.s
    fadd z3.s, z3.s, z15.s
    // st1w  {z0.s}, p0, [x6]
    // st1w  {z1.s}, p0, [x6, #1, mul VL]
    // st1w  {z2.s}, p0, [x6, #2, mul VL]
    // st1w  {z3.s}, p0, [x6, #3, mul VL]
    st1w  {z0.s - z3.s}, pn8, [x6]


    add w13, w13, #4
    add x6, x6, x10, lsl #2
    .endr







    // mova  {Z6.S - Z9.S}, ZA0H.S[W13, 0:3]
    // add w13, w13, #4
    // mova  {Z10.S - Z13.S}, ZA0H.S[W13, 0:3]
    // add w13, w13, #4
    // mova  {Z14.S - Z17.S}, ZA0H.S[W13, 0:3]
    // mov w13, #0


    // add w13, w13, #4
    // mova  {Z22.S - Z25.S}, ZA1H.S[W13, 0:3]
    // add w13, w13, #4
    // mova  {Z26.S - Z29.S}, ZA1H.S[W13, 0:3]
    // add w13, w13, #4
    // mova  {Z30.S - Z33.S}, ZA1H.S[W13, 0:3]
    // mov w13, #0




    // mov w13, #0
    // .rept 16
    // st1w { za0h.S[w13, 0] }, p0, [x6]
    // add w13, w13, #1
    // add x6, x6, x10
    // .endr 

    smstop
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ldp x29, x30, [sp], #16
    ret
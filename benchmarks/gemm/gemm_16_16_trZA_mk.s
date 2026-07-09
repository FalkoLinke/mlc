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
    .global FUNCLABEL(gemm_16_16_trZA_mk)
FUNCLABEL(gemm_16_16_trZA_mk):

    stp x29, x30, [sp, #-16]!
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!
    mov x29, sp


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

    // save accumulator tile za0 
    mov w14, #0
    mova { z16.S - z19.S }, za0h.S[w14, 0:3 ]
    add w14, w14, #4
    mova { z20.S - z23.S }, za0h.S[w14, 0:3 ]
    add w14, w14, #4
    mova { z24.S - z27.S }, za0h.S[w14, 0:3 ]
    add w14, w14, #4
    mova { z28.S - z31.S }, za0h.S[w14, 0:3 ]

    // Trans A: 
    mov w13, #0
    .rept 16
    ld1w { z0.S }, p0/Z, [x7]
    mova za0h.S[w13, #0], p0/m, z0.S
    add x7, x7, x11
    add w13, w13, #1
    .endr

    // read the transposed A block out of the vertical slices
    mov w14, #0
    mova { z0.S - z3.S }, za0v.S[w14, 0:3 ]
    add w14, w14, #4
    mova { z4.S - z7.S }, za0v.S[w14, 0:3 ]
    add w14, w14, #4
    mova { z8.S - z11.S }, za0v.S[w14, 0:3 ]
    add w14, w14, #4
    mova { z12.S - z15.S }, za0v.S[w14, 0:3 ]

    // restore accumulator tile za0
    mov w14, #0
    mova za0h.S[w14, 0:3 ], { z16.S - z19.S }
    add w14, w14, #4
    mova za0h.S[w14, 0:3 ], { z20.S - z23.S }
    add w14, w14, #4
    mova za0h.S[w14, 0:3 ], { z24.S - z27.S }
    add w14, w14, #4
    mova za0h.S[w14, 0:3 ], { z28.S - z31.S }
    
    // load A z0 and B z2 16 floats at a time and perform the outer product
    ld1w {z16.S}, p0/z, [x8]
    fmopa za0.s, p0/m, p0/m, z16.s, z0.s
    add x8, x8, x12

    ld1w {z16.S}, p0/z, [x8]
    fmopa za1.s, p0/m, p0/m, z16.s, z1.s
    add x8, x8, x12

    ld1w {z16.S}, p0/z, [x8]
    fmopa za2.s, p0/m, p0/m, z16.s, z2.s
    add x8, x8, x12
        
    ld1w {z16.S}, p0/z, [x8]
    fmopa za3.s, p0/m, p0/m, z16.s, z3.s
    add x8, x8, x12

    ld1w {z16.S}, p0/z, [x8]
    fmopa za0.s, p0/m, p0/m, z16.s, z4.s
    add x8, x8, x12

    ld1w {z16.S}, p0/z, [x8]
    fmopa za1.s, p0/m, p0/m, z16.s, z5.s
    add x8, x8, x12

    ld1w {z16.S}, p0/z, [x8]
    fmopa za2.s, p0/m, p0/m, z16.s, z6.s
    add x8, x8, x12
        
    ld1w {z16.S}, p0/z, [x8]
    fmopa za3.s, p0/m, p0/m, z16.s, z7.s
    add x8, x8, x12

    ld1w {z16.S}, p0/z, [x8]
    fmopa za0.s, p0/m, p0/m, z16.s, z8.s
    add x8, x8, x12

    ld1w {z16.S}, p0/z, [x8]
    fmopa za1.s, p0/m, p0/m, z16.s, z9.s
    add x8, x8, x12

    ld1w {z16.S}, p0/z, [x8]
    fmopa za2.s, p0/m, p0/m, z16.s, z10.s
    add x8, x8, x12
        
    ld1w {z16.S}, p0/z, [x8]
    fmopa za3.s, p0/m, p0/m, z16.s, z11.s
    add x8, x8, x12

    ld1w {z16.S}, p0/z, [x8]
    fmopa za0.s, p0/m, p0/m, z16.s, z12.s
    add x8, x8, x12

    ld1w {z16.S}, p0/z, [x8]
    fmopa za1.s, p0/m, p0/m, z16.s, z13.s
    add x8, x8, x12

    ld1w {z16.S}, p0/z, [x8]
    fmopa za2.s, p0/m, p0/m, z16.s, z14.s
    add x8, x8, x12
        
    ld1w {z16.S}, p0/z, [x8]
    fmopa za3.s, p0/m, p0/m, z16.s, z15.s
    add x8, x8, x12

    sub x7, x7, x11, lsl #4  // back up 16 rows
    add x7, x7, #64          // advance k by 16 floats

    subs x9, x9, #16
    b.ne K_loop


    // store C
    // mov w12, #0
    // mov w13, #4
    // mov w14, #8
    // mov w15, #12
    // mov x9, #64
    // mov x16, #128
    // mov x17, #192
    mov w8, #0
    mov x6, x2

    .rept 4
    mova { z0.s - z3.s }, za.s[w8, 1 , VGx4]
    fadd za.s[w8,0,VGx4], {z0.s - z3.s}

    mova { z4.s - z7.s }, za.s[w8, 2 , VGx4]
    fadd za.s[w8,0,VGx4], {z4.s - z7.s}

    mova { z8.s - z11.s }, za.s[w8, 3 , VGx4]
    fadd za.s[w8,0,VGx4], {z8.s - z11.s}
    add w8, w8, #4;

    // st1w { za0h.S[w12, 0] }, p0, [x6]
    // add w12, w12, #1
    // st1w { za0h.S[w13, 0] }, p0, [x6, x9, LSL #2]
    // add w13, w13, #1
    // st1w { za0h.S[w14, 0] }, p0, [x6, x16, LSL #2]
    // add w14, w14, #1
    // st1w { za0h.S[w15, 0] }, p0, [x6, x17, LSL #2]
    // add w15, w15, #1
    // add x6, x6, x10
    .endr

    mov w13, #0
    .rept 16
    st1w { za0h.S[w13, 0] }, p0, [x6]
    add w13, w13, #1
    add x6, x6, x10
    .endr 

    smstop
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ldp x29, x30, [sp], #16
    ret
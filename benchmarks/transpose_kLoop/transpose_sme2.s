

    .global _transpose_16x16_fp32_sme2
_transpose_16x16_fp32_sme2:
    stp x29, x30, [sp, #-16]!
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!
    mov x29, sp

    smstart

    ptrue p0.s
    ptrue pn8.s

    mov x4, x0
    mov x5, #50000

k_loop:
    mov x0, x4;

    ld1w {z0.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z1.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z2.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z3.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z4.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z5.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z6.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z7.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z8.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z9.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z10.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z11.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z12.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z13.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z14.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2
    ld1w {z15.S}, p0/z, [x0]
    add x0, x0, x2, lsl #2

    zip { z0.d - z3.d }, { z0.d - z3.d }
    zip { z4.d - z7.d }, { z4.d - z7.d }
    zip { z8.d - z11.d }, { z8.d - z11.d }
    zip { z12.d - z15.d }, { z12.d - z15.d }

    uzp { z16.s, z17.s}, z0.s, z4.s 
    uzp { z18.s, z19.s}, z8.s, z12.s

    uzp1 z26.d, z16.d, z18.d
    uzp2 z27.d, z16.d, z18.d
    // 2. Transponiere sie an die korrekten 128-Bit-Positionen
    trn1 z28.d, z26.d, z27.d   // z0 ist jetzt exakt das Ergebnis deines 'uzp1 .q'
    trn2 z30.d, z26.d, z27.d   // z1 ist jetzt exakt das Ergebnis deines 'uzp2 .q'


    uzp1 z26.d, z17.d, z19.d
    uzp2 z27.d, z17.d, z19.d
    // 2. Transponiere sie an die korrekten 128-Bit-Positionen
    trn1 z29.d, z26.d, z27.d   // z0 ist jetzt exakt das Ergebnis deines 'uzp1 .q'
    trn2 z31.d, z26.d, z27.d   // z1 ist jetzt exakt das Ergebnis deines 'uzp2 .q'

    //st1w { z28.s - z31.s }, pn8, [x1]
    //add x1, x1, x3, lsl #4

    uzp { z16.s, z17.s}, z1.s, z5.s 
    uzp { z18.s, z19.s}, z9.s, z13.s

    uzp1 z26.d, z16.d, z18.d
    uzp2 z27.d, z16.d, z18.d
    // 2. Transponiere sie an die korrekten 128-Bit-Positionen
    trn1 z28.d, z26.d, z27.d   // z0 ist jetzt exakt das Ergebnis deines 'uzp1 .q'
    trn2 z30.d, z26.d, z27.d   // z1 ist jetzt exakt das Ergebnis deines 'uzp2 .q'


    uzp1 z26.d, z17.d, z19.d
    uzp2 z27.d, z17.d, z19.d
    // 2. Transponiere sie an die korrekten 128-Bit-Positionen
    trn1 z29.d, z26.d, z27.d   // z0 ist jetzt exakt das Ergebnis deines 'uzp1 .q'
    trn2 z31.d, z26.d, z27.d   // z1 ist jetzt exakt das Ergebnis deines 'uzp2 .q'

    //st1w { z28.s - z31.s }, pn8, [x1, #4, mul vl]
    //add x1, x1, x3, lsl #4
    uzp { z16.s, z17.s}, z2.s, z6.s 
    uzp { z18.s, z19.s}, z10.s, z14.s

    uzp1 z26.d, z16.d, z18.d
    uzp2 z27.d, z16.d, z18.d
    // 2. Transponiere sie an die korrekten 128-Bit-Positionen
    trn1 z28.d, z26.d, z27.d   // z0 ist jetzt exakt das Ergebnis deines 'uzp1 .q'
    trn2 z30.d, z26.d, z27.d   // z1 ist jetzt exakt das Ergebnis deines 'uzp2 .q'


    uzp1 z26.d, z17.d, z19.d
    uzp2 z27.d, z17.d, z19.d
    // 2. Transponiere sie an die korrekten 128-Bit-Positionen
    trn1 z29.d, z26.d, z27.d   // z0 ist jetzt exakt das Ergebnis deines 'uzp1 .q'
    trn2 z31.d, z26.d, z27.d   // z1 ist jetzt exakt das Ergebnis deines 'uzp2 .q'

    //st1w { z28.s - z31.s }, pn8, [x1, #8, mul vl]
    //add x1, x1, x3, lsl #4
    uzp { z16.s, z17.s}, z3.s, z7.s 
    uzp { z18.s, z19.s}, z11.s, z15.s

    uzp1 z26.d, z16.d, z18.d
    uzp2 z27.d, z16.d, z18.d
    // 2. Transponiere sie an die korrekten 128-Bit-Positionen
    trn1 z28.d, z26.d, z27.d   // z0 ist jetzt exakt das Ergebnis deines 'uzp1 .q'
    trn2 z30.d, z26.d, z27.d   // z1 ist jetzt exakt das Ergebnis deines 'uzp2 .q'


    uzp1 z26.d, z17.d, z19.d
    uzp2 z27.d, z17.d, z19.d
    // 2. Transponiere sie an die korrekten 128-Bit-Positionen
    trn1 z29.d, z26.d, z27.d   // z0 ist jetzt exakt das Ergebnis deines 'uzp1 .q'
    trn2 z31.d, z26.d, z27.d   // z1 ist jetzt exakt das Ergebnis deines 'uzp2 .q'

    //st1w { z28.s - z31.s }, pn8, [x1, #12, mul vl]
    //add x1, x1, x3, lsl #4


    // uzp { z16.s, z17.s}, z1.s, z5.s 
    // uzp { z18.s, z19.s}, z9.s, z13.s
    // uzp1 z30.d, z16.d, z18.d
    // uzp2 z31.d, z16.d, z18.d
    // // 2. Transponiere sie an die korrekten 128-Bit-Positionen
    // trn1 z30.d, z30.d, z31.d   // z0 ist jetzt exakt das Ergebnis deines 'uzp1 .q'
    // trn2 z31.d, z30.d, z31.d   // z1 ist jetzt exakt das Ergebnis deines 'uzp2 .q'
    // st1w { z30.s - z31.s }, pn8, [x1]
    // add x1, x1, x3, lsl #2


    // st1w { z4.s - z7.s }, pn8, [x1]
    // add x1, x1, x3, lsl #4
    // // st1w { z8.s - z11.s }, pn8, [x1]
    // add x1, x1, x3, lsl #4
    // st1w { z30.s - z31.s }, pn8, [x1]
    // // add x1, x1, x3, lsl #4

    subs x5, x5, #1
    b.ne k_loop

    st1w z4.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z14.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z10.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z18.s, p0, [x1]
    add x1, x1, x3, LSL #2

    st1w z6.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z16.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z12.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z20.s, p0, [x1]
    add x1, x1, x3, LSL #2

    st1w z5.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z15.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z11.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z19.s, p0, [x1]
    add x1, x1, x3, LSL #2

    st1w z7.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z17.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z13.s, p0, [x1]
    add x1, x1, x3, LSL #2
    st1w z21.s, p0, [x1]
    add x1, x1, x3, LSL #2

    smstop

    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ldp x29, x30, [sp], #16
    ret




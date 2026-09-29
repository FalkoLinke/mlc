#ifdef __APPLE__
#define FUNCLABEL(NAME) _##NAME
#else
#define FUNCLABEL(NAME) NAME
#endif /* __APPLE__ */

    
    .text




/*
void zero_16_16( float* a,
                 int64_t ld_a );
*/
    .global FUNCLABEL(zero_16_16)
FUNCLABEL(zero_16_16):
    // smstart and smstop zero the vector registers: save the callee-saved d8 - d15
    stp d8, d9, [sp, #-16]!
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!
    smstart
    zero { za0.b }
    mov w12, #0
    ptrue p0.s
    
    mov x2, #16
zero_16_16_loop01:
    cbz x2, zero_16_16_end01
    
    st1w za0h.s[w12, 0], p0, [x0]

    add x0, x0, x1, LSL #2
    subs x2, x2, #1
    b zero_16_16_loop01
zero_16_16_end01:

    smstop
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ret
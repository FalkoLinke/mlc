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
    mov x2, #0
    
    .rept 16
    str x2, [x0, #0]
    str x2, [x0, #8]
    str x2, [x0, #16]
    str x2, [x0, #24]
    str x2, [x0, #32]
    str x2, [x0, #40]
    str x2, [x0, #48]
    str x2, [x0, #56]
    add x0, x0, x1, LSL #2
    .endr

    ret
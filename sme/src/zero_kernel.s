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
    smstart
    fmov z0.s, #0.0
    ptrue p0.s
    
    mov x2, #16
zero_16_16_loop01:
    cbz x2, zero_16_16_end01
    
    st1w z0.s, p0, [x0]

    add x0, x0, x1, LSL #2
    subs x2, x2, #1
    b zero_16_16_loop01
zero_16_16_end01:

    smstop
    ret
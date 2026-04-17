    .text   


;
;    int fmadd_kernel();
;
    .global _fmadd_kernel
_fmadd_kernel:
    movi v0.4s, #0
    movi v0.4s, #1

fmadd_insts_start:
    .rept  4000
    fmadd s0, s1, s0, s0
    .endr
fmadd_insts_end:

    adr x1, fmadd_insts_count
    ldr x0, [x1]
    ret
fmadd_insts_count:
    .long (fmadd_insts_end - fmadd_insts_start) / 4











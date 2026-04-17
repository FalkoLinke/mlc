    .text   


;
;    int fmadd_kernel();
;
    .global _fmadd_kernel
_fmadd_kernel:
    movi v0.4s, #0

insts_start:
    .rept   200
    fmadd s0, s0, s0, s0
    fmadd s0, s0, s0, s0
    fmadd s0, s0, s0, s0
    fmadd s0, s0, s0, s0
    fmadd s0, s0, s0, s0
    fmadd s0, s0, s0, s0
    fmadd s0, s0, s0, s0
    fmadd s0, s0, s0, s0
    fmadd s0, s0, s0, s0
    fmadd s0, s0, s0, s0
    fmadd s0, s0, s0, s0
    fmadd s0, s0, s0, s0
    fmadd s0, s0, s0, s0
    fmadd s0, s0, s0, s0
    fmadd s0, s0, s0, s0
    fmadd s0, s0, s0, s0
    fmadd s0, s0, s0, s0
    fmadd s0, s0, s0, s0
    .endr
insts_end:

    adr x1, insts_count
    ldr x0, [x1]
    ret
insts_count:
    .long (insts_end - insts_start) / 4











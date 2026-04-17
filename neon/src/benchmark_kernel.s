    .text   


;
;    int fmadd_kernel();
;
    .global _fmadd_kernel
_fmadd_kernel:
    mov s0, #0

insts_start:
    .rept   50
    fmadd s0, s0, s0, s0
    .endr
insts_end:

    mov x1, insts_count
    ldr x0, [x1]
    ret










    .section "rodata"
insts_count:
    .long insts_end - insts_start




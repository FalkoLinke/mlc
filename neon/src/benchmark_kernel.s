    .text   


;
;    int fmadd_kernel();
;
    .global _fmadd_kernel
_fmadd_kernel:
    movi v0.4s, #0
    movi v1.4s, #1

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








;
;    int fmla_4s_kernel();
;
    .global _fmla_4s_kernel
_fmla_4s_kernel:
    movi v0.4s, #0
    movi v1.4s, #1
    movi v2.4s, #0

fmla_4s_insts_start:
    .rept  4000
    fmla v2.4s, v0.4s, v1.4s
    .endr
fmla_4s_insts_end:

    adr x1, fmla_4s_insts_count
    ldr x0, [x1]
    ret
fmla_4s_insts_count:
    .long (fmla_4s_insts_end - fmla_4s_insts_start) / 4





;
;    int fmla_2s_kernel();
;
    .global _fmla_2s_kernel
_fmla_2s_kernel:
    movi v0.4s, #0
    movi v1.4s, #1
    movi v2.4s, #0

fmla_2s_insts_start:
    .rept  4000
    fmla v2.2s, v0.2s, v1.2s
    .endr
fmla_2s_insts_end:

    adr x1, fmla_2s_insts_count
    ldr x0, [x1]
    ret
fmla_2s_insts_count:
    .long (fmla_2s_insts_end - fmla_2s_insts_start) / 4


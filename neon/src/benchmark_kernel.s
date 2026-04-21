#ifdef __APPLE__
#define FUNCLABEL(NAME) _##NAME
#else
#define FUNCLABEL(NAME) NAME
#endif /* __APPLE__ */





    .macro INIT_FPREGS
    movi v0.4s, #0
    movi v1.4s, #0
    movi v2.4s, #0
    movi v3.4s, #0
    movi v4.4s, #0
    movi v5.4s, #0
    movi v6.4s, #0
    movi v7.4s, #0
    movi v8.4s, #0
    movi v9.4s, #0
    movi v10.4s, #0
    movi v11.4s, #0
    movi v12.4s, #0
    movi v13.4s, #0
    movi v14.4s, #0
    movi v15.4s, #0
    movi v16.4s, #0
    movi v17.4s, #0
    movi v18.4s, #0
    movi v19.4s, #0
    movi v20.4s, #0
    movi v21.4s, #0
    movi v22.4s, #0
    movi v23.4s, #0
    movi v24.4s, #0
    movi v25.4s, #0
    movi v26.4s, #0
    movi v27.4s, #0
    movi v28.4s, #0
    movi v29.4s, #0
    movi v30.4s, #0
    movi v31.4s, #0
    .endm





    .text


//
//    int fmadd_kernel(int repetitions);
//
    .global FUNCLABEL(fmadd_kernel)
FUNCLABEL(fmadd_kernel):
    mov x2, x0

    INIT_FPREGS

loop01:
    cbz x0, end01

fmadd_insts_start:
    .rept  4000
    fmadd s0, s1, s0, s0
    .endr
fmadd_insts_end:

    subs x0, x0, #1
    b loop01
end01:

    adr x1, fmadd_insts_count
    ldr x0, [x1]
    mul x0, x0, x2
    ret
fmadd_insts_count:
    .long (fmadd_insts_end - fmadd_insts_start) / 4








//
//    int fmla_4s_kernel(int repetitions);
//
    .global FUNCLABEL(fmla_4s_kernel)
FUNCLABEL(fmla_4s_kernel):
    mov x2, x0

    INIT_FPREGS

loop02:
    cbz x0, end02

fmla_4s_insts_start:
    .rept  4000
    fmla v2.4s, v0.4s, v1.4s
    .endr
fmla_4s_insts_end:

    subs x0, x0, #1
    b loop02
end02:

    adr x1, fmla_4s_insts_count
    ldr x0, [x1]
    mul x0, x0, x2
    ret
fmla_4s_insts_count:
    .long (fmla_4s_insts_end - fmla_4s_insts_start) / 4





//
//    int fmla_2s_kernel(int repetitions);
//
    .global FUNCLABEL(fmla_2s_kernel)
FUNCLABEL(fmla_2s_kernel):
    mov x2, x0

    INIT_FPREGS

loop03:
    cbz x0, end03

fmla_2s_insts_start:
    .rept  4000
    fmla v2.2s, v0.2s, v1.2s
    .endr
fmla_2s_insts_end:

    subs x0, x0, #1
    b loop03
end03:

    adr x1, fmla_2s_insts_count
    ldr x0, [x1]
    mul x0, x0, x2
    ret
fmla_2s_insts_count:
    .long (fmla_2s_insts_end - fmla_2s_insts_start) / 4










//
//    int fmadd_kernel_v2(int repetitions);
//
    .global FUNCLABEL(fmadd_kernel_v2)
FUNCLABEL(fmadd_kernel_v2):
    mov x2, x0

    INIT_FPREGS

loop04:
    cbz x0, end04

fmadd_v2_insts_start:
    .rept  4000
    fmadd s0, s8, s16, s24
    fmadd s1, s9, s17, s25
    fmadd s2, s10, s18, s26
    fmadd s3, s11, s19, s27
    fmadd s4, s12, s20, s28
    fmadd s5, s13, s21, s29
    fmadd s6, s14, s22, s30
    fmadd s7, s15, s23, s31

    fmadd s8, s17, s24, s0
    fmadd s9, s18, s25, s1
    fmadd s10, s19, s26, s2
    fmadd s11, s20, s27, s3
    fmadd s12, s21, s28, s4
    fmadd s13, s22, s29, s5
    fmadd s14, s23, s30, s6
    fmadd s15, s24, s31, s7
    .endr
fmadd_v2_insts_end:

    subs x0, x0, #1
    b loop04
end04:

    adr x1, fmadd_v2_insts_count
    ldr x0, [x1]
    mul x0, x0, x2
    ret
fmadd_v2_insts_count:
    .long (fmadd_v2_insts_end - fmadd_v2_insts_start) / 4

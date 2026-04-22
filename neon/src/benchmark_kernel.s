#ifdef __APPLE__
#define FUNCLABEL(NAME) _##NAME
#else
#define FUNCLABEL(NAME) NAME
#endif /* __APPLE__ */




    .macro KERNEL_PROLOGUE
    stp x29, x30, [sp, #-16]!               // push link- and frame-registers
    mov x29, sp                             // set frame-register to stack pointer

    stp d8, d9, [sp, #-16]!                 // push callee-saved float registers
    stp d10, d11, [sp, #-16]!
    stp d12, d13, [sp, #-16]!
    stp d14, d15, [sp, #-16]!
    .endm



    .macro KERNEL_EPILOGUE
    ldp d14, d15, [sp], #16                 // pop callee-saved float registers
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d18, d19, [sp], #16

    ldp x29, x30, [sp], #16                 // restore link- and frame-registers
    .endm



    .macro INIT_FPREGS
    movi v0.4s, #0                          // set all floating point registers to 0
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
    KERNEL_PROLOGUE
    
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

    KERNEL_EPILOGUE
    ret
fmadd_insts_count:
    .long (fmadd_insts_end - fmadd_insts_start) / 4
























//
//    int fmla_4s_kernel(int repetitions);
//
    .global FUNCLABEL(fmla_4s_kernel)
FUNCLABEL(fmla_4s_kernel):
    KERNEL_PROLOGUE

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

    KERNEL_EPILOGUE
    ret
fmla_4s_insts_count:
    .long (fmla_4s_insts_end - fmla_4s_insts_start) / 4




















//
//    int fmla_2s_kernel(int repetitions);
//
    .global FUNCLABEL(fmla_2s_kernel)
FUNCLABEL(fmla_2s_kernel):
    KERNEL_PROLOGUE

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

    KERNEL_EPILOGUE
    ret
fmla_2s_insts_count:
    .long (fmla_2s_insts_end - fmla_2s_insts_start) / 4























//
//    int fmadd_kernel_v2(int repetitions);
//
    .global FUNCLABEL(fmadd_kernel_v2)
FUNCLABEL(fmadd_kernel_v2):
    KERNEL_PROLOGUE

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

    KERNEL_EPILOGUE
    ret
fmadd_v2_insts_count:
    .long (fmadd_v2_insts_end - fmadd_v2_insts_start) / 4

























//
//    int fmla_4s_kernel_v2(int repetitions);
//
    .global FUNCLABEL(fmla_4s_kernel_v2)
FUNCLABEL(fmla_4s_kernel_v2):
    KERNEL_PROLOGUE

    mov x2, x0

    INIT_FPREGS

loop05:
    cbz x0, end05

fmla_4s_v2_insts_start:
    .rept  4000
    fmla v0.4s, v8.4s, v16.4s
    fmla v1.4s, v9.4s, v17.4s
    fmla v2.4s, v10.4s, v18.4s
    fmla v3.4s, v11.4s, v19.4s
    fmla v4.4s, v12.4s, v20.4s
    fmla v5.4s, v13.4s, v21.4s
    fmla v6.4s, v14.4s, v22.4s
    fmla v7.4s, v15.4s, v23.4s

    fmla v8.4s, v16.4s, v24.4s
    fmla v9.4s, v17.4s, v25.4s
    fmla v10.4s, v18.4s, v26.4s
    fmla v11.4s, v19.4s, v27.4s
    fmla v12.4s, v20.4s, v28.4s
    fmla v13.4s, v21.4s, v29.4s
    fmla v14.4s, v22.4s, v30.4s
    fmla v15.4s, v23.4s, v31.4s
    .endr
fmla_4s_v2_insts_end:

    subs x0, x0, #1
    b loop05
end05:

    adr x1, fmla_4s_v2_insts_count
    ldr x0, [x1]
    mul x0, x0, x2

    KERNEL_EPILOGUE
    ret
fmla_4s_v2_insts_count:
    .long (fmla_4s_v2_insts_end - fmla_4s_v2_insts_start) / 4
































//
//    int fmla_2s_kernel_v2(int repetitions);
//
    .global FUNCLABEL(fmla_2s_kernel_v2)
FUNCLABEL(fmla_2s_kernel_v2):
    KERNEL_PROLOGUE

    mov x2, x0

    INIT_FPREGS

loop06:
    cbz x0, end06

fmla_2s_v2_insts_start:
    .rept  4000
    fmla v0.2s, v8.2s, v16.2s
    fmla v1.2s, v9.2s, v17.2s
    fmla v2.2s, v10.2s, v18.2s
    fmla v3.2s, v11.2s, v19.2s
    fmla v4.2s, v12.2s, v20.2s
    fmla v5.2s, v13.2s, v21.2s
    fmla v6.2s, v14.2s, v22.2s
    fmla v7.2s, v15.2s, v23.2s

    fmla v8.2s, v16.2s, v24.2s
    fmla v9.2s, v17.2s, v25.2s
    fmla v10.2s, v18.2s, v26.2s
    fmla v11.2s, v19.2s, v27.2s
    fmla v12.2s, v20.2s, v28.2s
    fmla v13.2s, v21.2s, v29.2s
    fmla v14.2s, v22.2s, v30.2s
    fmla v15.2s, v23.2s, v31.2s
    .endr
fmla_2s_v2_insts_end:

    subs x0, x0, #1
    b loop06
end06:

    adr x1, fmla_2s_v2_insts_count
    ldr x0, [x1]
    mul x0, x0, x2

    KERNEL_EPILOGUE
    ret
fmla_2s_v2_insts_count:
    .long (fmla_2s_v2_insts_end - fmla_2s_v2_insts_start) / 4

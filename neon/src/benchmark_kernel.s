#ifdef __APPLE__
#define FUNCLABEL(NAME) _##NAME
#else
#define FUNCLABEL(NAME) NAME
#endif /* __APPLE__ */





    .text


//
//    int fmadd_kernel(int repetitions);
//
    .global FUNCLABEL(fmadd_kernel)
FUNCLABEL(fmadd_kernel):
    movi v0.4s, #0
    movi v1.4s, #1
    mov x2, x0

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
    movi v0.4s, #0
    movi v1.4s, #1
    movi v2.4s, #0
    mov x2, x0

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
    movi v0.4s, #0
    movi v1.4s, #1
    movi v2.4s, #0
    mov x2, x0

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


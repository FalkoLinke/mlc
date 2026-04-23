.text   
//    @brief Permutation operation abc->cba
//    @param size_c Size of dimension c.
//    @param abc    Pointer to row-major tensor abc.
//    @param cba    Pointer to row-major tensor cba.

//   void perm_neon_abc_cba(int64_t       size_c
//                          float const * abc,
//                          float       * cba);
    .global perm_neon_abc_cba
perm_neon_abc_cba:

    // c stride
    lsl x5, x0, #4
    // b stride
    lsl x6, x0, #2
    mov x8, xzr
    
    // c*b stride
    mul x7, x5, x6 
    // loop counter



c_loop:
    // local pointer
    mov x3, x1
    mov x4, x2

    //move in c dimension
    
    add x3, x3, x9
    add x4, x4, x9


    ldr q0, [x3]
    add x3, x3, x6   
    ldr q1, [x3]
    add x3, x3, x6   
    ldr q2, [x3]
    add x3, x3, x6   
    ldr q3, [x3]
    add x3, x3, x6   


    ldr q4, [x3]
    add x3, x3, x6   
    ldr q5, [x3]
    add x3, x3, x6   
    ldr q6, [x3]
    add x3, x3, x6   
    ldr q7, [x3]
    add x3, x3, x6   

    zip1 v16.4s, v0.4s, v4.4s 
    zip1 v17.4s, v1.4s, v5.4s 
    zip1 v18.4s, v2.4s, v6.4s 
    zip1 v19.4s, v3.4s, v7.4s 

    ldr q8, [x3]
    add x3, x3, x6   
    ldr q9, [x3]
    add x3, x3, x6   
    ldr q10, [x3]
    add x3, x3, x6   
    ldr q11, [x3]
    add x3, x3, x6   


    ldr q12, [x3]
    add x3, x3, x6   
    ldr q13, [x3]
    add x3, x3, x6   
    ldr q14, [x3]
    add x3, x3, x6   
    ldr q15, [x3]
    add x3, x3, x6   

    zip1 v20.4s, v8.4s, v12.4s 
    zip1 v21.4s, v9.4s, v13.4s 
    zip1 v22.4s, v10.4s, v14.4s 
    zip1 v23.4s, v11.4s, v15.4s 

    zip1 v24.2d, v16.2d, v20.2d
    zip1 v25.2d, v17.2d, v21.2d
    zip1 v26.2d, v18.2d, v22.2d
    zip1 v27.2d, v19.2d, v23.2d

    stp q24, q25, [x4]
    stp q26, q27, [x4, #32]

    zip2 v24.2d, v16.2d, v20.2d
    zip2 v25.2d, v17.2d, v21.2d
    zip2 v26.2d, v18.2d, v22.2d
    zip2 v27.2d, v19.2d, v23.2d
    
    add x4, x4, x7
    stp q24, q25, [x4]
    stp q26, q27, [x4, #32]

    // zweite paare der geladenen vektoren

    zip2 v16.4s, v0.4s, v4.4s 
    zip2 v17.4s, v1.4s, v5.4s 
    zip2 v18.4s, v2.4s, v6.4s 
    zip2 v19.4s, v3.4s, v7.4s 

    zip2 v20.4s, v8.4s, v12.4s 
    zip2 v21.4s, v9.4s, v13.4s 
    zip2 v22.4s, v10.4s, v14.4s 
    zip2 v23.4s, v11.4s, v15.4s 

    zip1 v24.2d, v16.2d, v20.2d
    zip1 v25.2d, v17.2d, v21.2d
    zip1 v26.2d, v18.2d, v22.2d
    zip1 v27.2d, v19.2d, v23.2d

    add x4, x4, x7
    stp q24, q25, [x4]
    stp q26, q27, [x4, #32]

    zip2 v24.2d, v16.2d, v20.2d
    zip2 v25.2d, v17.2d, v21.2d
    zip2 v26.2d, v18.2d, v22.2d
    zip2 v27.2d, v19.2d, v23.2d
    
    add x4, x4, x7
    stp q24, q25, [x4]
    stp q26, q27, [x4, #32]

    // jetzte wurde c size 4 verarbeitet damit kommt jetzt schleife für vielfache von 4 

    add x8, x8, #4
    mul x9, x8, x5
    cmp x8, x0
    b.ne c_loop

    ret

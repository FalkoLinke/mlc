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
    

    ldr q1, [x1], #16
    ldr q2, [x1], #16
    ldr q3, [x1], #16
    ldr q4, [x1], #16

    trn1 v5.4s, v1.4s, v2.4s
    trn1 v6.4s, v3.4s, v4.4s
    trn2 v7.4s, v1.4s, v2.4s
    trn2 v8.4s, v3.4s, v4.4s

    zip1 v9.2d, v5.2d, v6.2d
    zip2 v10.2d, v5.2d, v6.2d
    zip1 v11.2d, v7.2d, v8.2d
    zip2 v12.2d, v7.2d, v8.2d

    stp q9, q10, [x2]
    stp q11, q12, [x2, #32]

    ret



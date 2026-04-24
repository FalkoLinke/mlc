# Scalable Vector Extension/Scalable Matrix Extension


## Unary Primitives

### Identity 

This kernel implements the identity operation for a 16x16 FP32 matrix.

```c

  /**
   * @brief Identity operation with m=16 and n=16.
   * @param a       Pointer to column-major matrix A.
   * @param b       Pointer to matrix B.
   * @param ld_a    Leading dimension of A.
   * @param ld_b    Leading dimension of B.
   * @param trans_b Column-major B if 0, row-major B if 1. 
   **/
  void identity_16_16( float const * a,
                       float       * b,
                       int64_t       ld_a,
                       int64_t       ld_b,
                       int32_t       trans_b );
```

### Zero Primitive 

This zero primitive sets all entries of a 16x16 FP32 Matrix to zero. The input parameter a is unused since all entries of b are unconditionally set to zero.


```c
  /**
   * @brief Sets all entries of A to zero with size of A = 16x16. 
   * @param a    Pointer to column-major matrix A.
   * @param ld_a Leading dimension of A.
   **/
  void zero_16_16( float const * a,
                   int64_t       ld_a );
```


### ReLU Primitive 

The Rectified Linear Unit (ReLU) is computed by $f(x) = \max(0,x)$. This kernel computes the ReLU for a 16x16 FP32 Matrix.

```c

  /*
   * @brief Computes max(0,x) for every entry of column-major matrix A and writes the result to matrix B.
   * @param a       Pointer to column-major matrix A.
   * @param b       Pointer to matrix B.
   * @param ld_a    Leading dimension of A.
   * @param ld_b    Leading dimension of B.
   * @param trans_b Column-major B if 0, row-major B if 1. 
   **/
  void relu_16_16( float const * a,
                   float       * b,
                   int64_t       ld_a,
                   int64_t       ld_b,
                   int32_t       trans_b );
```

## GEMM

This section implements an FP32 SME microkernel for matrix-matrix multiplications.
The microkernel uses a 32x32 accumulator and is wrapped in the ``gemm_32_32_1`` function, which has the following signature:

```C
   /**
    * @brief GEMM that computes: C+=AB.
    * @param a    Pointer to column-major matrix A.
    * @param b    Pointer to row-major matrix B.
    * @param c    Pointer to column-major matrix C.
    * @param ld_a Leading dimension of A.
    * @param ld_b Leading dimension of B.
    * @param ld_c Leading dimension of C.
    **/
   void gemm_32_32_1( float   const * a,
                      float   const * b,
                      float         * c,
                      int64_t         ld_a,
                      int64_t         ld_b,
                      int64_t         ld_c );
```

## Loops

This section adds loops around the microkernel to implement GEMMs on larger matrices.
First, we add a loop over K and write an extended kernel with the following function signature:

```C

   /**
    * @brief GEMM that computes: C+=AB.
    * @param a    Pointer to column-major matrix A.
    * @param b    Pointer to row-major matrix B.
    * @param c    Pointer to column-major matrix C.
    * @param ld_a Leading dimension of A.
    * @param ld_b Leading dimension of B.
    * @param ld_c Leading dimension of C.
    **/
   void gemm_32_32_512( float   const * a,
                        float   const * b,
                        float         * c,
                        int64_t         ld_a,
                        int64_t         ld_b,
                        int64_t         ld_c );
```
Next, we add a loop over M to implement:

```C
   /**
    * @brief GEMM that computes: C+=AB.
    * @param a    Pointer to column-major matrix A.
    * @param b    Pointer to row-major matrix B.
    * @param c    Pointer to column-major matrix C.
    * @param ld_a Leading dimension of A.
    * @param ld_b Leading dimension of B.
    * @param ld_c Leading dimension of C.
    **/
   void gemm_512_32_512( float   const * a,
                         float   const * b,
                         float         * c,
                         int64_t         ld_a,
                         int64_t         ld_b,
                         int64_t         ld_c );
```

Finally, we add a loop over N to implement:

```C
   /**
    * @brief GEMM that computes: C+=AB.
    * @param a    Pointer to column-major matrix A.
    * @param b    Pointer to row-major matrix B.
    * @param c    Pointer to column-major matrix C.
    * @param ld_a Leading dimension of A.
    * @param ld_b Leading dimension of B.
    * @param ld_c Leading dimension of C.
    **/
   void gemm_512_512_512( float   const * a,
                          float   const * b,
                          float         * c,
                          int64_t         ld_a,
                          int64_t         ld_b,
                          int64_t         ld_c );
```

## Tasks

   1. Implement the SSVE unary kernels for permutation, zero and ReLU.
   2. Implement the SME GEMM kernels.
   3. Test and optimize the kernels. Report your performance in GiB/s for the unary kernel and GFLOPS for the GEMM kernels. 
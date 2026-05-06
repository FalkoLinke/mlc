Week 3/4
===========

.. toctree::
   :maxdepth: 2


In the third and fourth week we implemented the ``identity``, ``zero`` and ``relu`` unary primitives
on 16x16 FP32 matrices.
Furthermore we implemented FP32 SME microkernels for matrix-matrix multiplications.
Finally we determined the performance for all of the mentioned kernels.



Unary Primitives
------------------

Identity 
^^^^^^^^^

The ``identity`` operation copies the contents of the 16x16 matrix A
to the 16x16 matrix B. The matrix may optionally be transposed in the process if requested. 

If no transpose is requested, performing the identity operations simply
involves copying all of the data from A to B.

We implemented the 16x16 transposed identity using an implementation for
4x4 matrices.
Assuming that the 4x4 matrix to transpose is located in ``v0.4s`` through ``v3.4s``,
the following SSVE instructions may be used to transpose it's 2x2 submatrices:

```
    trn1 z4.s, z0.s, z1.s
    trn2 z5.s, z0.s, z1.s
    trn1 z6.s, z2.s, z3.s
    trn2 z7.s, z2.s, z3.s
```

To perform the full 4x4 transpose, we then use the following instructions
to  transpose the higher level 2x2 matrix of 2x2 submatrices:

```
    trn1 z0.d, z4.d, z6.d
    trn1 z1.d, z5.d, z7.d
    trn2 z2.d, z4.d, z6.d
    trn2 z3.d, z5.d, z7.d
```
The whole process is illustrated in the following figure:





The full 16x16 transpose is then implemented by splitting the input matrix up
into a 4x4 matrix of submatrices of size 4x4.
Each submatrix is transposed using the above process and then stored at it's
respective location.















Zero 
^^^^^^^^^

The zero operation simply fills the target 16x16 matrix with zeroes.
Our implementation is an unrolled version of a simple loop performing
this process and does not use any SSVE instructions.






Relu 
^^^^^^^^^

The RELU operation copies the input 16x16 matrix to the target matrix while 
setting all negative entries to zero.
Furthermore the output matrix may optionally be transposed in the process.

We implement the RELU operation by first copying the input matrix to the output matrix
using our previous 16x16 identity operation.
Negative entries are then set to zero by looping over the rows of the output matrix,
and using the ``fmax`` instruction of the following form:

```
    fmax z0.s, p0/m, z0.s, #0.0
```


















GEMM
----



















Benchmarks
----------










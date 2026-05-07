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
We chose to implement the nontransposing identity operation using 
load- and store-instructions into general purpose registers instead of 
the SSVE registers.

Our implementation for a transposing ``identity`` relies on the fact, that
the following two commands may be used to transpose 2x2 FP32 matrices stored 
in the registers ``z0`` and ``z1``::
    trn1 z4.s, z0.s, z1.s
    trn2 z5.s, z0.s, z1.s

.. figure:: transpose_2x2.svg
    :align: center

    Transposition of multiple 2x2 matrices.

This method may be used in order to transpose a 4x4 matrix.
First we transpose each of the 2x2 submatrices located in the corners of the
input matrix.
Then we use the following instructions to transpose the higher-level 
2x2 matrix of submatrices::
    trn1 z0.d, z4.d, z6.d
    trn1 z1.d, z5.d, z7.d
    trn2 z2.d, z4.d, z6.d
    trn2 z3.d, z5.d, z7.d

In that manner we implemented a 4x4 version of the ``identity`` operation.
The full 16x16 transpose is then implemented by splitting the input matrix up
into a 4x4 matrix of submatrices of size 4x4.
Each submatrix is transposed using the above process and then stored at it's
respective location.














Zero 
^^^^^^^^^

The zero operation simply fills the target 16x16 matrix with zeroes.
Our implementation is an unrolled version of a simple loop performing
this process using load- and store-instructions into general purpose registers
and does not make use of SSVE instructions.






Relu 
^^^^^^^^^

The RELU operation copies the input 16x16 matrix to the target matrix while 
setting all negative entries to zero.
Furthermore the output matrix may optionally be transposed in the process.

We implement the RELU operation by first copying the input matrix to the output matrix
using our previous 16x16 identity operation.
Negative entries are then set to zero by looping over the rows of the output matrix,
and using the ``fmax`` instruction of the following form::
    fmax z0.s, p0/m, z0.s, #0.0

We optimized our implementation by loading and processing four rows of the matrix in one 
loop iteration.


















GEMM
----


We implemented a General Matrix Multiply (GEMM) operation that computes :math:`C \leftarrow C + A \times B`.
The GEMM implementation was developed incrementally, starting from a small microkernel 
and scaling up to a full matrix multiplication using a block-based tiling approach.

The 32x32 Microkernel
^^^^^^^^^^^^^^^^^^^^^

The core of our GEMM implementation is a 32x32 microkernel. Since the SME ZA array is partitioned 
into four 32-bit tiles (``za0.s`` to ``za3.s``) for FP32 operations, a 32x32 matrix exactly 
fills the entire ZA array.

Scaling the K-Dimension
^^^^^^^^^^^^^^^^^^^^^^^

For the ``gemm_32_32_512`` kernel, we wrapped the outer product computation in a loop 
over the K-dimension. The C matrix is loaded into the ZA array once, followed by 512 iterations 
where vectors of A and B are loaded and accumulated via ``fmopa``.

Tiling over M and N
^^^^^^^^^^^^^^^^^^^

We iterate over the N and M dimensions of the global matrix in steps of 32. 
In each iteration, we calculate the appropriate memory offsets using the leading dimensions 
(``ld_a``, ``ld_b``, ``ld_c``) to isolate a specific 32x32 block of C. We then execute the K-loop 
to compute the partial results for this specific tile.



Benchmarks
----------


Unary Primitives
^^^^^^^^^^^^^^^^

We determine the performance of our implementations, by repeatedly executing
a given kernel on a contiguous 16x16 FP32 input matrix and a contiguous 16x16 FP32 
output matrix while measuring the time taken ``t`` for all repetitions.
If we let ``s`` be the total number of bytes processed by the kernel, then we
determine the number of bytes processed per second ``v`` as follows::
    v = s / t

When executing our benchmarks on the ``edward.inf-ra.uni-jena.de`` machine,
we obtain the following results:

* ``identity``: 35.3 GiBs
* transposing ``identity``: 1.05 GiBs
* ``zero``: 65.49 GiBs
* ``relu``: 1.55 GiBs
* transposing ``relu``: 0.99 GiBs




GEMM
^^^^^^^^^^^^^^^^


BENCHMARK ERGEBNISSE.

```
double flops_per_call = 2.0 * M * N * K;
double total_flops = flops_per_call * num_iterations;
double gflops = (total_flops / 1e9) / duration;
```

When executing our benchmarks on the ``edward.inf-ra.uni-jena.de`` machine,
we obtain the following results:
* ``gemm_kernel_32_32_1 ``: 50.498 GFLOPS
* ``gemm_kernel_32_32_512 ``: 1824.63 GFLOPS
* ``gemm_kernel_512_32_512 ``: 1821.79 GFLOPS
* ``gemm_kernel_512_512_512 ``: 1802.76 GFLOPS


Week 2
===========


.. toctree::
   :maxdepth: 2

In the second week we implemented microbenchmarks to measure the execution throughput 
in form of the floating point operations executed per second of the following instructions:

* `FMADD (scalar)`, FP32 variant.
* `FMLA (vector)` with arrangement specifier `4S`.
* `FMLA (vector)` with arrangement specifier `2S`.

Furthermore we implemented a kernel, which performs a permutation operation
on a tensor `abc` of the form `abc -> cba`.
The `a` and `b` dimensions were fixed to `8` and `4` respectively,
while the `c` dimensions was allowed to vary.




Execution Throughput
-------------------------

.. 
   1. Benchmark implementation
      1. General overview
      2. Optimizations
   2. Results on the raspberry pis
   3. Explanation of results



For each of the given instructions, we generally execute the same instruction repeatedly and 
measure the total time taken.
If we let ``i`` denote the total number of instructions executed, ``t`` the required execution time 
and ``f`` the floating point operations executed per instruction, then the floating point operations ``z``
executed per second of the benchmark may be determined as follows: 

``z = (i * f) / t``

The functions ``fmadd_kernel``, ``fmla_4s_kernel`` and ``fmla_2s_kernel`` execute
their respective instruction repeatedly as described.

However, these functions do not execute their respective instructions at the 
maximal possible rate.
This is because, in these functions, each instruction depends in it's arguments
on the results of it's preceding instruction.
This prevents the CPU from fully utilizing the instruction pipelines.
The functions ``fmadd_kernel_v2``, ``fmla_4s_kernel_v2`` and ``fmla_2s_kernel_v2``
avoid this issue.




We obtained the following results when running the benchmarks
on the provided Raspberry Pi machines.

* ``fmadd_kernel``:       1.12 GFlops
* ``fmla_4s_kernel``:     9.59 GFlops
* ``fmla_2s_kernel``:     4.80 GFlops
* ``fmadd_kernel_v2``:    9.59 GFlops
* ``fmla_4s_kernel_v2``:  38.37 GFlops
* ``fmla_2s_kernel_v2``:  19.17 GFlops









Permutation
-------------------------

The permutation kernel implements the operation :math:`abc \rightarrow cba` on row-major tensors
with :math:`|a| = 8` and :math:`|b| = 4`, while the size of :math:`c` is a parameter of the kernel.
Element :math:`(a, b, c)` of the input is stored at index :math:`(a \cdot 4 + b) \cdot |c| + c`
and element :math:`(c, b, a)` of the output at index :math:`(c \cdot 4 + b) \cdot 8 + a`.


Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^

For a fixed :math:`b` the operation is a transposition of the :math:`8 \times |c|` matrix formed by
the :math:`a` and :math:`c` dimensions. The kernel therefore consists of an outer loop over the four
:math:`b` values, which contains a loop over the :math:`c` dimension.

We first implemented the scalar kernel ``perm_neon_abc_cba_scalar``, which moves a single element
per iteration of its innermost loop. It serves as a baseline for the optimized version.

The optimized kernel ``perm_neon_abc_cba`` processes four :math:`c` values and all eight :math:`a`
values per iteration. It loads eight vector registers, each holding four consecutive :math:`c`
values of one :math:`a` value, and transposes the two resulting :math:`4 \times 4` blocks::

    trn1 v24.4s, v0.4s, v1.4s
    trn2 v25.4s, v0.4s, v1.4s
    trn1 v26.4s, v2.4s, v3.4s
    trn2 v27.4s, v2.4s, v3.4s
    zip1 v16.2d, v24.2d, v26.2d
    zip1 v17.2d, v25.2d, v27.2d
    zip2 v18.2d, v24.2d, v26.2d
    zip2 v19.2d, v25.2d, v27.2d

The ``trn1`` and ``trn2`` instructions transpose the four :math:`2 \times 2` submatrices,
afterwards ``zip1`` and ``zip2`` exchange the :math:`2 \times 2` submatrices themselves
by moving 64-bit elements.

After both blocks have been transposed, each of the four processed :math:`c` values is held in
two vector registers containing its eight :math:`a` values. Since these eight values are
contiguous in the output tensor, they are written with a single ``stp`` instruction.
This is the main advantage of transposing both :math:`4 \times 4` blocks at once:
the kernel performs full 32-byte stores instead of partial ones.

If :math:`|c|` is not a multiple of four, the remaining :math:`|c| \bmod 4` values are moved
element-wise by a tail loop. The kernel therefore supports arbitrary sizes of :math:`c`,
including :math:`|c| = 0`.

.. literalinclude:: ../../../neon/permutation_kernel.s
   :language: gas


Verification
^^^^^^^^^^^^^^^^^^^^^^^^^

Both kernels are tested in ``neon/permutation_tests.cpp`` against a C++ reference implementation
using `Catch2 <https://github.com/catchorg/Catch2>`_.
The tests cover the sizes :math:`|c| \in \{0, 1, 2, 3, 4, 5, 7, 8, 12, 15, 16, 31, 32, 64, 100, 128, 512\}`,
which include all remainders of the tail loop, and check two guard elements behind the output
tensor in order to detect out of bounds writes.


Results
^^^^^^^^^^^^^^^^^^^^^^^^^

We benchmarked both kernels by executing them repeatedly for at least 0.5 seconds.
Each call reads and writes the entire tensor once, so the number of bytes moved per call is
:math:`2 \cdot |a| \cdot |b| \cdot |c| \cdot 4`.
The results were obtained on an Apple M4.

.. list-table::
   :header-rows: 1

   * - :math:`|c|`
     - Tensor size [KiB]
     - ``perm_neon_abc_cba_scalar`` [GiB/s]
     - ``perm_neon_abc_cba`` [GiB/s]
     - Speed-up
   * - 4
     - 0.5
     - 26.41
     - 166.26
     - 6.3
   * - 8
     - 1
     - 24.62
     - 188.61
     - 7.7
   * - 16
     - 2
     - 27.47
     - 201.91
     - 7.3
   * - 32
     - 4
     - 29.08
     - 205.81
     - 7.1
   * - 64
     - 8
     - 28.71
     - 205.33
     - 7.2
   * - 128
     - 16
     - 27.24
     - 205.01
     - 7.5
   * - 256
     - 32
     - 27.21
     - 199.58
     - 7.3
   * - 512
     - 64
     - 29.15
     - 196.03
     - 6.7
   * - 1024
     - 128
     - 12.37
     - 88.72
     - 7.2
   * - 4096
     - 512
     - 12.02
     - 53.11
     - 4.4
   * - 16384
     - 2048
     - 12.14
     - 52.12
     - 4.3

The vectorized kernel is about seven times faster than the scalar one for all measured sizes.

Both kernels show a clear drop in bandwidth between :math:`|c| = 512` and :math:`|c| = 1024`.
The tensor occupies :math:`8 \cdot 4 \cdot |c| \cdot 4` bytes, so at :math:`|c| = 512` the input and
the output together need 128 KiB, which is the size of the L1 data cache of an M4 performance core.
For larger tensors the kernel is limited by the bandwidth of the L2 cache and finally of main memory,
where the optimized kernel reaches about 52 GiB/s.

The performance is nearly independent of :math:`|c|` as long as the tensor fits into the cache,
because the amount of work per element does not depend on :math:`|c|`.
Small tensors are slightly slower because the loop over the four :math:`b` values and the function
call overhead are amortized over fewer elements.

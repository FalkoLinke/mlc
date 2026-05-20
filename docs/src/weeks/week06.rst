Week 5
===========

.. toctree::
   :maxdepth: 2



In the sixth week we implemented a code generator to instantiate the unary and GEMM primitives.
We determined the performance for the instantiated primitives for different input sizes.



Unary Primitives
----------------

We implemented a code generator to instantiate the ``identity``, ``zero`` and ``relu`` primitives.
The matrix dimensions ``m`` and ``n``, the data type of the stored elements and the storage format of
the output matrix are passed to the code generator when instantiating the primitives.
The actual matrix pointers and stride information are passed when calling the generated kernel.
Currently code generation is only supported for primitives operating on matrices containing
32-bit floating point values.

Our code generator splits the input matrices into square submatrices of size 16x16.
It generates a microkernel to perform the desired operation on a 16x16 matrix.
The full kernel loops over the input matrices' tiles, executing the microkernel on each.






GEMM
----------------











Benchmarks
-----------------

Unary Primitives
^^^^^^^^^^^^^^^^^

We obtained the following benchmark results for the unary primitives on the ``edward.inf-ra.uni-jena.de``.








GEMM
^^^^^^^^^^^^^^^^^




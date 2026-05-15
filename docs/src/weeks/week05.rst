Week 5
===========

.. toctree::
   :maxdepth: 2


In the fifth week we converted the unary and GEMM kernels on 16x16 FP32 matrices to instruction words.
We determined the performance for all of the implemented kernels.


Unary Primitives
----------------

Our implementations of the unary primitives now assume, that the target processor's
scalable vector registers and rows of the ZA matrix hold exactly 16 FP32 values in streaming mode.

The implementation of the nontransposing ``identity`` operation consists of a simple loop
over the rows of the matrix, whichs performs the copy using simple SSVE load and store instructions.

The implementation of the transposing ``identity`` operation first loads the entire 16x16 FP32 input matrix
into the ``za0.s`` tile.
Afterwards the transpose is performed by writing each vertical tile slice of ``za0.s`` to the
respective row in the target matrix.

The ``RELU`` primitive is implemented in the same manner as the ``identity`` operation, while applying
an ``fmax`` instruction to the input matrix while it is stored in the registers.





GEMM
----------------









Benchmarks
-----------------

Unary Primitives
^^^^^^^^^^^^^^^^^

We obtained the following benchmark results for the unary primitives on the ``edward.inf-ra.uni-jena.de``.

* ``identity``: 35.3 GiBs
* transposing ``identity``: 1.05 GiBs
* ``zero``: 65.49 GiBs
* ``relu``: 1.55 GiBs
* transposing ``relu``: 0.99 GiBs





GEMM
^^^^^^^^^^^^^^^^^



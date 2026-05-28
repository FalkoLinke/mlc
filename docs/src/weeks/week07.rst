Week 7
===========

.. toctree::
   :maxdepth: 2



In the seventh week we implemented an interpreter and a compiler for the Tiled Execution Intermediate Representation (TEIR).




TEIR Interpreter
------------------




TEIR compiler
------------------

The TEIR compiler converts a TEIR operation into a function of the signature ``void func(void**)`` using AArch64 assembly instructions.
This function accepts a pointer to an array of the addresses of the tensor memory areas in the register ``x0``
and performs the compiled operation on these tensors.


Implementation
^^^^^^^^^^^^^^^^^^

The resulting function has the following general structure in pseudo-AArch64-assembly:

.. code-block:: none

    operation:
        // function prologue
        stp x29, x30, [sp, #-16]!
        mov x29, sp
        // ... save necessary registers

        // general setup
        mov x28, x0
        ldr x27, kernel_dispatch_table
        
        // loop nest
        // ...

        // function epilogue
        // ... restore necessary registers
        ldp x29, x30, [sp], #16
        ret

        // tensor shape data (extends, strides, offsets)
    shape_data:
        .quad #<extend01>
        .quad #<extend02>
        ...


This function makes use of the AArch64 general purpose registers as follows:

.. list-table::

    * - Registers
      - Usage
    * - ``x0`` through ``x7``
      - Scratch registers and parameter registers for JIT-kernels
    * - ``x19`` through ``x26``
      - Loop index registers
    * - ``x27``
      - Pointer to the kernel dispatch table
    * - ``x28``
      - Pointer to the tensor addresses


The compiler generates the loop nest recursively, keeping track of the current index of each axis in the loop index registers.
Currently this limits the maximum loop depth to the number of these registers.
The axis' strides and offsets will be applied by the corresponding loop, where offsets are added to the tensors before and strides inside the loop.
After the loop the offsets and total accumulated strides will be subtracted from the tensor pointers.

In order to load the extends, strides and offsets of the axes into registers at runtime, the compiler
appends all of the required data after the function in the executable memory area.

During the recursion over the nodes of the schedule, the compiler will encounter invocation nodes.
For each invocation node the compiler will attempt to lower the primitive referenced by that node.
This process inserts assembly code in the appropriate location of the loop nest to execute that primitive.
When lowering a primitive the compiler checks its type, the roles and other available data
to determine the assembly code to be generated.

The generated code may perform the required operation directly inside of the loop nest, or it may use an indirect branch instruction
to one of the functions stored in the ``kernel_dispatch_table`` to execute a JIT-compiled kernel.
The ``kernel_dispatch_table`` is maintained by the compiler and holds function pointers to the JIT-compiled kernels,
which are needed by the primitives.



Benchmark
^^^^^^^^^^^^^^^^^^

.. code-block:: none
   === TEIR @contraction Interpreter Benchmark ===

   Tensor shapes:
   %in0  [p=128 q=96 t=32 u=256]  384 MiB
   %in1  [t=32 r=96 u=256 s=64]  192 MiB
   %out  [p=128 q=96 r=96 s=64]  288 MiB

   FLOPs per run: 1.237 TFLOP
   Repetitions:   10

   Warmup ... done

   rep  1/10  time: 4.311 s    GFLOP/s: 286.96
   rep  2/10  time: 4.300 s    GFLOP/s: 287.67
   rep  3/10  time: 4.317 s    GFLOP/s: 286.54
   rep  4/10  time: 4.291 s    GFLOP/s: 288.25
   rep  5/10  time: 4.316 s    GFLOP/s: 286.62
   rep  6/10  time: 4.322 s    GFLOP/s: 286.17
   rep  7/10  time: 4.311 s    GFLOP/s: 286.95
   rep  8/10  time: 4.310 s    GFLOP/s: 286.99
   rep  9/10  time: 4.302 s    GFLOP/s: 287.53
   rep 10/10  time: 4.303 s    GFLOP/s: 287.45

   --- Summary ---
   best   4.291 s  ->  288.25 GFLOP/s
   avg    4.308 s  ->  287.11 GFLOP/s
   worst  4.322 s  ->  286.17 GFLOP/s


.. code-block:: none
   === TEIR @matmul Interpreter Benchmark ===

   Tensor shapes:
   %in0  [m0=256 k0=16 m1=32 k1=512]  256 MiB
   %in1  [k0=16 n0=128 k1=512 n1=64]  256 MiB
   %out  [m0=256 n0=128 m1=32 n1=64]  256 MiB

   FLOPs per run: 1.100 TFLOP
   Repetitions:   10

   Warmup ... done

   rep  1/10  time: 4.617 s   GFLOP/s: 238.16
   rep  2/10  time: 4.648 s   GFLOP/s: 236.55
   rep  3/10  time: 4.647 s   GFLOP/s: 236.63
   rep  4/10  time: 4.645 s   GFLOP/s: 236.72
   rep  5/10  time: 4.633 s   GFLOP/s: 237.31
   rep  6/10  time: 4.627 s   GFLOP/s: 237.62
   rep  7/10  time: 4.628 s   GFLOP/s: 237.55
   rep  8/10  time: 4.683 s   GFLOP/s: 234.81
   rep  9/10  time: 4.635 s   GFLOP/s: 237.21
   rep 10/10  time: 4.639 s   GFLOP/s: 237.03

   --- Summary ---
   best   4.617 s  ->  238.16 GFLOP/s
   avg    4.640 s  ->  236.96 GFLOP/s
   worst  4.683 s  ->  234.81 GFLOP/s

.. code-block:: none
   === TEIR @transposition Interpreter Benchmark ===

   Tensor shapes:
   %in   [a=96 b=128 c=48 d=32]  72 MiB
   %out  (Stride overlaps)                69 MiB

   Memory processed per run: 150.995 MB
   Repetitions:   10

   Warmup ... done

   rep  1/10  time: 0.068 s   GB/s: 2.22
   rep  2/10  time: 0.068 s   GB/s: 2.23
   rep  3/10  time: 0.068 s   GB/s: 2.22
   rep  4/10  time: 0.068 s   GB/s: 2.22
   rep  5/10  time: 0.069 s   GB/s: 2.20
   rep  6/10  time: 0.069 s   GB/s: 2.20
   rep  7/10  time: 0.069 s   GB/s: 2.19
   rep  8/10  time: 0.074 s   GB/s: 2.04
   rep  9/10  time: 0.068 s   GB/s: 2.21
   rep 10/10  time: 0.068 s   GB/s: 2.23

   --- Summary ---
   best   0.068 s  ->  2.23 GB/s
   avg    0.069 s  ->  2.20 GB/s
   worst  0.074 s  ->  2.04 GB/s
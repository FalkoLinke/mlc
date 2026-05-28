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




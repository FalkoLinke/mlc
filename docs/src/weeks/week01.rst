Week 1
===========

.. toctree::
   :maxdepth: 2


In the first week we implemented the functions ``inner_product`` and ``outer_product``
in AArch64 assembly, tested them against the provided C++ reference implementations
and stepped through an example call using the GNU Project Debugger (GDB).


Tests
-----

The tests are located in ``assembly/base_math_driver.cpp`` and use the unit test framework
`Catch2 <https://github.com/catchorg/Catch2>`_ (v3.8.1), which is fetched by CMake.
The test cases are registered with CTest via ``catch_discover_tests``, so all tests of the project
can be executed with a single ``ctest`` call.

.. list-table::
   :header-rows: 1

   * - Test case
     - Checks
   * - ``inner_product of fixed vectors``
     - Hand-computed results for several small vectors
   * - ``inner_product of empty vectors is zero``
     - ``size = 0`` returns ``0`` without touching memory
   * - ``inner_product does not overflow 32 bits``
     - Results larger than :math:`2^{32}` are returned correctly
   * - ``inner_product matches C++ reference``
     - Random vectors of sizes 1, 2, 3, 7, 16, 100 and 1000 against ``inner_product_cpp``
   * - ``outer_product of fixed vectors``
     - Hand-computed matrices, including non-symmetric inputs
   * - ``outer_product of empty vectors does not write``
     - ``size = 0`` leaves the output untouched
   * - ``outer_product does not overflow 32 bits``
     - :math:`(2^{32}-1)^2` is stored correctly
   * - ``outer_product matches C++ reference``
     - Random vectors of sizes 1, 2, 3, 7, 16 and 100 against ``outer_product_cpp``

The random tests use values below :math:`2^{15}`, because the provided reference implementations
multiply in 32 bits and would overflow for larger values.



Debugging with GDB
------------------

To step through a call of ``inner_product`` we use the following small program,
which computes :math:`(1, 2, 3) \cdot (4, 5, 6) = 32`:

.. code-block:: c++

   #include "base_math_s.h"

   int main() {
       uint32_t a[] = {1, 2, 3};
       uint32_t b[] = {4, 5, 6};
       return (int) inner_product(a, b, 3);
   }

It is compiled with debug information and started in GDB:

.. code-block:: none

   $ g++ -g -I. ip.cpp -x assembler-with-cpp base_math.s -o ip
   $ gdb ./ip
   (gdb) break inner_product
   (gdb) run
   (gdb) layout asm          # show the disassembly around the program counter
   (gdb) info registers x0 x1 x2
   (gdb) x/3uw $x0           # print the three elements of a
   (gdb) x/3uw $x1           # print the three elements of b
   (gdb) stepi               # execute one instruction
   (gdb) info registers x3 x4 x5 x6
   (gdb) finish              # run until the function returns
   (gdb) info registers x0

When the breakpoint is hit, ``x0`` and ``x1`` hold the stack addresses of ``a`` and ``b``
and ``x2`` holds the size ``3``. Examining the memory with ``x/3uw`` shows the elements ``1 2 3`` and ``4 5 6``.

Stepping through the loop with ``stepi`` shows the following register contents
at the end of each iteration (after ``add x3, x3, x6``):

.. list-table::
   :header-rows: 1

   * - Iteration
     - ``w4``
     - ``w5``
     - ``x6``
     - ``x3``
     - ``w2`` after ``subs``
   * - 1
     - 1
     - 4
     - 4
     - 4
     - 2
   * - 2
     - 2
     - 5
     - 10
     - 14
     - 1
   * - 3
     - 3
     - 6
     - 18
     - 32
     - 0

During each iteration the pointers ``x0`` and ``x1`` increase by 4 due to the post-indexed loads.
After the third iteration ``subs`` sets the zero flag, ``b.ne`` falls through,
the result is moved to ``x0`` and ``finish`` reports the return value ``32``.

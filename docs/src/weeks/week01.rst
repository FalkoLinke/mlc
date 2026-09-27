Week 1
===========

.. toctree::
   :maxdepth: 2


In the first week we implemented the functions ``inner_product`` and ``outer_product``
in AArch64 assembly, tested them against the provided C++ reference implementations
and stepped through an example call using the GNU Project Debugger (GDB).



Implementation
--------------

Both functions are implemented in ``assembly/base_math.s`` and follow the AArch64
procedure call standard (AAPCS64):
the arguments are passed in ``x0`` to ``x3`` and the result is returned in ``x0``.
Both functions only use the caller-saved registers ``x0`` to ``x15``,
therefore no registers have to be saved on the stack and neither function sets up a stack frame.

Since macOS prefixes C symbols with an underscore while Linux does not, the file is
preprocessed (``-x assembler-with-cpp``) and all labels are wrapped in a ``FUNCLABEL`` macro.
This way the same source assembles on the Apple M4 as well as on the Raspberry Pi.

.. code-block:: gas

   #ifdef __APPLE__
   #define FUNCLABEL(NAME) _##NAME
   #else
   #define FUNCLABEL(NAME) NAME
   #endif /* __APPLE__ */


Inner Product
^^^^^^^^^^^^^

.. code-block:: c

   int64_t inner_product(uint32_t const *i_a,
                         uint32_t const *i_b,
                         uint32_t const  i_size);

The function loops once over both vectors.
The loop uses post-indexed loads (``ldr w4, [x0], #4``), which read the current element
and advance the pointer to the next element in a single instruction.
The two 32-bit elements are multiplied with ``umull``, a widening multiplication,
which produces the full 64-bit product. The product is added to the 64-bit accumulator ``x3``.
The size register ``w2`` is used as down-counter, so the loop condition is simply the
zero flag set by ``subs``.
An empty input is handled by the ``cbz`` before the loop.

.. list-table::
   :header-rows: 1

   * - Register
     - Usage
   * - ``x0``
     - Pointer to the current element of ``a``, result on return
   * - ``x1``
     - Pointer to the current element of ``b``
   * - ``w2``
     - Number of remaining elements
   * - ``x3``
     - Accumulated result
   * - ``w4``, ``w5``
     - Current elements of ``a`` and ``b``
   * - ``x6``
     - Product of the current elements


Outer Product
^^^^^^^^^^^^^

.. code-block:: c

   void outer_product(uint32_t const *i_a,
                      uint32_t const *i_b,
                      uint32_t const  i_size,
                      uint64_t       *o_c);

The outer product consists of two nested loops.
The outer loop loads ``a[row]`` and resets the pointer into ``b`` to its base address ``x1``.
The inner loop loads ``b[col]``, computes the widening product ``a[row] * b[col]``
and stores it with a post-indexed store (``str x9, [x3], #8``).
Because the result matrix is written in row-major order, the output pointer ``x3``
only ever has to be advanced by one element and no index computation is needed.

.. list-table::
   :header-rows: 1

   * - Register
     - Usage
   * - ``x0``
     - Pointer to the current element of ``a``
   * - ``x1``
     - Base pointer to ``b``
   * - ``w2``
     - Size
   * - ``x3``
     - Pointer to the current element of ``c``
   * - ``w4``, ``w5``
     - Row and column down-counters
   * - ``x6``
     - Pointer to the current element of ``b``
   * - ``w7``, ``w8``
     - Current elements of ``a`` and ``b``
   * - ``x9``
     - Product of the current elements

The complete source code:

.. literalinclude:: ../../../assembly/base_math.s
   :language: gas


Bug fixes
^^^^^^^^^

Our first version of the kernels had the following issues, which were found during review
and fixed in the current version:

* ``outer_product`` used ``x20`` as scratch register.
  ``x19`` to ``x28`` are callee-saved, so the caller's value was destroyed.
* ``inner_product`` accumulated the result in the 32-bit register ``w0``,
  although the function returns a 64-bit integer. Large results were truncated.
* The comments used ``;``, which is not a comment character for the GNU assembler on AArch64,
  and the labels contained the macOS underscore prefix. The file did not assemble on Linux.



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


Continuous Integration
^^^^^^^^^^^^^^^^^^^^^^

The GitHub Actions workflow ``.github/workflows/tests.yaml`` builds the whole project on every push
and pull request using the build commands from the ``README.md`` and runs all tests with ``ctest``.
The hosted ``macos-latest`` runners are arm64 machines without SME support.
CMake detects whether the host supports SME (``sysctl hw.optional.arm.FEAT_SME`` on macOS,
``/proc/cpuinfo`` on Linux). On hosts without SME the SME and JIT tests are still built,
but not registered with CTest, so the pipeline tests the portable parts
and every test runs on SME capable machines such as the Apple M4.

The documentation workflow ``.github/workflows/sphinx.yaml`` builds this report with
``sphinx-build -W --keep-going``, so every Sphinx warning fails the pipeline.



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

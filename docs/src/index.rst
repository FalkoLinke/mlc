####################
Project Report
####################

This is the project report of Falko Linke and Johann Schwarze for the lecture Machine Learning Compilers.

Over the course of the project, we moved from hand-written AArch64 assembly towards a compiler for tensor operations:

* **Weeks 1 and 2:** AArch64 assembly and Neon kernels, including microbenchmarks and a permutation kernel.
* **Weeks 3 and 4:** unary and GEMM kernels using the Scalable Matrix Extension (SME).
* **Weeks 5 and 6:** a JIT code generator which instantiates unary and GEMM primitives at runtime.
* **Weeks 7 and 8:** the tensor runtime TEIR with an interpreter, a compiler, parallelization and optimization passes.
* **Group specific component:** an optimized JIT generator for GEMMs of any size with transposed inputs and outputs.

Each chapter states the machine the benchmarks were run on; most results were obtained on an Apple M4.
The source code, the build instructions and the tests are available in the
`project repository <https://github.com/FalkoLinke/mlc>`_.


.. toctree::
   :maxdepth: 2

   weeks/week01
   weeks/week02
   weeks/week03
   weeks/week05
   weeks/week06
   weeks/week07
   weeks/week08
   weeks/group_specific

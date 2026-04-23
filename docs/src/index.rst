.. Project Report master file, created by
   sphinx-quickstart on Wed Apr 15 16:08:33 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

####################
Project Report
####################

Add your content using ``reStructuredText`` syntax. See the
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
documentation for details.


.. toctree::
   :maxdepth: 2
   :caption: Contents:


*******************
Introduction
*******************





Weekly Tasks
===================

This section describes the solutions for the various weekly 
tasks over the course of this project.




Week 1
-------------------

In the first week we implemented the functions `inner_product` and `outer_product`
in AArch64 assembly.













Week 2
-------------------

In the second week we implemented microbenchmarks to measure the execution throughput 
in form of the floating point operations executed per second of the following instructions:

* `FMADD (scalar)`, FP32 variant.
* `FMLA (vector)` with arrangement specifier `4S`.
* `FMLA (vector)` with arrangement specifier `2S`.

Furthermore we implemented a kernel, which performs a permutation operation
on a tensor `abc` of the form `abc -> cba`.
The `a` and `b` dimensions were fixed to `8` and `4` respectively,
while the `c` dimensions was allowed to vary.




**Execution Throughput**
^^^^^^^^^^^^^^^^^^^^^^^^

.. 
   1. Benchmark implementation
      1. General overview
      2. Optimizations
   2. Results on the raspberry pis
   3. Explanation of results



For each of the given instructions, we generally execute the same instruction repeatedly and 
measure the total time taken.
If we let `i` denote the total number of instructions executed, `t` the required execution time 
and `f` the floating point operations executed per instruction, then the floating point operations `z`
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









**Permutation**
^^^^^^^^^^^^^^^^^^^^^


















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

In the second week we microbenchmarked the execution throughput of the
following instructions

* `FMADD (scalar)`, FP32 variant.
* `FMLA (vector)` with arrangement specifier `4S`.
* `FMLA (vector)` with arrangement specifier `2S`.

Furthermore we implemented a kernel, which performs a permutation operation
on a tensor `abc` of the form `abc -> cba`.
The `a` and `b` dimensions were fixed to `8` and `4` respectively,
while the `c` dimensions was allowed to vary.



**Execution Throughput**
^^^^^^^^^^^^^^^^^^^^^^^^














**Permutation**
^^^^^^^^^^^^^^^^^^^^^


















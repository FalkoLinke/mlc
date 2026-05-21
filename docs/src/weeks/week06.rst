Week 6
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

In order to benchmark our generated FP32 kernels, we measured the time taken to execute them repeatedly.
For each primitive we generated a kernel with each of the following settings:

* The number of rows ``m`` selected from ``{64, 128, 512}``.
* The number of columns ``n`` selected from ``{64, 128, 512}``.
* The storage format of the output matrix ``trans_b`` selected from ``{false, true}``.

We obtained the following benchmark results for the unary primitives on the ``edward.inf-ra.uni-jena.de``.
They are also available :download:`here<results.csv>`.
``trans_b = 0`` indicates a column-major output matrix, ``trans_b = 1`` a row-major output matrix. 

.. code-block::

   Operation		(m, n)		trans_b		Duration [s]		Bytes transferred [GiB]		GiBs
   identity		(64, 64)		0		1.73608		300		172.803
   identity		(64, 128)		0		1.73795		300		172.617
   identity		(64, 512)		0		6.94761		300		43.1803
   identity		(128, 64)		0		8.72457		300		34.3857
   identity		(128, 128)		0		9.52319		300		31.5021
   identity		(128, 512)		0		9.50211		300		31.5719
   identity		(512, 64)		0		8.97768		300		33.4162
   identity		(512, 128)		0		9.27079		300		32.3597
   identity		(512, 512)		0		8.70428		300		34.4658

   identity		(64, 64)		1		9.21632		300		32.551
   identity		(64, 128)		1		9.48204		300		31.6388
   identity		(64, 512)		1		9.29045		300		32.2912
   identity		(128, 64)		1		9.59753		300		31.258
   identity		(128, 128)		1		9.6224		300		31.1772
   identity		(128, 512)		1		9.47313		300		31.6685
   identity		(512, 64)		1		9.45504		300		31.7291
   identity		(512, 128)		1		8.71801		300		34.4115
   identity		(512, 512)		1		9.46249		300		31.7041

   zero		(64, 64)		0		4.43084		300		67.7073
   zero		(64, 128)		0		4.43103		300		67.7043
   zero		(64, 512)		0		4.45577		300		67.3284
   zero		(128, 64)		0		4.4169		300		67.9209
   zero		(128, 128)		0		4.45713		300		67.3079
   zero		(128, 512)		0		4.4104		300		68.021
   zero		(512, 64)		0		4.45651		300		67.3173
   zero		(512, 128)		0		4.45404		300		67.3546
   zero		(512, 512)		0		4.44726		300		67.4572

   zero		(64, 64)		1		4.50156		300		66.6436
   zero		(64, 128)		1		4.38322		300		68.4428
   zero		(64, 512)		1		4.4776		300		67.0002
   zero		(128, 64)		1		4.4496		300		67.4218
   zero		(128, 128)		1		4.44065		300		67.5577
   zero		(128, 512)		1		4.47058		300		67.1054
   zero		(512, 64)		1		4.48433		300		66.8996
   zero		(512, 128)		1		4.42612		300		67.7794
   zero		(512, 512)		1		4.47758		300		67.0004

   relu		(64, 64)		0		9.35826		300		32.0573
   relu		(64, 128)		0		9.12226		300		32.8866
   relu		(64, 512)		0		9.20762		300		32.5817
   relu		(128, 64)		0		9.21717		300		32.548
   relu		(128, 128)		0		9.49848		300		31.584
   relu		(128, 512)		0		9.52164		300		31.5072
   relu		(512, 64)		0		9.2242		300		32.5231
   relu		(512, 128)		0		9.39546		300		31.9303
   relu		(512, 512)		0		8.45415		300		35.4855

   relu		(64, 64)		1		8.95636		300		33.4958
   relu		(64, 128)		1		9.43993		300		31.7799
   relu		(64, 512)		1		9.41781		300		31.8546
   relu		(128, 64)		1		9.45306		300		31.7358
   relu		(128, 128)		1		9.38998		300		31.949
   relu		(128, 512)		1		9.06771		300		33.0844
   relu		(512, 64)		1		9.08826		300		33.0096
   relu		(512, 128)		1		9.2292		300		32.5055
   relu		(512, 512)		1		9.54665		300		31.4246





GEMM
^^^^^^^^^^^^^^^^^




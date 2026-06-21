# README

This directory contains various benchmarks measuring the performance of different implementations of several operations.






## GEMM

### Results

The following results have been obtained on ``edward.inf-ra.uni-jena.de``:

```
m	n	k	trans_a	trans_b	trans_c	Description	GFlops	Duration [s]
16	32	1	0	1	0	default	27.3149	3.74886
16	32	16	0	1	0	default	342.367	0.47855
16	32	512	0	1	0	default	935.583	5.60386
16	32	512	0	1	0	default	1536.09	3.41313
16	64	512	0	1	0	default	1789.51	5.85956
16	32	16	1	1	0	ZA transpose	147.284	1.11241
16	32	512	1	1	0	ZA transpose	206.121	2.54359
16	32	512	1	1	0	TBL transpose	687.143	7.62997
16	32	512	1	1	0	TBLV2 transpose	682.196	7.6853
16	16	512	0	1	0	default	469.393	5.58474
16	16	512	1	0	0	TBL stack transpose	204.42	12.8238
16	16	512	1	0	0	TBLV2 regs transpose	109.785	23.8779
```



- It is faster to use the vector registers for transposes than the ZA matrix.
- It is faster to read full 16x16 tiles and storing transposed matrices on the stack than reading 16x8 tiles and storing transposed matrices in registers.






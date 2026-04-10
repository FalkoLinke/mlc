	.global _add
_add:
	add x0, x0, x1
	ret




; int64_t inner_product_cpp(uint32_t const *i_a,
;                           uint32_t const *i_b,
;                           uint32_t const i_size); 
	.global _inner_product
_inner_product:
	mov x3, x2	; x3 - size
	mov x2, x1	; x2 - ptr to b
	mov x1, x0	; x1 - ptr to a
	mov x0, #0	; x0 - result register

loop01:
	cmp x3, #0
	beq end01 
	
	ldr w4, [x1]
	ldr w5, [x2]
	mul w4, w4, w5
	adds w0, w0, w4	

	adds x1, x1, #4
	adds x2, x2, #4
	subs x3, x3, #1
	b loop01
end01:
	ret






; void outer_product_cpp(uint32_t const *i_a,
;                        uint32_t const *i_b,
;                        uint32_t const i_size,
;                        uint64_t *o_c); 
	.global _outer_product
_outer_product:
	mov x4, x2
	mov x2, x3
	mov x3, x4
	mov x6, x1

	; x0 - ptr to a
	; x1 - ptr to b
	; x2 - ptr to c
	; x3 - size
	; x4 - row counter
	; x5 - column counter
	; x6 - copy of base ptr to b

	mov x4, x3
loop02:
	cmp x4, #0
	beq end02	


	mov x1, x6
	mov x5, x3
loop03:
	cmp x5, #0
	beq end03

	mov x7, #0
	mov x8, #0
	ldr w7, [x0]
	ldr w8, [x1]
	mul x7, x7, x8
	str x7, [x2]

	adds x2, x2, #8

	adds x1, x1, #4
	subs x5, x5, #1
	b loop03
end03:

	adds x0, x0, #4
	subs x4, x4, #1
	b loop02
end02:
	ret







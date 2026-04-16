	.global _add
_add:
	add x0, x0, x1
	ret




; int64_t inner_product(uint32_t const *i_a,
;                       uint32_t const *i_b,
;                       uint32_t const i_size); 
	.global _inner_product
_inner_product:

	movi v4.16b, #0
;	movi v5.16b, #0
	mov x6, xzr
	
	uxtw x2, w2

	and x3, x2, #0x3
	lsr x2, x2, #2
	add x2, x2, #1
	
loop_vector_inner:
	subs x2, x2, #1
	b.eq check_rest

;	ldp q0, q1, [x0], #32
;	ldp q2, q3, [x1], #32
	
;	mla v4.4s, v0.4s, v2.4s
;	mla v5.4s, v1.4s, v3.4s

	ldr q0, [x0], #16
	ldr q2, [x1], #16

	mla v4.4s, v0.4s, v2.4s

	b loop_vector_inner

check_rest:
    cbz x3, finish_inner

loop_rest_inner:
	ldr w4, [x0], #4
	ldr w5, [x1], #4
	madd x6, x4, x5, x6;


	subs x3, x3, #1
	b.ne loop_rest_inner

finish_inner:
;	add v4.4s, v4.4s, v5.4s
	addv s4, v4.4s
	umov w0, v4.s[0]

	add x0, x0, x6
	ret


; void outer_product(uint32_t const *i_a,
;                    uint32_t const *i_b,
;                    uint32_t const i_size,
;                    uint64_t *o_c); 
	.global _outer_product
_outer_product:

	movi v4.16b, #0

	mov x6, xzr
	
	uxtw x2, w2

	and x3, x2, #0x3
	lsr x2, x2, #2
	add x2, x2, #1
	
loop_vector_outer:
	subs x2, x2, #1
	b.eq check_rest_outer

	ldr q0, [x0], #16
	ldr q1, [x1], #16

	mla v4.4s, v1.4s, v0.s[0]
	mla v5.4s, v1.4s, v0.s[1]
	mla v6.4s, v1.4s, v0.s[2]
	mla v7.4s, v1.4s, v0.s[3]

	b loop_vector_outer

check_rest_outer:

	ret








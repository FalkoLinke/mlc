#ifdef __APPLE__
#define FUNCLABEL(NAME) _##NAME
#else
#define FUNCLABEL(NAME) NAME
#endif /* __APPLE__ */


    .text



/*
int32_t add(int32_t const a,
            int32_t const b);
*/
    .global FUNCLABEL(add)
FUNCLABEL(add):
    add w0, w0, w1
    ret




/*
int64_t inner_product(uint32_t const *i_a,
                      uint32_t const *i_b,
                      uint32_t const  i_size);

x0 - pointer to a
x1 - pointer to b
w2 - size (counts down to zero)
x3 - accumulated result
w4 - current element of a
w5 - current element of b
x6 - product of the current elements
*/
    .global FUNCLABEL(inner_product)
FUNCLABEL(inner_product):
    mov x3, #0
    cbz w2, inner_product_end

inner_product_loop:
    ldr w4, [x0], #4                        // load a[i], advance pointer
    ldr w5, [x1], #4                        // load b[i], advance pointer
    umull x6, w4, w5                        // widening 32x32 -> 64 bit multiply
    add x3, x3, x6
    subs w2, w2, #1
    b.ne inner_product_loop

inner_product_end:
    mov x0, x3
    ret




/*
void outer_product(uint32_t const *i_a,
                   uint32_t const *i_b,
                   uint32_t const  i_size,
                   uint64_t       *o_c);

Only caller-saved scratch registers (x0 - x15) are used,
therefore no registers have to be saved on the stack.

x0 - pointer to the current element of a
x1 - base pointer to b
w2 - size
x3 - pointer to the current element of c
w4 - row counter
w5 - column counter
x6 - pointer to the current element of b
w7 - current element of a
w8 - current element of b
x9 - product of the current elements
*/
    .global FUNCLABEL(outer_product)
FUNCLABEL(outer_product):
    cbz w2, outer_product_end

    mov w4, w2
outer_product_row_loop:
    ldr w7, [x0], #4                        // load a[row], advance pointer
    mov x6, x1                              // restart at b[0]
    mov w5, w2

outer_product_col_loop:
    ldr w8, [x6], #4                        // load b[col], advance pointer
    umull x9, w7, w8                        // widening 32x32 -> 64 bit multiply
    str x9, [x3], #8                        // store c[row * size + col], advance pointer
    subs w5, w5, #1
    b.ne outer_product_col_loop

    subs w4, w4, #1
    b.ne outer_product_row_loop

outer_product_end:
    ret

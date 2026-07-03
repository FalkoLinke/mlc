#ifndef ASSEMBLY_BASE_MATH_S_H
#define ASSEMBLY_BASE_MATH_S_H

extern "C" {




int32_t add(int32_t const a, int32_t const b);

int64_t inner_product(uint32_t const* a, 
                      uint32_t const* b, 
                      uint32_t const size);

void outer_product(uint32_t const *i_a,
                   uint32_t const *i_b,
                   uint32_t const i_size,
                   uint64_t *o_c); 




}

#endif /*ASSEMBLY_BASE_MATH_S_H*/
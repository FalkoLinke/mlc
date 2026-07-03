#ifndef ASSEMBLY_BASE_MATH_CPP_H
#define ASSEMBLY_BASE_MATH_CPP_H



int64_t inner_product_cpp(uint32_t const *i_a,
                          uint32_t const *i_b,
                          uint32_t const i_size); 
 

void outer_product_cpp(uint32_t const *i_a,
                       uint32_t const *i_b,
                       uint32_t const i_size,
                       uint64_t *o_c); 



#endif /*ASSEMBLY_BASE_MATH_CPP_H*/
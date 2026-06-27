#ifndef BASE_MATH_CPP_H
#define BASE_MATH_CPP_H



int64_t inner_product_cpp(uint32_t const *i_a,
                          uint32_t const *i_b,
                          uint32_t const i_size); 
 

void outer_product_cpp(uint32_t const *i_a,
                       uint32_t const *i_b,
                       uint32_t const i_size,
                       uint64_t *o_c); 



#endif /*BASE_MATH_CPP_H*/
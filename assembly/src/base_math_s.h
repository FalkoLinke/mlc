extern "C" {

int32_t add(const int32_t a, const int32_t b);

int64_t inner_product(uint32_t const* a, uint32_t const* b, uint32_t size);

void outer_product(uint32_t const *i_a,
                       uint32_t const *i_b,
                       uint32_t const i_size,
                       uint64_t *o_c); 







}

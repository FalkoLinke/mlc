#include <iostream>


uint64_t sve_vector_length() {
    uint64_t result;

    asm volatile(
        "smstart\n"
        "rdvl %0, #1\n"
        "smstop\n"
        : "=r"(result)
    );

    return result;
}

uint64_t get_za_dim() {
    uint64_t result;

    asm volatile(
        "smstart\n"
        "rdsvl %0, #1\n"
        "smstop\n"
        : "=r"(result)
    );

    return result;
}


int main() {
    std::cout << "SSVE vector length [B]: " << sve_vector_length() << std::endl;
    std::cout << "ZA dimension [B]: " << get_za_dim() << std::endl;
    return 0;
}
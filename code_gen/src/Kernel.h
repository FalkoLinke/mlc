#include <cstdint>
#include <cstddef>
#include <vector>
#include <map>

#ifndef MINI_JIT_KERNEL_H
#define MINI_JIT_KERNEL_H

namespace mini_jit {
  struct BranchRef;
  class Kernel;
}


struct mini_jit::BranchRef {
  private:

  public:
    //! the index of the instruction in the kernel
    uint32_t const idx;

    //! the label to be resolved
    std::string const label;

    //! the number of bits allowed for the offset
    uint32_t const offs_bits;

    //! the place in the instruction to insert the offset at
    uint32_t const offs_shift;

    BranchRef(uint32_t const idx, std::string const label, uint32_t offs_bits, uint32_t offs_shift);

    ~BranchRef() = default;

    BranchRef( BranchRef const & ) = default;
    BranchRef & operator=( BranchRef const & ) = delete;
    BranchRef( BranchRef && ) noexcept = default;
    BranchRef & operator=( BranchRef && ) noexcept = delete;
};


class mini_jit::Kernel {
  private:
    //! high-level code buffer
    std::vector< uint32_t > m_buffer;

    //! size of the kernel
    std::size_t m_size_alloc = 0;

    //! executable kernel
    void * m_kernel = nullptr;

    //! labeled instruction indices
    std::map<std::string, uint32_t> label_indices;

    //! unresolved branch instructions
    std::map<std::string, std::vector<BranchRef>> unresolved_branches;

    void resolve_branch( BranchRef branch );

    /**
     * Allocates memory through POSIX mmap.
     *
     * @param num_bytes number of bytes.
     **/
    void * alloc_mmap( std::size_t num_bytes ) const;

    /**
     * Release POSIX mmap allocated memory.
     *
     * @param num_bytes number of bytes.
     * @param mem pointer to memory which is released.
     **/
    void release_mmap( std::size_t   num_bytes,
                       void        * mem ) const;
  
    /**
     * Sets the given memory region executable.
     *
     * @param num_bytes number of bytes.
     * @param mem point to memory.
     **/
    void set_exec( std::size_t   num_bytes,
                   void        * mem ) const;

    /**
     * Release memory of the kernel if allocated.
     **/
    void release_memory();

  public:
    /**
     * Constructor
     **/
    Kernel(){};

    /**
     * Destructor
     **/
    ~Kernel() noexcept;

    Kernel( Kernel const & ) = delete;
    Kernel & operator=( Kernel const & ) = delete;
    Kernel( Kernel && ) noexcept = delete;
    Kernel & operator=( Kernel && ) noexcept = delete;

    /**
     * Adds an instruction to the code buffer.
     *
     * @param ins instruction which is added.
     **/
    void add_instr( uint32_t ins );

    void add_branch( uint32_t ins, std::string label, uint32_t offs_bits, uint32_t offs_shift );

    void add_label( std::string label );

    /**
     * Gets the size of the code buffer.
     *
     * @return size of the code buffer in bytes.
     **/
    std::size_t get_size() const;

    /**
     * Sets the kernel based on the code buffer.
     **/
    void set_kernel();

    /**
     * Gets a pointer to the executable kernel.
     **/
    void const * get_kernel() const;

    /**
     * Writes the code buffer to the given file.
     *
     * @param path path to the file.
     **/
    void write( char const * path ) const;
};

#endif
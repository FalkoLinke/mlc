#include "Kernel.h"
#include <sys/mman.h>
#include <fstream>
#include <stdexcept>
#include <cerrno>
#include <cstring>
#include <string>

#include "InsRef.h"













mini_jit::Kernel::~Kernel() noexcept {
  release_memory();
}

void mini_jit::Kernel::add_instr( uint32_t ins ) {
  m_buffer.push_back( ins );
}

void mini_jit::Kernel::add_labeled_instr( LabeledInstruction labeled_ins ) {
  uint32_t ins = labeled_ins.ins;

  uint32_t offs_mask = (0x1 << labeled_ins.offs_bits) - 1;
  ins &= ~(offs_mask << labeled_ins.offs_shift);
  m_buffer.push_back( ins );

  InsRef insRef(m_buffer.size()-1, labeled_ins);

  if (label_indices.find(labeled_ins.label) != label_indices.end()) {
    resolve_instruction(insRef);
  } else {
    unresolved_instructions[labeled_ins.label].push_back( insRef );
  }
}

void mini_jit::Kernel::add_label( std::string label ) {
  label_indices[label] = m_buffer.size();

  std::vector<InsRef> branches = unresolved_instructions[label];
  for (auto it = branches.begin(); it != branches.end(); it++) {
    InsRef branch = *it;
    resolve_instruction(branch);
  }
  unresolved_instructions[label] = std::vector<InsRef>();
}

void mini_jit::Kernel::resolve_instruction( InsRef insRef ) {
  uint32_t ins_idx = insRef.idx;
  uint32_t label_idx = label_indices[insRef.labeled_ins.label];

  int32_t offs = label_idx - ins_idx;
  offs += insRef.labeled_ins.bias;
  uint32_t offs_mask = (0x1 << insRef.labeled_ins.offs_bits) - 1;

  m_buffer[ins_idx] |= (offs & offs_mask) << insRef.labeled_ins.offs_shift;
}

std::size_t mini_jit::Kernel::get_size() const {
  return m_buffer.size() * 4;
}

void * mini_jit::Kernel::alloc_mmap( std::size_t num_bytes ) const {
  void* l_mem = mmap( 0,
                      num_bytes,
                      PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS,
                      -1,
                      0 );

  if( l_mem == MAP_FAILED ) {
    throw std::runtime_error( "Failed to allocate memory: "
                              + std::string( std::strerror(errno) ) );
  }

  return l_mem;
}

void mini_jit::Kernel::release_mmap( std::size_t   num_bytes,
                                     void        * mem ) const {
  int l_res = munmap( mem,
                      num_bytes );

  if( l_res == -1 ) {
    throw std::runtime_error( "Failed to release memory" );
  }
}

void mini_jit::Kernel::set_exec( std::size_t   num_bytes,
                                 void        * mem ) const {
  int l_res = mprotect( mem,
                        num_bytes,
                        PROT_READ | PROT_EXEC );

  if( l_res == -1 ) {
    throw std::runtime_error( "Failed to set memory executable: "
                              + std::string( std::strerror(errno) ) );
  }  
}

void mini_jit::Kernel::set_kernel() {
  release_memory();

  if( m_buffer.empty() ) {
    return;
  }

  // alloc kernel memory
  m_size_alloc = m_buffer.size() * 4;
  try {
    m_kernel = (void *) alloc_mmap( m_size_alloc );
  }
  catch( std::runtime_error & e ) {
    throw std::runtime_error( "Failed to allocate memory for kernel: "
                              + std::string(e.what()) );
  }

  // copy instruction words from buffer to kernel memory
  for( std::size_t l_in = 0; l_in < m_buffer.size(); l_in++ ) {
    reinterpret_cast< uint32_t * >(m_kernel)[l_in] = m_buffer[l_in];
  }

  // clear cache
  char * l_kernel_ptr = reinterpret_cast< char * >(m_kernel);
  __builtin___clear_cache( l_kernel_ptr,
                           l_kernel_ptr + m_buffer.size() * 4 );

  // set executable
  set_exec( m_size_alloc,
            m_kernel );
}

void const * mini_jit::Kernel::get_kernel() const {
  return m_kernel;
}

void mini_jit::Kernel::release_memory() {
  if( m_kernel != nullptr ) {
    release_mmap( m_size_alloc,
                  m_kernel );
  }
  m_size_alloc = 0;

  m_kernel = nullptr;
}

void mini_jit::Kernel::write( char const * path ) const {
  std::ofstream l_out ( path,
                        std::ios::out | std::ios::binary );
  if( !l_out ) {
    throw std::runtime_error( "Failed to open file: "
                              + std::string(path) );
  }

  l_out.write( reinterpret_cast< char const * >(m_buffer.data()),
               m_buffer.size()*4 );
}

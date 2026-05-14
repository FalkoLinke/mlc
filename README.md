# mlc

Published docs: https://falkolinke.github.io/mlc/



## Build source code

### code_gen (Week 5)

To build the source code of the `code_gen` subdirectory execute the following commands:
```
cd code_gen/src
mkdir build
cd build
cmake ..
cmake --build .
```
In order to use the `clang` compiler found in `homebrew` replace the last two commands as follows:
```
/opt/homebrew/opt/cmake/bin/cmake -D CMAKE_CXX_COMPILER="/opt/homebrew/opt/llvm/bin/clang++" ..
/opt/homebrew/opt/cmake/bin/cmake --build .
```

The following executables are provided:
- `build/unary_tests.out`: Executes the unit tests for the generated unary primitives.
- `build/unary_benchmarks_driver.out`: Executes the benchmarks for the generated unary primitives.
- `build/main.out`: Debugging driver.
- `build/kernel_examples.out`: Executes the kernel examples.
- `build/instgen_examples.out`: Executes the instruction generation examples.


### sme (Week 3/4)

To build the source code of the `sme` subdirectory, execute the following commands:
```
cd sme/src
make TOOLCHAIN=homebrew
```
Using `TOOLCHAIN=homebrew` selects the `clang` compiler found in `homebrew`.
Alternativly `TOOLCHAIN=standard` may be used to select the first `clang` compiler found on the `PATH`.


The following executables are provided:
- `unary_tests`: Executes the unit tests for the unary primitives.
- `unary_benchmarks_driver`: Executes the benchmarks for the unary primitives.
- `identity_driver`: Debugging driver for the identity operation.
- `zero_driver`: Debugging driver for the zero operation.
- `gemm_driver_32_32_1`: Exectues the gemm_32_32_1_kernel.s Kernel, verifies the results and benchmarks it.
- `gemm_driver_32_32_512`: Exectues the gemm_driver_32_32_512.s Kernel, verifies the results and benchmarks it.
- `gemm_driver_512_32_512`: Exectues the gemm_driver_512_32_512.s Kernel, verifies the results and benchmarks it.
- `gemm_driver_512_512_512`: Exectues the gemm_driver_512_512_512.s Kernel, verifies the results and benchmarks it.




### neon (Week 2)

To build the source code of the `neon` subdirectory, execute the following commands:
```
cd neon/src
make
```

The following executables are provided:
- `benchmark_driver`: Executes the microbenchmarks.
- `permutation_driver`: Benchmarks the implementation of `permutation`.


### assembly (Week 1)

To build the source code of the `assembly` subdirectory, execute the following commands:
```
cd assembly/src
make
```

The following executables are provided:
- `base_math_driver`: Executes the unit tests for the implemented functions.









## Build docs locally

1. Sync dependencies (includes docs group by default):
   `uv sync`
2. Activate the virtual environment
   `source .venv/bin/activate`
2. Build Sphinx HTML:
   `uv run sphinx-build -b html docs/src docs/_build/html`.
   Alternatively the following command may also be used:
   `cd docs && make html`
3. Open the result:
   `docs/_build/html/index.html`


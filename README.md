# mlc

Project of the lecture Machine Learning Compilers by Falko Linke and Johann Schwarze.
The project starts with AArch64 assembly and Neon kernels, continues with SME kernels and a JIT code generator
for unary and GEMM primitives, and ends with the tensor runtime TEIR (interpreter, compiler, optimization passes).

Published report: https://falkolinke.github.io/mlc/


## Requirements

- macOS on arm64. The SME kernels and the JIT generated code need a processor supporting SME and SME2, e.g. the Apple M4.
  On other arm64 processors everything builds, but only the portable tests run (see [Tests](#tests)).
- CMake 3.10 or newer and a C++17 compiler. Catch2 v3.8.1 is downloaded by CMake.
- OpenMP for the parallel TEIR runtime. We use the Homebrew packages:
  ```
  brew install cmake llvm libomp
  ```


## Build

The top-level CMake project builds all subdirectories.
Build commands using Homebrew clang and OpenMP (also used by the CI):
```
mkdir build
cd build
/opt/homebrew/opt/cmake/bin/cmake -D CMAKE_CXX_COMPILER="/opt/homebrew/opt/llvm/bin/clang++" -D CMAKE_PREFIX_PATH="/opt/homebrew/opt/libomp;$CMAKE_PREFIX_PATH" ..
/opt/homebrew/opt/cmake/bin/cmake --build .
```

Basic build commands with the default compiler:
```
mkdir build
cd build
cmake ..
cmake --build .
```
Without OpenMP, all TEIR iteration nodes run sequentially and `sme/gemm_driver_16_16_multi` is not built.

The executables are located in the subdirectories of `build` which match the source directories, e.g. `build/teir`.


## Tests

All unit tests use Catch2 and are registered with CTest. Run them from the `build` directory:
```
/opt/homebrew/opt/cmake/bin/ctest --output-on-failure
```
SME and JIT tests are only registered on processors supporting SME (e.g. Apple M4).
Use `-D MLC_SME_TESTS=ON|OFF` when configuring to override the detection.


## Continuous Integration

The GitHub Actions workflow `.github/workflows/tests.yaml` builds the whole project on every push
and pull request using the Homebrew build commands above and runs all tests with `ctest`.
The hosted `macos-latest` runners are arm64 machines without SME support.
CMake detects whether the host supports SME (`sysctl hw.optional.arm.FEAT_SME` on macOS,
`/proc/cpuinfo` on Linux). On hosts without SME, the SME and JIT tests are still built,
but not registered with CTest, so the pipeline tests the portable parts
and every test runs on SME capable machines such as the Apple M4.

The documentation workflow `.github/workflows/sphinx.yaml` builds the report with
`sphinx-build -W --keep-going`, so every Sphinx warning fails the pipeline, and publishes it on GitHub Pages.


## Repository layout

| Directory | Content |
|---|---|
| `assembly` | Week 1: AArch64 assembly |
| `neon` | Week 2: Neon microbenchmarks and permutation kernels |
| `sme` | Weeks 3/4: SME unary and GEMM kernels |
| `code_gen` | Weeks 5/6: JIT code generator for unary and GEMM primitives |
| `teir` | Weeks 7/8: TEIR interpreter, compiler and optimization passes |
| `benchmarks` | Group specific component: GEMM kernels with multiple K-tiles and transposed inputs |
| `common` | Helpers shared by the tests |
| `docs` | Sources of the project report |
| `scripts` | Shell scripts to disassemble and format generated machine code |
| `application` | Applications for the final report (not part of the build yet) |

`mlc.pdf` contains the slides of our presentation. Some subdirectories have their own `README` with more details.


## Executables

### assembly (Week 1)

Provided in `build/assembly`:
- `base_math_driver`: Executes the unit tests for the implemented functions.

### neon (Week 2)

Provided in `build/neon`:
- `benchmark_driver`: Executes the microbenchmarks.
- `permutation_driver`: Verifies and benchmarks the `permutation` kernels for several sizes of `c`.
- `permutation_tests`: Executes the unit tests for the `permutation` kernels.

### sme (Weeks 3/4)

Provided in `build/sme`:
- `unary_tests`: Executes the unit tests for the unary primitives.
- `unary_benchmarks_driver`: Executes the benchmarks for the unary primitives.
- `identity_driver`, `relu_driver`, `zero_driver`: Debugging drivers for the unary operations.
- `gemm_tests`: Executes the unit tests for the GEMM kernels.
- `gemm_driver_16_16`: Verifies and benchmarks the kernel `gemm_16_16_kernel.s` (M = N = 16, K = 512).
- `gemm_driver_16_16_multi`: Computes a 128×128×512 GEMM with `gemm_16_16_kernel.s` on multiple threads (only built with OpenMP).
- `gemm_driver_32_32_1`: Verifies and benchmarks the kernel `gemm_32_32_1_kernel.s`.
- `gemm_driver_32_32_512`: Verifies and benchmarks the kernel `gemm_32_32_512_kernel.s`.
- `gemm_driver_512_32_512`: Verifies and benchmarks the kernel `gemm_512_32_512_kernel.s`.
- `gemm_driver_512_512_512`: Verifies and benchmarks the kernel `gemm_512_512_512_kernel.s`.
- `gemm_driver_M_N_K_16th`: Verifies and benchmarks the kernel `gemm_M_N_K_16th_kernel.s` for M, N, K which are multiples of 16.

### code_gen (Weeks 5/6)

Provided in `build/code_gen`:
- `unary_tests.out`: Executes the unit tests for the generated unary primitives.
- `unary_benchmarks_driver.out`: Executes the benchmarks for the 16×16 unary kernels of the kernel factory (week 5).
- `unary_jit_benchmarks_driver.out`: Executes the benchmarks for the unary primitives generated by the JIT generator.
- `gemm_tests.out`: Executes the unit tests for the generated GEMM kernels.
- `gemm_benchmarks_driver.out [all|task|layouts]`: Benchmarks the generated GEMM kernels for the settings of the task
  and for all storage formats.
- `gemm_driver_M_N_K.out`: Verifies and benchmarks generated GEMM kernels for several sizes.
- `gemm_driver_512_512_512.out`: Verifies and benchmarks the generated 512×512×512 GEMM kernel.
- `main.out`: Debugging driver.
- `kernel_examples.out`: Executes the kernel examples.
- `instgen_examples.out`: Executes the instruction generation examples.

### teir (Weeks 7/8)

Parallel iteration nodes use OpenMP, so use the Homebrew build commands;
without OpenMP all iteration nodes run sequentially.

Provided in `build/teir`:
- `teir_interpreter_tests.out`, `teir_compiler_tests.out`, `teir_runtime_tests.out`: Execute the unit tests of the TEIR interpreter and compiler.
- `teir_optimization_tests.out`: Executes the unit tests of the TEIR transformations and optimization passes (week 8).
- `teir_benchmarks.out [matmul|contraction|transposition|all] [repetitions]`: Verifies and benchmarks the examples of `teir/data`
  with the interpreter and the compiler, with sequential and parallel policies. Set the number of threads with `OMP_NUM_THREADS`.
- `teir_ablation.out [matmul|contraction|einsum|all] [repetitions] [print]`: Ablation study of the optimization passes (week 8).
  `print` writes the pass log and the optimized operations to stderr.

### benchmarks (Group specific component)

Provided in `build/benchmarks/gemm`, `build/benchmarks/transpose` and `build/benchmarks/transpose_kLoop`:
- `gemm_benchmarks_driver_16_16`: Verifies and benchmarks the GEMM kernels with multiple K-tiles and with transposed A.
- `gemm_benchmarks_driver`: Benchmarks the GEMM kernels with M = 16 and different transposition strategies.
- `transpose_kloop_benchmarks_driver`: Benchmarks the transposition of a 16×16 block (ZA array, TBL, SME2),
  each kernel repeats the transposition in a loop.
- `transpose_benchmarks_driver`: Benchmarks the same transpositions with one kernel call per transposition.
- `gemm_main`, `transpose_main`, `transpose_kloop_main`: Debugging drivers.

Each benchmark directory is also a standalone CMake project, see the report for the commands and `benchmarks/README.md` for older results.


## Build docs locally

The report is built with Sphinx, the dependencies are managed with [uv](https://docs.astral.sh/uv/) in `docs/pyproject.toml`:
```
cd docs
uv sync
uv run sphinx-build -W --keep-going -b html src _build/html
```
Alternatively, activate the virtual environment with `source .venv/bin/activate` and run `make html`.
Open `docs/_build/html/index.html` to view the result.

The figures of the report are committed. The scripts in `docs/scripts` regenerate them from the data in `docs/src/weeks/data`
and need matplotlib, e.g.:
```
cd docs
uv run --with matplotlib python scripts/plot_week08.py
```

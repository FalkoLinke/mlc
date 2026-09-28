/**
 * Benchmarks the TEIR interpreter and compiler on the example operations of `teir/data`.
 *
 * Every example is executed with sequential policies (all iteration nodes sequential) and with the
 * parallel policies of the `.teir` file, once interpreted and once compiled.
 * The output of every configuration is compared to the output of the sequential interpreter.
 *
 * Usage: teir_benchmarks.out [matmul|contraction|transposition|all] [repetitions] [compiler-parallel]
 * The number of OpenMP threads is set with OMP_NUM_THREADS.
 * The optional third argument only runs the compiled parallel configuration without verification,
 * which is used to measure the scaling with the number of threads.
 *
 * Output: one tab-separated line per configuration.
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "mlc_common.hpp"
#include "teir_compiler.h"
#include "teir_examples.hpp"
#include "teir_interpreter.h"


struct example_t {
    std::string name;
    std::function<teir_operation(bool parallel)> build;
    std::vector<uint64_t> tensor_floats;
    /** Work per execution: FLOPs for contractions, bytes read and written for transpositions. */
    double work;
    char const* unit;
};


static int max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}


static double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}


static void benchmark(example_t const& example, int repetitions, bool only_compiler_parallel) {
    // the last tensor is the output, all others are inputs
    std::vector<std::vector<float>> tensors;
    for (uint64_t i = 0; i < example.tensor_floats.size(); i++) {
        tensors.emplace_back(example.tensor_floats[i]);
        teir_example_fill_random(tensors.back(), 1 + i);
    }
    std::vector<float>& out = tensors.back();
    std::vector<float> const out_init = out;

    std::vector<void*> args;
    for (std::vector<float>& tensor : tensors) {
        args.push_back(tensor.data());
    }

    std::vector<float> expected;

    for (bool parallel : {false, true}) {
        teir_operation operation = example.build(parallel);

        for (bool compiled : {false, true}) {
            if (only_compiler_parallel && !(parallel && compiled)) {
                continue;
            }
            teir_compiler compiler;
            teir_interpreter interpreter(operation, args);
            std::function<void()> run;
            if (compiled) {
                compiler.compile(operation);
                teir_compiler::teir_function_t function = compiler.get_function();
                run = [&args, function]() {
                    std::vector<void*> ptrs = args;
                    function(ptrs.data());
                };
            } else {
                run = [&interpreter]() { interpreter.run(); };
            }

            // verification run, which also serves as warm-up
            std::copy(out_init.begin(), out_init.end(), out.begin());
            run();
            if (expected.empty()) {
                // the first configuration is the sequential interpreter
                expected = out;
            }
            double error = only_compiler_parallel ? NAN : max_rel_diff(out.data(), expected.data(), out.size(), 1, out.size(), out.size());

            std::vector<double> times;
            for (int rep = 0; rep < repetitions; rep++) {
                auto start = std::chrono::steady_clock::now();
                run();
                times.push_back(seconds_since(start));
            }
            std::sort(times.begin(), times.end());
            double best = times.front();
            double median = times[times.size() / 2];

            std::printf("%s\t%s\t%s\t%d\t%.4f\t%.4f\t%.2f\t%.2f\t%s\t%.2e\n",
                example.name.c_str(),
                compiled ? "compiler" : "interpreter",
                parallel ? "parallel" : "sequential",
                parallel ? max_threads() : 1,
                best,
                median,
                example.work / best * 1e-9,
                example.work / median * 1e-9,
                example.unit,
                error);
            std::fflush(stdout);
        }
    }
}


int main(int argc, char* argv[]) {
    std::string selection = argc > 1 ? argv[1] : "all";
    int repetitions = argc > 2 ? std::max(1, std::stoi(argv[2])) : 5;
    bool only_compiler_parallel = argc > 3 && std::string(argv[3]) == "compiler-parallel";

    std::vector<example_t> examples;

    {
        uint64_t m0 = 256, m1 = 32, n0 = 128, n1 = 64, k0 = 16, k1 = 512;
        examples.push_back({
            "matmul",
            [=](bool parallel) { return teir_example_matmul(m0, m1, n0, n1, k0, k1, parallel); },
            {m0 * k0 * m1 * k1, k0 * n0 * k1 * n1, m0 * n0 * m1 * n1},
            2.0 * m0 * m1 * n0 * n1 * k0 * k1,
            "GFLOPS",
        });
    }
    {
        uint64_t p = 128, q = 96, r = 96, s = 64, t = 32, u = 256;
        examples.push_back({
            "contraction",
            [=](bool parallel) { return teir_example_contraction(p, q, r, s, t, u, parallel); },
            {p * q * t * u, t * r * u * s, p * q * r * s},
            2.0 * p * q * r * s * t * u,
            "GFLOPS",
        });
    }
    {
        uint64_t a = 96, b = 128, c = 48, d = 32;
        examples.push_back({
            "transposition",
            [=](bool parallel) { return teir_example_transposition(a, b, c, d, parallel); },
            {a * b * c * d, a * b * c * d},
            2.0 * a * b * c * d * sizeof(float),
            "GB/s",
        });
    }

    std::printf("example\truntime\tpolicy\tthreads\tbest_s\tmedian_s\tbest_perf\tmedian_perf\tunit\tmax_rel_diff\n");
    for (example_t const& example : examples) {
        if (selection == "all" || selection == example.name) {
            benchmark(example, repetitions, only_compiler_parallel);
        }
    }

    return 0;
}

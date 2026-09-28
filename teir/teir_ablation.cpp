/**
 * Ablation study of the TEIR optimization passes.
 *
 * For each operation (matmul and contraction of `teir/data`, and the einsum acspx,bspy->abcyx of week 8),
 * the unoptimized baseline, every single pass, all passes, and all passes but one are compiled and executed.
 * The output of every configuration is compared to the output of the baseline.
 *
 * Additional rows: `file` is matmul.teir with its original zero invocation (different result, see the report),
 * `sequential` are the examples with all policies set to sequential and `sequential + all` their optimized versions.
 *
 * Usage: teir_ablation.out [matmul|contraction|einsum|all] [repetitions] [print]
 * `print` writes the pass log and the optimized operation of every configuration to stderr.
 *
 * Output: one tab-separated line per configuration.
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "mlc_common.hpp"
#include "teir_compiler.h"
#include "teir_examples.hpp"
#include "teir_passes.h"
#include "teir_transform.h"


struct example_t {
    std::string name;
    teir_operation baseline;
    std::vector<uint64_t> tensor_floats;
    double flops;
    /** Additional configurations measured before the ablation, e.g. the unmodified example file. */
    std::vector<std::pair<std::string, teir_operation>> extra;
};

struct config_t {
    std::string name;
    uint32_t passes;
};


static std::vector<config_t> ablation_configs() {
    std::vector<teir_pass_t> const passes = {teir_opt_fusion, teir_opt_operands, teir_opt_blocking, teir_opt_parallel};
    std::vector<config_t> configs = {{"baseline", teir_opt_none}};
    for (teir_pass_t pass : passes) {
        configs.push_back({"only " + teir_pass_name(pass), pass});
    }
    configs.push_back({"all", teir_opt_all});
    for (teir_pass_t pass : passes) {
        configs.push_back({"all but " + teir_pass_name(pass), teir_opt_all & ~(uint32_t)pass});
    }
    return configs;
}


static double seconds_since(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}


static void run(example_t const& example, teir_target const& target, int repetitions, bool print) {
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
    double baseline_time = 0.0;
    std::string const baseline_text = teir_to_string(example.baseline);

    std::vector<std::pair<std::string, teir_operation>> operations = example.extra;
    std::vector<std::string> names;
    std::vector<uint32_t> pass_flags;
    for (auto const& extra : example.extra) {
        names.push_back(extra.first);
        pass_flags.push_back(0);
    }
    std::vector<std::vector<std::string>> logs(example.extra.size());
    for (config_t const& config : ablation_configs()) {
        std::vector<std::string> log;
        operations.emplace_back(config.name, teir_optimize(example.baseline, target, config.passes, &log));
        names.push_back(config.name);
        pass_flags.push_back(config.passes);
        logs.push_back(log);
    }

    // the baseline is executed first to obtain the reference output
    std::vector<uint64_t> order;
    for (uint64_t i = example.extra.size(); i < operations.size(); i++) {
        order.push_back(i);
    }
    for (uint64_t i = 0; i < example.extra.size(); i++) {
        order.push_back(i);
    }

    for (uint64_t i : order) {
        teir_operation const& operation = operations[i].second;
        std::string text = teir_to_string(operation);
        if (print) {
            std::fprintf(stderr, "==== %s: %s\n", example.name.c_str(), names[i].c_str());
            for (std::string const& line : logs[i]) {
                std::fprintf(stderr, "  %s\n", line.c_str());
            }
            std::fprintf(stderr, "%s\n", text.c_str());
        }

        teir_compiler compiler;
        compiler.compile(operation);
        teir_compiler::teir_function_t function = compiler.get_function();
        auto execute = [&]() {
            std::vector<void*> ptrs = args;
            function(ptrs.data());
        };

        std::copy(out_init.begin(), out_init.end(), out.begin());
        execute();
        if (expected.empty()) {
            expected = out;
        }
        double error = max_rel_diff(out.data(), expected.data(), out.size(), 1, out.size(), out.size());

        std::vector<double> times;
        for (int rep = 0; rep < repetitions; rep++) {
            auto start = std::chrono::steady_clock::now();
            execute();
            times.push_back(seconds_since(start));
        }
        std::sort(times.begin(), times.end());
        double best = times.front();
        double median = times[times.size() / 2];
        if (baseline_time == 0.0) {
            baseline_time = best;
        }

        std::printf("%s\t%s\t%u\t%s\t%.4f\t%.4f\t%.1f\t%.1f\t%.2f\t%.2e\n",
            example.name.c_str(),
            names[i].c_str(),
            pass_flags[i],
            text == baseline_text ? "unchanged" : "changed",
            best,
            median,
            example.flops / best * 1e-9,
            example.flops / median * 1e-9,
            baseline_time / best,
            error);
        std::fflush(stdout);
    }
}


int main(int argc, char* argv[]) {
    std::string selection = argc > 1 ? argv[1] : "all";
    int repetitions = argc > 2 ? std::max(1, std::stoi(argv[2])) : 5;
    bool print = argc > 3 && std::string(argv[3]) == "print";

    teir_target target = teir_target::host();
    std::fprintf(stderr, "target: L1d %llu KiB, L2 %llu MiB shared by %llu cores, %llu threads, SVL %llu bytes\n",
        (unsigned long long)(target.l1d_bytes / 1024), (unsigned long long)(target.l2_bytes / 1024 / 1024),
        (unsigned long long)target.cores_per_l2, (unsigned long long)target.threads, (unsigned long long)target.svl_bytes);

    std::vector<example_t> examples;
    {
        uint64_t m0 = 256, m1 = 32, n0 = 128, n1 = 64, k0 = 16, k1 = 512;
        examples.push_back({
            "matmul",
            teir_example_matmul(m0, m1, n0, n1, k0, k1, true, true),
            {m0 * k0 * m1 * k1, k0 * n0 * k1 * n1, m0 * n0 * m1 * n1},
            2.0 * m0 * m1 * n0 * n1 * k0 * k1,
            {
                {"file", teir_example_matmul(m0, m1, n0, n1, k0, k1, true, false)},
                {"sequential", teir_example_matmul(m0, m1, n0, n1, k0, k1, false, true)},
                {"sequential + all", teir_optimize(teir_example_matmul(m0, m1, n0, n1, k0, k1, false, true), target, teir_opt_all)},
            },
        });
    }
    {
        uint64_t p = 128, q = 96, r = 96, s = 64, t = 32, u = 256;
        examples.push_back({
            "contraction",
            teir_example_contraction(p, q, r, s, t, u, true),
            {p * q * t * u, t * r * u * s, p * q * r * s},
            2.0 * p * q * r * s * t * u,
            {
                {"sequential", teir_example_contraction(p, q, r, s, t, u, false)},
                {"sequential + all", teir_optimize(teir_example_contraction(p, q, r, s, t, u, false), target, teir_opt_all)},
            },
        });
    }
    {
        uint64_t a = 4, b = 4, c = 3, s = 64, p = 64, x = 1536, y = 1152;
        examples.push_back({
            "einsum",
            teir_example_einsum(a, b, c, s, p, x, y),
            {a * c * s * p * x, b * s * p * y, a * b * c * y * x},
            2.0 * a * b * c * s * p * x * y,
            {},
        });
    }

    std::printf("example\tconfig\tpasses\tschedule\tbest_s\tmedian_s\tbest_gflops\tmedian_gflops\tspeedup\tmax_rel_diff\n");
    for (example_t const& example : examples) {
        if (selection == "all" || selection == example.name) {
            run(example, target, repetitions, print);
        }
    }
    return 0;
}

"""
Generates the figures of the week 7 report from the benchmark results.

Input (tab separated, written by teir/teir_benchmarks.out):
    docs/src/weeks/data/week07_runtime.tsv   teir_benchmarks.out all 5
    docs/src/weeks/data/week07_scaling.tsv   teir_benchmarks.out <example> <reps> compiler-parallel
                                             with OMP_NUM_THREADS = 1, 2, 4, 6, 8, 10

Output:
    docs/src/weeks/week07_runtime.svg
    docs/src/weeks/week07_scaling.svg

The figures are committed, so building the report does not require matplotlib.
Usage: python docs/scripts/plot_week07.py
"""

from plot_week06 import (  # noqa: F401  (also applies the shared chart style)
    CATEGORICAL,
    DATA,
    INK_SECONDARY,
    BASELINE,
    WEEKS,
    grouped_bars,
    plt,
    read_rows,
    spread_labels,
)

CONFIGS = [
    ("interpreter", "sequential", "Interpreter, sequential"),
    ("compiler", "sequential", "Compiler, sequential"),
    ("interpreter", "parallel", "Interpreter, parallel"),
    ("compiler", "parallel", "Compiler, parallel"),
]


def label_bars(ax, fmt):
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, padding=2, fontsize=8, color=INK_SECONDARY)


def plot_runtime():
    rows = read_rows(DATA / "week07_runtime.tsv")[1:]
    perf = {(r[0], r[1], r[2]): float(r[6]) for r in rows}

    fig, (ax_flops, ax_bw) = plt.subplots(
        1, 2, figsize=(8, 3.6), gridspec_kw={"width_ratios": [2, 1]})

    examples = ["matmul", "contraction"]
    series = [[perf[(e, runtime, policy)] for e in examples] for runtime, policy, _ in CONFIGS]
    grouped_bars(ax_flops, examples, series, CATEGORICAL, [c[2] for c in CONFIGS])
    label_bars(ax_flops, "%.0f")
    ax_flops.set_ylabel("GFLOPS")
    ax_flops.set_ylim(0, 1000)
    ax_flops.set_title("Contractions", loc="left")

    series = [[perf[("transposition", runtime, policy)]] for runtime, policy, _ in CONFIGS]
    grouped_bars(ax_bw, ["transposition"], series, CATEGORICAL, [c[2] for c in CONFIGS])
    label_bars(ax_bw, "%.0f")
    ax_bw.set_ylabel("GB/s")
    ax_bw.set_ylim(0, 70)
    ax_bw.set_title("Transposition", loc="left")

    handles, labels = ax_flops.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, handlelength=1, handleheight=1,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(WEEKS / "week07_runtime.svg")
    plt.close(fig)


def plot_scaling():
    rows = read_rows(DATA / "week07_scaling.tsv")[1:]
    examples = ["matmul", "contraction", "transposition"]
    threads = sorted({int(r[3]) for r in rows})

    fig, ax = plt.subplots(figsize=(8, 3.6))
    speedups = {}
    for example, color in zip(examples, CATEGORICAL):
        perf = {int(r[3]): float(r[6]) for r in rows if r[0] == example}
        speedups[example] = [perf[t] / perf[1] for t in threads]
        ax.plot(threads, speedups[example], color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=example)

    # direct labels at the end of the lines
    ends = [speedups[e][-1] for e in examples]
    for example, y in zip(examples, spread_labels(ends, 0.12)):
        ax.annotate(example, (threads[-1], y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK_SECONDARY)

    ax.axvline(4, color=BASELINE, linewidth=1, linestyle=(0, (3, 3)))
    ax.annotate("4 P-cores", (4, 0.2), xytext=(4, 0), textcoords="offset points",
                fontsize=8, color=INK_SECONDARY)

    ax.set_xticks(threads)
    ax.set_xlim(0.5, threads[-1] + 1.8)
    ax.set_ylim(0, max(max(s) for s in speedups.values()) * 1.15)
    ax.set_xlabel("OpenMP threads")
    ax.set_ylabel("Speed-up over 1 thread")
    ax.set_title("Compiled parallel TEIR operations", loc="left")
    ax.legend(loc="upper left", ncol=3, handlelength=1.5)
    fig.tight_layout()
    fig.savefig(WEEKS / "week07_scaling.svg")
    plt.close(fig)


if __name__ == "__main__":
    plot_runtime()
    plot_scaling()

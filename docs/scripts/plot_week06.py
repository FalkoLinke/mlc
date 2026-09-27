"""
Generates the figures of the week 6 report from the benchmark results.

Input (tab separated, written by the benchmark drivers):
    docs/src/weeks/data/week06_unary.tsv          unary_jit_benchmarks_driver.out
    docs/src/weeks/data/week06_gemm_task.tsv      gemm_benchmarks_driver.out task
    docs/src/weeks/data/week06_gemm_layouts.tsv   gemm_benchmarks_driver.out layouts
    docs/src/weeks/data/week06_tiling_88_72.tsv   tile_matrix_v1 of code_gen/Gemm.cpp for m = 88, n = 72

Output:
    docs/src/weeks/week06_unary.svg
    docs/src/weeks/week06_gemm_task.svg
    docs/src/weeks/week06_gemm_layouts.svg
    docs/src/weeks/week06_tiling.svg

The figures are committed, so building the report does not require matplotlib.
Usage: python docs/scripts/plot_week06.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("svg")
import matplotlib.pyplot as plt  # noqa: E402

WEEKS = Path(__file__).resolve().parent.parent / "src" / "weeks"
DATA = WEEKS / "data"

# chart chrome (light surface) and validated series colors
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
ORDINAL = ["#86b6ef", "#2a78d6", "#104281"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "ytick.left": False,
    "legend.frameon": False,
    "legend.labelcolor": INK_SECONDARY,
    "svg.fonttype": "path",
})


def read_rows(path):
    """Reads a tab separated benchmark result, skipping empty columns."""
    with open(path) as f:
        rows = [[c for c in line.rstrip("\n").split("\t") if c != ""] for line in f]
    return [r for r in rows if r]


def grouped_bars(ax, groups, series, colors, labels, width=0.8):
    """Draws `len(series)` bars per group, separated by a thin surface gap."""
    n = len(series)
    bar = width / n
    for i, (values, color, label) in enumerate(zip(series, colors, labels)):
        xs = [g + (i - (n - 1) / 2) * bar for g in range(len(groups))]
        ax.bar(xs, values, bar, color=color, label=label, edgecolor=SURFACE, linewidth=1.5)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)
    ax.tick_params(axis="x", length=0)


def plot_gemm_task():
    rows = read_rows(DATA / "week06_gemm_task.tsv")[1:]
    perf = {(int(r[0]), int(r[1]), int(r[2])): float(r[6]) for r in rows}
    sizes = [64, 128, 512]
    groups = [f"{m}×{n}" for m in sizes for n in sizes]
    series = [[perf[(m, n, k)] for m in sizes for n in sizes] for k in sizes]

    fig, ax = plt.subplots(figsize=(8, 3.6))
    grouped_bars(ax, groups, series, ORDINAL, [f"k = {k}" for k in sizes])
    ax.set_ylabel("GFLOPS")
    ax.set_xlabel("m × n")
    ax.set_ylim(0, 2200)
    ax.set_title("Generated GEMM kernels: column-major A and C, row-major B", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.02), ncol=3, handlelength=1, handleheight=1)
    fig.tight_layout()
    fig.savefig(WEEKS / "week06_gemm_task.svg")
    plt.close(fig)


def spread_labels(ys, min_gap):
    """Moves label positions apart vertically so that neighbors keep `min_gap`."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    placed = list(ys)
    for prev, cur in zip(order, order[1:]):
        placed[cur] = max(placed[cur], placed[prev] + min_gap)
    return placed


def plot_gemm_layouts():
    rows = read_rows(DATA / "week06_gemm_layouts.tsv")[1:]
    perf = {}
    for r in rows:
        size, ta, tb, tc, gflops = int(r[0]), int(r[3]), int(r[4]), int(r[5]), float(r[6])
        perf[(ta, tb, tc, size)] = gflops
    sizes = sorted({k[3] for k in perf})

    # series: storage format of A and B, the panels separate the storage format of C
    ab = [(0, 1), (0, 0), (1, 1), (1, 0)]
    names = {
        (0, 1): "A col, B row",
        (0, 0): "A col, B col",
        (1, 1): "A row, B row",
        (1, 0): "A row, B col",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for tc, ax in zip((0, 1), axes):
        ends = []
        for (ta, tb), color in zip(ab, CATEGORICAL):
            ys = [perf[(ta, tb, tc, s)] for s in sizes]
            ax.plot(sizes, ys, color=color, linewidth=2, marker="o", markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=names[(ta, tb)])
            ends.append(ys[-1])
        # direct labels at the right end of every line
        for (ta, tb), y in zip(ab, spread_labels(ends, 110)):
            ax.annotate(names[(ta, tb)], xy=(sizes[-1], y), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK_SECONDARY)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(s) for s in sizes])
        ax.minorticks_off()
        ax.set_xlim(sizes[0] / 1.25, sizes[-1] * 3.2)
        ax.set_xlabel("m = n  (k = 512)")
        ax.set_title("C column-major" if tc == 0 else "C row-major", loc="left")
        ax.set_ylim(0, 2100)
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("GFLOPS")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=4, handlelength=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(WEEKS / "week06_gemm_layouts.svg")
    plt.close(fig)


def plot_tiling():
    from matplotlib.patches import Rectangle

    rows = read_rows(DATA / "week06_tiling_88_72.tsv")[1:]
    regions = [tuple(int(v) for v in r) for r in rows]
    m = max(r[1] for r in regions)
    n = max(r[3] for r in regions)

    def kind(mt, nt):
        if (mt, nt) == (32, 32):
            return 0, "32×32 microkernel (4 ZA tiles)"
        if (mt, nt) == (16, 16):
            return 1, "16×16 microkernel (1 ZA tile)"
        return 2, "predicated microkernel (partial tile)"

    fig, ax = plt.subplots(figsize=(5.6, 6.2))
    seen = set()
    gap = 0.35
    for mb, me, nb, ne, mt, nt in regions:
        idx, label = kind(mt, nt)
        for i in range(mb, me, mt):
            for j in range(nb, ne, nt):
                ax.add_patch(Rectangle((j + gap, i + gap), nt - 2 * gap, mt - 2 * gap,
                                       facecolor=CATEGORICAL[idx], edgecolor="none",
                                       label=None if idx in seen else label))
                seen.add(idx)
                if mt >= 16 and nt >= 16:
                    ax.text(j + nt / 2, i + mt / 2, f"{mt}×{nt}", ha="center", va="center",
                            color="#ffffff", fontsize=8)
    ax.set_xlim(0, n)
    ax.set_ylim(m, 0)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks(sorted({r[2] for r in regions} | {n}))
    ax.set_yticks(sorted({r[0] for r in regions} | {m}))
    ax.tick_params(length=0)
    ax.set_xlabel("n (columns of C)")
    ax.set_ylabel("m (rows of C)")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_title(f"Tiling of C for m = {m}, n = {n}", loc="left", pad=28)
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: [kind(32, 32)[1], kind(16, 16)[1]].index(labels[i]) if labels[i] in (kind(32, 32)[1], kind(16, 16)[1]) else 2)
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=1, handlelength=1, handleheight=1)
    fig.tight_layout()
    fig.savefig(WEEKS / "week06_tiling.svg")
    plt.close(fig)


def plot_unary():
    rows = read_rows(DATA / "week06_unary.tsv")[1:]
    perf = {(r[0], r[1], int(r[2])): float(r[5]) for r in rows}
    groups = [g for (op, g, t) in perf if op == "identity" and t == 0]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for op, ax in zip(("identity", "zero", "relu"), axes):
        series = [[perf[(op, g, t)] for g in groups] for t in (0, 1)]
        labels = [g.strip("()").replace(", ", "×") for g in groups]
        grouped_bars(ax, labels, series, CATEGORICAL[:2], ["B column-major", "B row-major (transposed)"])
        ax.set_title(op, loc="left")
        ax.set_xlabel("m × n")
        ax.tick_params(axis="x", labelrotation=45)
        ax.set_ylim(0, 260)
    axes[0].set_ylabel("GiB/s")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=2, handlelength=1, handleheight=1)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(WEEKS / "week06_unary.svg")
    plt.close(fig)


if __name__ == "__main__":
    plot_gemm_task()
    plot_gemm_layouts()
    plot_unary()
    plot_tiling()

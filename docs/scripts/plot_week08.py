"""
Generates the figure of the week 8 report from the ablation study.

Input (tab separated, written by teir/teir_ablation.out all 5):
    docs/src/weeks/data/week08_ablation.tsv

Output:
    docs/src/weeks/week08_ablation.svg

The figure is committed, so building the report does not require matplotlib.
Usage: python docs/scripts/plot_week08.py
"""

from plot_week06 import (  # noqa: F401  (also applies the shared chart style)
    BASELINE,
    CATEGORICAL,
    DATA,
    GRID,
    INK_MUTED,
    INK_SECONDARY,
    WEEKS,
    plt,
    read_rows,
)

EXAMPLES = ["einsum", "matmul", "contraction"]
CONFIGS = [
    "baseline",
    "only fusion",
    "only operands",
    "only blocking",
    "only parallel",
    "all",
    "all but fusion",
    "all but operands",
    "all but blocking",
    "all but parallel",
]


def plot_ablation():
    rows = read_rows(DATA / "week08_ablation.tsv")[1:]
    speedup = {(r[0], r[1]): float(r[8]) for r in rows}
    gflops = {(r[0], r[1]): float(r[6]) for r in rows}

    fig, axes = plt.subplots(1, 3, figsize=(8, 4.2), sharey=True)
    ys = list(range(len(CONFIGS)))[::-1]
    for ax, example in zip(axes, EXAMPLES):
        values = [speedup[(example, c)] for c in CONFIGS]
        colors = [INK_MUTED if c == "baseline" else CATEGORICAL[0] for c in CONFIGS]
        ax.barh(ys, values, height=0.7, color=colors, edgecolor="#fcfcfb", linewidth=1.5)
        for y, value in zip(ys, values):
            ax.annotate(f"{value:.2f}", (value, y), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK_SECONDARY)
        ax.axvline(1.0, color=BASELINE, linewidth=1, linestyle=(0, (3, 3)))
        ax.set_xlim(0, 2.2)
        ax.set_title(f"{example}\n{gflops[(example, 'baseline')]:.0f} GFLOPS baseline", loc="left", fontsize=10)
        ax.grid(axis="x", color=GRID)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel("Speed-up")
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(CONFIGS)
    fig.tight_layout()
    fig.savefig(WEEKS / "week08_ablation.svg")
    plt.close(fig)


if __name__ == "__main__":
    plot_ablation()

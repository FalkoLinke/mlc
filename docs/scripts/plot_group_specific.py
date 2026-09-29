"""
Generates the figures of the group specific component report.

The register contents of the SME2 transpose figures are computed by simulating
the zip/uzp/trn instruction sequence of benchmarks/gemm/gemm_16_16_trSME_mk.s
on a 16x16 fp32 matrix (SVL = 512 bit, element (r, c) holds the value 16 * r + c).

Output:
    docs/src/weeks/graphics/multi_k.svg
    docs/src/weeks/graphics/transpose_sme2_step1_zip.svg
    docs/src/weeks/graphics/transpose_sme2_step2_uzp.svg
    docs/src/weeks/graphics/transpose_sme2_step3_uzp1_uzp2.svg
    docs/src/weeks/graphics/transpose_sme2_step4_trn1_trn2.svg

The figures are committed, so building the report does not require matplotlib.
Usage: python docs/scripts/plot_group_specific.py
"""

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from plot_week06 import (  # noqa: F401  (also applies the shared chart style)
    BASELINE,
    CATEGORICAL,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    SURFACE,
    WEEKS,
    plt,
)

GRAPHICS = WEEKS / "graphics"
LANES = 16  # fp32 elements per vector register (SVL = 512 bit)
OTHER = "#e8e7e1"


def tint(color, alpha):
    """Blends a hex color with the surface color."""
    c = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    s = [int(SURFACE[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(a * alpha + b * (1 - alpha)):02x}" for a, b in zip(c, s))


# ---------------------------------------------------------------------------
# simulation of the SME2 transpose (registers hold lists of 16 fp32 values)
# ---------------------------------------------------------------------------
def pairs(reg):
    return [reg[i:i + 2] for i in range(0, LANES, 2)]


def flat(pair_list):
    return [v for p in pair_list for v in p]


def zip4_d(regs):
    """zip { z0.d - z3.d }, { z0.d - z3.d }: interleaves the 64-bit elements of four registers."""
    seq = [pairs(r)[e] for e in range(LANES // 2) for r in regs]
    return [flat(seq[i * 8:(i + 1) * 8]) for i in range(4)]


def uzp2_s(zn, zm):
    """uzp { zd1.s, zd2.s }, zn.s, zm.s: even / odd 32-bit elements of zn:zm."""
    cat = zn + zm
    return cat[0::2], cat[1::2]


def uzp_d(zn, zm):
    """uzp1 / uzp2 zd.d, zn.d, zm.d: even / odd 64-bit elements of zn:zm."""
    cat = pairs(zn) + pairs(zm)
    return flat(cat[0::2]), flat(cat[1::2])


def trn_d(zn, zm):
    """trn1 / trn2 zd.d, zn.d, zm.d."""
    pn, pm = pairs(zn), pairs(zm)
    t1 = flat([p for i in range(0, 8, 2) for p in (pn[i], pm[i])])
    t2 = flat([p for i in range(1, 8, 2) for p in (pn[i], pm[i])])
    return t1, t2


def simulate():
    z = {i: [LANES * i + c for c in range(LANES)] for i in range(16)}
    rows = [z[i][:] for i in range(4)]
    for base in (0, 4, 8, 12):
        z[base], z[base + 1], z[base + 2], z[base + 3] = zip4_d([z[base + i] for i in range(4)])
    z[16], z[17] = uzp2_s(z[0], z[4])
    z[18], z[19] = uzp2_s(z[8], z[12])
    z[26], z[27] = uzp_d(z[16], z[18])
    z[28], z[30] = trn_d(z[26], z[27])
    # the second half (z17 / z19) yields columns 1 and 3
    s26, s27 = uzp_d(z[17], z[19])
    z[29], z[31] = trn_d(s26, s27)
    for reg, col in ((28, 0), (29, 1), (30, 2), (31, 3)):
        assert z[reg] == [LANES * r + col for r in range(LANES)], reg
    return rows, z


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------
def element_style(value):
    col = value % LANES
    if col < 4:
        return tint(CATEGORICAL[col], 0.22), CATEGORICAL[col]
    return OTHER, BASELINE


def draw_register(ax, y, label, values):
    x = 0.0
    for i, value in enumerate(values):
        fill, edge = element_style(value)
        ax.add_patch(FancyBboxPatch((x, y), 0.9, 0.8, boxstyle="round,pad=0,rounding_size=0.12",
                                    facecolor=fill, edgecolor=edge, linewidth=1.0))
        ax.text(x + 0.45, y + 0.4, str(value), ha="center", va="center", fontsize=9, color=INK)
        x += 0.95 if i % 2 == 0 else 1.25  # 64-bit pairs are grouped
    ax.text(-0.35, y + 0.4, label, ha="right", va="center", fontsize=11, color=INK)


def transpose_step(filename, top, bottom, caption):
    fig, ax = plt.subplots(figsize=(10, 3.9))
    ax.set_axis_off()
    y = 0.0
    for label, values in top:
        draw_register(ax, y, label, values)
        y -= 1.0
    ax.add_patch(FancyArrowPatch((0.5, y + 0.6), (0.5, y - 0.6), arrowstyle="-|>", mutation_scale=12,
                                 color=INK_MUTED, linewidth=1.2))
    ax.text(1.0, y, caption, ha="left", va="center", fontsize=10, color=INK_SECONDARY, family="monospace")
    y -= 1.6
    for label, values in bottom:
        draw_register(ax, y, label, values)
        y -= 1.0

    # legend
    x = 0.0
    y -= 0.4
    entries = [(f"Column {c}", tint(CATEGORICAL[c], 0.22), CATEGORICAL[c]) for c in range(4)]
    entries.append(("other", OTHER, BASELINE))
    for text, fill, edge in entries:
        ax.add_patch(FancyBboxPatch((x, y + 0.1), 0.6, 0.6, boxstyle="round,pad=0,rounding_size=0.1",
                                    facecolor=fill, edgecolor=edge, linewidth=1.0))
        ax.text(x + 0.8, y + 0.4, text, ha="left", va="center", fontsize=10, color=INK_SECONDARY)
        x += 3.4

    ax.set_xlim(-3.6, 18.2)
    ax.set_ylim(y - 0.2, 1.0)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(GRAPHICS / filename)
    plt.close(fig)


def plot_transpose_steps():
    rows, z = simulate()
    transpose_step("transpose_sme2_step1_zip.svg",
                   [(f"row {r}", rows[r]) for r in range(4)],
                   [("z0 (new)", z[0])],
                   "zip { z0.d - z3.d }, { z0.d - z3.d }   (excerpt: new z0)")
    transpose_step("transpose_sme2_step2_uzp.svg",
                   [("z0", z[0]), ("z4", z[4])],
                   [("z16", z[16]), ("z17", z[17])],
                   "uzp { z16.s, z17.s }, z0.s, z4.s")
    transpose_step("transpose_sme2_step3_uzp1_uzp2.svg",
                   [("z16", z[16]), ("z18", z[18])],
                   [("z26 (uzp1)", z[26]), ("z27 (uzp2)", z[27])],
                   "uzp1 / uzp2 z26.d / z27.d, z16.d, z18.d")
    transpose_step("transpose_sme2_step4_trn1_trn2.svg",
                   [("z26", z[26]), ("z27", z[27])],
                   [("z28 = column 0", z[28]), ("z30 = column 2", z[30])],
                   "trn1 / trn2 z28.d / z30.d, z26.d, z27.d")


def plot_multi_k():
    """A and B are split into four K-tiles, K-tile t is accumulated in tile za<t>, the tiles are summed."""
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.set_axis_off()

    def block(x, y, w, h, t, text=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                    facecolor=tint(CATEGORICAL[t], 0.35), edgecolor=CATEGORICAL[t], linewidth=1.2))
        if text:
            ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, color=INK)

    # A (m x k): four k-slices side by side
    ax.text(-0.3, 0.5, "A", ha="right", va="center", fontsize=13, color=INK)
    for t in range(4):
        block(t * 1.1, 0.0, 1.0, 1.0, t, f"K{t}")
    ax.text(2.15, -0.35, "k", ha="center", va="top", fontsize=9, color=INK_SECONDARY)

    # B (k x n): four k-slices stacked
    ax.text(5.5, 4.4, "B", ha="center", va="bottom", fontsize=13, color=INK)
    for t in range(4):
        block(5.0, 3.3 - t * 1.1, 1.0, 1.0, t, f"K{t}")

    # ZA tiles
    ax.add_patch(FancyArrowPatch((6.4, 1.5), (7.6, 1.5), arrowstyle="-|>", mutation_scale=12,
                                 color=INK_MUTED, linewidth=1.2))
    ax.text(7.0, 1.75, "fmopa", ha="center", va="bottom", fontsize=9, color=INK_SECONDARY, family="monospace")
    for t in range(4):
        block(7.9 + (t % 2) * 1.3, 2.0 - (t // 2) * 1.3, 1.1, 1.1, t, f"za{t}")

    # sum
    ax.add_patch(FancyArrowPatch((10.6, 1.5), (11.8, 1.5), arrowstyle="-|>", mutation_scale=12,
                                 color=INK_MUTED, linewidth=1.2))
    ax.text(11.2, 1.75, "sum", ha="center", va="bottom", fontsize=9, color=INK_SECONDARY)
    ax.add_patch(FancyBboxPatch((12.0, 0.95), 1.1, 1.1, boxstyle="round,pad=0,rounding_size=0.08",
                                facecolor=OTHER, edgecolor=INK_MUTED, linewidth=1.2))
    ax.text(12.55, 1.5, "C", ha="center", va="center", fontsize=12, color=INK)

    ax.set_xlim(-0.8, 13.4)
    ax.set_ylim(-0.8, 4.9)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(GRAPHICS / "multi_k.svg")
    plt.close(fig)


if __name__ == "__main__":
    plot_multi_k()
    plot_transpose_steps()

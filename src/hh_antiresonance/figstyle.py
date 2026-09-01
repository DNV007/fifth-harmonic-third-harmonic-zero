"""Shared figure style for the manuscript figures (APS / PRL / PRE).

Why this module exists
----------------------
Every driver used to carry its own ``rcParams`` block and its own ``figsize``.
Label sizes ranged over 10, 11, 13, 14 pt and widths over 6.4--12.6 in, while
the manuscripts include everything at a 3.2 in column. The result was that each
figure was downscaled by a *different* factor (0.32--0.86), so a nominally
uniform "11 pt" label printed at anywhere from 3.5 to 9.5 pt. The figures were
inconsistent because they were authored at the wrong size, not because the fonts
were set wrong.

The rule enforced here: **author every figure at the exact width it will be
printed at**, so LaTeX applies no scaling and the point sizes below are the
point sizes on paper.

    fig, axes = plt.subplots(1, 2, figsize=column(0.46))   # single column
    fig, axes = plt.subplots(1, 3, figsize=full(0.30))     # double column

Print safety
------------
The print edition is black and white. Colour alone must therefore never be the
only thing distinguishing two series. ``SERIES`` below pairs each colour with a
distinct linestyle *and* a distinct marker, and the colours are ordered so that
consecutive entries are far apart in greyscale luminance as well as in hue. Run

    python -m hh_antiresonance.figstyle

to print the greyscale value of each colour and the pairwise separations.
The palette is Okabe--Ito (colour-vision-deficiency safe) plus a dark navy.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- geometry --
# APS two-column layout. A figure authored at these widths is placed with
# \includegraphics[width=\columnwidth]{...} (or \textwidth in a figure*) and
# is therefore *not* rescaled.
COLUMN_WIDTH = 3.375   # in  (8.6 cm), single column
FULL_WIDTH = 6.90      # in  (17.5 cm), double column / figure*


def column(aspect: float = 0.72) -> tuple[float, float]:
    """figsize for a single-column figure; ``aspect`` is height/width."""
    return (COLUMN_WIDTH, COLUMN_WIDTH * aspect)


def full(aspect: float = 0.32) -> tuple[float, float]:
    """figsize for a double-column (figure*) figure; ``aspect`` is height/width."""
    return (FULL_WIDTH, FULL_WIDTH * aspect)


# ------------------------------------------------------------------ colours --
# Ordered for greyscale separation: entry 0 is near-black, entry 1 is light,
# entry 2 sits between them. Two-series plots use 0 and 1 and remain legible
# in print even if the linestyles were stripped.
NAVY = "#08306B"    # very dark blue
AMBER = "#E69F00"   # Okabe-Ito orange   (light in greyscale)
BLUE = "#0072B2"    # Okabe-Ito blue     (mid)
GREEN = "#009E73"   # Okabe-Ito green
PURPLE = "#CC79A7"  # Okabe-Ito purple
VERMILLION = "#D55E00"  # Okabe-Ito vermillion -- reserved for accents
GREY = "#7F7F7F"

#: (colour, linestyle, marker) triples. Always take styles from here in order,
#: so that series 1 and series 2 differ in all three channels.
SERIES = [
    (NAVY,   "-",  "o"),
    (AMBER,  "--", "s"),
    (BLUE,   "-.", "^"),
    (GREEN,  ":",  "D"),
    (PURPLE, (0, (3, 1, 1, 1)), "v"),
]

#: For the fold/zero markers (stars). Black edge keeps them visible in print.
ACCENT = VERMILLION
#: Reference/analytic curves that must read as subordinate.
REFERENCE = GREY


def series(i: int) -> dict:
    """Style kwargs for the i-th series: colour + linestyle + marker."""
    c, ls, mk = SERIES[i % len(SERIES)]
    return dict(color=c, linestyle=ls, marker=mk)


def star(**kw) -> dict:
    """Style kwargs for an accent marker (a fold apex, a working point)."""
    d = dict(marker="*", markersize=9, markerfacecolor=ACCENT,
             markeredgecolor="black", markeredgewidth=0.6, linestyle="none",
             zorder=6)
    d.update(kw)
    return d


# ------------------------------------------------------------------- rcParams --
# These are FINAL PRINTED point sizes, valid only because figures are authored
# at their printed width. Nothing here should be below ~6 pt on paper.
RC = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    # APS: the smallest capitals/numerals must be >= 2 mm at printed size.
    # Measured ink height of a DejaVu Serif numeral: 7 pt -> 1.88 mm (fails),
    # 8 pt -> 2.12 mm (passes). Tick labels are numerals, so they must be 8 pt.
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.labelsize": 8,
    "axes.linewidth": 0.6,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.0,
    "lines.markeredgewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.45,
    "ytick.minor.width": 0.45,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.7",
    "legend.borderpad": 0.35,
    "legend.labelspacing": 0.3,
    "legend.handlelength": 1.8,
    "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.4,
    "grid.color": "0.85",
    "grid.linewidth": 0.4,
    "grid.linestyle": ":",
    "axes.grid": False,
    # No automatic offsets. Lowering the offset threshold to catch A_1 also made
    # matplotlib hoist "+1.2" out of the Omega axes, where the absolute value is
    # exactly what the reader needs. Rescaling an axis is a judgement call, so it
    # is made explicitly, per axis, with factor_axis() below.
    "axes.formatter.useoffset": False,
    # Constrained-layout padding. The defaults (h_pad = 0.042 in) leave a visible
    # band between the panels and an "outside lower center" legend. Tighten it so a
    # shared legend sits just under the x-labels instead of floating away from them.
    "figure.constrained_layout.h_pad": 0.030,
    "figure.constrained_layout.w_pad": 0.012,
    "figure.constrained_layout.hspace": 0.02,
    "figure.constrained_layout.wspace": 0.02,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,   # embed as TrueType, not Type 3 (APS requirement)
    "ps.fonttype": 42,
}


def use() -> None:
    """Apply the shared style. Call once, at driver import time."""
    mpl.rcParams.update(RC)


def grid(ax, which: str = "major") -> None:
    """Consistent, unobtrusive grid."""
    ax.grid(True, which=which, color="0.85", lw=0.4, ls=":", alpha=0.9)
    ax.set_axisbelow(True)


def panel(ax, label: str, dx: float = -0.20, dy: float = 1.02) -> None:
    """Bold panel letter, placed identically on every axes."""
    ax.text(dx, dy, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", ha="left")


def factor_axis(ax, exp: int, label: str, axis: str = "y") -> None:
    """Pull a common power of ten out of the tick labels and into the axis label.

    A column-width panel cannot afford tick labels like ``3.4e-16`` or
    ``6 x 10^-1`` -- they are wider than the data they annotate. Factor the
    exponent out once, so the ticks read ``2, 3, 4`` and the label carries the
    scale:

        factor_axis(ax, -16, r"$\\|R\\|_2$")   ->  ylabel  "||R||_2 (x10^-16)"

    The data are untouched; only the formatting changes.
    """
    from matplotlib.ticker import FuncFormatter

    fmt = FuncFormatter(lambda v, _pos: f"{v * 10.0 ** (-exp):g}")
    full = rf"{label} $(\times 10^{{{exp}}})$"
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
        ax.set_ylabel(full)
    else:
        ax.xaxis.set_major_formatter(fmt)
        ax.set_xlabel(full)


def offset_axis(ax, label: str, axis: str = "y") -> None:
    """Compact ticks for data with a large common offset and a tiny spread.

    ``A_1`` runs over 0.380814..0.380975 -- five decimals of tick label to show a
    spread of 1.6e-4. Let matplotlib hoist the shared offset out of the ticks.
    """
    from matplotlib.ticker import ScalarFormatter

    sf = ScalarFormatter(useOffset=True, useMathText=True)
    sf.set_powerlimits((-3, 3))
    if axis == "y":
        ax.yaxis.set_major_formatter(sf)
        ax.set_ylabel(label)
        ax.yaxis.get_offset_text().set_fontsize(6)
    else:
        ax.xaxis.set_major_formatter(sf)
        ax.set_xlabel(label)
        ax.xaxis.get_offset_text().set_fontsize(6)


def save(fig, figdir, stem: str) -> None:
    """Write the vector PDF (primary) and a 600 dpi PNG (APS raster minimum)."""
    fig.savefig(figdir / f"{stem}.pdf")
    fig.savefig(figdir / f"{stem}.png", dpi=600)


# --------------------------------------------------------- greyscale check --
def _luminance(hex_colour: str) -> float:
    """Relative luminance (WCAG) of an sRGB hex colour, in [0, 1]."""
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))

    def lin(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def greyscale_report() -> None:
    """Print the greyscale value of every series colour and check separation."""
    names = ["NAVY", "AMBER", "BLUE", "GREEN", "PURPLE"]
    lums = [(n, c, _luminance(c)) for (n, (c, _, _)) in zip(names, SERIES)]
    print("greyscale luminance of the series palette (print edition):")
    for n, c, L in lums:
        bar = "#" * int(round(L * 40))
        print(f"  {n:7s} {c}  L={L:5.3f}  |{bar:<40s}|")
    print(f"  {'ACCENT':7s} {ACCENT}  L={_luminance(ACCENT):5.3f}")
    print("\npairwise |dL| (want >= 0.15 for the pairs actually used together):")
    ok = True
    for i in range(len(lums)):
        for j in range(i + 1, len(lums)):
            d = abs(lums[i][2] - lums[j][2])
            flag = "" if d >= 0.15 else "   <-- rely on linestyle+marker"
            if d < 0.15:
                ok = False
            print(f"  {lums[i][0]:7s} vs {lums[j][0]:7s}  dL={d:5.3f}{flag}")
    print("\nseries 0 vs 1 (the default two-series pair):",
          f"dL={abs(lums[0][2] - lums[1][2]):.3f}",
          "OK" if abs(lums[0][2] - lums[1][2]) >= 0.15 else "TOO CLOSE")
    print("every series also carries a distinct linestyle and marker, so hue is",
          "never the sole discriminator." if not ok else "redundant.")


if __name__ == "__main__":
    greyscale_report()

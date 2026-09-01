"""Fundamental amplitude along the two zero loci.

The point of the panel is the contrast in how the two third-harmonic zeros of
the same model behave as the drive is reduced. The perturbative comparison root
follows A1 = 1.844 F to the origin. The branch reported here does not: it turns
at a fold near F = 0.2577, and the drive itself folds at F = 0.02099 and rises
again, so the locus never enters the weak-amplitude regime.

Both curves are drawn in CONTINUATION order. The reported locus folds twice and
is not single-valued in F, so it must not be sorted by F.

Inputs (resolved from data/):
  two_dof_k3_weak_drive_path_kappa010.csv   from run_k3_weak_drive_route.py
  k3_sister_family.csv                      from run_k3_sister_family.py

Usage:  PYTHONPATH=src uv run python scripts/make_k3_branch_vs_F.py [datadir] [outdir]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hh_antiresonance import figstyle as fs

DATADIR = sys.argv[1] if len(sys.argv) > 1 else "data"
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "figures"

LOCUS = "two_dof_k3_weak_drive_path_kappa010.csv"
PERT = "k3_sister_family.csv"

# Working points and folds, as reported in the Letter and SM.
F_WORK = 0.30
A1_WORK = 0.2924                                # reported branch at F = 0.30
A1_WORK_PERT = 1.794 * 0.30                     # perturbative comparison branch
F_FOLD, A1_FOLD = 0.25774, 0.2508               # turn of the tracked arm
F_DRIVE_FOLD, A1_DRIVE_FOLD = 0.02099, 0.66726  # minimum of the drive

# Authored at the width it is printed at, 0.46\textwidth of a figure*, so LaTeX
# applies no scaling. Matches k3_notch_vs_zero, the panel beside it.
FIGSIZE = (3.17, 2.60)


def find(name: str) -> str:
    """Resolve a CSV from the data directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    for d in (DATADIR, os.path.join(here, "..", "data")):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(name)


def load_locus():
    d = np.genfromtxt(find(LOCUS), delimiter=",", names=True)
    return np.asarray(d["force"], float), np.asarray(d["abs_X1"], float)


def load_pert():
    """The comparison branch; the file stores |X1|/F rather than |X1|."""
    d = np.genfromtxt(find(PERT), delimiter=",", names=True)
    F = np.asarray(d["F"], float)
    o = np.argsort(F)
    return F[o], (F * np.asarray(d["absX1_over_F"], float))[o]


def main():
    fs.use()
    fig, ax = plt.subplots(figsize=FIGSIZE, constrained_layout=True)

    cB, lsB, mkB = fs.SERIES[0]
    cP, lsP, mkP = fs.SERIES[1]

    F_b, A_b = load_locus()
    ax.plot(F_b, A_b, color=cB, linestyle=lsB, lw=1.3, marker=mkB, ms=2.6,
            markevery=70, zorder=3, label="reported branch")

    F_p, A_p = load_pert()
    ax.plot(F_p, A_p, color=cP, linestyle=lsP, lw=1.3, marker=mkP, ms=2.6,
            zorder=3, label="perturbative root")

    ax.plot([F_WORK], [A1_WORK], "o", ms=5.0, mfc="white", mec=cB, mew=1.2, zorder=5)
    ax.plot([F_WORK], [A1_WORK_PERT], "o", ms=5.0, mfc="white", mec=cP, mew=1.2, zorder=5)
    # Two distinct markers: the caption names them, and a reader should not have
    # to count "first" and "second" star.
    ax.plot([F_FOLD], [A1_FOLD], **fs.star(markersize=8))
    # A diamond, not a second star: filled-vs-open was decodable only from the
    # caption at printed size.
    ax.plot([F_DRIVE_FOLD], [A1_DRIVE_FOLD], marker="D", ms=5.0,
            markerfacecolor=fs.ACCENT, markeredgecolor="black", markeredgewidth=0.6,
            linestyle="none", zorder=6)

    # The amplitude floor is the claim the panel exists to support, so mark it.
    ax.axhline(A1_FOLD, color="0.55", lw=0.6, ls=(0, (4, 3)), zorder=1)
    ax.text(1.9, A1_FOLD * 0.80, r"$A_{1,\min}=0.2508$", fontsize=7, color="0.35",
            ha="right", va="top")

    ax.annotate("fold", xy=(F_FOLD, A1_FOLD), xytext=(-2, -16),
                textcoords="offset points", fontsize=7, color="0.25",
                ha="center", va="top",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="0.25", shrinkB=4))
    # Below the star: above it collides with the axes frame.
    ax.annotate("drive fold", xy=(F_DRIVE_FOLD, A1_DRIVE_FOLD), xytext=(-1, -14),
                textcoords="offset points", fontsize=7, color="0.25",
                ha="center", va="top",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="0.25", shrinkB=4))
    ax.text(2.2e-3, 6.0e-3, r"$A_1=1.844\,F$", fontsize=7, color=cP,
            ha="left", va="top", rotation=0)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"drive amplitude $F$")
    ax.set_ylabel(r"fundamental amplitude $A_1$")
    # Bound the axes by the data on both curves: the old limits stopped at
    # F = 0.45 and hid the return of the drive to F = 2.41, which is the
    # evidence that the locus is not heading toward F = 0.
    ax.set_xlim(7e-4, 3.2)
    ax.set_ylim(1.2e-3, 1.35)
    ax.legend(frameon=False, fontsize=7, loc="lower right", borderaxespad=0.3,
              handlelength=1.6, labelspacing=0.25)
    fs.grid(ax)
    fs.panel(ax, "(a)", dx=-0.20)

    out = Path(OUTDIR)
    out.mkdir(parents=True, exist_ok=True)
    fs.save(fig, out, "k3_branch_vs_F")
    print("wrote", out / "k3_branch_vs_F.pdf")


if __name__ == "__main__":
    main()

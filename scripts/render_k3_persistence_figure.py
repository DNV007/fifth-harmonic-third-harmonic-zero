"""Render the SM finite-amplitude-closure + structural-persistence figure.

Produces figures/k3_persistence_residual.{pdf,png}:
  (a) finite-amplitude closure: the raw oscillator-1 third-harmonic balance
      residual evaluated with U_{k>=5} = 0, and the contribution restored by the
      fifth harmonic, are equal and opposite in BOTH quadratures. The truncated
      residual is NOT the fundamental-only ("leading order") term: it retains the
      third-harmonic feedback. It is also a RAW residual, not the projected
      source combination of SM Eq. (S5), so its magnitude is not comparable with
      the projected channel magnitudes c_LO, c_3, c_5;
  (b) structural persistence: the root is recovered at the constrained-solver
      floor under four perturbation classes; at a large quintic (k5=0.15) no root
      was found in the searched window.

This is a pure renderer; the plotted values are verified solver outputs:
  panel (a): scripts/run_k3_residual_decomposition.py
             (Re,Im) leading residual (-6.54, +6.01)e-5, +A5 mixing equal and
             opposite, sum < 1e-17 (measured ~7e-18);
  panel (b): re-nulled |X3| from the persistence runs recorded in SM Sec. S3.
             Every point is a verified reproducer output:
               omega2 = 1.10 / 1.35 -> run_k3_persistence_renull.py
               mu = 0.7 / 1.3, k5 = 0.05, k5 = 0.15
                                    -> run_k3_extended_model_persistence.py
               alpha1 = 0.05        -> run_k3_mean_mode_corrections.py
                                       (extended basis: the two mean modes are
                                       free and the mean equations enforced,
                                       which a nonzero quadratic requires)
             The single-pathway reference |X3^(0)| = 2.77e-3 comes from
             run_k3_suppression.py. The k5 = 0.15 point is the best in-window
             residual of a 120-seed cold hunt, in which no root was found; the
             other six are re-nulled solver floors. A floor value is the
             least-squares residual of its solve rather than a spectral
             amplitude, so only its position inside the floor
             band carries meaning.

Run:  uv run python scripts/render_k3_persistence_figure.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
from hh_antiresonance import figstyle as fs
fs.use()

_D = os.path.dirname(os.path.abspath(__file__))
FIGDIR = Path(_D) / ".." / "figures"

fig = plt.figure(figsize=fs.column(1.02))
gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.25], hspace=0.55)
axA = fig.add_subplot(gs[0])
axB = fig.add_subplot(gs[1])

# ---- (a) quadrature decomposition: truncated residual vs restored U5 --------
# NB: `lead` is the RAW oscillator-1 third-harmonic balance residual evaluated
# with U_{k>=5} = 0. It still contains the third-harmonic feedback and is NOT
# the fundamental-only contribution c_LO; the two live in
# different residual coordinates.
lead = np.array([-6.54, 6.01])     # (Re, Im) of R3^(1) with m>=5 removed, x1e-5
mix = -lead                        # contribution restored by the 5th harmonic
x = np.arange(2); w = 0.34
axA.axhline(0, color="k", lw=0.6, zorder=1)
axA.bar(x - w/2, lead, w, color=fs.BLUE, edgecolor="k", lw=0.4,
        label=r"residual, $U_{k\geq5}{=}0$", zorder=3)
axA.bar(x + w/2, mix, w, color=fs.AMBER, edgecolor="k", lw=0.4,
        label=r"restored $U_5$", zorder=3, hatch="//////")
axA.set_xticks(x)
axA.set_xticklabels([r"$\mathrm{Re}\,R^{(1)}_3$", r"$\mathrm{Im}\,R^{(1)}_3$"])
axA.set_ylabel(r"residual $(\times 10^{-5})$")
axA.set_ylim(-9.0, 9.0)
axA.text(0.5, 0.0, r"sum $<10^{-17}$", ha="center", va="center", fontsize=8,
         bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.7", lw=0.4))
axA.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
           frameon=False, fontsize=8, handlelength=1.4, columnspacing=1.4)
fs.grid(axA)
fs.panel(axA, "(a)")

# ---- (b) structural persistence: |X3| at the re-solved zero -----------------
labels = [r"$\omega_2\!\downarrow$", r"$\omega_2\!\uparrow$", r"$\mu{=}0.7$",
          r"$\mu{=}1.3$", r"$k_5{=}0.05$", r"$\alpha_1{=}0.05$", r"$k_5{=}0.15$"]
# alpha1 = 0.05 breaks the half-period antisymmetry and forces a static offset,
# so that point is solved in the extended basis carrying the two mean modes
# (run_k3_mean_mode_corrections.py). Its root sits at (Omega,beta1) =
# (0.259634893, 0.244184214) with <x1> = -1.95e-3, not at the zero-mean value.
vals = np.array([3.4e-26, 8.9e-32, 3.2e-26, 2.9e-28, 3.3e-26, 1.6e-25, 6.1e-7])
robust = np.array([True]*6 + [False])
x = np.arange(len(labels))
FLOOR = 1e-13
# Constrained-solver termination band. NOT a physical null depth and not an
# "exact zero": X3 = 0 is imposed as a constraint, so a point inside this band
# reports where its least-squares solve terminated in the stated nondimensional
# scaling. Only position inside the band is meaningful.
axB.axhspan(1e-34, FLOOR, color=fs.GREEN, alpha=0.10, zorder=0)
axB.text(3.5, 4e-15, "solver floor", fontsize=8, color=fs.GREEN,
         ha="center", va="top")
# single-pathway reference
ref = 2.77e-3
axB.axhline(ref, color=fs.REFERENCE, ls="--", lw=0.9, zorder=1)
# The label is right-aligned at the last category, so the axes need explicit
# room for it: with autoscaled limits it ran off the right edge and was clipped.
axB.set_xlim(-0.7, 6.7)
axB.text(6.6, ref*0.28, "single pathway", fontsize=8,
         ha="right", va="top", color=fs.REFERENCE)
# stems + markers
for xi, v, r in zip(x, vals, robust):
    axB.plot([xi, xi], [1e-34, v], color="0.75", lw=0.7, zorder=2)
axB.scatter(x[robust], vals[robust], s=26, marker="o", color=fs.NAVY,
            edgecolor="k", lw=0.4, zorder=4, label="root recovered")
# The k5=0.15 point is "no root found in the searched window", NOT a located
# annihilation: SM Sec. S3 tracks the opposite-index pair only to k5=0.110 and
# states that the merger was never computed with an augmented fold solve.
axB.scatter(x[~robust], vals[~robust], s=30, marker="D", color=fs.VERMILLION,
            edgecolor="k", lw=0.4, zorder=4, label="no root found")
axB.set_yscale("log")
axB.set_ylim(1e-34, 3e-2)
axB.set_yticks([1e-30, 1e-20, 1e-10, 1e-3])
axB.set_xticks(x)
axB.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
axB.set_ylabel(r"$|X_3|$ at re-solved zero")
axB.legend(loc="upper left", bbox_to_anchor=(0.01, 0.92), ncol=1, fontsize=7,
           frameon=False, handletextpad=0.3, borderaxespad=0.2)
fs.grid(axB, which="major")
fs.panel(axB, "(b)")

fs.save(fig, FIGDIR, "k3_persistence_residual")
print("wrote k3_persistence_residual.{pdf,png} ->", FIGDIR)

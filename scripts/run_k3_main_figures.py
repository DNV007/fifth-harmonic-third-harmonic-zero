"""Main figures for the mechanical model.

Produces:
  k3_argand      -- Argand-plane trajectory of X3 through the origin as the on-site
                    cubic beta1 is tuned (two pathways cross zero; one pathway misses);
  k3_zero_map    -- two-parameter zero map in the drive/continuation plane: the
                    quadrature-null curves Re X3 = 0 and Im X3 = 0 cross at the exact
                    zero, with the continuous exact-zero branch overlaid.

These replace the leading-order-comparison panel and the perturbation-marker panel
in the core (those move to the SM). Uses the production jax solver.

Run:  JAX_ENABLE_X64=1 uv run python scripts/run_k3_main_figures.py
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np
from dataclasses import replace
from jax import config
config.update("jax_enable_x64", True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.optimize import least_squares

from hh_antiresonance.models import CoupledOscillatorParams
from hh_antiresonance.harmonic_balance import solve_harmonic_balance, harmonic_balance_residual, coefficient_index

OM, B1, KNL, F0 = 0.25972810, 0.23815428, -0.30, 0.30
BASE = CoupledOscillatorParams(omega1=1.0, omega2=1.25, zeta1=0.015, zeta2=0.02,
    alpha1=0.0, alpha2=0.0, beta1=B1, beta2=0.0, kappa=0.10, force=F0,
    drive_omega=OM, kappa_nl=KNL)
N, NT = 7, 512
IC = coefficient_index(oscillator=1, harmonic=3, component="cos", n_harmonics=N)
ISx = coefficient_index(oscillator=1, harmonic=3, component="sin", n_harmonics=N)
_D = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(_D, "..", "figures")


def solve(p, g=None):
    return np.asarray(solve_harmonic_balance(p, n_harmonics=N, initial_guess=g,
        n_time_samples=NT, tol=1e-13, max_nfev=4000), float)


def X3(c):
    return complex(c[IC], -c[ISx])


def main():
    from pathlib import Path
    from hh_antiresonance import figstyle as fs
    fs.use()
    FD = Path(FIGDIR)
    CACHE = FD / "k3_main_figures_cache.npz"
    cache = dict(np.load(CACHE)) if CACHE.exists() else None
    c0 = solve(BASE) if cache is None else None

    # ---------- (a) Argand trajectory of X3 through the origin (units 1e-3) ----------
    # Authored at its placed width: 0.46\textwidth = 3.17 in (no LaTeX rescaling).
    figA, ax = plt.subplots(figsize=(3.17, 2.75), constrained_layout=True)
    S = 1e3  # plot in units of 1e-3
    b1s = np.linspace(B1 - 0.05, B1 + 0.05, 81)
    trs = {}
    if cache is None:
        for knl in (0.0, KNL):
            g = solve(replace(BASE, kappa_nl=knl), c0); tr = []
            for bb in b1s:
                g = solve(replace(BASE, beta1=float(bb), kappa_nl=knl), g); tr.append(X3(g))
            trs[knl] = np.array(tr) * S
    else:
        trs[0.0], trs[KNL] = cache["tr0"], cache["trK"]
    ax.axhline(0, color="0.82", lw=0.5, zorder=0); ax.axvline(0, color="0.82", lw=0.5, zorder=0)
    ax.plot(trs[0.0].real, trs[0.0].imag, ls="--", color=fs.AMBER, lw=1.3,
            marker="s", ms=3.0, markevery=10, label="one pathway", zorder=2)
    ax.plot(trs[KNL].real, trs[KNL].imag, ls="-", color=fs.NAVY, lw=1.4,
            marker="o", ms=3.0, markevery=10, label="two pathways", zorder=3)
    # sweep-direction arrow (increasing beta1) on the two-pathway trajectory
    k = 22
    ax.annotate("", xy=(trs[KNL].real[k + 2], trs[KNL].imag[k + 2]),
                xytext=(trs[KNL].real[k], trs[KNL].imag[k]),
                arrowprops=dict(arrowstyle="-|>", color=fs.NAVY, lw=1.0), zorder=4)
    ax.plot(0, 0, "o", mfc="white", mec="black", mew=1.0, ms=6, zorder=5)
    ax.annotate("computed root", xy=(0, 0), xytext=(-36, -20), textcoords="offset points",
                ha="center", va="top", fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.7, color="black", shrinkB=4))
    ax.set_xlabel(r"$\mathrm{Re}\,X_3\ (\times10^{-3})$")
    ax.set_ylabel(r"$\mathrm{Im}\,X_3\ (\times10^{-3})$")
    ax.legend(loc="upper right", fontsize=8)
    fs.grid(ax); fs.panel(ax, "(a)", dx=-0.17)
    ax.margins(0.15)
    fs.save(figA, FD, "k3_argand")
    print("wrote k3_argand")

    # ---------- (b) two-parameter zero map + exact-zero branch ----------
    figB, ax = plt.subplots(figsize=(3.17, 2.75), constrained_layout=True)
    # beta1 window tracks the null curves (which span 0.2284..0.2490) rather than a
    # wide slab of featureless background; same 61 points now give ~4x the resolution.
    noms = np.linspace(OM - 0.012, OM + 0.012, 61); nb1 = np.linspace(0.2260, 0.2510, 61)
    if cache is None:
        RE = np.zeros((len(nb1), len(noms))); IM = np.zeros_like(RE); MAG = np.zeros_like(RE)
        for j, om in enumerate(noms):
            g = c0.copy()
            for i, bb in enumerate(nb1):
                g = solve(replace(BASE, drive_omega=float(om), beta1=float(bb)), g); y = X3(g)
                RE[i, j], IM[i, j], MAG[i, j] = y.real, y.imag, abs(y)
        np.savez(CACHE, tr0=trs[0.0], trK=trs[KNL], RE=RE, IM=IM, MAG=MAG)
    else:
        RE, IM, MAG = cache["RE"], cache["IM"], cache["MAG"]
    ext = [noms[0], noms[-1], nb1[0], nb1[-1]]
    # Suppressed = light, loud = dark, with the dark end capped well short of
    # black: the field is loud over most of the window, so a heavier map turns
    # the panel grey and swallows the two null contours it exists to show.
    import matplotlib as mpl
    from matplotlib.colors import LinearSegmentedColormap
    greys_soft = LinearSegmentedColormap.from_list(
        "greys_soft", mpl.colormaps["Greys"](np.linspace(0.0, 0.42, 256)))
    im = ax.imshow(np.log10(MAG + 1e-18), origin="lower", extent=ext, aspect="auto",
                   cmap=greys_soft, vmin=-7.0, vmax=-2.0)
    ax.contour(noms, nb1, RE, levels=[0], colors=[fs.NAVY], linewidths=1.5,
               linestyles="solid")
    ax.contour(noms, nb1, IM, levels=[0], colors=[fs.VERMILLION], linewidths=1.5,
               linestyles="dashed")
    ax.plot(OM, B1, **fs.star(markersize=10))
    # Contours are labelled where they run; a legend box would cover the
    # Im X_3 = 0 crest, which is the feature that makes the crossing transverse.
    # The two contours cross the region where the labels sit, so give the
    # glyphs a white halo rather than hunting for a gap that does not exist.
    halo = [pe.withStroke(linewidth=1.8, foreground="white")]
    ax.text(noms[0] + 0.0008, 0.2396, r"$\mathrm{Re}\,X_3=0$", color=fs.NAVY,
            fontsize=7, ha="left", va="top", path_effects=halo, zorder=7)
    ax.text(0.2568, 0.2487, r"$\mathrm{Im}\,X_3=0$", color=fs.VERMILLION,
            fontsize=7, ha="left", va="center", path_effects=halo, zorder=7)
    ax.annotate("computed root", xy=(OM, B1), xytext=(14, -20),
                textcoords="offset points", fontsize=7, ha="left", va="top",
                arrowprops=dict(arrowstyle="->", lw=0.6, color="0.25", shrinkB=5))
    ax.set_xlabel(r"drive frequency $\Omega$"); ax.set_ylabel(r"on-site cubic $\beta_1$")
    cb = figB.colorbar(im, ax=ax, pad=0.02); cb.set_label(r"$\log_{10}|X_3|$", fontsize=8)
    cb.ax.tick_params(labelsize=8)
    # Single-panel figure: no panel letter.
    fs.save(figB, FD, "k3_zero_map")
    print("wrote k3_zero_map")


if __name__ == "__main__":
    main()

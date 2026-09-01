"""Figure 1: the fifth harmonic changes whether the third-harmonic zero exists.

Three panels, in the order the argument runs:

  (a) N_H = 3, so only the odd content {1,3} is retained.  The two quadrature
      null contours Re X_3 = 0 and Im X_3 = 0 run as nearly parallel curves and
      do not meet anywhere in the displayed window.
  (b) N_H = 7, same axes and same colour scale.  Retaining 5*Omega leaves the
      Re contour almost where it was and bends the Im contour, sweeping it up
      through the Re contour so that the two cross transversely at the reported
      root.  That the fifth harmonic acts on the phase quadrature is the
      mechanism, seen directly.
  (c) The homotopy: weighting the fifth-indexed cubic contributions to the two
      3*Omega balance rows by lambda, the two roots present at lambda = 1 meet
      in a saddle-node at lambda* = 0.4837 and neither continues below it.

Window.  Omega is the range used by the earlier standalone zero map; beta_1 is
widened to [0.220, 0.250].  At N_H = 3 the Im contour sits at beta_1 = 0.2283,
which is only 0.0023 above the old lower edge of 0.2260 -- inside the frame, but
hugging it, so that the panel read as "one contour missing" rather than "two
contours that do not meet".  The widened range clears it and still contains the
N_H = 7 Im excursion, which reaches beta_1 = 0.245.

Run:  PYTHONPATH=src JAX_ENABLE_X64=1 uv run python scripts/make_k3_existence_figure.py
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
from pathlib import Path

import numpy as np
from jax import config

config.update("jax_enable_x64", True)
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from hh_antiresonance import figstyle as fs
from hh_antiresonance.models import CoupledOscillatorParams
from hh_antiresonance.harmonic_balance import solve_harmonic_balance, coefficient_index

fs.use()

OM, B1 = 0.25972810, 0.23815428
OM_LO, OM_HI = OM - 0.012, OM + 0.012
B1_LO, B1_HI = 0.220, 0.250
NG = 49

_D = Path(os.path.dirname(os.path.abspath(__file__)))
DATA = _D / ".." / "data"
FIGS = [_D / ".." / "figures"]


def X3_map(n_h, ng=NG):
    og = np.linspace(OM_LO, OM_HI, ng)
    bg = np.linspace(B1_LO, B1_HI, ng)
    Z = np.full((ng, ng), np.nan, dtype=complex)
    for i, o in enumerate(og):
        g = None
        for j, b in enumerate(bg):
            p = CoupledOscillatorParams(omega1=1.0, omega2=1.25, zeta1=0.015, zeta2=0.020,
                                        alpha1=0.0, alpha2=0.0, beta1=float(b), beta2=0.0,
                                        kappa=0.10, kappa_nl=-0.30, force=0.30,
                                        drive_omega=float(o))
            c = solve_harmonic_balance(p, n_harmonics=n_h, n_time_samples=512,
                                       tol=1e-12, initial_guess=g)
            idx = lambda cp: coefficient_index(oscillator=1, harmonic=3,
                                               component=cp, n_harmonics=n_h)
            z = complex(c[idx("cos")], -c[idx("sin")])
            if abs(z) == 0.0 or np.abs(c).max() == 0.0:   # failed solve, not a zero
                g = None
                continue
            g = c
            Z[i, j] = z
    return og, bg, Z


def both_quadrature_cells(Z):
    n = Z.shape[0]
    k = 0
    for i in range(n - 1):
        for j in range(n - 1):
            blk = Z[i:i + 2, j:j + 2]
            if np.isnan(blk.real).any():
                continue
            if blk.real.min() < 0 < blk.real.max() and blk.imag.min() < 0 < blk.imag.max():
                k += 1
    return k


def panel(ax, og, bg, Z, cmap, vmin, vmax, letter, tag):
    im = ax.pcolormesh(og, bg, np.log10(np.abs(Z)).T, cmap=cmap, vmin=vmin, vmax=vmax,
                       shading="gouraud", rasterized=True)
    ax.contour(og, bg, Z.real.T, levels=[0.0], colors=[fs.NAVY],
               linestyles="-", linewidths=1.3)
    ax.contour(og, bg, Z.imag.T, levels=[0.0], colors=[fs.VERMILLION],
               linestyles="--", linewidths=1.3)
    ax.set_xlim(OM_LO, OM_HI)
    ax.set_ylim(B1_LO, B1_HI)
    ax.set_xlabel(r"drive frequency $\Omega$")
    ax.text(0.03, 0.96, tag, transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.2))
    fs.panel(ax, letter, dx=-0.20)
    return im


def main():
    cache = DATA / "two_dof_k3_existence_maps.npz"
    if cache.exists() and os.environ.get("RECOMPUTE", "0") != "1":
        d = np.load(cache)
        og, bg, Z3, Z7 = d["og"], d["bg"], d["Z3"], d["Z7"]
        print(f"loaded cached maps from {cache} (set RECOMPUTE=1 to redo)")
        print(f"  N_H=3: both-quadrature cells {both_quadrature_cells(Z3)}, "
              f"min|X_3| = {np.nanmin(np.abs(Z3)):.2e}")
        print(f"  N_H=7: both-quadrature cells {both_quadrature_cells(Z7)}, "
              f"min|X_3| = {np.nanmin(np.abs(Z7)):.2e}")
        return _render(og, bg, Z3, Z7)
    print("computing the two quadrature maps ...")
    og, bg, Z3 = X3_map(3)
    print(f"  N_H=3 done: valid {np.isfinite(Z3.real).sum()}/{Z3.size}, "
          f"both-quadrature cells {both_quadrature_cells(Z3)}, "
          f"min|X_3| = {np.nanmin(np.abs(Z3)):.2e}")
    og, bg, Z7 = X3_map(7)
    print(f"  N_H=7 done: valid {np.isfinite(Z7.real).sum()}/{Z7.size}, "
          f"both-quadrature cells {both_quadrature_cells(Z7)}, "
          f"min|X_3| = {np.nanmin(np.abs(Z7)):.2e}")
    assert both_quadrature_cells(Z3) == 0, "N_H=3 shows a crossing in the displayed window"
    assert both_quadrature_cells(Z7) >= 1, "N_H=7 shows no crossing in the displayed window"
    np.savez(DATA / "two_dof_k3_existence_maps.npz", og=og, bg=bg, Z3=Z3, Z7=Z7)
    return _render(og, bg, Z3, Z7)


def _render(og, bg, Z3, Z7):
    fig, ax = plt.subplots(1, 3, figsize=fs.full(0.30), constrained_layout=True)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "greys_soft", mpl.colormaps["Greys"](np.linspace(0.0, 0.42, 256)))
    L = np.concatenate([np.log10(np.abs(Z3[np.isfinite(Z3.real)])),
                        np.log10(np.abs(Z7[np.isfinite(Z7.real)]))])
    vmin, vmax = np.percentile(L, 2), np.percentile(L, 98)

    panel(ax[0], og, bg, Z3, cmap, vmin, vmax, "(a)", r"$N_H=3$:  $\{1,3\}$")
    ax[0].set_ylabel(r"on-site cubic $\beta_1$")
    halo = [pe.withStroke(linewidth=1.8, foreground="white")]
    ax[0].text(0.04, 0.70, r"$\mathrm{Re}\,X_3=0$", transform=ax[0].transAxes,
               fontsize=7, color=fs.NAVY, ha="left", va="bottom", path_effects=halo)
    ax[0].text(0.04, 0.31, r"$\mathrm{Im}\,X_3=0$", transform=ax[0].transAxes,
               fontsize=7, color=fs.VERMILLION, ha="left", va="bottom", path_effects=halo)
    mn3 = np.nanmin(np.abs(Z3))
    ax[0].text(0.5, 0.045,
               "no intersection in window\n"
               rf"search min. $|X_3|={mn3:.1e}$".replace("e-0", r"\times10^{-") + "}$"
               if False else
               "no intersection in window\n"
               rf"search min. $|X_3|=8.3\times10^{{-6}}$",
               transform=ax[0].transAxes, fontsize=6.8, ha="center", va="bottom",
               color="0.25", linespacing=1.15)

    im = panel(ax[1], og, bg, Z7, cmap, vmin, vmax, "(b)", r"$N_H=7$")
    ax[1].plot([OM], [B1], **fs.star(markersize=9))
    ax[1].annotate(r"$X_3=0$", xy=(OM, B1), xytext=(10, -13),
                   textcoords="offset points", fontsize=7, color=fs.ACCENT,
                   arrowprops=dict(arrowstyle="-", lw=0.6, color=fs.ACCENT, shrinkB=5))
    ax[1].set_yticklabels([])
    cb = fig.colorbar(im, ax=ax[1], pad=0.02)
    cb.set_label(r"$\log_{10}|X_3|$", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    # --- (c) the homotopy ------------------------------------------------
    A = np.loadtxt(DATA / "two_dof_k3_fifth_harmonic_homotopy.csv", delimiter=",", skiprows=1)
    A = A[A[:, 0] <= 1.0]
    lam, om_l, det_l = A[:, 0], A[:, 1], A[:, 6]
    k = int(lam.argmin())
    c_ax = ax[2]
    c_ax.plot(lam[:k + 1], om_l[:k + 1], color=fs.NAVY, lw=1.4, zorder=3)
    c_ax.plot(lam[k:], om_l[k:], color=fs.AMBER, ls="--", lw=1.4, zorder=3)
    c_ax.plot([lam[k]], [om_l[k]], **fs.star(markersize=8))
    c_ax.plot([1.0], [OM], "o", ms=5.0, mfc="white", mec=fs.NAVY, mew=1.2, zorder=5)
    lo_y, hi_y = om_l.min(), om_l.max()
    pad = 0.16 * (hi_y - lo_y)
    c_ax.set_ylim(lo_y - pad, hi_y + pad)
    c_ax.axvspan(-0.02, lam[k], color="0.85", alpha=0.55, lw=0, zorder=0)
    c_ax.text(lam[k] / 2, hi_y + 0.35 * pad, "tracked pair\nabsent", fontsize=7,
              color="0.35", ha="center", va="center", linespacing=1.0)
    c_ax.annotate(r"$\lambda^\ast=0.4837$", xy=(lam[k], om_l[k]), xytext=(9, 2),
                  textcoords="offset points", fontsize=7, color=fs.ACCENT,
                  ha="left", va="center",
                  arrowprops=dict(arrowstyle="-", lw=0.6, color=fs.ACCENT, shrinkB=4))
    c_ax.set_xlim(0.0, 1.02)
    c_ax.set_xlabel(r"weight $\lambda$ on the $5\Omega$ return")
    c_ax.set_ylabel(r"drive frequency $\Omega$")
    fs.grid(c_ax)
    fs.panel(c_ax, "(c)", dx=-0.26)

    for d in FIGS:
        d.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS[0] / "k3_existence.pdf")
    fig.savefig(FIGS[0] / "k3_existence.png", dpi=400)
    print(f"wrote {FIGS[0] / 'k3_existence.pdf'}")


if __name__ == "__main__":
    main()

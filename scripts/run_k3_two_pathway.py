"""Two-pathway origin of the third-harmonic exact zero.

The third harmonic of oscillator 1 is sourced by two cubic terms in the model:
  (1) the on-site cubic       beta1 * x1^3
  (2) the reaction coupling   kappa_nl * (x1 - x2)^3
Both feed the 3*Omega component of x1. A NONtrivial exact spectral zero,
X3 = (X3^cos, X3^sin) = (0,0), requires the two contributions to cancel.

This script demonstrates that cancellation is a genuine two-pathway
interference, not fine tuning:

  * With kappa_nl = 0 (only source 1) the third harmonic X3(beta1), evaluated
    at the exact-zero frequency Omega*, is a fixed-phase RAY: |X3| grows
    monotonically with beta1 and both quadratures keep their sign. A single
    cubic source cannot cancel itself, so the only zero is the trivial
    beta1 -> 0 limit. A scan over (Omega, beta1) confirms |X3| never
    approaches zero anywhere nontrivial.

  * With kappa_nl = -0.30 (both sources) the second, out-of-phase pathway is
    present. As beta1 increases the two contributions destructively interfere:
    X3(beta1) sweeps THROUGH the origin at beta1* ~ 0.238, both quadratures
    change sign together, and |X3| dips to the floor. This is the nonlinear,
    higher-harmonic analogue of a two-path transmission (Fano-type) zero.

Outputs
  data/two_dof_k3_two_pathway.csv       (beta1 sweep, both kappa_nl, at Omega*)
  figures/k3_two_pathway_interference.{pdf,png}       (wide 1x3, for the PRE figure*)
  figures/k3_two_pathway_interference_col.{pdf,png}   (stacked 3x1, single column, for the Letter)

Run:  PYTHONPATH=src uv run python scripts/run_k3_two_pathway.py
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, least_squares

jax.config.update("jax_enable_x64", True)

from hh_antiresonance import figstyle as fs

fs.use()

from hh_antiresonance.harmonic_balance import (
    solve_harmonic_balance, target_harmonic_indices,
)
from hh_antiresonance.models import CoupledOscillatorParams

DATADIR = Path("data"); FIGDIR = Path("figures")
DATADIR.mkdir(exist_ok=True); FIGDIR.mkdir(exist_ok=True)

# ---- fixed working point ----
N_H, N_T = 7, 512
F_FIX, KA0 = 0.30, 0.10
OM_STAR, B1_STAR = 0.259728097, 0.238154276     # exact zero at kappa_nl=-0.30
KNL_ON = -0.30

BASE = CoupledOscillatorParams(
    omega1=1.0, omega2=1.25, zeta1=0.015, zeta2=0.02,
    alpha1=0.0, alpha2=0.0, beta1=0.15, beta2=0.0,
    kappa=KA0, force=F_FIX, drive_omega=OM_STAR, kappa_nl=0.0,
)
I3C, I3S = target_harmonic_indices(oscillator=1, harmonic=3, n_harmonics=N_H)
I1C, I1S = target_harmonic_indices(oscillator=1, harmonic=1, n_harmonics=N_H)

# Full-HB exact zeros at kappa=0.10
NUM_ARMS = np.array([[0.259728, 0.238154], [0.2515, 0.2402]])


def analytic_beta1_curves(n=60):
    """Leading-order design curve beta1(Omega) that nulls Re X3 at fixed
    (kappa, F, kappa_nl), from the two-pathway condition
        (L2+kappa) beta1 X1^3 + kappa_nl L2 (X1-X2)^3 = 0,   L2 = w2^2-9 Om^2+2i z2(3 Om).
    X1, X2 are the fundamental amplitudes; both sources are filtered through the
    linear 2-DOF response at 3*Omega. Returns (Omega grid, beta1 from the linear
    fundamental, beta1 from a single-harmonic nonlinear fundamental).
    """
    w2, z2 = 1.25, 0.02
    kap, knl, F = KA0, KNL_ON, F_FIX

    def a(j, Om):
        wj, zj = (1.0, 0.015) if j == 1 else (w2, z2)
        return wj**2 + kap - Om**2 + 2j * zj * Om

    def L2(Om):
        return w2**2 - 9 * Om**2 + 2j * z2 * (3 * Om)

    def lin(Om):
        a1, a2 = a(1, Om), a(2, Om)
        det = a1 * a2 - kap**2
        return F * a2 / det, F * kap / det

    def nl(Om, b1):
        a1, a2 = a(1, Om), a(2, Om)

        def res(v):
            X1 = v[0] + 1j * v[1]
            X2 = v[2] + 1j * v[3]
            U = X1 - X2
            f1 = a1 * X1 - kap * X2 + 0.75 * b1 * abs(X1)**2 * X1 \
                + 0.75 * knl * abs(U)**2 * U - F
            f2 = a2 * X2 - kap * X1 - 0.75 * knl * abs(U)**2 * U
            return [f1.real, f1.imag, f2.real, f2.imag]

        x0 = lin(Om)
        s = least_squares(res, [x0[0].real, x0[0].imag, x0[1].real, x0[1].imag],
                          xtol=1e-14, ftol=1e-14)
        return s.x[0] + 1j * s.x[1], s.x[2] + 1j * s.x[3]

    def reN(Om, b1, fund):
        X1, X2 = fund(Om, b1) if fund is nl else fund(Om)
        U = X1 - X2
        return ((L2(Om) + kap) * b1 * X1**3 + knl * L2(Om) * U**3).real

    Oms = np.linspace(0.244, 0.276, n)
    b_lin = np.array([brentq(lambda b: reN(O, b, lin), 0.05, 0.60) for O in Oms])
    b_nl = np.array([brentq(lambda b: reN(O, b, nl), 0.05, 0.60) for O in Oms])
    return Oms, b_lin, b_nl


def log(m: str) -> None:
    print(m, flush=True)


def x3_of(Om, b1, knl):
    """Return (X3^cos, X3^sin, |X1|) of oscillator 1 from an HB solve."""
    p = replace(BASE, drive_omega=float(Om), beta1=float(b1), kappa=KA0,
                force=F_FIX, kappa_nl=float(knl))
    c = np.asarray(solve_harmonic_balance(p, n_harmonics=N_H, n_time_samples=N_T,
                                          tol=1e-13, max_nfev=800), float)
    return c[I3C], c[I3S], float(np.hypot(c[I1C], c[I1S]))


def main():
    log("=== two-pathway origin of the k=3 exact zero ===")
    log(f"fixed: Omega*={OM_STAR:.6f}, kappa={KA0}, F={F_FIX}, N_H={N_H}")

    # beta1 grid: dense cluster around beta1* so the interference dip resolves
    grid = np.unique(np.concatenate([
        np.linspace(0.02, 0.45, 44),
        np.linspace(0.225, 0.252, 28),
        [B1_STAR],
    ]))

    rows = []
    log(f"\n{'beta1':>8} | {'kNL=0  ReX3':>12}{'ImX3':>11}{'|X3|':>10} "
        f"| {'kNL=-.30 ReX3':>13}{'ImX3':>11}{'|X3|':>10}")
    for b1 in grid:
        c0, s0, a1_0 = x3_of(OM_STAR, b1, 0.0)
        cN, sN, a1_N = x3_of(OM_STAR, b1, KNL_ON)
        rows.append((b1, c0, s0, np.hypot(c0, s0), a1_0,
                     cN, sN, np.hypot(cN, sN), a1_N))
        if b1 in grid[::6] or abs(b1 - B1_STAR) < 1e-9:
            log(f"{b1:8.4f} | {c0:12.3e}{s0:11.3e}{np.hypot(c0,s0):10.2e} "
                f"| {cN:13.3e}{sN:11.3e}{np.hypot(cN,sN):10.2e}")
    R = np.array(rows)

    # sign-change diagnostics at Omega*
    def nsign(col):
        s = np.sign(R[:, col]); s = s[s != 0]
        return int(np.sum(np.abs(np.diff(s)) > 0))
    log("\nsign changes at Omega* over the beta1 sweep:")
    log(f"  kappa_nl= 0.00 : ReX3 -> {nsign(1)}, ImX3 -> {nsign(2)}  "
        f"(min|X3|={R[:,3].min():.2e})   [ray: no nontrivial zero]")
    log(f"  kappa_nl=-0.30 : ReX3 -> {nsign(5)}, ImX3 -> {nsign(6)}  "
        f"(min|X3|={R[:,7].min():.2e})   [origin crossing: exact zero]")

    # persist
    np.savetxt(DATADIR / "two_dof_k3_two_pathway.csv", R, delimiter=",",
               header=("beta1,ReX3_knl0,ImX3_knl0,absX3_knl0,absX1_knl0,"
                       "ReX3_knlON,ImX3_knlON,absX3_knlON,absX1_knlON"),
               comments="")

    # ---------------------- figures ----------------------
    # The same three panels are emitted twice: a wide 1x3 layout for the PRE
    # (used at \textwidth in a figure*) and a stacked 3x1 layout for the Letter
    # (used at \columnwidth). Rescaling the wide render to one column makes the
    # fonts ~30% of design size and illegible; the stacked layout renders each
    # panel at full column width instead.
    Oms, b_lin, b_nl = analytic_beta1_curves()
    s0, s1 = fs.series(0), fs.series(1)   # navy solid o / amber dashed s

    def draw_a(ax):
        # |X3| vs beta1: single-source ray (no nontrivial zero) vs two-source dip.
        # No text annotation: the caption names beta1* and the dotted line marks it.
        # markevery thins the markers over the deliberately dense beta1 cluster;
        # the line itself still resolves the dip.
        ax.semilogy(R[:, 0], R[:, 3], **dict(s0, markevery=4, ms=3.0),
                    label=r"one ($\kappa_{\rm nl}\!=\!0$)")
        ax.semilogy(R[:, 0], R[:, 7], **dict(s1, markevery=4, ms=3.0),
                    label=r"two ($\kappa_{\rm nl}\!=\!-0.30$)")
        ax.axvline(B1_STAR, color=fs.REFERENCE, ls=":", lw=0.7)
        ax.set_xlabel(r"$\beta_1$")
        ax.set_ylabel(r"$|X_3|$ at $\Omega^{\ast}$")
        # both traces sit high across the panel; the dip at beta1* leaves the
        # lower-left free
        ax.legend(loc="lower left", fontsize=7, frameon=False,
                  handlelength=1.4, handletextpad=0.5)
        fs.grid(ax, which="both")

    def draw_b(ax):
        # both quadratures cross zero together at beta1* (the codim-2 signature).
        # Twin axes because Im is ~12x smaller than Re; each y-axis is coloured and
        # each series also carries its own linestyle+marker, so the pairing survives
        # a black-and-white print.
        ax.axhline(0, color="0.6", lw=0.6)
        ax.axvline(B1_STAR, color=fs.REFERENCE, ls=":", lw=0.7)
        l1, = ax.plot(R[:, 0], R[:, 5] * 1e3, **dict(s0, markevery=4, ms=3.0))
        ax.set_xlabel(r"$\beta_1$")
        ax.set_ylabel(r"$\mathrm{Re}\,X_3\ (\times10^{-3})$", color=fs.NAVY)
        ax.tick_params(axis="y", labelcolor=fs.NAVY)
        axt = ax.twinx()
        axt.tick_params(axis="y", labelcolor=fs.AMBER, right=True)
        l2, = axt.plot(R[:, 0], R[:, 6] * 1e3, **dict(s1, markevery=4, ms=3.0))
        axt.set_ylabel(r"$\mathrm{Im}\,X_3\ (\times10^{-3})$", color=fs.AMBER)
        ax.set_xlim(0.02, 0.45)
        # Align the twin-axis zeros exactly: both axes symmetric about zero, so
        # the shared zero line is truthful for both quadratures and the common
        # crossing at beta1* is a statement about the data, not the autoscale.
        m1 = 1.06 * float(np.max(np.abs(R[:, 5]))) * 1e3
        m2 = 1.06 * float(np.max(np.abs(R[:, 6]))) * 1e3
        ax.set_ylim(-m1, m1); axt.set_ylim(-m2, m2)
        ax.legend([l1, l2], [r"$\mathrm{Re}\,X_3$", r"$\mathrm{Im}\,X_3$"],
                  loc="upper right", title=r"$\kappa_{\rm nl}=-0.30$",
                  title_fontsize=8, fontsize=7, frameon=False,
                  handlelength=1.4, handletextpad=0.5)
        fs.grid(ax)

    def draw_c(ax):
        # analytic design curve: beta1(Omega) that nulls X3 vs the
        # two full-HB exact zeros. Reverse-designs the zero: pick Omega, read beta1.
        ax.plot(Oms, b_lin, ls="--", color=fs.REFERENCE, lw=1.0,
                label="linear fund.")
        ax.plot(Oms, b_nl, ls="-", color=fs.NAVY, lw=1.2, label="nonlinear fund.")
        ax.plot(NUM_ARMS[:, 0], NUM_ARMS[:, 1], **fs.star(markersize=10),
                label="HB zeros")
        ax.set_xlabel(r"$\Omega$")
        ax.set_ylabel(r"$\beta_1$ nulling $X_3$")
        # HB zeros sit at beta1 = 0.2382 and 0.2402, both at Omega <= 0.26, so a
        # compact upper-right legend clears them without inflating the ceiling:
        # the axis closes just above the data instead of framing the legend.
        ax.set_ylim(0.2195, 0.2435)
        ax.legend(loc="upper right", fontsize=7, frameon=False, handlelength=1.2,
                  handletextpad=0.4, labelspacing=0.25, borderaxespad=0.3)
        fs.grid(ax)

    # wide 1x3 (PRE figure*, authored at full text width)
    fig, axes = plt.subplots(1, 3, figsize=fs.full(0.30), constrained_layout=True)
    for ax, draw, lab in zip(axes, (draw_a, draw_b, draw_c), ("(a)", "(b)", "(c)")):
        draw(ax)
        fs.panel(ax, lab, dx=-0.24)
    fs.save(fig, FIGDIR, "k3_two_pathway_interference")

    # stacked 3x1 (Letter, authored at single-column width)
    figc, axesc = plt.subplots(3, 1, figsize=fs.column(1.75),
                               constrained_layout=True)
    for ax, draw, lab in zip(axesc, (draw_a, draw_b, draw_c), ("(a)", "(b)", "(c)")):
        draw(ax)
        fs.panel(ax, lab, dx=-0.17)
    fs.save(figc, FIGDIR, "k3_two_pathway_interference_col")

    log("\nWrote:")
    log("  data/two_dof_k3_two_pathway.csv")
    log("  figures/k3_two_pathway_interference.{pdf,png}      (wide, PRE)")
    log("  figures/k3_two_pathway_interference_col.{pdf,png}  (stacked, Letter)")


if __name__ == "__main__":
    main()

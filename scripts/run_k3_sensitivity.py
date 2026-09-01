"""Sensitivity geometry for the exact zero. Three panels:

  (a) tolerance ellipses in the (dOmega, dbeta1) control plane at 20/40/60 dB
      suppression, from the SVD of the control Jacobian (anisotropic: the drive
      frequency is the soft direction);
  (b) suppression S(dB) vs |mismatch| in beta1 and kappa_nl (log-x -> the
      decade-per-20 dB linear law of a transversal zero);
  (c) Floquet spectral radius across the admissible drive band F in [0.26,0.35]
      (the exact zero stays asymptotically stable along the branch).

Run:  JAX_ENABLE_X64=1 uv run python scripts/run_k3_sensitivity.py
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
from scipy.optimize import least_squares

from hh_antiresonance.models import CoupledOscillatorParams
from hh_antiresonance.harmonic_balance import solve_harmonic_balance, harmonic_balance_residual, coefficient_index
from hh_antiresonance.stability import floquet_multipliers_from_coeffs

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


def ab3(p, g=None):
    c = solve(p, g); return np.array([c[IC], c[ISx]]), c


def amp3(p, g=None):
    v, _ = ab3(p, g); return float(np.hypot(*v))


def renull_F(F, seed):
    """free (Omega, beta1) so X3=0 at fixed F."""
    def r(z):
        c = z[:4 * N]
        p = replace(BASE, drive_omega=float(z[-2]), beta1=float(z[-1]), force=float(F))
        return np.concatenate([harmonic_balance_residual(c, p, n_harmonics=N, n_time_samples=NT), [c[IC], c[ISx]]])
    s = least_squares(r, seed, xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=8000)
    return s.x


def main():
    from pathlib import Path
    from hh_antiresonance import figstyle as fs
    fs.use()
    FD = Path(FIGDIR)
    c0 = solve(BASE)
    X30 = amp3(replace(BASE, kappa_nl=0.0), c0)   # single-pathway reference
    # Authored at its placed width: \textwidth in a figure* (no LaTeX rescaling).
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=fs.full(0.34), constrained_layout=True)

    # ---- (a) tolerance ellipses from the control Jacobian SVD ----
    h = 1e-5
    J = np.column_stack([(ab3(replace(BASE, drive_omega=OM + h))[0] - ab3(replace(BASE, drive_omega=OM - h))[0]) / (2 * h),
                         (ab3(replace(BASE, beta1=B1 + h))[0] - ab3(replace(BASE, beta1=B1 - h))[0]) / (2 * h)])
    U, sv, Vt = np.linalg.svd(J)
    th = np.linspace(0, 2 * np.pi, 400)
    circ = np.array([np.cos(th), np.sin(th)])
    Jinv = np.linalg.inv(J)
    ells = {}
    for S in (20, 40, 60):
        t = X30 * 10 ** (-S / 20.0)
        ells[S] = Jinv @ (t * circ)                # {d : |J d| = t}
    # principal axes of the 20 dB ellipse (soft / stiff control directions)
    t20 = X30 * 10 ** (-1.0)
    for i in (0, 1):
        v = Vt[i] * (t20 / sv[i])
        axA.plot([-v[0], v[0]], [-v[1], v[1]], color="0.75", lw=0.6, ls=":", zorder=1)
    for S, ls in ((20, "-"), (40, "--"), (60, ":")):
        axA.plot(ells[S][0], ells[S][1], ls=ls, color=fs.NAVY, lw=1.1, label=f"{S} dB")
    axA.plot(0, 0, "+", color="black", ms=6, mew=1.0)
    xr = 1.15 * float(np.max(np.abs(ells[20][0]))); yr = 1.15 * float(np.max(np.abs(ells[20][1])))
    axA.set_xlim(-xr, xr); axA.set_ylim(-yr, yr)
    # x10 zoom inset: the 40/60 dB ellipses repeat the 20/40 dB pattern exactly
    # (the transversal zero is self-similar: one decade of tolerance per 20 dB).
    axin = axA.inset_axes([0.04, 0.05, 0.32, 0.32])
    for S, ls in ((40, "--"), (60, ":")):
        axin.plot(ells[S][0], ells[S][1], ls=ls, color=fs.NAVY, lw=0.9)
    axin.set_xlim(-xr / 10, xr / 10); axin.set_ylim(-yr / 10, yr / 10)
    axin.set_xticks([]); axin.set_yticks([])
    for sp in axin.spines.values():
        sp.set_color("0.6"); sp.set_linewidth(0.5)
    # upper-right corner: the ellipse major axis runs upper-left to lower-right,
    # so the upper-left corner (where this label used to sit) is occupied.
    axin.text(0.95, 0.92, r"$\times 10$", transform=axin.transAxes, fontsize=8,
              color="0.35", ha="right", va="top")
    fs.factor_axis(axA, -1, r"$\delta\Omega$", axis="x")
    fs.factor_axis(axA, -2, r"$\delta\beta_1$", axis="y")
    axA.legend(loc="lower center", bbox_to_anchor=(0.55, 1.0), ncol=3,
               frameon=False, fontsize=8, handlelength=1.5,
               columnspacing=1.0, handletextpad=0.4)
    fs.grid(axA); fs.panel(axA, "(a)")

    # ---- (b) suppression vs fractional mismatch ----
    # Plot against |delta|/|coefficient| so the Letter's "percent-level -> 40 dB"
    # reads directly off the axis (delta_beta1 = 0.9% and delta_knl = 0.85% at 40 dB).
    ds = np.logspace(-4, -1.3, 40)
    for (name, key, base_val), si in [((r"$\beta_1$", "beta1", B1), 0),
                                      ((r"$\kappa_{\rm nl}$", "kappa_nl", KNL), 1)]:
        S = []
        g = c0
        for d in ds:
            g = solve(replace(BASE, **{key: float(base_val + d)}), g)
            S.append(20 * np.log10(X30 / np.hypot(g[IC], g[ISx])))
        c, ls, mk = fs.SERIES[si]
        axB.semilogx(ds / abs(base_val), S, ls=ls, color=c, lw=1.1,
                     marker=mk, ms=3.0, markevery=4, label=name)
    for lvl in (20, 40, 60):
        axB.axhline(lvl, color="0.87", lw=0.4, zorder=0)
    axB.axvline(1e-2, color=fs.REFERENCE, lw=0.7, ls=":", zorder=1)
    axB.text(1.25e-2, 2, "1%", ha="left", va="bottom", fontsize=8, color="0.35")
    axB.set_xlabel(r"fractional mismatch $|\delta|$"); axB.set_ylabel("suppression (dB)")
    axB.set_ylim(0, 75)
    axB.legend(loc="upper right", fontsize=8)
    fs.grid(axB); fs.panel(axB, "(b)")

    # ---- (c) Floquet across the admissible band ----
    Fs = np.linspace(0.26, 0.35, 10)
    z = np.concatenate([c0, [OM, B1]]); rho = []
    for F in Fs:
        z = renull_F(F, z)
        p = replace(BASE, drive_omega=float(z[-2]), beta1=float(z[-1]), force=float(F))
        rep = floquet_multipliers_from_coeffs(p, z[:4 * N], n_harmonics=N)
        rho.append(float(np.max(np.abs(rep.multipliers))))
    axC.plot(Fs, rho, ls="-", color=fs.NAVY, lw=1.1, marker="o", ms=3.0)
    axC.axhline(1.0, color=fs.VERMILLION, lw=0.9, ls="--")
    axC.text(0.349, 1.005, "unit circle", ha="right", va="bottom",
             fontsize=8, color=fs.VERMILLION)
    axC.set_ylim(0.6, 1.08)
    axC.set_xlabel("drive amplitude $F$"); axC.set_ylabel(r"Floquet radius $\rho$")
    fs.grid(axC); fs.panel(axC, "(c)")

    fs.save(fig, FD, "k3_sensitivity")
    print(f"X3_single={X30:.3e}, sigma(J)={sv}, cond={sv[0]/sv[-1]:.2f}")
    print(f"Floquet rho over F[0.26,0.35]: {np.min(rho):.3f}..{np.max(rho):.3f}")
    print("wrote k3_sensitivity")


if __name__ == "__main__":
    main()

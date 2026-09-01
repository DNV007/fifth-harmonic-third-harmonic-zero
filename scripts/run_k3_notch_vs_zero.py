"""A notch and a zero on the same axes, along the experimental sweep.

The distinction the Letter opens with is reachability in control space, not
suppression depth, and it is easiest to see in the sweep an experiment actually
performs: hold the coefficients and sweep the drive frequency.

  * one pathway (kappa_nl = 0): a finite notch, minimum ~3e-3;
  * two pathways with the on-site cubic detuned by +-1% from beta_1^*: still a
    finite notch, four orders deeper but bounded;
  * two pathways at beta_1 = beta_1^*: the sweep passes through the root and
    |X_3| falls to the nonlinear-solver floor at Omega = Omega^*.

The last curve is not what a generic sweep finds. It is what a sweep finds when
the second control is already at the value that puts the zero on the sweep line,
which is the content of the codimension-two statement. The depth reached there
is a solver floor, not a physical null depth.

Run:  JAX_ENABLE_X64=1 PYTHONPATH=src uv run python scripts/run_k3_notch_vs_zero.py
"""
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
from dataclasses import replace
from pathlib import Path

import numpy as np
from jax import config

config.update("jax_enable_x64", True)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hh_antiresonance import figstyle as fs
from hh_antiresonance.harmonic_balance import (
    coefficient_index, solve_harmonic_balance,
)
from hh_antiresonance.models import CoupledOscillatorParams

OM, B1, KNL, F0 = 0.25972810, 0.23815428, -0.30, 0.30
BASE = CoupledOscillatorParams(
    omega1=1.0, omega2=1.25, zeta1=0.015, zeta2=0.02, alpha1=0.0, alpha2=0.0,
    beta1=B1, beta2=0.0, kappa=0.10, force=F0, drive_omega=OM, kappa_nl=KNL)
N, NT = 7, 512
IC = coefficient_index(oscillator=1, harmonic=3, component="cos", n_harmonics=N)
ISx = coefficient_index(oscillator=1, harmonic=3, component="sin", n_harmonics=N)

_D = Path(os.path.dirname(os.path.abspath(__file__)))
FIGS = [_D / ".." / "figures"]
DATA = _D / ".." / "data"


def solve(p, g=None):
    return np.asarray(solve_harmonic_balance(p, n_harmonics=N, initial_guess=g,
                                             n_time_samples=NT, tol=1e-14,
                                             max_nfev=6000), float)


def absX3(c):
    return float(np.hypot(c[IC], c[ISx]))


def sweep(oms, beta1, knl):
    """|X_3| along a frequency sweep at fixed coefficients, warm-started."""
    out = np.empty(len(oms))
    g = solve(replace(BASE, drive_omega=float(oms[0]), beta1=beta1, kappa_nl=knl))
    for i, om in enumerate(oms):
        g = solve(replace(BASE, drive_omega=float(om), beta1=beta1, kappa_nl=knl), g)
        out[i] = absX3(g)
    return out


def main():
    fs.use()
    half = 0.010
    oms = np.unique(np.concatenate([
        np.linspace(OM - half, OM + half, 161),
        OM + np.geomspace(1e-6, half, 40),            # resolve the cusp at the root
        OM - np.geomspace(1e-6, half, 40),
        [OM],
    ]))

    curves = {
        "one pathway": (B1, 0.0),
        r"two pathways, $\beta_1$ off by $1\%$": (B1 * 1.01, KNL),
        r"two pathways, $\beta_1=\beta_1^\ast$": (B1, KNL),
    }
    res = {}
    for lab, (b, k) in curves.items():
        res[lab] = sweep(oms, float(b), float(k))
        print(f"{lab:<40} min |X3| = {res[lab].min():.3e} at "
              f"Omega = {oms[np.argmin(res[lab])]:.7f}")

    fig, ax = plt.subplots(figsize=(3.17, 2.60), constrained_layout=True)
    keys = list(res)
    for i, lab in enumerate(keys):
        col, ls, mk = fs.SERIES[i]
        ax.semilogy(oms, np.maximum(res[lab], 1e-15), color=col, linestyle=ls,
                    lw=1.3, marker=mk, ms=2.6, markevery=23)
    ax.set_xlabel(r"drive frequency $\Omega$")
    ax.set_ylabel(r"$|X_3|$")
    ax.set_ylim(6e-12, 6e-2)
    ax.set_xlim(oms[0], oms[-1])

    # The three curves are far apart in |X_3| and each occupies its own band, so
    # they are labelled where they run: a legend box would have to sit on the
    # plunge, which is the one feature the panel exists to show.
    ax.text(0.2508, 9e-3, "one pathway", color=fs.SERIES[0][0], fontsize=7,
            ha="left", va="bottom")
    ax.text(0.2645, 7e-5, r"two pathways, $\beta_1$ off by $1\%$",
            color=fs.SERIES[1][0], fontsize=7, ha="right", va="bottom")
    ax.text(0.2519, 4e-7, r"tuned, $\beta_1=\beta_1^\ast$", color=fs.SERIES[2][0],
            fontsize=7, ha="left", va="center")
    ax.text(OM, 1.15e-11, r"$\Omega^\ast$", fontsize=8, ha="center", va="bottom",
            color="0.25")
    fs.grid(ax)
    # Lower panel of the paired branch figure; the upper panel is k3_branch_vs_F.
    fs.panel(ax, "(b)", dx=-0.20)
    for d in FIGS:
        Path(d).mkdir(parents=True, exist_ok=True)
    fs.save(fig, Path(FIGS[0]), "k3_notch_vs_zero")
    # Copy rather than re-render: with constrained_layout a second savefig after
    # the 600 dpi PNG re-solves the layout and lands sub-pixel away, so the
    # deposited figure and the submitted one would not be the same file.
    DATA.mkdir(exist_ok=True)
    np.savetxt(DATA / "two_dof_k3_notch_vs_zero.csv",
               np.column_stack([oms] + [res[k] for k in curves]), delimiter=",",
               comments="", header="omega,one_pathway,two_pathways_beta1_plus1pct,"
                                   "two_pathways_tuned")
    print(f"wrote {FIGS[0]}/k3_notch_vs_zero.pdf")


if __name__ == "__main__":
    main()

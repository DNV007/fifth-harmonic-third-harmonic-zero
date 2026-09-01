"""Quadratic-perturbation and gate-path results with the mean modes restored.

The production harmonic-balance basis carries harmonics k = 1..N_H only. That is
exact for the baseline root, whose orbit is half-period antisymmetric and so has
zero mean, but it is NOT a solution basis once a quadratic term is present:
averaging the equations of motion over one period gives

    0 = (w1^2 + ka) xbar1 - ka xbar2 + alpha1 <x1^2> + beta1 <x1^3> + knl <d^3>
    0 = (w2^2 + ka) xbar2 - ka xbar1 + alpha2 <x2^2> + beta2 <x2^3> - knl <d^3>

and alpha1 <x1^2> is strictly nonzero, so both means must be nonzero. This driver
repeats the two affected calculations with include_mean=True:

  (A) the alpha1 = 0.05 quadratic-perturbation point;
  (B) the electrostatic gate path, including
      the constant electrostatic force -g*dg^3/4 that accompanies the tied
      (w1^2, alpha1, beta1) shifts and which the coefficient-only path drops.

Run:  JAX_ENABLE_X64=1 uv run python scripts/run_k3_mean_mode_corrections.py
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np
from dataclasses import replace
from pathlib import Path
from jax import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from scipy.optimize import least_squares

from hh_antiresonance.models import CoupledOscillatorParams
from hh_antiresonance.harmonic_balance import (
    solve_harmonic_balance, harmonic_balance_residual,
    coefficient_index, coefficient_count, mean_index,
)

OM_P, B1_P = 0.25972810, 0.23815428
BASE = CoupledOscillatorParams(omega1=1.0, omega2=1.25, zeta1=0.015, zeta2=0.02,
                               alpha1=0.0, alpha2=0.0, beta1=B1_P, beta2=0.0,
                               kappa=0.10, force=0.30, drive_omega=OM_P, kappa_nl=-0.30)
N, NT = 7, 2048
DG = 1.0                                   # gate gap in displacement-scale units

i3c = coefficient_index(oscillator=1, harmonic=3, component="cos", n_harmonics=N)
i3s = coefficient_index(oscillator=1, harmonic=3, component="sin", n_harmonics=N)
m1 = mean_index(oscillator=1, n_harmonics=N)
m2 = mean_index(oscillator=2, n_harmonics=N)
amp = lambda c, o, h: float(np.hypot(
    c[coefficient_index(oscillator=o, harmonic=h, component="cos", n_harmonics=N)],
    c[coefficient_index(oscillator=o, harmonic=h, component="sin", n_harmonics=N)]))
_D = Path(os.path.dirname(os.path.abspath(__file__)))
OUT = _D / ".." / "data"


def solve(p, mean, fdc=0.0, guess=None):
    return np.asarray(solve_harmonic_balance(
        p, n_harmonics=N, initial_guess=guess, n_time_samples=NT, tol=1e-13,
        max_nfev=8000, include_mean=mean, force_dc=fdc), float)


def renull(p0, mean, free, fdc_of=None, guess=None):
    """Augmented solve: balance + Re X3 = Im X3 = 0, free two controls."""
    nc = coefficient_count(N, include_mean=mean)

    def resid(z):
        c, u0, u1 = z[:nc], z[nc], z[nc + 1]
        p, fdc = free(p0, u0, u1)
        if fdc_of is not None:
            fdc = fdc_of(u1)
        r = harmonic_balance_residual(c, p, n_harmonics=N, n_time_samples=NT,
                                      include_mean=mean, force_dc=fdc)
        return np.concatenate([r, [c[i3c], c[i3s]]])

    z0 = np.concatenate([guess, [OM_P, 0.0]]) if guess is not None else None
    lo = np.full(nc + 2, -np.inf); hi = np.full(nc + 2, np.inf)
    lo[nc], hi[nc] = 0.15, 0.45
    s = least_squares(resid, z0, bounds=(lo, hi), xtol=1e-14, ftol=1e-14,
                      gtol=1e-14, max_nfev=20000)
    return s.x, float(np.linalg.norm(resid(s.x)))


print("=" * 78)
print("(A) QUADRATIC PERTURBATION alpha1 = 0.05")
print("=" * 78)
rows_a = []
for mean, tag in ((False, "zero-mean comparison"), (True, "mean modes included")):
    nc = coefficient_count(N, include_mean=mean)

    def free(p0, om, b1, _m=mean):
        return replace(p0, drive_omega=float(om), beta1=float(b1), alpha1=0.05), 0.0

    seed = solve(replace(BASE, alpha1=0.05), mean)
    z, r = renull(BASE, mean, free, guess=seed)
    c = z[:nc]
    mu1 = c[m1] if mean else 0.0
    mu2 = c[m2] if mean else 0.0
    print(f"  {tag:30s} Omega*={z[nc]:.9f}  beta1*={z[nc+1]:.9f}")
    print(f"  {'':30s} <x1>={mu1:+.4e}  <x2>={mu2:+.4e}")
    print(f"  {'':30s} |X3|={amp(c,1,3):.2e}  |X2|={amp(c,1,2):.4e}  |res|={r:.1e}")
    rows_a.append((tag, z[nc], z[nc + 1], mu1, mu2, amp(c, 1, 3), amp(c, 1, 2)))
    if not mean:
        z_nomean, c_nomean = z, c

# score the zero-mean solution against the full system, mean equations included
c_pad = np.concatenate([c_nomean, [0.0, 0.0]])
p_bad = replace(BASE, drive_omega=float(z_nomean[4 * N]),
                beta1=float(z_nomean[4 * N + 1]), alpha1=0.05)
r_full = harmonic_balance_residual(c_pad, p_bad, n_harmonics=N, n_time_samples=NT,
                                   include_mean=True)
print(f"\n  zero-mean comparison re-scored in the full system: mean-equation residuals "
      f"= ({r_full[-2]:+.3e}, {r_full[-1]:+.3e})")
print(f"  its constrained |X3| was {amp(c_nomean,1,3):.2e} -- the mean "
      f"balance is not measured by that number.")

print()
print("=" * 78)
print("(B) SECOND-HARMONIC SLOPE  A2/A1 vs alpha1   [Letter End Matter]")
print("=" * 78)
for mean, tag in ((False, "zero-mean comparison"), (True, "mean modes included")):
    al, ratios = [], []
    for a1 in (0.01, 0.02, 0.03, 0.04, 0.05):
        c = solve(replace(BASE, alpha1=a1), mean)
        al.append(a1); ratios.append(amp(c, 1, 2) / amp(c, 1, 1))
    print(f"  {tag:30s} fitted slope A2/A1 = {np.polyfit(al, ratios, 1)[0]:.4f}")

print()
print("=" * 78)
print("(C) GATE PATH, free (Omega, g), constant electrostatic force included")
print("=" * 78)
print(f"  {'F':>6} {'basis':>28} {'Omega*':>13} {'g*':>13} {'<x1>':>12} {'|X3|':>9}")
rows_c = []
for F in (0.34, 0.32, 0.30, 0.28, 0.27):
    for mean, with_const, tag in ((False, False, "no mean, no const force"),
                                  (True, True, "mean + const force")):
        nc = coefficient_count(N, include_mean=mean)

        def free(p0, om, g, _F=F):
            w1 = float(np.sqrt(1.0 + (DG ** 2 / 2.0) * g))
            return replace(p0, drive_omega=float(om), force=float(_F),
                           omega1=w1, alpha1=float((3 * DG / 4.0) * g),
                           beta1=float(B1_P + g)), 0.0

        fdc = (lambda g: float(-g * DG ** 3 / 4.0)) if with_const else None
        seed = solve(replace(BASE, force=F), mean)
        z, r = renull(BASE, mean, free, fdc_of=fdc, guess=seed)
        c = z[:nc]
        mu1 = c[m1] if mean else 0.0
        print(f"  {F:>6.2f} {tag:>28} {z[nc]:>13.8f} {z[nc+1]:>+13.6e} "
              f"{mu1:>+12.4e} {amp(c,1,3):>9.1e}")
        rows_c.append((F, tag, z[nc], z[nc + 1], mu1, amp(c, 1, 3)))

OUT.mkdir(exist_ok=True)
np.savetxt(OUT / "two_dof_k3_mean_mode_alpha1.csv",
           np.array([[r[1], r[2], r[3], r[4], r[5], r[6]] for r in rows_a]),
           delimiter=",", comments="",
           header="omega_star,beta1_star,mean_x1,mean_x2,abs_X3,abs_X2")
np.savetxt(OUT / "two_dof_k3_mean_mode_gate.csv",
           np.array([[r[0], r[2], r[3], r[4], r[5]] for r in rows_c]),
           delimiter=",", comments="",
           header="F,omega_star,g_star,mean_x1,abs_X3")
print(f"\nwrote {OUT/'two_dof_k3_mean_mode_alpha1.csv'}")
print(f"wrote {OUT/'two_dof_k3_mean_mode_gate.csv'}")

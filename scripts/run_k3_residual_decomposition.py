"""Finite-amplitude closure of the k=3 exact zero: fifth-harmonic residual decomposition.

Quantifies the complex balance behind the exact third-harmonic zero (Letter P6,
End Matter EM-3, SI Sec. S2). At the converged working point the third-harmonic
balance of oscillator 1 vanishes. Removing the fifth-and-higher harmonics from
the converged orbit leaves a residual in the two quadratures; the cubic mixes the
fifth harmonic back into the third (5*Omega with two -Omega sum to 3*Omega) and
supplies the equal and opposite, closing both quadratures to the solver floor.

Reproduces:
  * residual with fifth+ removed     : (-6.5, +6.0)e-5   (cos, sin projections)
  * fifth-harmonic contribution      : (+6.5, -6.0)e-5   (equal and opposite)
  * seventh-harmonic contribution    : < 1e-9  (closure is a fifth-harmonic effect)
  * full converged residual          : ~1e-18 (both quadratures)
  * A5/|X1| = 2.9e-4, A7/|X1| = 1.2e-5

The reported pair are the raw harmonic-balance residual components (cos, sin
projections of the oscillator-1 equation at 3*Omega); they carry the quadrature
weight T/2 of the collocation projection.
"""
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np
from jax import config

config.update("jax_enable_x64", True)

from hh_antiresonance.models import CoupledOscillatorParams
from hh_antiresonance.harmonic_balance import (
    solve_harmonic_balance,
    harmonic_balance_residual,
    coefficient_index,
)

# Baseline parameters + k=3 exact-zero working point
OM_STAR = 0.25972810
B1_STAR = 0.23815428
P = CoupledOscillatorParams(
    omega1=1.0, omega2=1.25, zeta1=0.015, zeta2=0.02,
    alpha1=0.0, alpha2=0.0, beta1=B1_STAR, beta2=0.0,
    kappa=0.10, force=0.30, drive_omega=OM_STAR, kappa_nl=-0.30,
)
N_H = 7
N_TIME = 2048  # dense enough that the 5-1-1 -> 3 cubic mixing is not aliased


def _amp(coeffs, osc, h):
    ci = coefficient_index(oscillator=osc, harmonic=h, component="cos", n_harmonics=N_H)
    si = coefficient_index(oscillator=osc, harmonic=h, component="sin", n_harmonics=N_H)
    return float(np.hypot(coeffs[ci], coeffs[si]))


def _r3(coeffs):
    """Raw (cos, sin) projections of the oscillator-1 balance residual at 3*Omega."""
    r = harmonic_balance_residual(coeffs, P, n_harmonics=N_H, n_time_samples=N_TIME)
    return np.array([r[4 * (3 - 1) + 0], r[4 * (3 - 1) + 1]])  # osc1, harmonic 3


def _zero_harmonics(coeffs, harmonics):
    cc = coeffs.copy()
    for h in harmonics:
        for osc in (1, 2):
            for comp in ("cos", "sin"):
                cc[coefficient_index(oscillator=osc, harmonic=h, component=comp, n_harmonics=N_H)] = 0.0
    return cc


def main():
    c = np.asarray(
        solve_harmonic_balance(P, n_harmonics=N_H, n_time_samples=N_TIME, tol=1e-13, max_nfev=4000),
        dtype=float,
    )
    a1, a3, a5, a7 = (_amp(c, 1, h) for h in (1, 3, 5, 7))
    print(f"working point: Omega*={OM_STAR}, beta1*={B1_STAR}, kappa=0.10, kappa_nl=-0.30, F=0.30")
    print(f"N_H={N_H}, N_time={N_TIME}")
    print(f"|X1|={a1:.6f}  |X3|={a3:.3e}  |X5|={a5:.3e}  |X7|={a7:.3e}")
    print(f"A5/|X1|={a5 / a1:.3e}  A7/|X1|={a7 / a1:.3e}\n")

    full = _r3(c)
    no5 = _r3(_zero_harmonics(c, [5]))
    no5up = _r3(_zero_harmonics(c, [5, 6, 7]))
    no7 = _r3(_zero_harmonics(c, [7]))
    fifth = full - no5
    seventh = full - no7

    print("third-harmonic balance residual, oscillator 1 (cos, sin):")
    print(f"  full (converged)      = ({full[0]:+.4e}, {full[1]:+.4e})")
    print(f"  fifth+higher removed  = ({no5up[0]:+.4e}, {no5up[1]:+.4e})")
    print(f"  fifth contribution    = ({fifth[0]:+.4e}, {fifth[1]:+.4e})   (equal and opposite)")
    print(f"  seventh contribution  = ({seventh[0]:+.4e}, {seventh[1]:+.4e})   (negligible)")
    print(f"\n  in 1e-5 units: residual=({no5up[0] * 1e5:+.2f}, {no5up[1] * 1e5:+.2f})"
          f"  fifth=({fifth[0] * 1e5:+.2f}, {fifth[1] * 1e5:+.2f})")


if __name__ == "__main__":
    main()

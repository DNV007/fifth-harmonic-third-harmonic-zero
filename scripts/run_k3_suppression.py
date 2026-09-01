"""Same-frequency suppression metric for the k=3 exact zero.

Suppression is referenced to the UNCANCELLED single-pathway third harmonic
|X3^(0)| = |X3| at kappa_nl = 0 (on-site cubic only), at the design operating
point (Omega*, F). Both |X3^(0)| and the mismatched |X3| are the third harmonic
at 3*Omega -- a same-frequency ratio, NOT a cross-frequency figure of merit:

    S = 20 * log10(|X3^(0)| / |X3|)   [dB].

Reproduces:
  * |X3^(0)| = 2.8e-3  (= -40.5 dBc relative to the fundamental)
  * a +-1% fabrication mismatch in either cubic -> ~39 dB suppression
  * exactly 40 dB needs ~0.9% of each cubic: d(beta1)=2.2e-3, d(knl)=2.6e-3
    (from full harmonic-balance re-solves)

The operating point (Omega*, F) is held fixed: this is a FABRICATION tolerance
on the two cubics, not an active retune.

Run:  JAX_ENABLE_X64=1 uv run python scripts/run_k3_suppression.py
"""
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np
from dataclasses import replace
from jax import config

config.update("jax_enable_x64", True)
from scipy.optimize import brentq

from hh_antiresonance.models import CoupledOscillatorParams
from hh_antiresonance.harmonic_balance import solve_harmonic_balance, coefficient_index

OM, B1, KNL, F0 = 0.25972810, 0.23815428, -0.30, 0.30
BASE = CoupledOscillatorParams(
    omega1=1.0, omega2=1.25, zeta1=0.015, zeta2=0.02,
    alpha1=0.0, alpha2=0.0, beta1=B1, beta2=0.0,
    kappa=0.10, force=F0, drive_omega=OM, kappa_nl=KNL,
)
N, NT = 7, 2048


def _idx(o, h, c):
    return coefficient_index(oscillator=o, harmonic=h, component=c, n_harmonics=N)


def _solve(p, guess=None):
    return np.asarray(
        solve_harmonic_balance(p, n_harmonics=N, initial_guess=guess, n_time_samples=NT, tol=1e-13, max_nfev=6000),
        dtype=float,
    )


def _amp(c, o, h):
    return float(np.hypot(c[_idx(o, h, "cos")], c[_idx(o, h, "sin")]))


def main():
    c_match = _solve(BASE)
    a1 = _amp(c_match, 1, 1)
    c0 = _solve(replace(BASE, kappa_nl=0.0), guess=c_match)
    x30 = _amp(c0, 1, 3)  # single-pathway reference

    print(f"single-pathway reference |X3^(0)| (knl=0) = {x30:.3e}   (reference value 2.8e-3)")
    print(f"   = {20 * np.log10(x30 / a1):.1f} dBc rel. to |X1|={_amp(c0, 1, 1):.4f}   (Letter: -40.5 dBc)\n")

    def x3_at(beta1=B1, knl=KNL):
        return _amp(_solve(replace(BASE, beta1=beta1, kappa_nl=knl), guess=c_match), 1, 3)

    def S(x3):
        return 20 * np.log10(x30 / x3)

    print("same-frequency suppression S = 20 log10(|X3^(0)|/|X3|), (Om*,F) held fixed:")
    xb, xk = x3_at(beta1=B1 * 1.01), x3_at(knl=KNL * 1.01)
    print(f"  +1% mismatch in beta1: |X3|={xb:.3e}  ->  S = {S(xb):.1f} dB")
    print(f"  +1% mismatch in knl:   |X3|={xk:.3e}  ->  S = {S(xk):.1f} dB")

    target = x30 / 100.0
    db1 = brentq(lambda d: x3_at(beta1=B1 + d) - target, 1e-5, 0.05)
    dkn = brentq(lambda d: x3_at(knl=KNL - d) - target, 1e-5, 0.05)
    print(f"\nexact 40 dB tolerance (|X3| = |X3^(0)|/100 = {target:.2e}):")
    print(f"  d(beta1) = {db1:.2e} = {100 * db1 / B1:.2f}% of beta1   (Letter: 2.2e-3)")
    print(f"  d(knl)   = {dkn:.2e} = {100 * dkn / abs(KNL):.2f}% of knl     (Letter: 2.6e-3)")


if __name__ == "__main__":
    main()

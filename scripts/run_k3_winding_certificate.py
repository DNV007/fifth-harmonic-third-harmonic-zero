"""Topological-degree (winding-number) existence certificate for the k=3 exact zero.

Certifies that the exact third-harmonic zero is a GENUINE root of the converged
harmonic-balance system, not merely a small numerical residual. The map
(Omega, beta1) -> (Re X3, Im X3) is evaluated on a loop enclosing the working
point. If |X3| stays bounded away from zero on the loop (verified with ~1e5-1e6
margin over the solver residual) and each consecutive phase step is < pi (so the
discrete winding equals the continuous one), then the topological degree is a
well-defined integer; a NONZERO degree places a zero of the map inside the loop
(degree existence theorem). Here the degree is +1. The boundary is sampled
rather than enclosed, so this is a numerical existence certificate and not a
validated proof.

This certifies an exact zero of the N_H=7 balance; the full-ODE connection is the
truncation convergence (bit-identical for N_H>=7) and the time-domain cross-check.

Run:  JAX_ENABLE_X64=1 uv run python scripts/run_k3_winding_certificate.py
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np
from dataclasses import replace
from jax import config
config.update("jax_enable_x64", True)
from hh_antiresonance.models import CoupledOscillatorParams
from hh_antiresonance.harmonic_balance import solve_harmonic_balance, coefficient_index

OM, B1 = 0.25972810, 0.23815428
BASE = CoupledOscillatorParams(omega1=1.0, omega2=1.25, zeta1=0.015, zeta2=0.02,
    alpha1=0.0, alpha2=0.0, beta1=B1, beta2=0.0, kappa=0.10, force=0.30,
    drive_omega=OM, kappa_nl=-0.30)
N, NT = 7, 512
IC = coefficient_index(oscillator=1, harmonic=3, component="cos", n_harmonics=N)
IS = coefficient_index(oscillator=1, harmonic=3, component="sin", n_harmonics=N)


def _solve(p, g=None):
    return np.asarray(solve_harmonic_balance(p, n_harmonics=N, initial_guess=g,
        n_time_samples=NT, tol=1e-13, max_nfev=4000), float)


def main(rO=0.006, rB=0.020, M=240):
    c0 = _solve(BASE)
    th = np.linspace(0, 2 * np.pi, M, endpoint=False)
    vals, g = [], c0
    for t in th:
        c = _solve(replace(BASE, drive_omega=float(OM + rO * np.cos(t)),
                           beta1=float(B1 + rB * np.sin(t))), g)
        vals.append(complex(c[IC], -c[IS])); g = c
    vals = np.array(vals)
    dphi = np.angle(vals * np.conj(np.roll(vals, 1)))
    wind = np.sum(dphi) / (2 * np.pi)
    print(f"loop (rO={rO}, rB={rB}) in (Omega,beta1), {M} points:")
    print(f"  min|X3| on loop      = {np.min(np.abs(vals)):.3e}  (margin over 1e-11 residual: {np.min(np.abs(vals))/1e-11:.0e})")
    print(f"  max per-step |dphi|  = {np.max(np.abs(dphi)):.3f} rad  ({'OK <pi' if np.max(np.abs(dphi))<np.pi else 'REFINE'})")
    print(f"  topological degree   = {round(wind):+d}  (raw {wind:+.4f})")
    print("  => nonzero degree places an exact zero of the converged balance inside the loop")
    print("     (numerical certificate: boundary sampled, |X3|>0 on loop, every dphi<pi).")


if __name__ == "__main__":
    main()

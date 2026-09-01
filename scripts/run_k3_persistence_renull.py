"""Persistence re-nulls of the exact third-harmonic zero.

Two scripted persistence legs, both plain augmented re-solves of the production
model (the quintic k5 and mass-ratio mu legs modify the equations of motion and
are archived one-off runs; the gate quadratic alpha1 leg is
run_k3_gate_certificates.py):

(1) Absorber detuning, re-nulled with the drive (Letter wording): at
    omega2 = 1.10 and 1.35, free (Omega, F) at fixed beta1, kappa_nl returns
    |X3| to the solver floor.
(2) Absorber loss: zeta2 relocates the zero rather than removing it. At each
    zeta2 in {0.005, 0.010, 0.040}, re-matching both cubic coefficients (free
    beta1, kappa_nl at fixed drive) restores an exact null at a shifted point.
(3) Whether the DRIVE alone restores the null under zeta2 variation at fixed
    cubics, counting a re-null only if F lands inside the window [0.26, 0.35].
(4) The same question by a grid over the drive plane, which depends on no
    continuation path and so cannot report a false negative by losing the branch.

Parts (3) and (4) agree: the drive re-nulls up to zeta2 ~ 0.03 and is exhausted
by zeta2 = 0.04, where both give a best in-window |X3| ~ 9e-6.

SLOW. The augmented re-nulls of parts (1)-(3) take tens of minutes; part (4) is
~2000 forward solves. Run it in the background and expect roughly an hour.

Run:  JAX_ENABLE_X64=1 uv run python scripts/run_k3_persistence_renull.py
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np
from dataclasses import replace
from jax import config
config.update("jax_enable_x64", True)
from scipy.optimize import least_squares

from hh_antiresonance.models import CoupledOscillatorParams
from hh_antiresonance.harmonic_balance import (
    solve_harmonic_balance, harmonic_balance_residual, coefficient_index)

OM, B1, KNL, F0 = 0.25972810, 0.23815428, -0.30, 0.30
BASE = CoupledOscillatorParams(omega1=1.0, omega2=1.25, zeta1=0.015, zeta2=0.02,
    alpha1=0.0, alpha2=0.0, beta1=B1, beta2=0.0, kappa=0.10, force=F0,
    drive_omega=OM, kappa_nl=KNL)
N, NT = 7, 512
IC = coefficient_index(oscillator=1, harmonic=3, component="cos", n_harmonics=N)
IS = coefficient_index(oscillator=1, harmonic=3, component="sin", n_harmonics=N)


def solve(p, g=None):
    return np.asarray(solve_harmonic_balance(p, n_harmonics=N, initial_guess=g,
        n_time_samples=NT, tol=1e-13, max_nfev=4000), float)


def renull(pbase, free, seed_vals, seed_coeffs):
    """Augmented solve: coefficients + two free parameters, X3 quadratures = 0."""
    def r(z):
        c = z[:4 * N]
        p = replace(pbase, **{free[0]: float(z[-2]), free[1]: float(z[-1])})
        return np.concatenate([harmonic_balance_residual(c, p, n_harmonics=N,
                                                         n_time_samples=NT), [c[IC], c[IS]]])
    z0 = np.concatenate([seed_coeffs, seed_vals])
    s = least_squares(r, z0, xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=9000)
    c = s.x[:4 * N]
    return s.x[-2], s.x[-1], float(np.hypot(c[IC], c[IS]))


def main():
    c0 = solve(BASE)

    # Large detuning needs continuation: warm-step omega2 from the nominal 1.25,
    # re-nulling at each step, so the augmented Newton stays on the genuine
    # branch instead of escaping to the trivial Omega->0 or F->0 roots.
    print("(1) absorber detuning, re-nulled by warm-stepped continuation:")
    for w2t, nstep in ((1.10, 7), (1.35, 5)):
        for free, seed_vals in ((("drive_omega", "force"), [OM, F0]),
                                (("drive_omega", "beta1"), [OM, B1])):
            v, cs, ok = list(seed_vals), c0, True
            for w2 in np.linspace(1.25, w2t, nstep + 1)[1:]:
                p = replace(BASE, omega2=float(w2),
                            **{free[0]: float(v[0]), free[1]: float(v[1])})
                cs = solve(p, cs)
                a, b, m = renull(replace(BASE, omega2=float(w2)), free, v, cs)
                v = [a, b]
                p = replace(BASE, omega2=float(w2),
                            **{free[0]: float(a), free[1]: float(b)})
                cs = solve(p, cs)
            trivial = (free[1] == "force" and not 0.10 < v[1] < 0.36) or abs(v[0]) < 0.05
            tag = "TRIVIAL/OFF-BAND" if (trivial or m > 1e-12) else "genuine re-null"
            print(f"    omega2 = {w2t}, free ({free[0]}, {free[1]}): "
                  f"({v[0]:.6f}, {v[1]:.6f}), |X3| = {m:.1e}  [{tag}]")

    print("(2) absorber loss, re-matched cubics (free beta1, kappa_nl):")
    for z2 in (0.005, 0.010, 0.040):
        p = replace(BASE, zeta2=z2)
        b1, knl, m = renull(p, ("beta1", "kappa_nl"), [B1, KNL], solve(p, c0))
        print(f"    zeta2 = {z2}: (beta1, kappa_nl) = ({b1:.6f}, {knl:.6f}), |X3| = {m:.1e}")

    # (3) Does the DRIVE alone restore the null under zeta2 variation? The Letter
    # says a high-Q2 device is "redesigned rather than re-driven", which is a claim
    # about what (Omega, F) cannot do at fixed cubics. Test it directly, warm-stepping
    # zeta2 so the augmented Newton stays on the branch. A re-null only counts if the
    # forcing lands inside the admissible window F in [0.26, 0.35].
    print("(3) absorber loss, drive basis at fixed cubics (free Omega, F):")
    for z2t, nstep in ((0.005, 6), (0.010, 5), (0.040, 5)):
        v, cs = [OM, F0], c0
        for z2 in np.linspace(0.02, z2t, nstep + 1)[1:]:
            p = replace(BASE, zeta2=float(z2), drive_omega=float(v[0]), force=float(v[1]))
            cs = solve(p, cs)
            a, b, m = renull(replace(BASE, zeta2=float(z2)), ("drive_omega", "force"), v, cs)
            v = [a, b]
            cs = solve(replace(BASE, zeta2=float(z2), drive_omega=float(a), force=float(b)), cs)
        inband = 0.26 <= v[1] <= 0.35
        tag = ("re-driven inside the window" if (inband and m < 1e-12)
               else "OUTSIDE admissible window" if m < 1e-12 else "no re-null")
        print(f"    zeta2 = {z2t}: (Omega, F) = ({v[0]:.6f}, {v[1]:.6f}), "
              f"|X3| = {m:.1e}  [{tag}]")

    # (4) The same question without any continuation path, which could lose the
    # branch and report a false negative. Grid the drive plane at fixed cubics and
    # ask whether BOTH quadratures change sign inside F in [0.26, 0.35]. A crossing
    # is necessary for a drive re-null; its absence is a property of the map, not of
    # a solver trajectory. zeta2 = 0.020 is the nominal control and must show one.
    print("(4) drive-plane crossing at fixed cubics (no continuation):")
    print(f"    {'zeta2':>7} {'min|X3| in window':>18} {'crossing?':>10}")
    oms = np.linspace(0.240, 0.285, 19)
    Fs = np.linspace(0.220, 0.420, 21)
    for z2 in (0.005, 0.020, 0.030, 0.040, 0.060):
        RE = np.zeros((len(Fs), len(oms))); IM = np.zeros_like(RE); MG = np.zeros_like(RE)
        for j, om in enumerate(oms):
            g = None
            for i, F in enumerate(Fs):
                g = solve(replace(BASE, zeta2=float(z2), drive_omega=float(om),
                                  force=float(F)), g)
                RE[i, j], IM[i, j] = g[IC], -g[IS]
                MG[i, j] = float(np.hypot(g[IC], g[IS]))
        sre, sim = np.sign(RE), np.sign(IM)
        hit = any(sre[i:i+2, j:j+2].min() < 0 < sre[i:i+2, j:j+2].max()
                  and sim[i:i+2, j:j+2].min() < 0 < sim[i:i+2, j:j+2].max()
                  and 0.26 <= Fs[i] <= 0.35
                  for i in range(len(Fs) - 1) for j in range(len(oms) - 1))
        inb = (Fs >= 0.26) & (Fs <= 0.35)
        print(f"    {z2:>7.3f} {MG[inb, :].min():>18.2e} {('YES' if hit else 'no'):>10}")


if __name__ == "__main__":
    main()

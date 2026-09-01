"""Two-family certificate: the perturbative sister zero and the N_H=3 absence
of the reported zero.

Four parts:
(1) The closed-form leading-order root, solved directly from the scale-free
    condition Z22(3W) b1 + knl L2(3W) (1 - k/Z22(W))^3 = 0 (Letter Eq. (EM2)).
    Expected (Omega, beta1) = (1.24849746, -0.02488117).
(2) The sister family: exact third-harmonic zeros (free Omega, beta1; X3 = 0)
    continued in forcing F = 0.30 -> 0.001 at N_H = 7. Reproduces:
    convergence onto the closed-form root as F -> 0, |X1|/F -> 1.844, |X3| at
    the solver floor at every F.
(3) The sister at the coarsest truncation N_H = 3: it exists there, |X3| at the
    solver floor (the reported zero does not exist at N_H = 3 at all).
(4) The reported zero's absence at N_H = 3, the scripted legs of SM Sec. S2:
    a 67x67 scan over Omega in [0.12, 0.45], beta1 in [0.05, 0.60] (min |X3|
    ~ 4e-6, both quadratures never zero together) and the winding number on
    loops about the N_H = 7 working point (0 at every radius, against +1 at
    N_H = 7 from run_k3_winding_certificate.py). Pass --seeds 1156 to also run
    the seeded-Newton refinement leg at full scale (adds ~10-20 min).

Run:  JAX_ENABLE_X64=1 uv run python scripts/run_k3_sister_family.py
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import argparse
import numpy as np
from dataclasses import replace
from jax import config
config.update("jax_enable_x64", True)
from scipy.optimize import least_squares, root

from hh_antiresonance.models import CoupledOscillatorParams
from hh_antiresonance.harmonic_balance import (
    solve_harmonic_balance, harmonic_balance_residual, coefficient_index)

W2, Z1, Z2, KA, KNL = 1.25, 0.015, 0.02, 0.10, -0.30
OM7, B17 = 0.25972810, 0.23815428          # reported zero (N_H=7 working point)
BASE = CoupledOscillatorParams(omega1=1.0, omega2=W2, zeta1=Z1, zeta2=Z2,
    alpha1=0.0, alpha2=0.0, beta1=0.0, beta2=0.0, kappa=KA, force=0.30,
    drive_omega=1.25, kappa_nl=KNL)
NT = 512
_D = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(_D, "..", "data")


def idx(n_h, comp, harmonic=3):
    return coefficient_index(oscillator=1, harmonic=harmonic, component=comp, n_harmonics=n_h)


def solve(p, n_h, g=None):
    return np.asarray(solve_harmonic_balance(p, n_harmonics=n_h, initial_guess=g,
        n_time_samples=NT, tol=1e-13, max_nfev=4000), float)


def x3(c, n_h):
    return complex(c[idx(n_h, "cos")], -c[idx(n_h, "sin")])


def x1(c, n_h):
    return complex(c[idx(n_h, "cos", 1)], -c[idx(n_h, "sin", 1)])


def solve_zero(F, seed, n_h):
    """Free (Omega, beta1) so the third-harmonic quadratures of x1 vanish."""
    ic, isx = idx(n_h, "cos"), idx(n_h, "sin")
    def r(z):
        c = z[:4 * n_h]
        p = replace(BASE, drive_omega=float(z[-2]), beta1=float(z[-1]), force=float(F))
        return np.concatenate([harmonic_balance_residual(c, p, n_harmonics=n_h,
                                                         n_time_samples=NT), [c[ic], c[isx]]])
    return least_squares(r, seed, xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=9000).x


def closed_form_root():
    """Real root of Z22(3W) b1 + knl L2(3W) (1 - k/Z22(W))^3 = 0 at beta1 < 0."""
    def L2(w):
        return W2 ** 2 - w ** 2 + 2j * Z2 * w
    def G(v):
        Om, b1 = v
        z3, z1v = L2(3 * Om) + KA, L2(Om) + KA
        g = z3 * b1 + KNL * L2(3 * Om) * (1 - KA / z1v) ** 3
        return [g.real, g.imag]
    s = root(G, [1.25, -0.025], tol=1e-14)
    return s.x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=0,
                    help="run the N_H=3 seeded-Newton absence leg with this many seeds")
    args = ap.parse_args()

    # (1) closed-form leading-order root
    Om0, b10 = closed_form_root()
    print(f"(1) closed-form leading-order root: (Omega, beta1) = ({Om0:.8f}, {b10:.8f})")
    print(f"    expected                        (1.24849746, -0.02488117)")

    # (2) sister family, descending F ladder at N_H=7
    n_h = 7
    p0 = replace(BASE, drive_omega=1.2507, beta1=-0.0453, force=0.30)
    z = np.concatenate([solve(p0, n_h), [1.2507, -0.0453]])
    print(f"\n(2) sister family (N_H=7, free Omega, beta1; X3=0):")
    print(f"    {'F':>7} {'Omega*':>12} {'beta1*':>13} {'|X1|/F':>8} {'|X3|':>10}")
    rows = []
    for F in (0.30, 0.10, 0.05, 0.01, 0.001):
        z = solve_zero(F, z, n_h)
        c = z[:4 * n_h]
        rows.append((F, z[-2], z[-1], abs(x1(c, n_h)) / F, abs(x3(c, n_h))))
        print(f"    {F:>7} {z[-2]:>12.8f} {z[-1]:>13.8f} {rows[-1][3]:>8.3f} {rows[-1][4]:>10.1e}")
    print(f"    F->0 limit vs closed form: dOmega = {abs(rows[-1][1]-Om0):.1e}, "
          f"dbeta1 = {abs(rows[-1][2]-b10):.1e}")
    np.savetxt(os.path.join(DATADIR, "k3_sister_family.csv"), np.array(rows),
               delimiter=",", header="F,Omega,beta1,absX1_over_F,absX3", comments="")

    # (3) sister at the coarsest truncation N_H=3
    n3 = 3
    p3 = replace(BASE, drive_omega=1.2507, beta1=-0.0453, force=0.30)
    z3 = solve_zero(0.30, np.concatenate([solve(p3, n3), [1.2507, -0.0453]]), n3)
    c3 = z3[:4 * n3]
    print(f"\n(3) sister at N_H=3, F=0.30: (Omega, beta1) = ({z3[-2]:.6f}, {z3[-1]:.6f}), "
          f"|X3| = {abs(x3(c3, n3)):.1e}  (exists at the coarsest truncation)")

    # (4) reported-zero absence at N_H=3
    print(f"\n(4) reported zero at N_H=3 over Omega in [0.12,0.45], beta1 in [0.05,0.60]:")
    noms = np.linspace(0.12, 0.45, 67); nb1 = np.linspace(0.05, 0.60, 67)
    best = np.inf
    for om in noms:
        g = None
        for bb in nb1:
            g = solve(replace(BASE, drive_omega=float(om), beta1=float(bb), force=0.30), n3, g)
            m = abs(x3(g, n3))
            if m < best:
                best = m
    print(f"    67x67 scan: min |X3| = {best:.2e}  (expected ~4e-6; never at a zero)")
    for rO, rB in ((0.006, 0.020), (0.012, 0.040), (0.020, 0.080)):
        M = 240
        th = np.linspace(0, 2 * np.pi, M, endpoint=False)
        vals = []
        g = None
        for t in th:
            om, bb = OM7 + rO * np.cos(t), B17 + rB * np.sin(t)
            g = solve(replace(BASE, drive_omega=float(om), beta1=float(bb), force=0.30), n3, g)
            vals.append(x3(g, n3))
        vals = np.array(vals)
        ph = np.unwrap(np.angle(np.append(vals, vals[0])))
        wind = round(float((ph[-1] - ph[0]) / (2 * np.pi)))
        print(f"    winding on loop (rO={rO}, rB={rB}): degree = {wind:+d}, "
              f"min |X3| on loop = {np.min(np.abs(vals)):.1e}")
    if args.seeds > 0:
        rng = np.random.default_rng(0)
        ic3, is3 = idx(n3, "cos"), idx(n3, "sin")
        hits = 0
        for k in range(args.seeds):
            om = rng.uniform(0.12, 0.45); bb = rng.uniform(0.05, 0.60)
            seed = np.concatenate([solve(replace(BASE, drive_omega=om, beta1=bb, force=0.30), n3), [om, bb]])
            zz = solve_zero(0.30, seed, n3)
            if abs(x3(zz[:4 * n3], n3)) < 1e-10 and 0.10 < zz[-2] < 0.50 and 0.02 < zz[-1] < 0.65:
                hits += 1
        print(f"    seeded Newton refinement: {args.seeds} seeds -> {hits} exact zeros found")


if __name__ == "__main__":
    main()

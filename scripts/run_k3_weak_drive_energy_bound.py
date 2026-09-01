"""Why the zero locus has a minimum drive: dissipation.

The minimum drive is bounded below by an exact dissipation identity.  A linear
transfer-function maximum is not such a bound: no theorem makes it an upper
bound on a nonlinear steady-state gain.  The exact statement follows.

Exact periodic energy balance.  Multiply each equation of motion by its own
velocity and average over a period.  Every conservative term averages to zero
and the coupling cancels between the two equations, leaving, exactly,

    F <cos(Omega t) xdot_1>  =  2 zeta_1 <xdot_1^2> + 2 zeta_2 <xdot_2^2>.

With x_j = sum_n [a_n cos(n Omega t) + b_n sin(n Omega t)] this gives

    |<cos(Omega t) xdot_1>| <= Omega |X_{1,1}| / 2,
    2 z1 <v1^2> + 2 z2 <v2^2> = Omega^2 sum_n n^2 (z1 |X_{1,n}|^2 + z2 |X_{2,n}|^2),

so every periodic orbit of the model obeys

    F >= (2 Omega / |X_{1,1}|) sum_n n^2 (z1|X_{1,n}|^2 + z2|X_{2,n}|^2)
      >= 2 zeta_1 Omega |X_{1,1}|.                                    (bound)

This is a theorem about the ODE, not a fit, and it is what actually forbids a
finite-amplitude orbit at arbitrarily small drive at finite frequency and loss.

Why the bound is nearly SATURATED at the fold.  On the low-forcing arc the
branch sits where the effective fundamental coupling cancels,

    kappa_eff = kappa + (3/4) kappa_nl |Delta_1|^2 = 0
        =>  |Delta_1| = sqrt(-4 kappa / (3 kappa_nl)),

so the absorber decouples from the driven mass AT THE FUNDAMENTAL (while
staying lit at 3*Omega, which is what the Letter's mechanism needs).  The
driven mass is then effectively an isolated Duffing oscillator, zeta_1 carries
essentially all the dissipation, and the drive is minimised at its backbone,

    Omega_backbone = sqrt(omega_1^2 + (3/4) beta_1 |X_{1,1}|^2),

where the response is in quadrature and the inequality above is tight.  That is
the whole low-forcing arc in closed form: the amplitude is pinned by coupling
cancellation, Omega and F trade along the bare Duffing response, and the fold
sits at the backbone peak.

Reported:
  (1) the exact energy identity on the solved fold orbit, and the bound
  (2) the bound at every coupling tracked by run_k3_weak_drive_route.py
  (3) the locked amplitude along the arc against sqrt(-4 kappa / 3 kappa_nl)

Run:  PYTHONPATH=src JAX_ENABLE_X64=1 uv run python scripts/run_k3_weak_drive_energy_bound.py
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
from pathlib import Path

import numpy as np
from jax import config

config.update("jax_enable_x64", True)

from hh_antiresonance.models import CoupledOscillatorParams
from hh_antiresonance.harmonic_balance import (
    solve_harmonic_balance, harmonic_balance_residual)

W1, W2, Z1, Z2 = 1.0, 1.25, 0.015, 0.020
KA, KNL = 0.10, -0.30
NH = 9

_D = Path(os.path.dirname(os.path.abspath(__file__)))
DATA = _D / ".." / "data"


def orbit(F, Om, b1, ka=KA, n_h=NH):
    p = CoupledOscillatorParams(omega1=W1, omega2=W2, zeta1=Z1, zeta2=Z2,
                                alpha1=0.0, alpha2=0.0, beta1=b1, beta2=0.0,
                                kappa=ka, kappa_nl=KNL, force=F, drive_omega=Om)
    c = solve_harmonic_balance(p, n_harmonics=n_h, n_time_samples=2048, tol=1e-13)
    r = float(np.linalg.norm(harmonic_balance_residual(
        c, p, n_harmonics=n_h, n_time_samples=2048)))
    x1 = c[:2 * n_h].reshape(n_h, 2)
    x2 = c[2 * n_h:].reshape(n_h, 2)
    return x1, x2, r


def main():
    hdr = "=" * 96
    print(hdr)
    print("MINIMUM DRIVE ON THE ZERO LOCUS: EXACT PERIODIC ENERGY BALANCE")
    print(hdr)
    print(f"fixed: omega1={W1}, omega2={W2}, zeta1={Z1}, zeta2={Z2}, kappa_nl={KNL}\n")

    path = np.loadtxt(DATA / "two_dof_k3_weak_drive_path_kappa010.csv",
                      delimiter=",", skiprows=1)
    j = int(path[:, 0].argmin())
    F, Om, b1 = path[j, 0], path[j, 1], path[j, 2]

    # ---- (1) the identity and the bound on the solved fold orbit ------------
    x1, x2, res = orbit(F, Om, b1)
    A1 = np.hypot(x1[:, 0], x1[:, 1])
    A2 = np.hypot(x2[:, 0], x2[:, 1])
    n = np.arange(1, NH + 1)
    dissip = Z1 * Om ** 2 * np.sum(n ** 2 * A1 ** 2) + Z2 * Om ** 2 * np.sum(n ** 2 * A2 ** 2)
    inject = F * 0.5 * Om * x1[0, 1]          # <cos(Om t) v_1> = Om b_1 / 2

    print(f"(1) FOLD ORBIT  F={F:.9f}  Omega={Om:.9f}  beta1={b1:.9f}   "
          f"(N_H={NH}, HB residual {res:.1e})")
    print(f"    |X_1,1| = {A1[0]:.9f}     |X_2,1| = {A2[0]:.3e}     "
          f"|X_2,3| = {A2[2]:.3e}")
    print(f"    the absorber is dark at Omega and lit at 3*Omega: "
          f"|X_2,3|/|X_2,1| = {A2[2] / A2[0]:.1f}")
    print(f"\n    exact balance   dissipation = {dissip:.12e}")
    print(f"                    injection   = {inject:.12e}")
    print(f"                    closure     = {abs(inject - dissip) / dissip:.1e}")

    b_lead = 2 * Z1 * Om * A1[0]
    b_full = 2 * Om * (Z1 * np.sum(n ** 2 * A1 ** 2) + Z2 * np.sum(n ** 2 * A2 ** 2)) / A1[0]
    print(f"\n    dissipation split: D1/(D1+D2) = "
          f"{Z1 * Om ** 2 * np.sum(n ** 2 * A1 ** 2) / dissip:.6f}")
    print(f"\n    bound  2 zeta1 Omega |X_1,1|        = {b_lead:.9f}   "
          f"F/bound = {F / b_lead:.6f}  {'OK' if F >= b_lead else '** VIOLATED **'}")
    print(f"    bound  all harmonics, both masses   = {b_full:.9f}   "
          f"F/bound = {F / b_full:.6f}  {'OK' if F >= b_full else '** VIOLATED **'}")
    quad = abs(0.5 * Om * x1[0, 1]) / (0.5 * Om * A1[0])
    print(f"    quadrature factor |<cos v1>|/(Omega|X_1,1|/2) = {quad:.6f}  (1 = exact quadrature)")

    d_lock = np.sqrt(-4 * KA / (3 * KNL))
    D1 = abs((x1[0, 0] - 1j * x1[0, 1]) - (x2[0, 0] - 1j * x2[0, 1]))
    keff = KA + 0.75 * KNL * D1 ** 2
    om_bb = np.sqrt(W1 ** 2 + 0.75 * b1 * A1[0] ** 2)
    # Eq. (coupling-cancellation) is the FUNDAMENTAL-ONLY effective coupling.
    # The exact harmonic-resolved one uses A_1[d^3] on the full orbit.
    from itertools import product as _pr
    _v = np.zeros(2 * NH + 1, complex)
    _d = (x1[:, 0] - 1j * x1[:, 1]) - (x2[:, 0] - 1j * x2[:, 1])
    _v[NH + 1:] = _d / 2; _v[:NH] = (_d.conj() / 2)[::-1]
    _a1 = 2 * sum(_v[p + NH] * _v[q + NH] * _v[r + NH]
                  for p, q, r in _pr(range(-NH, NH + 1), repeat=3)
                  if p + q + r == 1 and 0 not in (p, q, r))
    _K = KA + KNL * _a1 / _d[0]
    print(f"\n    exact effective coupling K_eff^(1) = kappa + kappa_nl A_1[d^3]/Delta_1")
    print(f"                          = {_K.real:.3e}{_K.imag:+.1e}j   "
          f"|K|/kappa = {abs(_K) / KA:.2e}")
    print(f"\n    coupling cancellation: |Delta_1| = {D1:.9f}  vs "
          f"sqrt(-4k/3knl) = {d_lock:.9f}  ({100 * (D1 - d_lock) / d_lock:+.3f}%)")
    print(f"                           kappa_eff = {keff:.3e}   ({abs(keff) / KA:.2%} of kappa)")
    print(f"    Duffing backbone:      Omega_bb  = {om_bb:.6f}  vs fold Omega = {Om:.6f}"
          f"  ({100 * (om_bb - Om) / Om:+.3f}%)")

    # ---- (2) the bound at every tracked coupling ---------------------------
    print(f"\n(2) BOUND AT EVERY TRACKED COUPLING")
    route = np.loadtxt(DATA / "two_dof_k3_weak_drive_route.csv", delimiter=",", skiprows=1)
    print(f"    {'kappa':>6}{'F_min':>12}{'Omega':>10}{'|X_1,1|':>11}"
          f"{'2 z1 Om |X1|':>14}{'margin':>9}{'sqrt(-4k/3knl)':>16}{'dev':>8}")
    rows = []
    for r in np.atleast_2d(route):
        ka, fmin, om, x1a, cap = r[0], r[4], r[5], r[6], r[8]
        assert cap == 0.0, f"kappa={ka}: continuation hit its step cap; F_min is not a fold"
        bl = 2 * Z1 * om * x1a
        lk = np.sqrt(-4 * ka / (3 * KNL))
        rows.append([ka, fmin, om, x1a, bl, lk])
        print(f"    {ka:6.2f}{fmin:12.7f}{om:10.5f}{x1a:11.6f}{bl:14.7f}"
              f"{100 * (fmin - bl) / bl:8.3f}%{lk:16.6f}{100 * (x1a - lk) / lk:7.2f}%")
    m = np.array([100 * (r[1] - r[4]) / r[4] for r in rows])
    assert (m > 0).all(), "energy bound violated"
    print(f"\n    every fold satisfies the bound, by {m.min():.3f}% to {m.max():.3f}%.")

    # ---- (3) the amplitude lock along the arc ------------------------------
    print(f"\n(3) AMPLITUDE LOCK ALONG THE LOW-FORCING ARC (kappa = {KA})")
    print(f"    {'Omega':>9}{'F':>10}{'|X_1,1|':>11}{'|X_2,1|':>11}"
          f"{'|Delta_1|':>11}{'dev vs lock':>13}{'kappa_eff/kappa':>17}")
    for tgt in (1.00, 1.048134, 1.15, 1.40, 1.80, 2.17):
        k = int(np.abs(path[:, 1] - tgt).argmin())
        Fk, Omk, b1k = path[k, 0], path[k, 1], path[k, 2]
        u1, u2, _ = orbit(Fk, Omk, b1k)
        X11 = u1[0, 0] - 1j * u1[0, 1]
        X21 = u2[0, 0] - 1j * u2[0, 1]
        Dk = abs(X11 - X21)
        print(f"    {Omk:9.5f}{Fk:10.5f}{abs(X11):11.6f}{abs(X21):11.6f}{Dk:11.6f}"
              f"{100 * (Dk - d_lock) / d_lock:12.3f}%"
              f"{(KA + 0.75 * KNL * Dk ** 2) / KA:17.2e}")
    print(f"\n    predicted lock |Delta_1| = sqrt(-4 kappa / 3 kappa_nl) = {d_lock:.9f}")

    np.savetxt(DATA / "two_dof_k3_weak_drive_energy_bound.csv", np.array(rows),
               delimiter=",",
               header="kappa,F_min,omega_min,abs_X1,bound_2z1_Om_absX1,lock_sqrt_m4k_3knl",
               comments="")
    print(f"\nwrote {DATA / 'two_dof_k3_weak_drive_energy_bound.csv'}")


if __name__ == "__main__":
    main()

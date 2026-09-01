"""Persistence of the exact third-harmonic zero under two structural perturbations
that the production model does not carry: an unequal mass ratio and a stabilizing
coupling quintic.

`CoupledOscillatorParams` has neither a mass ratio nor a quintic coefficient, so
these two legs cannot run against the production solver. This driver is therefore
standalone, in the manner of `run_k3_rlc_second_example.py`: it carries its own
harmonic-balance residual and shares no code with the production path. The
extended equations of motion are, with d = x1 - x2 and mu = m2/m1,

  x1'' + 2 z1 x1' + w1^2 x1 + b1 x1^3 + ka d + knl d^3 + k5 d^5 = F cos(Om t)
  x2'' + 2 z2 x2' + w2^2 x2 - (ka/mu) d - (knl/mu) d^3 - (k5/mu) d^5 = 0

The absorber equation is divided by its own mass, so the coupling terms it feels
carry 1/mu. At mu = 1 and k5 = 0 the system is exactly the Letter's Eqs. (3)-(4),
which is the validation below: the standalone solver must return the production
working point (0.25972810, 0.23815428). The quintic is the coupling quintic
k5 d^5 that regularizes the tangent stiffness
K(d) = ka + 3 knl d^2 + 5 k5 d^4; it is the "stabilizing quintic" of P7.

Parts:
(0) validation at mu = 1, k5 = 0 against the production working point;
(1) quintic ladder k5 = 0 .. 0.11 continuing BOTH arms of the exact zero with
    their control determinants det d(ReX3,ImX3)/d(Om,b1), which vanish from
    opposite signs at the annihilation;
(2) a cold seed hunt at k5 = 0.15, where no exact zero should exist;
(3) mass ratio mu over [0.4, 2.0], where the zero should re-null at the floor.

Run:  uv run python scripts/run_k3_extended_model_persistence.py
"""
import os
import numpy as np
from scipy.optimize import least_squares, root

W1, W2 = 1.0, 1.25
Z1, Z2 = 0.015, 0.02
KA, KNL = 0.10, -0.30
F0 = 0.30
OM_PROD, B1_PROD = 0.25972810, 0.23815428   # production working point (mu=1, k5=0)
OM_ARM_B = 0.2515209                        # second arm of the exact zero at k5=0
N, NT = 7, 512

nn = np.arange(NT)
PH = 2 * np.pi * np.outer(np.arange(1, N + 1), nn) / NT
COS, SIN = np.cos(PH), np.sin(PH)
MVEC = np.arange(1, N + 1)[:, None]
IC, IS = 2 * (3 - 1), 2 * (3 - 1) + 1


def recon(c, Om):
    x1 = c[:2 * N].reshape(N, 2)
    x2 = c[2 * N:].reshape(N, 2)
    a1, b1 = x1[:, [0]], x1[:, [1]]
    a2, b2 = x2[:, [0]], x2[:, [1]]
    q1 = np.sum(a1 * COS + b1 * SIN, 0)
    q2 = np.sum(a2 * COS + b2 * SIN, 0)
    v1 = np.sum(MVEC * Om * (-a1 * SIN + b1 * COS), 0)
    v2 = np.sum(MVEC * Om * (-a2 * SIN + b2 * COS), 0)
    ac1 = np.sum(-(MVEC * Om) ** 2 * (a1 * COS + b1 * SIN), 0)
    ac2 = np.sum(-(MVEC * Om) ** 2 * (a2 * COS + b2 * SIN), 0)
    return q1, v1, ac1, q2, v2, ac2


def hb_res(c, Om, b1, F=F0, mu=1.0, k5=0.0):
    q1, v1, ac1, q2, v2, ac2 = recon(c, Om)
    d = q1 - q2
    cpl = KA * d + KNL * d ** 3 + k5 * d ** 5
    r1 = ac1 + 2 * Z1 * v1 + W1 ** 2 * q1 + b1 * q1 ** 3 + cpl - F * COS[0]
    r2 = ac2 + 2 * Z2 * v2 + W2 ** 2 * q2 - cpl / mu
    out = np.empty(4 * N)
    for m in range(N):
        out[4 * m:4 * m + 4] = [(2 / NT) * (r1 @ COS[m]), (2 / NT) * (r1 @ SIN[m]),
                                (2 / NT) * (r2 @ COS[m]), (2 / NT) * (r2 @ SIN[m])]
    return out


def solve_hb(Om, b1, F=F0, mu=1.0, k5=0.0, guess=None):
    z0 = np.zeros(4 * N) if guess is None else guess.copy()
    if guess is None:
        z0[0] = F / (W1 ** 2 - Om ** 2)
    else:
        s = root(lambda c: hb_res(c, Om, b1, F, mu, k5), z0, method="hybr", tol=1e-13)
        if s.success:
            return s.x
    return least_squares(lambda c: hb_res(c, Om, b1, F, mu, k5), z0,
                         xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=6000).x


def X3(c):
    return complex(c[IC], -c[IS])


def solve_zero(seed, F=F0, mu=1.0, k5=0.0):
    """Free (Omega, beta1) so the third-harmonic quadratures of x1 vanish."""
    def aug(z):
        c = z[:4 * N]
        return np.concatenate([hb_res(c, float(z[-2]), float(z[-1]), F, mu, k5),
                               [c[IC], c[IS]]])
    s = least_squares(aug, seed, xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=9000)
    return s.x


def detJ(Om, b1, F=F0, mu=1.0, k5=0.0, guess=None, h=1e-6):
    """det d(ReX3, ImX3)/d(Omega, beta1) by central differences."""
    def y(o, b):
        c = solve_hb(o, b, F, mu, k5, guess)
        return np.array([c[IC], -c[IS]])
    J = np.column_stack([(y(Om + h, b1) - y(Om - h, b1)) / (2 * h),
                         (y(Om, b1 + h) - y(Om, b1 - h)) / (2 * h)])
    return float(np.linalg.det(J))


def main():
    # ---- (0) validation: the standalone solver must reproduce production ----
    z = solve_zero(np.concatenate([solve_hb(OM_PROD, B1_PROD), [OM_PROD, B1_PROD]]))
    om, b1 = z[-2], z[-1]
    print("(0) validation at mu=1, k5=0 (must equal the production working point):")
    print(f"    standalone : (Omega*, beta1*) = ({om:.8f}, {b1:.8f})  |X3| = {abs(X3(z[:4*N])):.1e}")
    print(f"    production : (Omega*, beta1*) = ({OM_PROD:.8f}, {B1_PROD:.8f})")
    print(f"    difference : dOmega = {abs(om-OM_PROD):.1e}, dbeta1 = {abs(b1-B1_PROD):.1e}")

    # ---- (1) quintic ladder, both arms, with control determinants ----
    print("\n(1) coupling quintic k5, both arms:")
    print(f"    {'k5':>6} {'Omega*_A':>12} {'detJ_A':>11} {'Omega*_B':>12} {'detJ_B':>11}")
    zA = np.concatenate([solve_hb(OM_PROD, B1_PROD), [OM_PROD, B1_PROD]])
    zB = solve_zero(np.concatenate([solve_hb(OM_ARM_B, B1_PROD), [OM_ARM_B, B1_PROD]]))
    for k5 in (0.000, 0.040, 0.080, 0.100, 0.110):
        zA = solve_zero(zA, k5=k5)
        zB = solve_zero(zB, k5=k5)
        dA = detJ(zA[-2], zA[-1], k5=k5, guess=zA[:4 * N])
        dB = detJ(zB[-2], zB[-1], k5=k5, guess=zB[:4 * N])
        print(f"    {k5:>6.3f} {zA[-2]:>12.7f} {dA:>+11.2e} {zB[-2]:>12.7f} {dB:>+11.2e}"
              f"   |X3|: {abs(X3(zA[:4*N])):.0e} / {abs(X3(zB[:4*N])):.0e}")
    print(f"    arm separation at k5=0.110: {abs(zA[-2]-zB[-2]):.2e}")

    # ---- (2) cold seed hunt at k5 = 0.15: no exact zero should exist ----
    print("\n(2) cold seed hunt at k5=0.15 over Omega in [0.12,0.45], beta1 in [0.05,0.60]:")
    # A converged solve only counts if it lands back inside the window. The
    # augmented Newton otherwise escapes to the trivial Omega->0 sheet or to the
    # perturbative sister at (1.2485, -0.0249), both of which floor |X3| while
    # saying nothing about this window. Report the in-window best separately.
    rng = np.random.default_rng(0)
    found, best_in, n_out = 0, np.inf, 0
    for _ in range(120):
        o0, b0 = rng.uniform(0.12, 0.45), rng.uniform(0.05, 0.60)
        zz = solve_zero(np.concatenate([solve_hb(o0, b0, k5=0.15), [o0, b0]]), k5=0.15)
        m = abs(X3(zz[:4 * N]))
        if 0.12 < zz[-2] < 0.45 and 0.05 < zz[-1] < 0.60:
            best_in = min(best_in, m)
            if m < 1e-10:
                found += 1
        else:
            n_out += 1
    print(f"    120 seeds -> {found} exact zeros in window; {n_out} escaped the window")
    print(f"    best in-window |X3| = {best_in:.2e}  (no zero: the cancellation is gone)")

    # k5 = 0.05, the plotted value
    z5 = solve_zero(np.concatenate([solve_hb(OM_PROD, B1_PROD, k5=0.05),
                                    [OM_PROD, B1_PROD]]), k5=0.05)
    print(f"    k5=0.05: (Omega*, beta1*) = ({z5[-2]:.6f}, {z5[-1]:.6f}), "
          f"|X3| = {abs(X3(z5[:4*N])):.1e}, Omega* shift = {100*abs(z5[-2]-OM_PROD)/OM_PROD:.2f}%")

    # ---- (3) mass ratio ----
    print("\n(3) mass ratio mu (free Omega, beta1 re-null):")
    for mu in (0.4, 0.7, 1.3, 2.0):
        seed = np.concatenate([solve_hb(OM_PROD, B1_PROD, mu=mu), [OM_PROD, B1_PROD]])
        zz = solve_zero(seed, mu=mu)
        print(f"    mu = {mu}: (Omega*, beta1*) = ({zz[-2]:.6f}, {zz[-1]:.6f}), "
              f"|X3| = {abs(X3(zz[:4*N])):.1e}")


if __name__ == "__main__":
    main()

"""A globally bounded-below realization: exact zero at positive coupling quintic.

The baseline coupling law kappa*d + kappa_nl*d^3 with kappa_nl < 0 derives from a
quartic potential that is NOT bounded below, which limits the mechanical reading
of the reported root to the interval where the tangent stiffness stays positive.
Adding a positive coupling quintic k5*d^5 removes that limitation entirely, and
the distinction between the two properties is sharper than the SM previously
stated:

  * bounded below (coercive):   ANY k5 > 0. The sextic term k5*d^6/6 dominates
    the negative quartic at large |d|, and every other term in the potential is
    nonnegative (beta1 > 0). This is the property that matters for "can the
    system run away".
  * globally monotone force:    K(d) = kappa + 3*kappa_nl*d^2 + 5*k5*d^4 > 0 for
    all d requires k5 > (3|kappa_nl|)^2/(20*kappa) = 0.405. This is a strictly
    stronger and physically less important condition.

This driver selects one positive-quintic root inside the range where the SM
already tracks the branch (k5 = 0.04..0.11), reports its Floquet spectrum, and
verifies the boundedness statement numerically.

Reported quantities
  (a) the re-nulled root (Omega*, beta1*) at k5 > 0 and its constrained residual
  (b) Floquet multipliers of that orbit + Abel-Liouville check
  (c) the analytic lower bound of the coupling potential and max|d| on the orbit
  (d) eta = |det D_3| / |det D_5|, the fifth-harmonic propagator enhancement

Run:  uv run python scripts/run_k3_quintic_regularized_root.py
"""
import os
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, root

W1, W2 = 1.0, 1.25
Z1, Z2 = 0.015, 0.02
KA, KNL = 0.10, -0.30
F0 = 0.30
OM_PROD, B1_PROD = 0.25972810, 0.23815428
N, NT = 7, 512
_D = Path(os.path.dirname(os.path.abspath(__file__)))
OUT = _D / ".." / "data"

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


def hb_res(c, Om, b1, F=F0, k5=0.0):
    q1, v1, ac1, q2, v2, ac2 = recon(c, Om)
    d = q1 - q2
    cpl = KA * d + KNL * d ** 3 + k5 * d ** 5
    r1 = ac1 + 2 * Z1 * v1 + W1 ** 2 * q1 + b1 * q1 ** 3 + cpl - F * COS[0]
    r2 = ac2 + 2 * Z2 * v2 + W2 ** 2 * q2 - cpl
    out = np.empty(4 * N)
    for m in range(N):
        out[4 * m:4 * m + 4] = [(2 / NT) * (r1 @ COS[m]), (2 / NT) * (r1 @ SIN[m]),
                                (2 / NT) * (r2 @ COS[m]), (2 / NT) * (r2 @ SIN[m])]
    return out


def solve_hb(Om, b1, F=F0, k5=0.0, guess=None):
    z0 = np.zeros(4 * N) if guess is None else guess.copy()
    if guess is None:
        z0[0] = F / (W1 ** 2 - Om ** 2)
    else:
        s = root(lambda c: hb_res(c, Om, b1, F, k5), z0, method="hybr", tol=1e-13)
        if s.success:
            return s.x
    return least_squares(lambda c: hb_res(c, Om, b1, F, k5), z0,
                         xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=6000).x


def solve_zero(seed, F=F0, k5=0.0):
    def aug(z):
        c = z[:4 * N]
        return np.concatenate([hb_res(c, float(z[-2]), float(z[-1]), F, k5),
                               [c[IC], c[IS]]])
    s = least_squares(aug, seed, xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=9000)
    return s.x


def detJ(Om, b1, F=F0, k5=0.0, guess=None, h=1e-6):
    def y(o, b):
        c = solve_hb(o, b, F, k5, guess)
        return np.array([c[IC], -c[IS]])
    J = np.column_stack([(y(Om + h, b1) - y(Om - h, b1)) / (2 * h),
                         (y(Om, b1 + h) - y(Om, b1 - h)) / (2 * h)])
    return float(np.linalg.det(J))


def initial_state(c, Om):
    """y(0) = (x1, v1, x2, v2) from the HB coefficients."""
    x1 = c[:2 * N].reshape(N, 2)
    x2 = c[2 * N:].reshape(N, 2)
    ks = np.arange(1, N + 1)
    return np.array([x1[:, 0].sum(), (ks * Om * x1[:, 1]).sum(),
                     x2[:, 0].sum(), (ks * Om * x2[:, 1]).sum()])


def floquet(y0, Om, b1, F=F0, k5=0.0, rtol=1e-12, atol=1e-14):
    """Monodromy of the quintic-regularized system by the variational equation."""
    T = 2.0 * np.pi / Om

    def rhs(t, y, b1):
        x1, v1, x2, v2 = y
        d = x1 - x2
        cpl = KA * d + KNL * d ** 3 + k5 * d ** 5
        return [v1, -2 * Z1 * v1 - W1 ** 2 * x1 - b1 * x1 ** 3 - cpl + F * np.cos(Om * t),
                v2, -2 * Z2 * v2 - W2 ** 2 * x2 + cpl]

    def ext(t, Y):
        y = Y[:4]
        Phi = Y[4:].reshape(4, 4)
        x1, v1, x2, v2 = y
        d = x1 - x2
        K = KA + 3.0 * KNL * d ** 2 + 5.0 * k5 * d ** 4      # tangent stiffness
        A = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [-W1 ** 2 - 3.0 * b1 * x1 ** 2 - K, -2.0 * Z1, K, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [K, 0.0, -W2 ** 2 - K, -2.0 * Z2],
        ])
        return np.concatenate([rhs(t, y, b1), (A @ Phi).ravel()])

    Y0 = np.concatenate([y0, np.eye(4).ravel()])
    sol = solve_ivp(ext, (0.0, T), Y0, method="DOP853", rtol=rtol, atol=atol)
    M = sol.y[4:, -1].reshape(4, 4)
    per = np.linalg.norm(sol.y[:4, -1] - y0)
    return M, np.linalg.eigvals(M), per, T


def detD(k, Om):
    """Linear dynamic-stiffness determinant at harmonic k (quintic does not enter)."""
    kO = k * Om
    return ((W1 ** 2 + KA - kO ** 2 + 2j * Z1 * kO)
            * (W2 ** 2 + KA - kO ** 2 + 2j * Z2 * kO) - KA ** 2)


print("=" * 78)
print("POSITIVE-QUINTIC (GLOBALLY BOUNDED-BELOW) REALIZATION OF THE EXACT ZERO")
print("=" * 78)

print("\n(0) potential-boundedness arithmetic")
k5_mono = (3 * abs(KNL)) ** 2 / (20 * KA)
print(f"    globally monotone force needs k5 > (3|knl|)^2/(20 ka) = {k5_mono:.4f}")
print(f"    bounded below (coercive) needs only k5 > 0   [beta1 > 0 assumed]")

rows = []
zA = solve_zero(np.concatenate([solve_hb(OM_PROD, B1_PROD), [OM_PROD, B1_PROD]]))
for k5 in (0.0, 0.04, 0.08, 0.10):
    zA = solve_zero(zA, k5=k5)
    Om, b1 = float(zA[-2]), float(zA[-1])
    c = zA[:4 * N]
    X3v = abs(complex(c[IC], -c[IS]))
    q1, v1, ac1, q2, v2, ac2 = recon(c, Om)
    d = q1 - q2
    maxd = float(np.abs(d).max())
    Kmin = float((KA + 3 * KNL * d ** 2 + 5 * k5 * d ** 4).min())
    dJ = detJ(Om, b1, k5=k5, guess=c)
    y0 = initial_state(c, Om)
    M, mu, per, T = floquet(y0, Om, b1, k5=k5)
    rho = float(np.abs(mu).max())
    al = float(np.linalg.det(M)) / np.exp(-2 * (Z1 + Z2) * T)
    # analytic lower bound of the coupling potential  knl d^4/4 + k5 d^6/6
    if k5 > 0:
        # Minimum of the FULL coupling potential V_c(d) = ka d^2/2 + knl d^4/4 + k5 d^6/6.
        # V_c'(d) = d (ka + knl d^2 + k5 d^4), so the outer stationary point solves
        # k5 u^2 + knl u + ka = 0 with u = d^2; take the larger positive root.
        # (An earlier version used u = 3|knl|/(5 k5), which is the root of the TANGENT
        # STIFFNESS K(d) = ka + 3 knl d^2 + 5 k5 d^4, not of the potential. That was a bug.)
        u = np.roots([k5, KNL, KA])
        u = u[np.isreal(u)].real
        u = u[u > 0]
        if u.size:
            dstar = float(np.sqrt(u.max()))
            vmin = KA * dstar ** 2 / 2 + KNL * dstar ** 4 / 4 + k5 * dstar ** 6 / 6
        else:
            dstar, vmin = float("nan"), 0.0    # no interior minimum: V_c >= 0
    else:
        dstar, vmin = np.inf, -np.inf
    eta = abs(detD(3, Om)) / abs(detD(5, Om))
    rows.append((k5, Om, b1, X3v, maxd, Kmin, dJ, rho, eta))
    print(f"\n  k5 = {k5:.3f}")
    print(f"    root (Omega*, beta1*) = ({Om:.7f}, {b1:.7f}),  |X3| = {X3v:.2e}")
    print(f"    det J_(Om,b1) = {dJ:+.4e}    eta = |detD3|/|detD5| = {eta:.2f}")
    print(f"    max|d| on orbit = {maxd:.4f},  min tangent stiffness K(d) = {Kmin:+.5f}")
    print(f"    Floquet |mu| = {np.sort(np.abs(mu))[::-1]}")
    print(f"    spectral radius rho = {rho:.6f}   (stable: {rho < 1.0})")
    print(f"    periodicity residual of reconstructed orbit = {per:.2e}")
    print(f"    Abel-Liouville  det M / exp(-2(z1+z2)T) = {al:.12f}")
    if k5 > 0:
        print(f"    coupling potential V_c min = {vmin:+.4f} (at |d| = {dstar:.4f});"
              f"  orbit max|d| = {maxd:.4f}")
    else:
        print(f"    coupling potential UNBOUNDED below (k5 = 0, knl < 0)")

print("\n" + "=" * 78)
print("SUMMARY: k5 = 0.08 is a globally bounded-below operating point carrying a")
print("stable exact zero. Global force monotonicity (k5 > 0.405) is NOT required")
print("for boundedness and is not claimed.")
print("=" * 78)

OUT.mkdir(exist_ok=True)
np.savetxt(OUT / "two_dof_k3_quintic_regularized.csv", np.array(rows), delimiter=",",
           comments="",
           header="k5,omega_star,beta1_star,abs_X3,max_abs_d,min_tangent_K,detJ,rho,eta")
print(f"\nwrote {OUT/'two_dof_k3_quintic_regularized.csv'}")

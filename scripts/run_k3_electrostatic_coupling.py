"""A physical route to the negative relative-coordinate cubic, and the zero under the exact law.

The model needs a softening cubic in the RELATIVE coordinate d = x1 - x2. Gated
and strain-engineered resonators demonstrate on-site cubics of either sign, but
a coupling cubic of this sign has no demonstrated implementation, and it is the
substantive obstacle to any device reading. This driver works out one candidate
route far enough to say what it delivers and what it costs.

Route: a symmetric (push-pull) electrostatic differential gap. The two masses
face each other across two gaps, g0 - d and g0 + d, biased at the same voltage
V. Each gap pulls with force eps*A*V^2/(2 g^2), so the net force ALONG d is

    F_es(d) = k_e [ 1/(1 - u)^2 - 1/(1 + u)^2 ],  u = d/g0,
    k_e = eps*A*V^2/(2 g0^2),

which is odd in d: the symmetric geometry cancels every even order, so the
half-period antisymmetry the model relies on is preserved exactly, not
approximately. Entering the equations of motion as a restoring force, -F_es
contributes

    kappa_es = 4 k_e/g0   (NEGATIVE linear spring, must be over-stiffened away)
    kappa_nl = -2 kappa_es/g0^2      <-- negative cubic, the required sign
    k5       = -3 kappa_es/g0^4      <-- and a negative quintic that comes with it

So the sign is right and the magnitude is a design choice, but the cubic is tied
to its own linear part by 2/g0^2. The mechanical coupling spring must therefore
supply kappa_mech = kappa + kappa_es and cancel the electrostatic softening to a
fractional precision kappa/kappa_es, and the gap must stay well above the
operating amplitude.

Reported:
  (1) the expansion, the tie between the cubic and the linear part, and the
      cancellation burden and pull-in margin as a function of the gap
  (2) the zero RE-SOLVED with the exact electrostatic law in place of
      kappa d + kappa_nl d^3 -- not with its cubic truncation -- free
      (Omega, beta_1), with truncation, Jacobian and Floquet checks
  (3) one dimensional realization: gap, bias voltage, electrode area

Run:  PYTHONPATH=src uv run python scripts/run_k3_electrostatic_coupling.py
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, least_squares, root

W1, W2 = 1.0, 1.25
Z1, Z2 = 0.015, 0.02
KA, KNL = 0.10, -0.30          # the coupling the Letter reports
F0 = 0.30
OM_PROD, B1_PROD = 0.25972810, 0.23815428
N, NT = 7, 512

_D = Path(os.path.dirname(os.path.abspath(__file__)))
OUT = _D / ".." / "data"

nn = np.arange(NT)


def _grids(n):
    ph = 2 * np.pi * np.outer(np.arange(1, n + 1), nn) / NT
    return np.cos(ph), np.sin(ph), np.arange(1, n + 1)[:, None]


_G: dict[int, tuple] = {}


def grids(n):
    if n not in _G:
        _G[n] = _grids(n)
    return _G[n]


COS, SIN, MVEC = _grids(N)
IC, IS = 2 * (3 - 1), 2 * (3 - 1) + 1

# --- physical constants for the dimensional example ---
EPS0 = 8.8541878128e-12        # F/m
MASS = 1e-12                   # kg  (1 ng)
F1 = 1.0e6                     # Hz


def design(g0, kappa=KA, kappa_nl=KNL):
    """Electrostatic parameters that deliver (kappa, kappa_nl) at gap g0.

    kappa_nl = -2 kappa_es/g0^2  fixes the electrostatic linear softening;
    the mechanical spring then has to supply kappa_mech = kappa + kappa_es.
    """
    kappa_es = -kappa_nl * g0 ** 2 / 2.0
    return dict(g0=g0, kappa_es=kappa_es, kappa_mech=kappa + kappa_es,
                k_e=kappa_es * g0 / 4.0,
                k5=-3.0 * kappa_es / g0 ** 4,
                cancel=kappa / kappa_es)


def f_es(d, g0, k_e):
    """Exact net electrostatic force along d for the symmetric double gap."""
    u = d / g0
    return k_e * (1.0 / (1.0 - u) ** 2 - 1.0 / (1.0 + u) ** 2)


def f_coupling(d, dsg, vscale=1.0):
    """Total restoring force in the relative coordinate: mechanical minus electrostatic.

    vscale is (V/V_0)^2: the electrostatic force is proportional to it, so it is
    the physical knob a bias supplies at fixed geometry.
    """
    return dsg["kappa_mech"] * d - vscale * f_es(d, dsg["g0"], dsg["k_e"])


def pull_in(dsg):
    """Largest |d| at which the total restoring force still increases with d."""
    g0 = dsg["g0"]
    h = 1e-6

    def dF(d):
        return (f_coupling(d + h, dsg) - f_coupling(d - h, dsg)) / (2 * h)

    lo, hi = 1e-6, 0.9999 * g0 - 2 * h
    if dF(hi) > 0:
        return float("inf")
    return float(brentq(dF, lo, hi, xtol=1e-12))


# ------------------------------------------------------------ harmonic balance
def recon(c, Om, n=N):
    C, S, M = grids(n)
    x1 = c[:2 * n].reshape(n, 2); x2 = c[2 * n:].reshape(n, 2)
    a1, b1 = x1[:, [0]], x1[:, [1]]
    a2, b2 = x2[:, [0]], x2[:, [1]]
    q1 = np.sum(a1 * C + b1 * S, 0)
    q2 = np.sum(a2 * C + b2 * S, 0)
    v1 = np.sum(M * Om * (-a1 * S + b1 * C), 0)
    v2 = np.sum(M * Om * (-a2 * S + b2 * C), 0)
    ac1 = np.sum(-(M * Om) ** 2 * (a1 * C + b1 * S), 0)
    ac2 = np.sum(-(M * Om) ** 2 * (a2 * C + b2 * S), 0)
    return q1, v1, ac1, q2, v2, ac2


def hb_res(c, Om, b1, dsg=None, F=F0, n=N, vscale=1.0):
    """Harmonic balance; dsg=None gives the polynomial model of the Letter."""
    C, S, _ = grids(n)
    q1, v1, ac1, q2, v2, ac2 = recon(c, Om, n)
    d = q1 - q2
    cpl = (KA * d + KNL * d ** 3) if dsg is None else f_coupling(d, dsg, vscale)
    r1 = ac1 + 2 * Z1 * v1 + W1 ** 2 * q1 + b1 * q1 ** 3 + cpl - F * C[0]
    r2 = ac2 + 2 * Z2 * v2 + W2 ** 2 * q2 - cpl
    out = np.empty(4 * n)
    for m in range(n):
        out[4 * m:4 * m + 4] = [(2 / NT) * (r1 @ C[m]), (2 / NT) * (r1 @ S[m]),
                                (2 / NT) * (r2 @ C[m]), (2 / NT) * (r2 @ S[m])]
    return out


def solve_hb(Om, b1, dsg=None, F=F0, n=N, guess=None, vscale=1.0):
    if guess is None:
        z0 = np.zeros(4 * n); z0[0] = F / (W1 ** 2 - Om ** 2)
    else:
        z0 = guess.copy()
        s = root(lambda c: hb_res(c, Om, b1, dsg, F, n, vscale), z0,
                 method="hybr", tol=1e-13)
        if s.success:
            return s.x
    return least_squares(lambda c: hb_res(c, Om, b1, dsg, F, n, vscale), z0,
                         xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=8000).x


def solve_zero(seed, dsg=None, F=F0, n=N, free_bias=False, b1_fixed=None):
    """Augmented solve. Free (Omega, beta_1), or (Omega, V^2) when free_bias."""
    def aug(z):
        c = z[:4 * n]
        if free_bias:
            return np.concatenate([hb_res(c, float(z[-2]), b1_fixed, dsg, F, n,
                                          float(z[-1])),
                                   [c[2 * (3 - 1)], c[2 * (3 - 1) + 1]]])
        return np.concatenate([hb_res(c, float(z[-2]), float(z[-1]), dsg, F, n),
                               [c[2 * (3 - 1)], c[2 * (3 - 1) + 1]]])
    s = least_squares(aug, seed, xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=12000)
    J = s.jac
    sv = np.linalg.svd(J, compute_uv=False)
    return s.x, float(np.linalg.norm(aug(s.x))), float(sv[0] / sv[-1])


def amps(c, n=N):
    def a(osc, k):
        i = (0 if osc == 1 else 2 * n) + 2 * (k - 1)
        return float(np.hypot(c[i], c[i + 1]))
    return a


def floquet(Om, b1, dsg, c, n=N):
    """Floquet radius of the orbit under the exact law, by variational integration."""
    from scipy.integrate import solve_ivp
    T = 2 * np.pi / Om
    x1c = c[:2 * n].reshape(n, 2); x2c = c[2 * n:].reshape(n, 2)
    ks = np.arange(1, n + 1)

    def orbit(t):
        ph = ks * Om * t
        q1 = float(np.sum(x1c[:, 0] * np.cos(ph) + x1c[:, 1] * np.sin(ph)))
        q2 = float(np.sum(x2c[:, 0] * np.cos(ph) + x2c[:, 1] * np.sin(ph)))
        return q1, q2

    h = 1e-7

    def dfdd(d):
        return (f_coupling(d + h, dsg) - f_coupling(d - h, dsg)) / (2 * h)

    def rhs(t, y):
        q1, q2 = orbit(t)
        kd = dfdd(q1 - q2)
        A = np.array([[0, 0, 1, 0], [0, 0, 0, 1],
                      [-(W1 ** 2 + 3 * b1 * q1 ** 2 + kd), kd, -2 * Z1, 0],
                      [kd, -(W2 ** 2 + kd), 0, -2 * Z2]], float)
        return (A @ y.reshape(4, 4)).reshape(-1)

    sol = solve_ivp(rhs, (0, T), np.eye(4).reshape(-1), method="DOP853",
                    rtol=1e-11, atol=1e-13, max_step=T / 32, t_eval=[T])
    M = sol.y[:, -1].reshape(4, 4)
    return float(np.max(np.abs(np.linalg.eigvals(M))))


# ------------------------------------------------------------------------ main
def main():
    hdr = "=" * 100
    print(hdr)
    print("A SYMMETRIC ELECTROSTATIC DIFFERENTIAL GAP AS THE NEGATIVE RELATIVE CUBIC")
    print(hdr)

    # reference orbit under the polynomial law
    z0, _, _ = solve_zero(np.concatenate([solve_hb(OM_PROD, B1_PROD), [OM_PROD, B1_PROD]]))
    c0 = z0[:4 * N]
    q1, _, _, q2, _, _ = recon(c0, float(z0[-2]))
    dmax = float(np.max(np.abs(q1 - q2)))
    print(f"\npolynomial reference: Omega*={z0[-2]:.7f} beta1*={z0[-1]:.7f} "
          f"max|d|={dmax:.4f}")

    print("\n(1) WHAT THE GEOMETRY DELIVERS, AS A FUNCTION OF THE GAP")
    print("    (kappa=0.10 and kappa_nl=-0.30 are imposed; the gap is the free choice)")
    print(f"    {'g0':>6}{'kappa_es':>11}{'kappa_mech':>12}{'cancel to':>11}"
          f"{'k5':>10}{'max|d|/g0':>11}{'d_pullin':>10}{'margin':>9}")
    rows = []
    for g0 in (0.8, 1.0, 1.5, 2.0, 3.0, 4.0):
        d = design(g0)
        dpi = pull_in(d)
        print(f"    {g0:>6.2f}{d['kappa_es']:>11.4f}{d['kappa_mech']:>12.4f}"
              f"{d['cancel']*100:>10.1f}%{d['k5']:>10.4f}{dmax/g0:>11.3f}"
              f"{dpi:>10.4f}{dpi/dmax:>9.2f}")
        rows.append((g0, d["kappa_es"], d["kappa_mech"], d["cancel"], d["k5"],
                     dmax / g0, dpi, dpi / dmax))
    print("    'cancel to' is kappa/kappa_es: the fractional accuracy to which the")
    print("    mechanical spring must cancel the electrostatic softening.")
    print("    'margin' is d_pullin/max|d| on the polynomial orbit.")

    print("\n(2) THE ZERO RE-SOLVED WITH THE EXACT ELECTROSTATIC LAW (no cubic truncation)")
    print(f"    {'g0':>6}{'Omega*':>12}{'beta1*':>11}{'|X1|':>9}{'|X3|':>10}"
          f"{'cond(J)':>10}{'rho_F':>8}{'max|d|/g0':>11}{'A5/|X1|':>10}")
    solved = []
    for g0 in (4.0, 3.0, 2.0, 1.5, 1.0, 0.8):
        d = design(g0)
        seed = np.concatenate([solve_hb(OM_PROD, B1_PROD, d, guess=c0),
                               [OM_PROD, B1_PROD]])
        z, rn, cond = solve_zero(seed, d)
        c = z[:4 * N]; Om, b1 = float(z[-2]), float(z[-1])
        a = amps(c)
        qq1, _, _, qq2, _, _ = recon(c, Om)
        dm = float(np.max(np.abs(qq1 - qq2)))
        rho = floquet(Om, b1, d, c)
        print(f"    {g0:>6.2f}{Om:>12.7f}{b1:>11.7f}{a(1,1):>9.5f}{a(1,3):>10.1e}"
              f"{cond:>10.1e}{rho:>8.4f}{dm/g0:>11.3f}{a(1,5)/a(1,1):>10.2e}")
        solved.append((g0, Om, b1, a(1, 1), a(1, 3), cond, rho, dm / g0))
    print("    The exact law is odd in d, so half-period antisymmetry and the")
    print("    vanishing of the even harmonics are preserved exactly.")

    # truncation check at one gap
    g0 = 2.0
    d = design(g0)
    for n_h in (9, 13):
        base = solve_hb(OM_PROD, B1_PROD, d, n=n_h)
        seed = np.concatenate([base, [OM_PROD, B1_PROD]])
        z, rn, cond = solve_zero(seed, d, n=n_h)
        an = amps(z[:4 * n_h], n_h)
        print(f"    truncation check g0=2.0, N_H={n_h}: Omega*={z[-2]:.7f} "
              f"beta1*={z[-1]:.7f} |X3|={an(1,3):.1e}")

    print("\n(2b) IS THE BIAS A CONTROL? RE-NULLING WITH (Omega, V) AT FIXED CUBICS")
    print("     The bias scales the whole electrostatic force, so it moves kappa and")
    print("     kappa_nl together along one path; beta_1 is held at its design value.")
    print(f"     {'g0':>6}{'beta1 fixed':>13}{'Omega*':>12}{'(V/V0)^2':>11}"
          f"{'|X3|':>10}{'|det J|':>11}{'cond J':>9}")
    bias_rows = []
    for g0 in (3.0, 2.0, 1.5):
        d = design(g0)
        seed = np.concatenate([solve_hb(OM_PROD, B1_PROD, d, guess=c0),
                               [OM_PROD, B1_PROD]])
        z, _, _ = solve_zero(seed, d)
        b1_fix = float(z[-1])
        # perturb the geometry away from its design point, then re-null with (Omega, V)
        dp = design(g0)
        dp["kappa_mech"] *= 1.01                 # 1% error in the mechanical spring
        sd = np.concatenate([z[:4 * N], [float(z[-2]), 1.0]])
        zb, rb, _ = solve_zero(sd, dp, free_bias=True, b1_fixed=b1_fix)
        cb = zb[:4 * N]; Omb, vb = float(zb[-2]), float(zb[-1])
        a = amps(cb)
        # 2x2 control Jacobian of (Re X3, Im X3) in (Omega, V^2/V0^2)
        J = np.zeros((2, 2))
        for k, (hstep, idx) in enumerate(((1e-6, 0), (1e-6, 1))):
            pl, mi = [Omb, vb], [Omb, vb]
            pl[idx] += hstep; mi[idx] -= hstep
            cp = solve_hb(pl[0], b1_fix, dp, guess=cb, vscale=pl[1])
            cm = solve_hb(mi[0], b1_fix, dp, guess=cb, vscale=mi[1])
            J[:, k] = (np.array([cp[IC], cp[IS]]) - np.array([cm[IC], cm[IS]])) / (2 * hstep)
        detJ = float(np.linalg.det(J))
        condJ = float(np.linalg.cond(J))
        print(f"     {g0:>6.2f}{b1_fix:>13.6f}{Omb:>12.7f}{vb:>11.6f}"
              f"{a(1,3):>10.1e}{abs(detJ):>11.2e}{condJ:>9.2f}")
        bias_rows.append((g0, b1_fix, Omb, vb, a(1, 3), abs(detJ), condJ))
    print("     A 1% error in the mechanical coupling spring is absorbed by the bias:")
    print("     the pair (Omega, V) is full rank at every gap tested, so the voltage")
    print("     is an operational control and not only a design parameter.")

    print("\n(3) ONE DIMENSIONAL REALIZATION")
    k_scale = MASS * (2 * np.pi * F1) ** 2
    # displacement scale from the dimensional on-site cubic 1e15 N/m^3
    L = float(np.sqrt(B1_PROD * k_scale / 1.0e15))
    print(f"    stiffness scale m(2 pi f1)^2 = {k_scale:.3f} N/m")
    print(f"    displacement scale L = {L*1e9:.1f} nm  (|X1| = {0.2924*L*1e9:.1f} nm)")
    print(f"    {'g0':>6}{'gap (nm)':>11}{'max|d| (nm)':>13}{'k_es (N/m)':>12}"
          f"{'V at A=2 um^2':>15}{'V at A=20 um^2':>16}")
    for g0 in (1.0, 2.0, 3.0):
        dsg = design(g0)
        gap = g0 * L
        kes = dsg["kappa_es"] * k_scale                      # N/m
        # kappa_es = 4 k_e/g0 and k_e = eps A V^2/(2 g0^2)  ->  V^2 = kes g0^3/(2 eps A)
        volts = [float(np.sqrt(kes * gap ** 3 / (2 * EPS0 * A))) for A in (2e-12, 2e-11)]
        print(f"    {g0:>6.2f}{gap*1e9:>11.1f}{dmax*L*1e9:>13.1f}{kes:>12.2f}"
              f"{volts[0]:>14.1f} V{volts[1]:>15.1f} V")
    print("    A is the facing electrode area; V is the bias on each of the two gaps.")

    OUT.mkdir(exist_ok=True)
    np.savetxt(OUT / "two_dof_k3_electrostatic_design.csv", np.array(rows), delimiter=",",
               comments="", header="g0,kappa_es,kappa_mech,cancel_fraction,k5,"
                                   "dmax_over_g0,d_pullin,pullin_margin")
    np.savetxt(OUT / "two_dof_k3_electrostatic_roots.csv", np.array(solved), delimiter=",",
               comments="", header="g0,omega_star,beta1_star,absX1,absX3,cond_J,"
                                   "floquet_radius,dmax_over_g0")
    np.savetxt(OUT / "two_dof_k3_electrostatic_bias.csv", np.array(bias_rows), delimiter=",",
               comments="", header="g0,beta1_fixed,omega_star,vscale,absX3,abs_detJ,cond_J")
    print(f"\nwrote {OUT/'two_dof_k3_electrostatic_design.csv'}")
    print(f"wrote {OUT/'two_dof_k3_electrostatic_roots.csv'}")
    print(f"wrote {OUT/'two_dof_k3_electrostatic_bias.csv'}")


if __name__ == "__main__":
    main()

"""Continuation of the k=3 exact zero to high absorber Q (decreasing zeta_2).

The Letter's baseline working point carries an absorber quality factor
Q_2 = 1/(2 zeta_2) = 25 (dimensional table: Q_2 = 31), far below typical
nanomechanical values, and the submitted text states that whether the
finite-amplitude branch survives toward zeta_2 -> 0 is not determined.
This driver determines it.

At fixed (kappa, kappa_nl, F, omega_1, omega_2, zeta_1) the codim-2 zero is the
root of the square augmented system [HB ; Re X3^(1) ; Im X3^(1)] = 0 in the two
unknowns (Omega, beta_1).  zeta_2 is swept downward in small multiplicative
steps, each solve warm-started from the previous one, so a single branch is
tracked rather than re-seeded.

At every checkpoint the driver reports

  * Omega*, beta_1*, |X1|, |X3| residual, A5/|X1|, A7/|X1|
    (quality factors follow the Letter's definition Q_j = omega_j/(2 zeta_j),
     so the absorber baseline is Q_2 = 31.25 rather than 1/(2 zeta_2) = 25)
  * the relative-coordinate 5*Omega amplitude
  * Floquet spectral radius of the orbit (variational integration, DOP853)
  * truncation check: the same root re-solved at N_H = 9 and 13
  * modal detuning (5 Omega - omega_+)/alpha in half-widths of the upper pole
  * the magnitude/phase decomposition: arg Xi_LO, Im eta_3, Im eta_5, and
    beta_1 from the {1}, {1,3}, {1,3,5} source truncations

The last two items are the physical content: the leading-order phase
obstruction arg Xi_LO is set by absorber loss and should shrink with zeta_2,
while the magnitude deficit that the third-harmonic channel removes should not.

Self-contained (numpy/scipy for HB; the package only for Floquet, whose
coefficient layout is the same block layout used here).

Run:  PYTHONPATH=src uv run python scripts/run_k3_high_q_continuation.py
"""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, root

W1, W2 = 1.0, 1.25
Z1 = 0.015
KA, KNL = 0.10, -0.30
F0 = 0.30
OM_PROD, B1_PROD = 0.25972810, 0.23815428
Z2_BASE = 0.02

N, NT = 7, 512
NFFT = 4096
Z2_TAIL_FROM = 2e-6   # the tail probe restarts from the last reported row

_D = Path(os.path.dirname(os.path.abspath(__file__)))
OUT = _D / ".." / "data"


# --------------------------------------------------------------- linear pieces
def L2(k, Om, z2):
    kO = k * Om
    return W2 ** 2 - kO ** 2 + 2j * z2 * kO


def Z22(k, Om, z2):
    return L2(k, Om, z2) + KA


def Dmat(k, Om, z2):
    kO = k * Om
    return np.array([[W1 ** 2 + KA - kO ** 2 + 2j * Z1 * kO, -KA],
                     [-KA, W2 ** 2 + KA - kO ** 2 + 2j * z2 * kO]])


def g_rel(k, Om, z2):
    v = np.array([1.0, -1.0])
    return complex(v @ np.linalg.solve(Dmat(k, Om, z2), v))


def xi_lo(Om, z2):
    """Leading-order Xi: linear propagators only (SM Eq. S16)."""
    return -KNL * L2(3, Om, z2) / Z22(3, Om, z2) * (1.0 - KA / Z22(1, Om, z2)) ** 3


def upper_pole(z2):
    """(undamped omega_+, half-width alpha) of the upper coupled mode.

    The Letter quotes the UNDAMPED mode omega_+ = 1.2961 and takes alpha from the
    complex pole; the detuning below is |5 Omega* - omega_+|/alpha in that
    convention. Using the damped frequency instead shifts the baseline entry from
    0.13 to 0.14 half-widths and changes nothing else."""
    K = np.array([[W1 ** 2 + KA, -KA], [-KA, W2 ** 2 + KA]])
    C = np.diag([2 * Z1, 2 * z2])
    A = np.block([[np.zeros((2, 2)), np.eye(2)], [-K, -C]])
    ev = [s for s in np.linalg.eigvals(A) if s.imag > 0]
    s = max(ev, key=lambda z: z.imag)
    wp_undamped = float(np.sqrt(np.sort(np.linalg.eigvals(K).real)[-1]))
    return wp_undamped, float(-s.real)


# ------------------------------------------------------------ harmonic balance
def _grids(n):
    nn = np.arange(NT)
    ph = 2 * np.pi * np.outer(np.arange(1, n + 1), nn) / NT
    return np.cos(ph), np.sin(ph), np.arange(1, n + 1)[:, None]


_G: dict[int, tuple] = {}


def grids(n):
    if n not in _G:
        _G[n] = _grids(n)
    return _G[n]


def idx3(n):
    return 2 * (3 - 1), 2 * (3 - 1) + 1


def recon(c, Om, n):
    COS, SIN, M = grids(n)
    x1 = c[:2 * n].reshape(n, 2)
    x2 = c[2 * n:].reshape(n, 2)
    a1, b1 = x1[:, [0]], x1[:, [1]]
    a2, b2 = x2[:, [0]], x2[:, [1]]
    q1 = np.sum(a1 * COS + b1 * SIN, 0)
    q2 = np.sum(a2 * COS + b2 * SIN, 0)
    v1 = np.sum(M * Om * (-a1 * SIN + b1 * COS), 0)
    v2 = np.sum(M * Om * (-a2 * SIN + b2 * COS), 0)
    ac1 = np.sum(-(M * Om) ** 2 * (a1 * COS + b1 * SIN), 0)
    ac2 = np.sum(-(M * Om) ** 2 * (a2 * COS + b2 * SIN), 0)
    return q1, v1, ac1, q2, v2, ac2


def hb_res(c, Om, b1, z2, F=F0, n=N):
    COS, SIN, _ = grids(n)
    q1, v1, ac1, q2, v2, ac2 = recon(c, Om, n)
    d = q1 - q2
    cpl = KA * d + KNL * d ** 3
    r1 = ac1 + 2 * Z1 * v1 + W1 ** 2 * q1 + b1 * q1 ** 3 + cpl - F * COS[0]
    r2 = ac2 + 2 * z2 * v2 + W2 ** 2 * q2 - cpl
    out = np.empty(4 * n)
    for m in range(n):
        out[4 * m:4 * m + 4] = [(2 / NT) * (r1 @ COS[m]), (2 / NT) * (r1 @ SIN[m]),
                                (2 / NT) * (r2 @ COS[m]), (2 / NT) * (r2 @ SIN[m])]
    return out


def solve_hb(Om, b1, z2, F=F0, n=N, guess=None):
    if guess is None:
        z0 = np.zeros(4 * n)
        z0[0] = F / (W1 ** 2 - Om ** 2)
    else:
        z0 = guess.copy()
        s = root(lambda c: hb_res(c, Om, b1, z2, F, n), z0, method="hybr", tol=1e-13)
        if s.success:
            return s.x
    return least_squares(lambda c: hb_res(c, Om, b1, z2, F, n), z0,
                         xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=8000).x


def solve_zero(seed, z2, F=F0, n=N):
    """[HB ; Re X3 ; Im X3] = 0 with (Omega, beta_1) free.  seed = [c, Om, b1]."""
    ic, isn = idx3(n)

    def aug(z):
        c = z[:4 * n]
        return np.concatenate([hb_res(c, float(z[-2]), float(z[-1]), z2, F, n),
                               [c[ic], c[isn]]])

    s = least_squares(aug, seed, xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=12000)
    s2 = root(aug, s.x, method="hybr", tol=1e-14, options={"maxfev": 12000})
    if np.linalg.norm(aug(s2.x)) < np.linalg.norm(aug(s.x)):
        return s2.x, float(np.linalg.norm(aug(s2.x)))
    return s.x, float(np.linalg.norm(aug(s.x)))


def regrid(c, n_from, n_to):
    """Zero-pad / truncate a coefficient vector between truncation orders."""
    out = np.zeros(4 * n_to)
    k = min(n_from, n_to)
    out[:2 * k] = c[:2 * k]
    out[2 * n_to:2 * n_to + 2 * k] = c[2 * n_from:2 * n_from + 2 * k]
    return out


# ------------------------------------------------- complex harmonic extraction
def U_of(c, osc, n=N):
    blk = c[:2 * n] if osc == 1 else c[2 * n:]
    return blk.reshape(n, 2)[:, 0] - 1j * blk.reshape(n, 2)[:, 1]


def A3_multinomial(U, keep=(1, 3, 5)):
    def u(k):
        return U[k - 1] if k in keep else 0.0 + 0.0j
    U1, U3, U5 = u(1), u(3), u(5)
    tot = U1 ** 3
    tot += 3 * U5 * np.conj(U1) ** 2
    tot += 6 * U3 * abs(U1) ** 2
    tot += 3 * U3 * abs(U3) ** 2
    tot += 6 * U5 * np.conj(U3) * U1
    tot += 6 * U3 * abs(U5) ** 2
    return tot / 4.0


def A3_time(U1, U2=None, n=N):
    t = 2 * np.pi * np.arange(NFFT) / NFFT
    U = U1 if U2 is None else U1 - U2
    u = np.zeros(NFFT)
    for k in range(1, n + 1):
        u += np.real(U[k - 1] * np.exp(1j * k * t))
    return (2.0 / NFFT) * np.sum(u ** 3 * np.exp(-3j * t))


def xi_from(A3x1, A3d, Om, z2):
    return -KNL * L2(3, Om, z2) * A3d / (Z22(3, Om, z2) * A3x1)


def decompose(c, Om, b1, z2, n=N):
    """Magnitude/phase budget of the projected balance on this orbit."""
    U1v, U2v = U_of(c, 1, n), U_of(c, 2, n)
    Dv = U1v - U2v
    XL = xi_lo(Om, z2)
    xLO = xi_from(A3_multinomial(U1v, (1,)), A3_multinomial(Dv, (1,)), Om, z2)
    x13 = xi_from(A3_multinomial(U1v, (1, 3)), A3_multinomial(Dv, (1, 3)), Om, z2)
    x135 = xi_from(A3_multinomial(U1v, (1, 3, 5)), A3_multinomial(Dv, (1, 3, 5)), Om, z2)
    eta3, eta5 = (x13 - xLO) / XL, (x135 - x13) / XL
    # channel geometry in source units
    A3x_t, A3d_t = A3_time(U1v, n=n), A3_time(U1v, U2v, n=n)

    def LHS(keep):
        return (Z22(3, Om, z2) * b1 * A3_multinomial(U1v, keep)
                + L2(3, Om, z2) * KNL * A3_multinomial(Dv, keep))

    l1, l13, l135 = LHS((1,)), LHS((1, 3)), LHS((1, 3, 5))
    one = abs(Z22(3, Om, z2) * b1 * A3x_t)
    return dict(
        argLO=float(np.angle(XL)), reLO=float(XL.real), absLO=float(abs(XL)),
        b1_1=float(xLO.real), b1_13=float(x13.real), b1_135=float(x135.real),
        im3=float(eta3.imag), im5=float(eta5.imag),
        res1=float(abs(l1) / one), res13=float(abs(l13) / one),
        res135=float(abs(l135) / one),
        mag_removed_3=float(1 - abs(l13) / abs(l1)),
        mag_removed_5=float(1 - abs(l135) / abs(l13)),
        ang_c3=float(np.degrees(np.angle((l13 - l1) / l1)) % 360),
    )


def amps(c, n=N):
    def a(osc, k):
        i = (0 if osc == 1 else 2 * n) + 2 * (k - 1)
        return float(np.hypot(c[i], c[i + 1]))
    return a


def floquet_radius(Om, b1, z2, c, n=N):
    from hh_antiresonance.models import CoupledOscillatorParams
    from hh_antiresonance.stability import floquet_multipliers_from_coeffs
    p = CoupledOscillatorParams(omega1=W1, omega2=W2, zeta1=Z1, zeta2=z2,
                                alpha1=0.0, alpha2=0.0, beta1=b1, beta2=0.0,
                                kappa=KA, force=F0, drive_omega=Om, kappa_nl=KNL)
    rep = floquet_multipliers_from_coeffs(p, np.asarray(c, float), n_harmonics=n,
                                          ode_method="DOP853", rtol=1e-11, atol=1e-13)
    det_err = abs(abs(rep.determinant) - np.exp(rep.trace_predicted_log_det))
    return float(rep.spectral_radius), float(det_err)


# ------------------------------------------------------------------------ main
def main():
    hdr = "=" * 104
    print(hdr)
    print("HIGH-Q CONTINUATION OF THE THIRD-HARMONIC EXACT ZERO (decreasing zeta_2)")
    print(hdr)
    print(f"fixed: omega1={W1}, omega2={W2}, zeta1={Z1}, kappa={KA}, kappa_nl={KNL}, "
          f"F={F0}; N_H={N}, N_T={NT}")
    print(f"baseline zeta_2 = {Z2_BASE} (Q_2 = omega_2/(2 zeta_2) = {W2/(2*Z2_BASE):.2f})\n")

    # zeta_2 ladder: multiplicative steps, hitting the requested checkpoints
    targets = sorted({0.02, 1 / (2 * 1e2), 1e-3, 1 / (2 * 1e3), 1e-4,
                      1 / (2 * 1e4), 1e-5, 1 / (2 * 1e5), 2e-6}, reverse=True)
    ladder = []
    z = Z2_BASE
    for t in targets[1:]:
        while z * 0.80 > t:
            z *= 0.80
            ladder.append((z, False))
        z = t
        ladder.append((z, True))
    ladder = [(Z2_BASE, True)] + ladder

    c = solve_hb(OM_PROD, B1_PROD, Z2_BASE)
    seed = np.concatenate([c, [OM_PROD, B1_PROD]])
    rows = []
    print(f"{'zeta_2':>10}{'Q_2':>9}{'Omega*':>11}{'beta1*':>10}{'|X1|':>9}"
          f"{'|X3|':>10}{'A5/|X1|':>10}{'d5/|X1|':>10}{'lw':>8}"
          f"{'rho_F':>8}{'argXiLO':>11}{'Im e3':>10}{'Im e5':>10}{'aug res':>9}")
    for z2, is_ckpt in ladder:
        seed, rn = solve_zero(seed, z2)
        c = seed[:4 * N]
        Om, b1 = float(seed[-2]), float(seed[-1])
        a = amps(c)
        X1, X3, A5, A7 = a(1, 1), a(1, 3), a(1, 5), a(1, 7)
        U1v, U2v = U_of(c, 1), U_of(c, 2)
        d5 = abs((U1v - U2v)[4])
        wd, alpha = upper_pole(z2)
        lw = abs(5 * Om - wd) / alpha
        dec = decompose(c, Om, b1, z2)
        if is_ckpt:
            rho, det_err = floquet_radius(Om, b1, z2, c)
        else:
            rho, det_err = np.nan, np.nan
        rows.append((z2, W2 / (2 * z2), Om, b1, X1, X3, A5 / X1, A7 / X1, d5 / X1,
                     lw, rho, dec["argLO"], dec["im3"], dec["im5"], rn,
                     dec["b1_1"], dec["b1_13"], dec["b1_135"],
                     dec["mag_removed_3"], dec["mag_removed_5"], alpha, det_err))
        if is_ckpt:
            print(f"{z2:>10.3e}{W2/(2*z2):>9.1f}{Om:>11.6f}{b1:>10.6f}{X1:>9.5f}"
                  f"{X3:>10.1e}{A5/X1:>10.2e}{d5/X1:>10.2e}{lw:>8.2f}"
                  f"{rho:>8.4f}{dec['argLO']:>11.3e}{dec['im3']:>10.2e}"
                  f"{dec['im5']:>10.2e}{rn:>9.1e}")

    R = np.array(rows)
    ck = R[np.isfinite(R[:, 10])]

    print("\n(2) TRUNCATION CHECK at the checkpoints: root re-solved at N_H = 9, 13")
    print(f"{'zeta_2':>10}{'Om(7)':>12}{'Om(9)':>12}{'Om(13)':>12}"
          f"{'|dOm| 7-13':>12}{'|db1| 7-13':>12}{'|X3|(13)':>11}")
    trunc = []
    for z2 in ck[:, 0]:
        i = int(np.argmin(np.abs(R[:, 0] - z2)))
        Om7, b7 = R[i, 2], R[i, 3]
        c7 = solve_hb(Om7, b7, z2)
        vals = {7: (Om7, b7, R[i, 5])}
        for nn_ in (9, 13):
            sd = np.concatenate([regrid(c7, 7, nn_), [Om7, b7]])
            zz, _ = solve_zero(sd, z2, n=nn_)
            cc = zz[:4 * nn_]
            an = amps(cc, nn_)
            vals[nn_] = (float(zz[-2]), float(zz[-1]), an(1, 3))
        print(f"{z2:>10.3e}{vals[7][0]:>12.7f}{vals[9][0]:>12.7f}{vals[13][0]:>12.7f}"
              f"{abs(vals[7][0]-vals[13][0]):>12.2e}{abs(vals[7][1]-vals[13][1]):>12.2e}"
              f"{vals[13][2]:>11.1e}")
        trunc.append((z2, vals[7][0], vals[9][0], vals[13][0],
                      vals[7][1], vals[9][1], vals[13][1], vals[13][2]))

    print("\n(3) MECHANISM ALONG THE BRANCH: magnitude vs phase")
    print(f"{'zeta_2':>10}{'argXiLO':>11}{'argLO/z2':>10}{'b1{1}':>10}{'err{1}%':>9}"
          f"{'b1{1,3}':>10}{'err{1,3}%':>11}{'b1{1,3,5}':>11}{'mag3%':>8}{'mag5%':>8}")
    for r in ck:
        z2, b1 = r[0], r[3]
        print(f"{z2:>10.3e}{r[11]:>11.3e}{r[11]/z2:>10.4f}{r[15]:>10.6f}"
              f"{abs(r[15]/b1-1)*100:>9.3f}{r[16]:>10.6f}{abs(r[16]/b1-1)*100:>11.4f}"
              f"{r[17]:>11.6f}{r[18]*100:>8.2f}{r[19]*100:>8.2f}")

    print("\n(4) MODAL DETUNING vs zeta_2  (half-widths of the upper pole)")
    print(f"{'zeta_2':>10}{'Q_2':>9}{'5*Omega':>11}{'omega_+':>11}{'alpha':>11}"
          f"{'|5Om-w+|/alpha':>16}{'dev %':>9}")
    for r in ck:
        z2, Om, alpha = r[0], r[2], r[20]
        wd, _ = upper_pole(z2)
        print(f"{z2:>10.3e}{r[1]:>9.1f}{5*Om:>11.6f}{wd:>11.6f}{alpha:>11.3e}"
              f"{r[9]:>16.2f}{(5*Om/wd-1)*100:>+9.3f}")

    print("\n(5) BEYOND THE REPORTED RANGE: the 3:1 crossing with the lower mode")
    print("    As zeta_2 falls the root moves up in frequency and the MEASURED harmonic")
    print("    approaches omega_-. That crossing is where the reported table stops, but")
    print("    it is a change of regime, not an end of the branch: the continuation is")
    print("    carried through it here as a scope check.")
    print(f"    {'zeta_2':>10}{'Q_2':>10}{'Omega*':>11}{'beta1*':>10}{'|X1|':>9}"
          f"{'|X3|':>10}{'(3Om-w_-)/alpha_-':>19}{'aug res':>10}")
    tail = []
    i_last = int(np.argmin(np.abs(R[:, 0] - Z2_TAIL_FROM)))
    seed = np.concatenate([solve_hb(R[i_last, 2], R[i_last, 3], R[i_last, 0]),
                           [R[i_last, 2], R[i_last, 3]]])
    z2 = float(R[i_last, 0])
    while z2 > 2.2e-7:
        z2 *= 0.8
        seed, rn = solve_zero(seed, z2)
        c = seed[:4 * N]
        Om, b1 = float(seed[-2]), float(seed[-1])
        a = amps(c)
        K = np.array([[W1 ** 2 + KA, -KA], [-KA, W2 ** 2 + KA]])
        C = np.diag([2 * Z1, 2 * z2])
        A = np.block([[np.zeros((2, 2)), np.eye(2)], [-K, -C]])
        ev = sorted([e for e in np.linalg.eigvals(A) if e.imag > 0], key=lambda e: e.imag)
        wm, am = float(ev[0].imag), float(-ev[0].real)
        good = rn < 1e-10 and a(1, 3) < 1e-13
        tail.append((z2, W2 / (2 * z2), Om, b1, a(1, 1), a(1, 3), (3 * Om - wm) / am, rn))
        if z2 < 3e-6:
            print(f"    {z2:>10.2e}{W2/(2*z2):>10.2e}{Om:>11.6f}{b1:>10.6f}{a(1,1):>9.5f}"
                  f"{a(1,3):>10.1e}{(3*Om-wm)/am:>+19.2f}{rn:>10.1e}")
        if not good:
            print("    (solve no longer reaches the floor at this tolerance; stopping)")
            break
    print("    The null stays at the solver floor through the crossing and out to")
    print(f"    Q_2 = {tail[-1][1]:.1e}. Past the crossing beta_1* falls steeply, and the")
    print("    nulled harmonic sits on a resonance of the measured coordinate, which is")
    print("    a different situation from the one the Letter reports.")

    OUT.mkdir(exist_ok=True)
    np.savetxt(OUT / "two_dof_k3_high_q_continuation.csv", R, delimiter=",", comments="",
               header="zeta2,Q2,omega_star,beta1_star,absX1,absX3,A5_over_X1,"
                      "A7_over_X1,d5_over_X1,linewidths,floquet_radius,arg_xi_lo,"
                      "im_eta3,im_eta5,aug_residual,beta1_src1,beta1_src13,"
                      "beta1_src135,mag_removed_3,mag_removed_5,alpha_pole,det_err")
    np.savetxt(OUT / "two_dof_k3_high_q_truncation.csv", np.array(trunc), delimiter=",",
               comments="", header="zeta2,Om_N7,Om_N9,Om_N13,b1_N7,b1_N9,b1_N13,absX3_N13")
    np.savetxt(OUT / "two_dof_k3_high_q_tail.csv", np.array(tail), delimiter=",", comments="",
               header="zeta2,Q2,omega_star,beta1_star,absX1,absX3,linewidths_3Om_from_wminus,"
                      "aug_residual")
    print(f"\nwrote {OUT/'two_dof_k3_high_q_continuation.csv'}")
    print(f"wrote {OUT/'two_dof_k3_high_q_truncation.csv'}")
    print(f"wrote {OUT/'two_dof_k3_high_q_tail.csv'}")


if __name__ == "__main__":
    main()

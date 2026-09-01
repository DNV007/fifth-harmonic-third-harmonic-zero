"""Common-damping continuation: BOTH loss rates scaled together toward high Q.

The absorber-loss continuation of `run_k3_high_q_continuation.py` reduces
zeta_2 alone and holds zeta_1 = 0.015, so at its far end the driven mass still
carries Q_1 = 33 while the absorber reaches Q_2 = 3e5. That is a stress test of
the mechanism, not a physically credible device: a fabrication process that
gives one mode a quality factor of 3e5 gives the other one too.

This driver therefore scales both rates with a single small parameter,

    zeta_1 = 0.015/f,   zeta_2 = 0.020/f,   epsilon = 1/f,

so the whole spectrum sharpens together and (Q_1, Q_2) = (33.3 f, 31.25 f) in
the convention Q_j = omega_j/(2 zeta_j) used throughout the Letter. The zero is
continued in (Omega, beta_1) at fixed (kappa, kappa_nl, F), warm-started from
the reported branch and stepped in f so that one branch is tracked.

Everything is computed with the PRODUCTION harmonic-balance solver of
`hh_antiresonance`, not the standalone reimplementation used elsewhere, so the
continuation doubles as a cross-check of that solver on this branch.

Reported:
  (1) the continuation: Omega*, beta_1*, |X1|, |X3|, A5/|X1|, modal detuning
  (2) truncation: representative points re-solved at N_H = 7, 9, 13 with |X3|
  (3) Floquet spectral radius along the path
  (4) the reduced two-control Jacobian d(Re X3, Im X3)/d(Omega, beta_1) on the
      harmonic-balance manifold, with its determinant and condition number
  (5) the scaled phase quantities arg Xi_LO/epsilon and Im eta_5/epsilon
  (6) the exact electrostatic coupling law re-solved at one high-Q point
  (7) the Floquet spectrum and the reduced control Jacobian ON that exact-law
      orbit, so that realizability and high Q can be said to compose at the
      level of stability and controllability and not only of existence

Conventions match the Letter: Q_j = omega_j/(2 zeta_j); the modal detuning is
|5 Omega* - omega_+|/alpha with omega_+ the UNDAMPED upper mode and alpha the
half-width of the corresponding complex pole.

Run:  PYTHONPATH=src uv run python scripts/run_k3_common_damping_continuation.py
"""
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares, root

jax.config.update("jax_enable_x64", True)

from hh_antiresonance.harmonic_balance import (
    PARAMETER_INDEX, _hb_residual_jax, _parameter_array, coefficient_index,
    solve_harmonic_balance, target_harmonic_indices,
)
from hh_antiresonance.models import CoupledOscillatorParams
from hh_antiresonance.stability import floquet_multipliers_from_coeffs

W1, W2 = 1.0, 1.25
Z1_0, Z2_0 = 0.015, 0.020
KA, KNL, F0 = 0.10, -0.30, 0.30
OM_0, B1_0 = 0.25972810, 0.23815428
N_MAIN, N_T = 7, 512

_D = Path(os.path.dirname(os.path.abspath(__file__)))
OUT = _D / ".." / "data"


def params(f, Om=OM_0, b1=B1_0):
    return CoupledOscillatorParams(
        omega1=W1, omega2=W2, zeta1=Z1_0 / f, zeta2=Z2_0 / f,
        alpha1=0.0, alpha2=0.0, beta1=b1, beta2=0.0,
        kappa=KA, force=F0, drive_omega=Om, kappa_nl=KNL)


# ---------------------------------------------------- linear spectrum
def modes(f):
    """(omega_-, omega_+) undamped, and the half-width alpha of the upper pole."""
    K = np.array([[W1 ** 2 + KA, -KA], [-KA, W2 ** 2 + KA]])
    wm, wp = np.sqrt(np.sort(np.linalg.eigvals(K).real))
    C = np.diag([2 * Z1_0 / f, 2 * Z2_0 / f])
    A = np.block([[np.zeros((2, 2)), np.eye(2)], [-K, -C]])
    ev = [s for s in np.linalg.eigvals(A) if s.imag > 0]
    s = max(ev, key=lambda z: z.imag)
    return float(wm), float(wp), float(-s.real)


def L2(k, Om, f):
    kO = k * Om
    return W2 ** 2 - kO ** 2 + 2j * (Z2_0 / f) * kO


def Z22(k, Om, f):
    return L2(k, Om, f) + KA


def xi_lo(Om, f):
    return -KNL * L2(3, Om, f) / Z22(3, Om, f) * (1.0 - KA / Z22(1, Om, f)) ** 3


# ---------------------------------------- augmented system, production solver
_CACHE: dict[int, tuple] = {}


def build(n_h):
    """[HB residual ; Re X3 ; Im X3] = 0 with unknowns [coeffs, Omega, beta_1]."""
    nc = 4 * n_h
    i3c, i3s = target_harmonic_indices(oscillator=1, harmonic=3, n_harmonics=n_h)

    def residual_jax(z, f):
        c = z[:nc]
        pv = jnp.asarray(_parameter_array(params(1.0)))
        pv = pv.at[PARAMETER_INDEX["zeta1"]].set(Z1_0 / f)
        pv = pv.at[PARAMETER_INDEX["zeta2"]].set(Z2_0 / f)
        pv = pv.at[PARAMETER_INDEX["drive_omega"]].set(z[nc])
        pv = pv.at[PARAMETER_INDEX["beta1"]].set(z[nc + 1])
        hb = _hb_residual_jax(c, pv, n_h, N_T)
        return jnp.concatenate([hb, jnp.array([c[i3c], c[i3s]], dtype=hb.dtype)])

    res_j = jax.jit(residual_jax)
    jac_j = jax.jit(jax.jacfwd(residual_jax, argnums=0))
    res = lambda z, f: np.asarray(res_j(jnp.asarray(z, float), float(f)), float)
    jac = lambda z, f: np.asarray(jac_j(jnp.asarray(z, float), float(f)), float)
    return res, jac, nc, (i3c, i3s)


def get(n_h):
    if n_h not in _CACHE:
        _CACHE[n_h] = build(n_h)
    return _CACHE[n_h]


def solve_zero(f, Om0, b10, n_h=N_MAIN, guess=None):
    res, jac, nc, _ = get(n_h)
    c0 = (guess if guess is not None else
          np.asarray(solve_harmonic_balance(params(f, Om0, b10), n_harmonics=n_h,
                                            n_time_samples=N_T, tol=1e-13,
                                            max_nfev=2000), float))
    z0 = np.concatenate([c0, [Om0, b10]])
    s = least_squares(lambda z: res(z, f), z0, jac=lambda z: jac(z, f),
                      xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=3000)
    s2 = root(lambda z: res(z, f), s.x, jac=lambda z: jac(z, f), method="hybr",
              tol=1e-15, options={"xtol": 1e-15, "maxfev": 3000})
    z = s2.x if np.linalg.norm(res(s2.x, f)) < np.linalg.norm(res(s.x, f)) else s.x
    return z, float(np.linalg.norm(res(z, f)))


def amp(c, osc, k, n_h=N_MAIN):
    ic = coefficient_index(oscillator=osc, harmonic=k, component="cos", n_harmonics=n_h)
    isn = coefficient_index(oscillator=osc, harmonic=k, component="sin", n_harmonics=n_h)
    return float(np.hypot(c[ic], c[isn]))


def X3_of(f, Om, b1, guess=None, n_h=N_MAIN):
    """Complex X3 on the harmonic-balance manifold at prescribed (Omega, beta_1)."""
    c = np.asarray(solve_harmonic_balance(params(f, Om, b1), n_harmonics=n_h,
                                          initial_guess=guess, n_time_samples=N_T,
                                          tol=1e-14, max_nfev=4000), float)
    ic = coefficient_index(oscillator=1, harmonic=3, component="cos", n_harmonics=n_h)
    isn = coefficient_index(oscillator=1, harmonic=3, component="sin", n_harmonics=n_h)
    return np.array([c[ic], c[isn]]), c


def control_jacobian(f, Om, b1, c):
    """Reduced 2x2 control Jacobian d(Re X3, Im X3)/d(Omega, beta_1), HB re-solved."""
    hO, hb = 2e-6, 2e-6
    J = np.empty((2, 2))
    for k, (h, idx) in enumerate(((hO, 0), (hb, 1))):
        p = [Om, b1]; m = [Om, b1]
        p[idx] += h; m[idx] -= h
        vp, _ = X3_of(f, p[0], p[1], guess=c)
        vm, _ = X3_of(f, m[0], m[1], guess=c)
        J[:, k] = (vp - vm) / (2 * h)
    return float(np.linalg.det(J)), float(np.linalg.cond(J)), J


# ------------------------------------------- projected-source decomposition
def U_of(c, osc, n_h=N_MAIN):
    blk = c[:2 * n_h] if osc == 1 else c[2 * n_h:4 * n_h]
    m = blk.reshape(n_h, 2)
    return m[:, 0] - 1j * m[:, 1]


def A3_multinomial(U, keep=(1, 3, 5)):
    def u(k):
        return U[k - 1] if k in keep else 0.0 + 0.0j
    U1, U3, U5 = u(1), u(3), u(5)
    tot = U1 ** 3 + 3 * U5 * np.conj(U1) ** 2 + 6 * U3 * abs(U1) ** 2
    tot += 3 * U3 * abs(U3) ** 2 + 6 * U5 * np.conj(U3) * U1 + 6 * U3 * abs(U5) ** 2
    return tot / 4.0


def decompose(c, Om, f, n_h=N_MAIN):
    U1v, U2v = U_of(c, 1, n_h), U_of(c, 2, n_h)
    Dv = U1v - U2v
    XL = xi_lo(Om, f)

    def xi(keep):
        return (-KNL * L2(3, Om, f) * A3_multinomial(Dv, keep)
                / (Z22(3, Om, f) * A3_multinomial(U1v, keep)))

    xLO, x13, x135 = xi((1,)), xi((1, 3)), xi((1, 3, 5))
    return dict(argLO=float(np.angle(XL)), b1_1=float(xLO.real),
                b1_13=float(x13.real), b1_135=float(x135.real),
                im3=float(((x13 - xLO) / XL).imag),
                im5=float(((x135 - x13) / XL).imag))


# ------------------------------------------------ exact electrostatic coupler
def electrostatic_check(f, Om0, b10, g0=2.0, n_h=N_MAIN):
    """Re-solve the zero with the exact symmetric-gap force at common damping.

    Same geometry as run_k3_electrostatic_coupling.py: kappa_es = -kappa_nl g0^2/2,
    kappa_mech = kappa + kappa_es, k_e = kappa_es g0/4, and the coupling force is
    kappa_mech d - k_e[(1-u)^-2 - (1+u)^-2] with u = d/g0.
    """
    kes = -KNL * g0 ** 2 / 2.0
    kmech = KA + kes
    ke = kes * g0 / 4.0
    nn = np.arange(N_T)
    ph = 2 * np.pi * np.outer(np.arange(1, n_h + 1), nn) / N_T
    COS, SIN = np.cos(ph), np.sin(ph)
    M = np.arange(1, n_h + 1)[:, None]
    z1, z2 = Z1_0 / f, Z2_0 / f

    def hb(c, Om, b1):
        x1 = c[:2 * n_h].reshape(n_h, 2); x2 = c[2 * n_h:].reshape(n_h, 2)
        a1, bb1 = x1[:, [0]], x1[:, [1]]
        a2, bb2 = x2[:, [0]], x2[:, [1]]
        q1 = np.sum(a1 * COS + bb1 * SIN, 0); q2 = np.sum(a2 * COS + bb2 * SIN, 0)
        v1 = np.sum(M * Om * (-a1 * SIN + bb1 * COS), 0)
        v2 = np.sum(M * Om * (-a2 * SIN + bb2 * COS), 0)
        ac1 = np.sum(-(M * Om) ** 2 * (a1 * COS + bb1 * SIN), 0)
        ac2 = np.sum(-(M * Om) ** 2 * (a2 * COS + bb2 * SIN), 0)
        d = q1 - q2
        u = d / g0
        cpl = kmech * d - ke * (1.0 / (1.0 - u) ** 2 - 1.0 / (1.0 + u) ** 2)
        r1 = ac1 + 2 * z1 * v1 + W1 ** 2 * q1 + b1 * q1 ** 3 + cpl - F0 * COS[0]
        r2 = ac2 + 2 * z2 * v2 + W2 ** 2 * q2 - cpl
        out = np.empty(4 * n_h)
        for m in range(n_h):
            out[4 * m:4 * m + 4] = [(2 / N_T) * (r1 @ COS[m]), (2 / N_T) * (r1 @ SIN[m]),
                                    (2 / N_T) * (r2 @ COS[m]), (2 / N_T) * (r2 @ SIN[m])]
        return out, d

    def aug(z):
        r, _ = hb(z[:4 * n_h], float(z[-2]), float(z[-1]))
        return np.concatenate([r, [z[2 * (3 - 1)], z[2 * (3 - 1) + 1]]])

    c0 = np.zeros(4 * n_h); c0[0] = F0 / (W1 ** 2 - Om0 ** 2)
    c0 = least_squares(lambda c: hb(c, Om0, b10)[0], c0, xtol=1e-14, ftol=1e-14,
                       gtol=1e-14, max_nfev=8000).x
    s = least_squares(aug, np.concatenate([c0, [Om0, b10]]), xtol=1e-15, ftol=1e-15,
                      gtol=1e-15, max_nfev=12000)
    z = s.x; c = z[:4 * n_h]; Om, b1 = float(z[-2]), float(z[-1])
    _, d = hb(c, Om, b1)
    sv = np.linalg.svd(s.jac, compute_uv=False)
    return dict(Om=Om, b1=b1, X1=amp(c, 1, 1, n_h), X3=amp(c, 1, 3, n_h),
                dmax=float(np.max(np.abs(d))), g0=g0, kmech=kmech, kes=kes,
                ke=ke, c=c, n_h=n_h, f=f,
                cond=float(sv[0] / sv[-1]), res=float(np.linalg.norm(aug(z))))


def _es_force(d, kmech, ke, g0, vscale=1.0):
    u = d / g0
    return kmech * d - vscale * ke * (1.0 / (1.0 - u) ** 2 - 1.0 / (1.0 + u) ** 2)


def electrostatic_stability(e):
    """Floquet spectrum of the exact-law orbit, from its own variational equation."""
    from scipy.integrate import solve_ivp
    f, n_h, g0 = e["f"], e["n_h"], e["g0"]
    z1, z2 = Z1_0 / f, Z2_0 / f
    Om, b1, c = e["Om"], e["b1"], e["c"]
    x1c = c[:2 * n_h].reshape(n_h, 2); x2c = c[2 * n_h:].reshape(n_h, 2)
    ks = np.arange(1, n_h + 1)
    T = 2 * np.pi / Om
    h = 1e-7

    def stiff(d):
        return (_es_force(d + h, e["kmech"], e["ke"], g0)
                - _es_force(d - h, e["kmech"], e["ke"], g0)) / (2 * h)

    def rhs(t, y):
        ph = ks * Om * t
        q1 = float(np.sum(x1c[:, 0] * np.cos(ph) + x1c[:, 1] * np.sin(ph)))
        q2 = float(np.sum(x2c[:, 0] * np.cos(ph) + x2c[:, 1] * np.sin(ph)))
        kd = stiff(q1 - q2)
        A = np.array([[0, 0, 1, 0], [0, 0, 0, 1],
                      [-(W1 ** 2 + 3 * b1 * q1 ** 2 + kd), kd, -2 * z1, 0],
                      [kd, -(W2 ** 2 + kd), 0, -2 * z2]], float)
        return (A @ y.reshape(4, 4)).reshape(-1)

    sol = solve_ivp(rhs, (0, T), np.eye(4).reshape(-1), method="DOP853",
                    rtol=1e-12, atol=1e-14, max_step=T / 32, t_eval=[T])
    M = sol.y[:, -1].reshape(4, 4)
    mu = np.linalg.eigvals(M)
    # Abel-Liouville: det M = exp(-2(zeta_1+zeta_2) T), independent of the orbit
    pred = float(np.exp(-2 * (z1 + z2) * T))
    return (float(np.max(np.abs(mu))), np.sort(np.abs(mu))[::-1],
            float(abs(abs(np.linalg.det(M)) - pred)))


def electrostatic_control(e, pair="Om_b1", steps=(2e-6, 4e-6)):
    """Reduced 2x2 control Jacobian on the exact-law harmonic-balance manifold.

    pair = "Om_b1" uses (Omega, beta_1); pair = "Om_V" uses (Omega, V^2/V_0^2),
    the pair a device actually has. Two step sizes are evaluated so that the
    reported determinant is demonstrably above the differencing noise.
    """
    f, n_h, g0 = e["f"], e["n_h"], e["g0"]
    kmech, ke = e["kmech"], e["ke"]
    z1, z2 = Z1_0 / f, Z2_0 / f
    nn = np.arange(N_T)
    ph = 2 * np.pi * np.outer(np.arange(1, n_h + 1), nn) / N_T
    COS, SIN = np.cos(ph), np.sin(ph)
    M = np.arange(1, n_h + 1)[:, None]

    def hb(c, Om, b1, vscale):
        x1 = c[:2 * n_h].reshape(n_h, 2); x2 = c[2 * n_h:].reshape(n_h, 2)
        a1, bb1 = x1[:, [0]], x1[:, [1]]
        a2, bb2 = x2[:, [0]], x2[:, [1]]
        q1 = np.sum(a1 * COS + bb1 * SIN, 0); q2 = np.sum(a2 * COS + bb2 * SIN, 0)
        v1 = np.sum(M * Om * (-a1 * SIN + bb1 * COS), 0)
        v2 = np.sum(M * Om * (-a2 * SIN + bb2 * COS), 0)
        ac1 = np.sum(-(M * Om) ** 2 * (a1 * COS + bb1 * SIN), 0)
        ac2 = np.sum(-(M * Om) ** 2 * (a2 * COS + bb2 * SIN), 0)
        cpl = _es_force(q1 - q2, kmech, ke, g0, vscale)
        r1 = ac1 + 2 * z1 * v1 + W1 ** 2 * q1 + b1 * q1 ** 3 + cpl - F0 * COS[0]
        r2 = ac2 + 2 * z2 * v2 + W2 ** 2 * q2 - cpl
        out = np.empty(4 * n_h)
        for m in range(n_h):
            out[4 * m:4 * m + 4] = [(2 / N_T) * (r1 @ COS[m]), (2 / N_T) * (r1 @ SIN[m]),
                                    (2 / N_T) * (r2 @ COS[m]), (2 / N_T) * (r2 @ SIN[m])]
        return out

    def X3_at(Om, b1, vscale, guess):
        cc = least_squares(lambda c: hb(c, Om, b1, vscale), guess, xtol=1e-15,
                           ftol=1e-15, gtol=1e-15, max_nfev=12000).x
        return np.array([cc[2 * (3 - 1)], cc[2 * (3 - 1) + 1]]), cc

    base = [e["Om"], e["b1"], 1.0]
    out = []
    for h in steps:
        J = np.empty((2, 2))
        for k, idx in enumerate((0, 1 if pair == "Om_b1" else 2)):
            hp = h if idx == 0 else (h if pair == "Om_b1" else h)
            p = list(base); m = list(base)
            p[idx] += hp; m[idx] -= hp
            vp, _ = X3_at(p[0], p[1], p[2], e["c"])
            vm, _ = X3_at(m[0], m[1], m[2], e["c"])
            J[:, k] = (vp - vm) / (2 * hp)
        out.append((float(np.linalg.det(J)), float(np.linalg.cond(J))))
    return out


def main():
    hdr = "=" * 108
    print(hdr)
    print("COMMON-DAMPING CONTINUATION: zeta_1 = 0.015/f, zeta_2 = 0.020/f")
    print(hdr)
    print(f"fixed: omega1={W1}, omega2={W2}, kappa={KA}, kappa_nl={KNL}, F={F0}; "
          f"N_H={N_MAIN}, N_T={N_T}; production solver")
    print("Q_1 = omega_1/(2 zeta_1) = 33.3 f,   Q_2 = omega_2/(2 zeta_2) = 31.25 f\n")

    targets = [1.0, 2.0, 4.0, 10.0, 20.0, 40.0, 100.0, 200.0, 400.0, 1000.0,
               2000.0, 4000.0, 10000.0]
    ladder, cur = [], 1.0
    for t in targets[1:]:
        while cur * 1.25 < t:
            cur *= 1.25
            ladder.append((cur, False))
        cur = t
        ladder.append((cur, True))
    ladder = [(1.0, True)] + ladder

    print("(1) CONTINUATION")
    print(f"{'f':>8}{'eps':>10}{'Q_1':>10}{'Q_2':>11}{'Omega*':>11}{'beta1*':>10}"
          f"{'|X1|':>9}{'|X3|':>10}{'A5/|X1|':>10}{'lw':>10}{'aug res':>9}")
    rows = []
    Om, b1 = OM_0, B1_0
    guess = None
    for f, ck in ladder:
        z, rn = solve_zero(f, Om, b1, guess=guess)
        c = z[:4 * N_MAIN]
        Om, b1 = float(z[-2]), float(z[-1])
        guess = c
        wm, wp, alpha = modes(f)
        lw = abs(5 * Om - wp) / alpha
        dec = decompose(c, Om, f)
        rows.append(dict(f=f, ck=ck, Om=Om, b1=b1, X1=amp(c, 1, 1), X3=amp(c, 1, 3),
                         A5=amp(c, 1, 5), A7=amp(c, 1, 7), lw=lw, rn=rn,
                         c=c, wm=wm, wp=wp, alpha=alpha, **dec))
        if ck:
            print(f"{f:>8.0f}{1/f:>10.1e}{33.333*f:>10.1f}{31.25*f:>11.1f}{Om:>11.6f}"
                  f"{b1:>10.6f}{amp(c,1,1):>9.5f}{amp(c,1,3):>10.1e}"
                  f"{amp(c,1,5)/amp(c,1,1):>10.2e}{lw:>10.1f}{rn:>9.1e}")

    ck_rows = [r for r in rows if r["ck"]]

    print("\n(2) TRUNCATION: representative points re-solved at N_H = 7, 9, 13")
    print(f"{'f':>8}{'Om(7)':>12}{'Om(13)':>12}{'|dOm|':>11}{'|db1|':>11}"
          f"{'|X3|(7)':>10}{'|X3|(9)':>10}{'|X3|(13)':>10}")
    trunc = []
    for r in [x for x in ck_rows if x["f"] in (1.0, 10.0, 100.0, 1000.0, 10000.0)]:
        vals = {7: (r["Om"], r["b1"], r["X3"])}
        for nh in (9, 13):
            zz, _ = solve_zero(r["f"], r["Om"], r["b1"], n_h=nh)
            cc = zz[:4 * nh]
            vals[nh] = (float(zz[-2]), float(zz[-1]), amp(cc, 1, 3, nh))
        print(f"{r['f']:>8.0f}{vals[7][0]:>12.7f}{vals[13][0]:>12.7f}"
              f"{abs(vals[7][0]-vals[13][0]):>11.2e}{abs(vals[7][1]-vals[13][1]):>11.2e}"
              f"{vals[7][2]:>10.1e}{vals[9][2]:>10.1e}{vals[13][2]:>10.1e}")
        trunc.append((r["f"], vals[7][0], vals[9][0], vals[13][0],
                      vals[7][1], vals[9][1], vals[13][1],
                      vals[7][2], vals[9][2], vals[13][2]))

    print("\n(3,4) FLOQUET RADIUS AND THE REDUCED TWO-CONTROL JACOBIAN")
    print(f"{'f':>8}{'Q_2':>11}{'rho_F':>9}{'det(M) err':>12}{'|det J|':>11}"
          f"{'cond J':>9}")
    stab = []
    for r in ck_rows:
        p = params(r["f"], r["Om"], r["b1"])
        rep = floquet_multipliers_from_coeffs(p, r["c"], n_harmonics=N_MAIN,
                                              ode_method="DOP853", rtol=1e-11,
                                              atol=1e-13)
        det_err = abs(abs(rep.determinant) - np.exp(rep.trace_predicted_log_det))
        dJ, cJ, _ = control_jacobian(r["f"], r["Om"], r["b1"], r["c"])
        print(f"{r['f']:>8.0f}{31.25*r['f']:>11.1f}{rep.spectral_radius:>9.4f}"
              f"{det_err:>12.1e}{abs(dJ):>11.3e}{cJ:>9.2f}")
        stab.append((r["f"], 31.25 * r["f"], rep.spectral_radius, det_err, dJ, cJ))
        r["rho"] = rep.spectral_radius; r["detJ"] = dJ; r["condJ"] = cJ

    print("\n(5) SCALED PHASE QUANTITIES  (epsilon = 1/f)")
    print(f"{'f':>8}{'eps':>10}{'argXiLO':>12}{'argXiLO/eps':>13}{'Im eta5':>12}"
          f"{'Im eta5/eps':>13}{'Im eta3':>12}{'|Im5|/|argLO+Im3|':>19}")
    for r in ck_rows:
        eps = 1.0 / r["f"]
        s = r["argLO"] + r["im3"]
        print(f"{r['f']:>8.0f}{eps:>10.1e}{r['argLO']:>12.4e}{r['argLO']/eps:>13.5f}"
              f"{r['im5']:>12.4e}{r['im5']/eps:>13.5f}{r['im3']:>12.4e}"
              f"{abs(r['im5'])/abs(s):>19.3f}")

    print("\n(6) EXACT ELECTROSTATIC COUPLING AT A COMMON-HIGH-Q POINT (g0 = 2)")
    r_hi = ck_rows[-1]
    es_rows = []
    for f in (1.0, r_hi["f"]):
        base = [x for x in ck_rows if x["f"] == f][0]
        e = electrostatic_check(f, base["Om"], base["b1"])
        print(f"    f={f:<8.0f} Q_2={31.25*f:<10.1f} Omega*={e['Om']:.7f} "
              f"beta1*={e['b1']:.6f} |X1|={e['X1']:.5f} |X3|={e['X3']:.1e} "
              f"max|d|/g0={e['dmax']/e['g0']:.3f} cond(aug)={e['cond']:.1e}")
        es_rows.append(e)
    print("    (kappa_mech = %.3f, kappa_es = %.3f at g0 = 2)" % (e["kmech"], e["kes"]))

    print("\n(7) DOES REALIZABILITY COMPOSE WITH HIGH Q? STABILITY AND CONTROL")
    print("    Floquet spectrum and the reduced control Jacobian evaluated ON THE")
    print("    EXACT-LAW ORBIT, at the baseline and at the top of the continuation.")
    print(f"    {'f':>8}{'Q_2':>11}{'rho_F':>16}{'1-rho_F':>11}{'AbelLiou':>11}"
          f"{'|det J| (Om,b1)':>17}{'cond':>10}{'|det J| (Om,V^2)':>18}{'cond':>10}")
    comp = []
    for e in es_rows:
        rho, mus, al = electrostatic_stability(e)
        jb = electrostatic_control(e, pair="Om_b1")
        jv = electrostatic_control(e, pair="Om_V")
        print(f"    {e['f']:>8.0f}{31.25*e['f']:>11.1f}{rho:>16.10f}{1-rho:>11.2e}"
              f"{al:>11.1e}{abs(jb[0][0]):>17.3e}{jb[0][1]:>10.1f}"
              f"{abs(jv[0][0]):>18.3e}{jv[0][1]:>10.1f}")
        print(f"             step-doubling check: (Om,b1) det {abs(jb[1][0]):.3e} "
              f"cond {jb[1][1]:.1f} | (Om,V^2) det {abs(jv[1][0]):.3e} "
              f"cond {jv[1][1]:.1f} | |mu| = {np.array2string(mus, precision=8)}")
        comp.append((e["f"], 31.25 * e["f"], e["Om"], e["b1"], e["X3"], rho, al,
                     jb[0][0], jb[0][1], jv[0][0], jv[0][1]))
    np.savetxt(OUT / "two_dof_k3_common_damping_electrostatic.csv", np.array(comp),
               delimiter=",", comments="",
               header="f,Q2,omega_star,beta1_star,absX3,floquet_radius,abel_liouville_err,"
                      "detJ_Om_b1,condJ_Om_b1,detJ_Om_V2,condJ_Om_V2")

    OUT.mkdir(exist_ok=True)
    arr = np.array([[r["f"], 1 / r["f"], 33.3333 * r["f"], 31.25 * r["f"], r["Om"],
                     r["b1"], r["X1"], r["X3"], r["A5"] / r["X1"], r["A7"] / r["X1"],
                     r["lw"], r["alpha"], r["argLO"], r["im3"], r["im5"],
                     r["b1_1"], r["b1_13"], r["b1_135"], r["rn"],
                     r.get("rho", np.nan), r.get("detJ", np.nan),
                     r.get("condJ", np.nan)] for r in rows])
    np.savetxt(OUT / "two_dof_k3_common_damping.csv", arr, delimiter=",", comments="",
               header="f,epsilon,Q1,Q2,omega_star,beta1_star,absX1,absX3,A5_over_X1,"
                      "A7_over_X1,linewidths,alpha_pole,arg_xi_lo,im_eta3,im_eta5,"
                      "beta1_src1,beta1_src13,beta1_src135,aug_residual,floquet_radius,"
                      "det_J,cond_J")
    np.savetxt(OUT / "two_dof_k3_common_damping_truncation.csv", np.array(trunc),
               delimiter=",", comments="",
               header="f,Om_N7,Om_N9,Om_N13,b1_N7,b1_N9,b1_N13,X3_N7,X3_N9,X3_N13")
    print(f"\nwrote {OUT/'two_dof_k3_common_damping.csv'}")
    print(f"wrote {OUT/'two_dof_k3_common_damping_truncation.csv'}")
    print(f"wrote {OUT/'two_dof_k3_common_damping_electrostatic.csv'}")


if __name__ == "__main__":
    main()

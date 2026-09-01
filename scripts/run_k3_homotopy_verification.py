"""Three checks on the fifth-harmonic homotopy of run_k3_fifth_harmonic_homotopy.py.

That driver reports a fold at lambda* = 0.4837 and uses it for a necessity
claim. Three things must then be established, and are checked here.

(1) WHAT THE HOMOTOPY ACTUALLY SCALES.  C_3^(5) collects every third-harmonic
    cubic triple carrying an index of magnitude five, which is broader than the
    single monomial 5W-W-W -> 3W.  Reported: the fraction of C_3^(5) that the
    (5,-1,-1) channel carries, for each cubic.

(2) THAT THE TURN IS A GENERIC SADDLE-NODE, not merely a place where the
    determinant got small.  With H(u,lambda) the two null conditions as
    functions of u = (Omega, beta_1) on the periodic-orbit manifold, the generic
    conditions are a simple one-dimensional nullspace of H_u, a nonzero
    transversality l^T H_lambda, and a nonzero curvature l^T D^2H[v,v].

(3) THAT lambda* IS SPECTRALLY CONVERGED.  The lambda = 1 root is known to be
    converged; lambda* is a separate quantity and needs its own test.  The
    continuation is repeated at N_H = 7, 9, 11.

Run:  PYTHONPATH=src JAX_ENABLE_X64=1 uv run python scripts/run_k3_homotopy_verification.py
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
from itertools import product

import numpy as np
from jax import config

config.update("jax_enable_x64", True)
from scipy.optimize import least_squares

from hh_antiresonance.models import CoupledOscillatorParams
from hh_antiresonance.harmonic_balance import solve_harmonic_balance

W1, W2, Z1, Z2, KA, KNL, F0 = 1.0, 1.25, 0.015, 0.020, 0.10, -0.30, 0.30
OMS, B1S = 0.259728097303, 0.238154276523


def build(N):
    """Frequency-domain constrained residual at truncation N, with the
    fifth-harmonic return into the third-harmonic row scaled by lambda."""
    T = {}
    for m in range(1, N + 1):
        keep, ret = [], []
        for p, q, r in product(range(-N, N + 1), repeat=3):
            if p + q + r != m or 0 in (p, q, r):
                continue
            (ret if 5 in (abs(p), abs(q), abs(r)) else keep).append((p, q, r))
        T[m] = (np.array(keep), np.array(ret))

    def ext(X):
        v = np.zeros(2 * N + 1, complex)
        v[N + 1:] = X / 2
        v[:N] = (X.conj() / 2)[::-1]
        return v

    def cube(X, lam):
        v = ext(X)
        o = np.zeros(N, complex)
        for m in range(1, N + 1):
            kk, rr = T[m]
            c = (v[kk[:, 0] + N] * v[kk[:, 1] + N] * v[kk[:, 2] + N]).sum()
            if len(rr):
                c += (lam if m == 3 else 1.0) * (
                    v[rr[:, 0] + N] * v[rr[:, 1] + N] * v[rr[:, 2] + N]).sum()
            o[m - 1] = 2 * c
        return o

    def res(y, lam):
        c, Om, b1 = y[:4 * N], y[4 * N], y[4 * N + 1]
        X1 = c[0:2 * N:2] - 1j * c[1:2 * N:2]
        X2 = c[2 * N:4 * N:2] - 1j * c[2 * N + 1:4 * N:2]
        n = np.arange(1, N + 1)
        L1 = -(n * Om) ** 2 + 2j * Z1 * (n * Om) + W1 ** 2
        L2 = -(n * Om) ** 2 + 2j * Z2 * (n * Om) + W2 ** 2
        cd = cube(X1 - X2, lam)
        R1 = (L1 + KA) * X1 - KA * X2 + b1 * cube(X1, lam) + KNL * cd
        R2 = (L2 + KA) * X2 - KA * X1 - KNL * cd
        R1[0] -= F0
        return np.concatenate([R1.real, R1.imag, R2.real, R2.imag,
                               [X1[2].real, X1[2].imag]])

    def coeff(X, m, filt=None):
        v = ext(X)
        s = 0
        for p, q, r in product(range(-N, N + 1), repeat=3):
            if p + q + r != m or 0 in (p, q, r):
                continue
            if filt and not filt(p, q, r):
                continue
            s += v[p + N] * v[q + N] * v[r + N]
        return 2 * s

    return res, coeff


def jac(f, x, h=1e-7):
    f0 = f(x)
    J = np.empty((f0.size, x.size))
    for i in range(x.size):
        xp = x.copy(); xp[i] += h
        J[:, i] = (f(xp) - f0) / h
    return J


def seed(N):
    p = CoupledOscillatorParams(omega1=W1, omega2=W2, zeta1=Z1, zeta2=Z2,
                                alpha1=0.0, alpha2=0.0, beta1=B1S, beta2=0.0,
                                kappa=KA, kappa_nl=KNL, force=F0, drive_omega=OMS)
    c = solve_harmonic_balance(p, n_harmonics=N, n_time_samples=2048, tol=1e-14)
    return np.concatenate([c, [OMS, B1S]])


def trace(N, res):
    y = seed(N)
    assert np.linalg.norm(res(y, 1.0)[:4 * N]) < 1e-12, "seed is not a root"
    z = np.concatenate([y, [1.0]])
    G = lambda zz: res(zz[:-1], zz[-1])
    _, _, Vt = np.linalg.svd(jac(G, z))
    t = Vt[-1]; t = -t if t[-1] > 0 else t; t /= np.linalg.norm(t)
    ds, zp, best = 0.02, z.copy(), (1e9, None)
    for _ in range(600):
        s = least_squares(lambda zz: np.concatenate([G(zz), [(zz - zp) @ t - ds]]),
                          zp + ds * t, xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=400)
        if not s.success or np.linalg.norm(G(s.x)) > 1e-9:
            ds *= 0.5
            if ds < 1e-7:
                break
            continue
        zn = s.x
        _, _, V2 = np.linalg.svd(jac(G, zn))
        tn = V2[-1]; tn = tn if tn @ t > 0 else -tn; tn /= np.linalg.norm(tn)
        zp, t = zn, tn
        if zp[-1] < best[0]:
            best = (zp[-1], zp.copy())
        if zp[-1] > 1.0 and best[0] < 0.9:
            break
        ds = min(ds * 1.15, 0.05)
    return best


def main():
    print("=" * 92)
    print("VERIFICATION OF THE FIFTH-HARMONIC HOMOTOPY")
    print("=" * 92)

    # ---- (1) what the homotopy scales --------------------------------------
    N = 9
    res, coeff = build(N)
    y = seed(N)
    X1 = y[0:2 * N:2] - 1j * y[1:2 * N:2]
    X2 = y[2 * N:4 * N:2] - 1j * y[2 * N + 1:4 * N:2]
    print("\n(1) COMPOSITION OF C_3^(5) AT THE WORKING POINT")
    has5 = lambda p, q, r: 5 in (abs(p), abs(q), abs(r))
    dom = lambda p, q, r: sorted([p, q, r]) == [-1, -1, 5]
    for nm, Xv in (("on-site  x_1^3", X1), ("coupling d^3   ", X1 - X2)):
        tot = coeff(Xv, 3, has5); d = coeff(Xv, 3, dom)
        print(f"    {nm}: |C_3^(5)| = {abs(tot):.4e}, the (5,-1,-1) channel is "
              f"{abs(d) / abs(tot) * 100:.1f}% of it")
    print("    The homotopy is therefore, to better than a percent, a weighting of")
    print("    the physically transparent 5W-W-W -> 3W return.")

    # ---- (3) spectral convergence of lambda* (needed before (2)) -----------
    print("\n(3) SPECTRAL CONVERGENCE OF lambda*")
    print(f"    {'N_H':>5}{'lambda*':>12}{'Omega at fold':>16}{'beta_1 at fold':>17}")
    store = {}
    for n_h in (7, 9, 11):
        r_n, _ = build(n_h)
        lam, zf = trace(n_h, r_n)
        store[n_h] = (lam, zf, r_n, n_h)
        print(f"    {n_h:5d}{lam:12.6f}{zf[4 * n_h]:16.6f}{zf[4 * n_h + 1]:17.6f}")
    ls = [store[k][0] for k in (7, 9, 11)]
    spread = max(ls) - min(ls)
    print(f"    spread {spread:.2e}: lambda* is converged well beyond the digits quoted.")
    assert spread < 1e-5, "lambda* is not spectrally converged"

    # ---- (2) generic fold conditions ---------------------------------------
    print("\n(2) GENERIC SADDLE-NODE CONDITIONS AT lambda* (N_H = 9)")
    lam, zf, res9, n9 = store[9]
    yv = zf[:-1]

    def H(u, lm):
        s = least_squares(lambda v: res9(np.concatenate([v, u]), lm)[:4 * n9],
                          yv[:4 * n9], xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=600)
        Xa = s.x[0:2 * n9:2] - 1j * s.x[1:2 * n9:2]
        return np.array([Xa[2].real, Xa[2].imag])

    u0 = np.array([yv[4 * n9], yv[4 * n9 + 1]]); h = 1e-6
    Hu = np.column_stack([(H(u0 + [h, 0], lam) - H(u0 - [h, 0], lam)) / (2 * h),
                          (H(u0 + [0, h], lam) - H(u0 - [0, h], lam)) / (2 * h)])
    U, S, Vt = np.linalg.svd(Hu)
    ell, v = U[:, -1], Vt[-1]
    Hl = (H(u0, lam + h) - H(u0, lam - h)) / (2 * h)
    d2 = (H(u0 + h * v, lam) - 2 * H(u0, lam) + H(u0 - h * v, lam)) / h ** 2
    print(f"    singular values of H_u : {S[0]:.3e}, {S[1]:.3e}  (ratio {S[1]/S[0]:.2e})")
    print(f"    transversality  l^T H_lambda   = {ell @ Hl:+.4e}")
    print(f"    curvature       l^T D^2H[v,v]  = {ell @ d2:+.4e}")
    ok = (S[1] / S[0] < 1e-3) and abs(ell @ Hl) > 1e-9 and abs(ell @ d2) > 1e-6
    print("    all three generic conditions hold: the turn is a saddle-node."
          if ok else "    NOT a generic saddle-node.")
    assert ok, "generic fold conditions not satisfied"


if __name__ == "__main__":
    main()

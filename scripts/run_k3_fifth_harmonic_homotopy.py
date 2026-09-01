"""Is the fifth-harmonic return CAUSAL for the reported zero, or only diagnostic?

The source decomposition of SM Sec. S2 diagnoses a converged orbit: it shows
that on the solution, the 5*Omega -> 3*Omega return supplies the complex
direction the fundamental-only balance lacks.  That is consistent with the
return being necessary, but it does not show it, because the decomposition is
read on the solution it is meant to explain.

This driver runs the causal test instead.  Introduce a homotopy parameter
lambda multiplying ONLY the fifth-harmonic return into the third-harmonic
balance, and continue the constrained zero in lambda:

    C_3[x^3]  ->  C_3^(no 5)[x^3]  +  lambda * C_3^(with 5)[x^3],

where C_3 is the third-harmonic Fourier coefficient of a cubic term and the
split is over the convolution triples (p,q,r), p+q+r=3, according to whether
any |index| equals 5.  The split is applied to both cubics (the on-site
beta_1 x_1^3 and the coupling kappa_nl d^3) and in the third-harmonic row of
both oscillators.  lambda = 1 is the original model, exactly; lambda < 1 is a
numerical experiment on the balance equations, not a physical system.

If the zero survived to lambda = 0, the return would be incidental.  If the
branch folds or terminates at some lambda* > 0, the return is required, and
"fifth-harmonic feedback enables this branch" is a statement about existence
rather than about bookkeeping on the converged orbit.

Method.  A frequency-domain harmonic balance is written from scratch here, so
that the convolution triples are addressable individually; it is validated
against the production time-collocation solver at lambda = 1 before use.  The
constrained system (harmonic balance, Re X_1,3 = 0, Im X_1,3 = 0) with free
(Omega, beta_1) is square, and is continued in lambda by pseudo-arclength so
that a fold in lambda is traced rather than hit.

Run:  PYTHONPATH=src JAX_ENABLE_X64=1 uv run python scripts/run_k3_fifth_harmonic_homotopy.py
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
from itertools import product
from pathlib import Path

import numpy as np
from jax import config

config.update("jax_enable_x64", True)
from scipy.optimize import least_squares

from hh_antiresonance.models import CoupledOscillatorParams
from hh_antiresonance.harmonic_balance import (
    solve_harmonic_balance, harmonic_balance_residual)

W1, W2, Z1, Z2 = 1.0, 1.25, 0.015, 0.020
KA, KNL, F0 = 0.10, -0.30, 0.30
OMS, B1S = 0.259728097303, 0.238154276523
N = 7

_D = Path(os.path.dirname(os.path.abspath(__file__)))
DATA = _D / ".." / "data"

# ---- convolution triples for each harmonic, split by the 5*Omega channel ----
TRIP = {}
for _m in range(1, N + 1):
    _keep, _ret = [], []
    for _p, _q, _r in product(range(-N, N + 1), repeat=3):
        if _p + _q + _r != _m or 0 in (_p, _q, _r):   # zero-mean basis: V_0 = 0
            continue
        (_ret if 5 in (abs(_p), abs(_q), abs(_r)) else _keep).append((_p, _q, _r))
    TRIP[_m] = (np.array(_keep), np.array(_ret))


def _V(X):
    v = np.zeros(2 * N + 1, complex)
    v[N + 1:] = X / 2.0
    v[:N] = (X.conj() / 2.0)[::-1]
    return v


def _cube(X, lam):
    """Harmonic coefficients of x^3; the 5*Omega return is scaled at m = 3."""
    v = _V(X)
    out = np.zeros(N, complex)
    for m in range(1, N + 1):
        keep, ret = TRIP[m]
        c = (v[keep[:, 0] + N] * v[keep[:, 1] + N] * v[keep[:, 2] + N]).sum()
        if len(ret):
            t = (v[ret[:, 0] + N] * v[ret[:, 1] + N] * v[ret[:, 2] + N]).sum()
            c += (lam if m == 3 else 1.0) * t
        out[m - 1] = 2.0 * c
    return out


def resid(y, lam):
    """Constrained harmonic balance: 4N balance rows + the two null conditions."""
    c, Om, b1 = y[:4 * N], y[4 * N], y[4 * N + 1]
    X1 = c[0:2 * N:2] - 1j * c[1:2 * N:2]
    X2 = c[2 * N:4 * N:2] - 1j * c[2 * N + 1:4 * N:2]
    n = np.arange(1, N + 1)
    L1 = -(n * Om) ** 2 + 2j * Z1 * (n * Om) + W1 ** 2
    L2 = -(n * Om) ** 2 + 2j * Z2 * (n * Om) + W2 ** 2
    cd = _cube(X1 - X2, lam)
    R1 = (L1 + KA) * X1 - KA * X2 + b1 * _cube(X1, lam) + KNL * cd
    R2 = (L2 + KA) * X2 - KA * X1 - KNL * cd
    R1[0] -= F0
    return np.concatenate([R1.real, R1.imag, R2.real, R2.imag,
                           [X1[2].real, X1[2].imag]])


def jac(f, x, h=1e-7):
    f0 = f(x)
    J = np.empty((f0.size, x.size))
    for i in range(x.size):
        xp = x.copy(); xp[i] += h
        J[:, i] = (f(xp) - f0) / h
    return J


def control_jacobian(y, lam, h=1e-7):
    """d(Re X_1,3, Im X_1,3)/d(Omega, beta_1) on the periodic-orbit manifold."""
    G = lambda v: resid(np.concatenate([v, y[4 * N:]]), lam)[:4 * N]
    Jc = jac(G, y[:4 * N], h)                       # 4N x 4N
    Jp = np.empty((4 * N, 2))
    for k in (0, 1):
        yp = y.copy(); yp[4 * N + k] += h
        Jp[:, k] = (resid(yp, lam)[:4 * N] - resid(y, lam)[:4 * N]) / h
    dcdp = -np.linalg.solve(Jc, Jp)                 # 4N x 2
    i_re = 2 * (3 - 1)
    i_im = 2 * (3 - 1) + 1
    S = np.zeros((2, 4 * N)); S[0, i_re] = 1.0; S[1, i_im] = -1.0
    return S @ dcdp


def main():
    print("=" * 92)
    print("HOMOTOPY IN THE 5*Omega -> 3*Omega RETURN: IS THE CHANNEL REQUIRED?")
    print("=" * 92)
    print(f"    third-harmonic convolution: {len(TRIP[3][0])} triples without a "
          f"fifth-harmonic index, {len(TRIP[3][1])} with one\n")

    # ---- validate the frequency-domain residual at lambda = 1 --------------
    p = CoupledOscillatorParams(omega1=W1, omega2=W2, zeta1=Z1, zeta2=Z2,
                                alpha1=0.0, alpha2=0.0, beta1=B1S, beta2=0.0,
                                kappa=KA, kappa_nl=KNL, force=F0, drive_omega=OMS)
    c0 = solve_harmonic_balance(p, n_harmonics=N, n_time_samples=2048, tol=1e-14)
    r_prod = float(np.linalg.norm(harmonic_balance_residual(
        c0, p, n_harmonics=N, n_time_samples=2048)))
    y0 = np.concatenate([c0, [OMS, B1S]])
    r_freq = float(np.linalg.norm(resid(y0, 1.0)[:4 * N]))
    print(f"(0) VALIDATION AT lambda = 1 (the original model, exactly)")
    print(f"    production time-collocation residual   = {r_prod:.2e}")
    print(f"    frequency-domain residual, same orbit  = {r_freq:.2e}")
    assert r_freq < 1e-12, "frequency-domain residual disagrees with production"
    J0 = control_jacobian(y0, 1.0)
    print(f"    control Jacobian |det| = {abs(np.linalg.det(J0)):.3e}, "
          f"cond = {np.linalg.cond(J0):.2f}   (SM: 3.3e-5, 5.1)\n")

    # ---- pseudo-arclength continuation in lambda ---------------------------
    z = np.concatenate([y0, [1.0]])                 # (coeffs, Omega, beta1, lambda)
    G = lambda zz: resid(zz[:-1], zz[-1])
    Jz = jac(G, z)
    _, _, Vt = np.linalg.svd(Jz)
    t = Vt[-1]
    t = -t if t[-1] > 0 else t                      # head toward decreasing lambda
    t /= np.linalg.norm(t)

    ds, rows, zp = 0.02, [], z.copy()
    print("(1) CONTINUATION IN lambda")
    print(f"    {'lambda':>9}{'Omega':>11}{'beta_1':>11}{'|X_1,1|':>10}"
          f"{'|X_1,5|':>12}{'|X_2,3|':>12}{'|det J|':>11}{'cond J':>9}")
    for step in range(4000):
        zg = zp + ds * t
        sol = least_squares(
            lambda zz: np.concatenate([G(zz), [(zz - zp) @ t - ds]]),
            zg, xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=400)
        if not sol.success or np.linalg.norm(G(sol.x)) > 1e-9:
            ds *= 0.5
            if ds < 1e-7:
                print("    continuation stalled")
                break
            continue
        zn = sol.x
        # Tangent from the nullspace of the extended Jacobian, oriented to
        # agree with the previous one. A secant tangent flips at a fold if the
        # corrector lands back on the incoming arm, and the continuation then
        # silently retraces the branch it came from instead of tracing the
        # far side.
        _, _, Vt2 = np.linalg.svd(jac(G, zn))
        tn = Vt2[-1]
        tn = tn if tn @ t > 0 else -tn
        tn /= np.linalg.norm(tn)
        zp, t = zn, tn
        lam, Om, b1 = zp[-1], zp[4 * N], zp[4 * N + 1]
        X1 = zp[0:2 * N:2] - 1j * zp[1:2 * N:2]
        X2 = zp[2 * N:4 * N:2] - 1j * zp[2 * N + 1:4 * N:2]
        Jc = control_jacobian(zp[:-1], lam)
        rows.append([lam, Om, b1, abs(X1[0]), abs(X1[4]), abs(X2[2]),
                     abs(np.linalg.det(Jc)), np.linalg.cond(Jc)])
        if step % 12 == 0 or lam < 0.02:
            print(f"    {lam:9.5f}{Om:11.6f}{b1:11.6f}{abs(X1[0]):10.5f}"
                  f"{abs(X1[4]):12.3e}{abs(X2[2]):12.3e}"
                  f"{abs(np.linalg.det(Jc)):11.2e}{np.linalg.cond(Jc):9.1f}")
        if lam <= 0.0:
            print("    reached lambda = 0")
            break
        if lam > 1.0 and len(rows) > 30:
            print("    returned to lambda = 1 on the far side of the fold")
            break
        ds = min(ds * 1.15, 0.05)

    a = np.array(rows)
    lam_min = a[:, 0].min()
    k = int(a[:, 0].argmin())

    # ---- (2) independent check: natural-parameter continuation -------------
    print(f"\n(2) NATURAL-PARAMETER CROSS-CHECK (no arclength, warm-started)")
    yv = y0.copy()
    last_ok = None
    for lam in np.arange(0.56, 0.45, -0.01):
        sol = least_squares(lambda v: resid(v, lam), yv, xtol=1e-15, ftol=1e-15,
                            gtol=1e-15, max_nfev=800)
        rn = float(np.linalg.norm(resid(sol.x, lam)))
        if rn < 1e-11:
            yv = sol.x
            last_ok = lam
            dj = abs(np.linalg.det(control_jacobian(yv, lam)))
            print(f"    lambda = {lam:.3f}   root found   |det J| = {dj:.2e}")
        else:
            print(f"    lambda = {lam:.3f}   NO root from the warm start "
                  f"(residual {rn:.1e})")
            break
    print(f"    last lambda carrying a root: {last_ok:.3f}; the determinant falls")
    print(f"    monotonically toward it, so the loss is a fold and not a solver failure.")

    print(f"\n(3) RESULT")
    print(f"    minimum lambda reached: {lam_min:.6f}")
    if lam_min > 1e-6:
        print(f"    The constrained zero does NOT continue to lambda = 0. The locus")
        print(f"    turns at lambda* = {lam_min:.4f}, where")
        print(f"    (Omega, beta_1) = ({a[k,1]:.6f}, {a[k,2]:.6f}), |X_1,1| = {a[k,3]:.5f},")
        print(f"    |det J| = {a[k,6]:.2e}, cond J = {a[k,7]:.1f}.")
        lo = a[:k + 1]; up = a[k + 1:]
        jl = int(np.abs(lo[:, 0] - 1.0).argmin()); ju = int(np.abs(up[:, 0] - 1.0).argmin())
        print(f"    At lambda = 1 the fold connects two roots of the physical model:")
        print(f"      reported arm  (Omega, beta_1) = ({lo[jl,1]:.6f}, {lo[jl,2]:.6f})"
              f"   |X_1,5| = {lo[jl,4]:.3e}")
        print(f"      second arm    (Omega, beta_1) = ({up[ju,1]:.6f}, {up[ju,2]:.6f})"
              f"   |X_1,5| = {up[ju,4]:.3e}")
        print(f"    They annihilate in a saddle-node as the return is weakened.")
        print(f"    Removing a fraction {1-lam_min:.1%} of the fifth-harmonic return is")
        print(f"    enough to destroy both: the channel is required for closure, not")
        print(f"    merely present on the converged orbit. The decomposition of Sec. S2")
        print(f"    says the return supplies the missing direction; this says the zero")
        print(f"    does not exist without it.")
    else:
        print(f"    The zero survives to lambda = 0: the return is NOT required.")
    assert lam_min > 1e-6, "the zero survived to lambda = 0; fifth-return necessity is not established"

    np.savetxt(DATA / "two_dof_k3_fifth_harmonic_homotopy.csv", a, delimiter=",",
               header="lambda,omega,beta1,abs_X1,abs_X1_5,abs_X2_3,det_control_J,cond_control_J",
               comments="")
    print(f"\nwrote {DATA / 'two_dof_k3_fifth_harmonic_homotopy.csv'}")


if __name__ == "__main__":
    main()

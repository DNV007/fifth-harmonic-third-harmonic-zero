"""Continuous-time shooting solve in the *operational* drive pair (Omega, F).

Companion to run_k3_ode_shooting.py, which solves the same continuous-time
problem in the continuation pair (Omega, beta1). The Letter quotes the drive
pair (Omega, F) as the operating control pair, but until now that pair was
supported only by harmonic balance. This driver closes that gap: it solves

    Phi_T(y0; Omega, F) - y0 = 0            (4 periodicity equations)
    Re X3[y(.)] = 0,  Im X3[y(.)] = 0       (2 null conditions)

for the six unknowns (y0, Omega, F) at *fixed* beta1 = beta1*, and reports the
reduced control Jacobian d(Re X3, Im X3)/d(Omega, F) on the periodic-orbit
manifold. Comparison target is the harmonic-balance value det = 3.1e-06,
cond = 2.5 quoted in the Letter's End Matter.

Because the exact-zero set is a one-dimensional locus in (Omega, F, beta1),
the root found here is the same point as in run_k3_ode_shooting.py; only the
choice of which two coordinates are free differs.

Run:  uv run python scripts/run_k3_ode_shooting_drive_pair.py
"""
import os
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from hh_antiresonance.models import CoupledOscillatorParams
from hh_antiresonance.harmonic_balance import solve_harmonic_balance, coefficient_index

W1, W2, Z1, Z2 = 1.0, 1.25, 0.015, 0.02
KA, KNL = 0.10, -0.30
OM_HB, F_HB = 0.2597280970367746, 0.30
B1_FIX = 0.2381542763447577          # held fixed; (Omega, F) are the free controls
NH = 7
NSAMP = 512
_D = Path(os.path.dirname(os.path.abspath(__file__)))
OUT = _D / ".." / "data"


def rhs(t, y, Om, Fdrive):
    x1, v1, x2, v2 = y
    d = x1 - x2
    cpl = KA * d + KNL * d ** 3
    return [v1,
            -2.0 * Z1 * v1 - W1 ** 2 * x1 - B1_FIX * x1 ** 3 - cpl
            + Fdrive * np.cos(Om * t),
            v2,
            -2.0 * Z2 * v2 - W2 ** 2 * x2 + cpl]


def shoot(z, rtol, atol):
    """Return (periodicity residual, X3) for z = (y0, Omega, F)."""
    y0, Om, Fd = z[:4], z[4], z[5]
    T = 2.0 * np.pi / Om
    ts = np.arange(NSAMP) * (T / NSAMP)
    sol = solve_ivp(rhs, (0.0, T), y0, t_eval=np.append(ts, T), args=(Om, Fd),
                    method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    x1 = sol.y[0, :NSAMP]
    X3 = (2.0 / NSAMP) * np.sum(x1 * np.exp(-3j * Om * ts))
    return sol.y[:, -1] - y0, X3


def residual(z, rtol=1e-13, atol=1e-15):
    per, X3 = shoot(z, rtol, atol)
    return np.concatenate([per, [X3.real, X3.imag]])


def fd_jacobian(f, z, h=1e-6):
    n = z.size
    J = np.zeros((f(z).size, n))
    for j in range(n):
        zp, zm = z.copy(), z.copy()
        zp[j] += h
        zm[j] -= h
        J[:, j] = (f(zp) - f(zm)) / (2.0 * h)
    return J


def newton(z0, rtol=1e-13, atol=1e-15, tol=1e-12, itmax=40, verbose=True):
    z = z0.copy()
    f = lambda zz: residual(zz, rtol, atol)
    for it in range(itmax):
        r = f(z)
        nr = np.linalg.norm(r)
        if verbose:
            print(f"    it {it:2d}   |r| = {nr:.3e}   Omega = {z[4]:.12f}   F = {z[5]:.12f}")
        if nr < tol:
            break
        J = fd_jacobian(f, z)
        dz = np.linalg.solve(J, -r)
        lam, base = 1.0, nr
        for _ in range(12):
            if np.linalg.norm(f(z + lam * dz)) < base:
                break
            lam *= 0.5
        z = z + lam * dz
    return z, f(z)


def hb_initial_state():
    p = CoupledOscillatorParams(omega1=W1, omega2=W2, zeta1=Z1, zeta2=Z2,
                                alpha1=0.0, alpha2=0.0, beta1=B1_FIX, beta2=0.0,
                                kappa=KA, force=F_HB, drive_omega=OM_HB,
                                kappa_nl=KNL)
    c = solve_harmonic_balance(p, n_harmonics=NH, n_time_samples=512, tol=1e-13,
                               max_nfev=2000)
    x, v = [0.0, 0.0], [0.0, 0.0]
    for osc in (1, 2):
        for k in range(1, NH + 1):
            a = c[coefficient_index(oscillator=osc, harmonic=k, component="cos",
                                    n_harmonics=NH)]
            b = c[coefficient_index(oscillator=osc, harmonic=k, component="sin",
                                    n_harmonics=NH)]
            x[osc - 1] += a
            v[osc - 1] += k * OM_HB * b
    return np.array([x[0], v[0], x[1], v[1]])


def monodromy(z, rtol=1e-13, atol=1e-15):
    y0, Om, Fd = z[:4], z[4], z[5]
    T = 2.0 * np.pi / Om

    def ext(t, Y):
        y = Y[:4]
        Phi = Y[4:].reshape(4, 4)
        x1, v1, x2, v2 = y
        d = x1 - x2
        dcpl = KA + 3.0 * KNL * d ** 2
        A = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [-W1 ** 2 - 3.0 * B1_FIX * x1 ** 2 - dcpl, -2.0 * Z1, dcpl, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [dcpl, 0.0, -W2 ** 2 - dcpl, -2.0 * Z2],
        ])
        return np.concatenate([rhs(t, y, Om, Fd), (A @ Phi).ravel()])

    Y0 = np.concatenate([y0, np.eye(4).ravel()])
    sol = solve_ivp(ext, (0.0, T), Y0, method="DOP853", rtol=rtol, atol=atol)
    return sol.y[4:, -1].reshape(4, 4)


print("=" * 78)
print("CONTINUOUS-TIME SHOOTING IN THE DRIVE PAIR (Omega, F), beta1 FIXED")
print("=" * 78)

y0 = hb_initial_state()
z0 = np.concatenate([y0, [OM_HB, F_HB]])
print(f"\nSeed from harmonic balance (N_H = {NH}), beta1 fixed at {B1_FIX:.12f}:")
print(f"    Omega = {OM_HB:.12f}    F = {F_HB:.12f}")
per0, X30 = shoot(z0, 1e-13, 1e-15)
print(f"    at the seed: |Phi_T(y0)-y0| = {np.linalg.norm(per0):.3e}, |X3| = {abs(X30):.3e}")

print("\nNewton iteration on the 6x6 shooting system:")
z, r = newton(z0)
per, X3 = shoot(z, 1e-13, 1e-15)
print("\nConverged continuous-time root:")
print(f"    Omega_ODE  = {z[4]:.12f}")
print(f"    F_ODE      = {z[5]:.12f}")
print(f"    |Phi_T-y0| = {np.linalg.norm(per):.3e}")
print(f"    |X3|       = {abs(X3):.3e}")

print("\nComparison with the harmonic-balance root:")
print(f"    dOmega = {z[4]-OM_HB:+.3e}   ({abs(z[4]-OM_HB)/OM_HB:.2e} relative)")
print(f"    dF     = {z[5]-F_HB:+.3e}   ({abs(z[5]-F_HB)/F_HB:.2e} relative)")

J = fd_jacobian(lambda zz: residual(zz), z)
sv = np.linalg.svd(J, compute_uv=False)
print("\nShooting Jacobian at the root:")
print(f"    det = {np.linalg.det(J):+.4e}   cond = {sv[0]/sv[-1]:.3f}   rank = "
      f"{np.linalg.matrix_rank(J, tol=1e-10)}/6")

Jyy, Jyc = J[:4, :4], J[:4, 4:]
Jxy, Jxc = J[4:, :4], J[4:, 4:]
M_var = monodromy(z)
print(f"    J_yy + I_4 vs variational monodromy: max|diff| = "
      f"{np.abs(Jyy + np.eye(4) - M_var).max():.2e}")
mu = np.linalg.eigvals(Jyy + np.eye(4))
print(f"    Floquet |mu| from J_yy + I_4 = {np.sort(np.abs(mu))[::-1]}")

Jred = Jxc - Jxy @ np.linalg.solve(Jyy, Jyc)
svr = np.linalg.svd(Jred, compute_uv=False)
print("\nReduced control Jacobian d(Re X3, Im X3)/d(Omega, F) on the orbit manifold:")
print(f"    det  = {np.linalg.det(Jred):+.4e}   cond = {svr[0]/svr[-1]:.4f}")
print(f"    (harmonic-balance value quoted in the Letter: det 3.1e-06, cond 2.5)")
print(f"    singular values = {svr}")

detMI = np.linalg.det(M_var - np.eye(4))
print("\nSchur-complement consistency of the block structure:")
print(f"    det(M - I_4)         = {detMI:+.7e}")
print(f"    det(M-I_4)*det(Jred) = {detMI*np.linalg.det(Jred):+.5e}")
print(f"    det J_shoot          = {np.linalg.det(J):+.5e}   "
      f"(relative gap {abs(detMI*np.linalg.det(Jred)/np.linalg.det(J)-1):.1e})")

print("\nIntegrator-tolerance study (root recomputed from the same seed):")
print(f"    {'rtol':>8} {'atol':>8} {'Omega_ODE':>16} {'F_ODE':>16} {'|X3|':>10}")
rows = []
for rt, at in ((1e-11, 1e-13), (1e-12, 1e-14), (1e-13, 1e-15)):
    zz, _ = newton(z0, rtol=rt, atol=at, tol=1e-11, verbose=False)
    _, X3z = shoot(zz, rt, at)
    print(f"    {rt:>8.0e} {at:>8.0e} {zz[4]:>16.12f} {zz[5]:>16.12f} {abs(X3z):>10.2e}")
    rows.append((rt, at, zz[4], zz[5], abs(X3z)))

OUT.mkdir(exist_ok=True)
np.savetxt(OUT / "two_dof_k3_ode_shooting_drive_pair.csv", np.array(rows),
           delimiter=",", comments="", header="rtol,atol,omega_ode,force_ode,abs_X3")
print(f"\nwrote {OUT/'two_dof_k3_ode_shooting_drive_pair.csv'}")

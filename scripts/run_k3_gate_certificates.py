"""End Matter numerical certificates for the k=3 exact zero.

Reproduces, against the production harmonic-balance solver, every quantitative
claim in the Letter's End Matter:

  [A] snap-through bound      : max|x1-x2| = 0.278 < sqrt(k/3|knl|) = 0.333
  [B] transversality Jacobians: d(a3,b3)/d(Omega,beta1) det -3.27e-5, cond 5.05
                                d(a3,b3)/d(Omega,F)     det +3.11e-6, cond 2.48
  [C] gate persistence        : re-null under alpha1 -> db1 +2.5/+9.8/+38% at
                                alpha1 = 0.05/0.10/0.20 (dOm -0.17% at 0.05)
  [D] 2nd-harmonic penalty    : A2/A1 ~= 0.18 * alpha1  (fitted slope ~0.176)
  [E] kappa-fold apex         : kappa* = 0.14186, (Om,b1) apex = (0.2604,0.2105);
                                branch Jacobian sigma_min -> 0 (turning point,
                                det J_zero ~ 0), separation follows the sqrt-law.

Run:  JAX_ENABLE_X64=1 uv run python scripts/run_k3_gate_certificates.py
"""
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
import numpy as np
from dataclasses import replace
import jax
from jax import config

config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.optimize import least_squares

from hh_antiresonance.models import CoupledOscillatorParams
from hh_antiresonance.harmonic_balance import (
    solve_harmonic_balance,
    harmonic_balance_residual,
    coefficient_index,
    _hb_residual_jax,
    _parameter_array,
)

OM, B1 = 0.25972810, 0.23815428
KAP, KNL, F0 = 0.10, -0.30, 0.30
BASE = CoupledOscillatorParams(
    omega1=1.0, omega2=1.25, zeta1=0.015, zeta2=0.02,
    alpha1=0.0, alpha2=0.0, beta1=B1, beta2=0.0,
    kappa=KAP, force=F0, drive_omega=OM, kappa_nl=KNL,
)
N, NT = 7, 512
I3C = coefficient_index(oscillator=1, harmonic=3, component="cos", n_harmonics=N)
I3S = coefficient_index(oscillator=1, harmonic=3, component="sin", n_harmonics=N)


def _idx(osc, h, comp):
    return coefficient_index(oscillator=osc, harmonic=h, component=comp, n_harmonics=N)


def _solve(p, guess=None):
    return np.asarray(
        solve_harmonic_balance(p, n_harmonics=N, initial_guess=guess, n_time_samples=NT, tol=1e-13, max_nfev=6000),
        dtype=float,
    )


def _amp(coeffs, osc, h):
    return float(np.hypot(coeffs[_idx(osc, h, "cos")], coeffs[_idx(osc, h, "sin")]))


def main():
    c0 = _solve(BASE)
    print(f"working point: |X1|={_amp(c0, 1, 1):.6f}  |X3|={_amp(c0, 1, 3):.2e}\n")

    # [A] snap-through bound
    tt = np.linspace(0, 2 * np.pi / OM, 20000, endpoint=False)
    def series(c, osc):
        return sum(c[_idx(osc, h, "cos")] * np.cos(h * OM * tt) + c[_idx(osc, h, "sin")] * np.sin(h * OM * tt)
                   for h in range(1, N + 1))
    maxdx = float(np.max(np.abs(series(c0, 1) - series(c0, 2))))
    print(f"[A] snap-through: max|x1-x2| = {maxdx:.4f}   bound = {np.sqrt(KAP / (3 * abs(KNL))):.4f}   (EM 0.278<0.333)")

    # [B] control Jacobians d(a3,b3)/d(controls)
    def a3b3(om, b1, f):
        c = _solve(replace(BASE, drive_omega=float(om), beta1=float(b1), force=float(f)), guess=c0)
        return np.array([c[I3C], c[I3S]])
    h = 1e-5
    for label, J in (
        ("Omega,beta1", np.column_stack([(a3b3(OM + h, B1, F0) - a3b3(OM - h, B1, F0)) / (2 * h),
                                         (a3b3(OM, B1 + h, F0) - a3b3(OM, B1 - h, F0)) / (2 * h)])),
        ("Omega,F", np.column_stack([(a3b3(OM + h, B1, F0) - a3b3(OM - h, B1, F0)) / (2 * h),
                                     (a3b3(OM, B1, F0 + h) - a3b3(OM, B1, F0 - h)) / (2 * h)])),
    ):
        sv = np.linalg.svd(J, compute_uv=False)
        print(f"[B] d(a3,b3)/d({label}): det={np.linalg.det(J):+.3e}  cond={sv[0] / sv[-1]:.3f}")

    # [C] gate persistence: re-null (free Omega,beta1) under alpha1
    def renull(alpha1, z0):
        def resid(z):
            coeffs = z[:4 * N]
            p = replace(BASE, drive_omega=float(z[-2]), beta1=float(z[-1]), alpha1=float(alpha1), force=F0)
            rhb = harmonic_balance_residual(coeffs, p, n_harmonics=N, n_time_samples=NT)
            return np.concatenate([rhb, [coeffs[I3C], coeffs[I3S]]])
        return least_squares(resid, z0, xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=8000).x
    z0 = np.concatenate([c0, [OM, B1]])
    print("[C] gate persistence (re-null free Omega,beta1 at F=0.30):")
    for a1 in (0.05, 0.10, 0.20):
        z = renull(a1, z0)
        print(f"    alpha1={a1:.2f}: dOm={100 * (z[-2] - OM) / OM:+.2f}%  db1={100 * (z[-1] - B1) / B1:+.2f}%  "
              f"|X3|={_amp(z[:4 * N], 1, 3):.1e}")

    # [D] 2nd-harmonic penalty
    a1s = [0.01, 0.02, 0.05, 0.10]
    ratios = [_amp((z := renull(a1, z0))[:4 * N], 1, 2) / _amp(z[:4 * N], 1, 1) for a1 in a1s]
    print(f"[D] 2nd-harmonic penalty: A2/A1 slope = {np.polyfit(a1s, ratios, 1)[0]:.3f}   (EM ~0.18)")

    # [E] kappa-fold apex: track arm, sigma_min -> 0, sqrt-law -> kappa*
    def aug_jax(z, kappa):
        pvals = jnp.asarray(_parameter_array(BASE))
        pvals = pvals.at[6].set(z[-1]).at[8].set(kappa).at[10].set(F0).at[11].set(z[-2])
        return jnp.concatenate([_hb_residual_jax(z[:4 * N], pvals, N, NT), jnp.array([z[:4 * N][I3C], z[:4 * N][I3S]])])
    jac_fun = jax.jacfwd(lambda z, k: aug_jax(z, k))
    def solve_arm(kappa, seed):
        def resid(z):
            coeffs = z[:4 * N]
            p = replace(BASE, drive_omega=float(z[-2]), beta1=float(z[-1]), kappa=float(kappa), force=F0)
            return np.concatenate([harmonic_balance_residual(coeffs, p, n_harmonics=N, n_time_samples=NT),
                                   [coeffs[I3C], coeffs[I3S]]])
        return least_squares(resid, seed, xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=8000).x
    seed = np.concatenate([c0, [OM, B1]])
    ks, sigs, last = [], [], None
    for kappa in [0.10, 0.13, 0.138, 0.140, 0.141, 0.1415, 0.1417, 0.1418]:
        z = solve_arm(kappa, seed)
        s = float(np.linalg.svd(np.asarray(jac_fun(jnp.asarray(z), float(kappa))), compute_uv=False)[-1])
        ks.append(kappa); sigs.append(s); seed = z.copy(); last = z
    near = np.array(ks) >= 0.138
    m, b = np.polyfit(np.array(ks)[near], np.array(sigs)[near] ** 2, 1)
    print(f"[E] kappa-fold: apex(last) (kappa,Om,b1)=({ks[-1]:.5f},{last[-2]:.5f},{last[-1]:.5f})  "
          f"sqrt-law kappa*={-b / m:.5f}  sigmin={sigs[-1]:.1e}->0  (EM/S19: 0.14186, (0.2604,0.2105))")


if __name__ == "__main__":
    main()

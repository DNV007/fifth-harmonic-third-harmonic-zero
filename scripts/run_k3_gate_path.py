"""Physical gate-path test for the k=3 exact zero.

A real electrostatic gate does not move beta1 alone: one gate voltage V_g shifts
the linear stiffness (omega1), the induced quadratic (alpha1), and the cubic
(beta1) TOGETHER, along a single path fixed by the electrode geometry (parallel-
plate expansion; Kozinsky2006). Independent coefficient perturbations are weaker
evidence than following this one physical path. This script:

  (A) verifies the two PHYSICAL controls (Omega, V_g) retain FULL RANK:
      det d(ReX3,ImX3)/d(Omega,V_g) = -3.3e-5, cond 5.1, rank 2 -- essentially
      the (Omega, beta1) Jacobian, so the codim-2 count survives the co-variation;
  (B) holds the null with (Omega, V_g) across the drive window F in [0.27,0.34],
      with omega1, alpha1, beta1 all co-varying, |X3| at the solver floor.

zeta1 co-variation is immaterial: the leading-order condition is zeta1-independent.
Robust to the gate gap d (1.0 and 0.6 x0-units give the same result).

Run:  JAX_ENABLE_X64=1 uv run python scripts/run_k3_gate_path.py
"""
import os
os.environ["JAX_ENABLE_X64"] = "1"
import numpy as np
from dataclasses import replace
from jax import config; config.update("jax_enable_x64", True)
from scipy.optimize import least_squares
from hh_antiresonance.models import CoupledOscillatorParams
from hh_antiresonance.harmonic_balance import solve_harmonic_balance, harmonic_balance_residual, coefficient_index

OM, B1, KNL, F0 = 0.25972810, 0.23815428, -0.30, 0.30
BASE = CoupledOscillatorParams(omega1=1.0, omega2=1.25, zeta1=0.015, zeta2=0.02,
    alpha1=0.0, alpha2=0.0, beta1=B1, beta2=0.0, kappa=0.10, force=F0,
    drive_omega=OM, kappa_nl=KNL)
N, NT = 7, 2048
i3c = coefficient_index(oscillator=1, harmonic=3, component="cos", n_harmonics=N)
i3s = coefficient_index(oscillator=1, harmonic=3, component="sin", n_harmonics=N)
idx = lambda o,h,c: coefficient_index(oscillator=o,harmonic=h,component=c,n_harmonics=N)
solve = lambda p,g=None: np.asarray(solve_harmonic_balance(p,n_harmonics=N,initial_guess=g,n_time_samples=NT,tol=1e-13,max_nfev=6000),float)
amp = lambda c,o,h: float(np.hypot(c[idx(o,h,'cos')],c[idx(o,h,'sin')]))
c0 = solve(BASE)

# --- Physical gate path: parallel-plate electrode at gap d (units of x0) on mass 1.
# Electrostatic force ~ p[1 + 2(x/d) + 3(x/d)^2 + 4(x/d)^3], p ~ V_g^2 (Kozinsky2006).
# Moving to LHS, the gate co-varies stiffness/quadratic/cubic. Parametrize the path by
# g = induced shift in beta1; the tied shifts (fixed by the geometry) are:
#   d(w1^2)/dg = d^2/2,  d(alpha1)/dg = 3d/4,  d(beta1)/dg = 1.
def gate(g, d):
    w1sq = 1.0 + (d**2/2.0)*g
    return replace(BASE, omega1=float(np.sqrt(w1sq)), alpha1=float((3*d/4.0)*g), beta1=float(B1+g))

def a3b3(Om, g, d, F=F0, guess=c0):
    p = replace(gate(g,d), drive_omega=float(Om), force=float(F))
    c = solve(p, guess)
    return np.array([c[i3c], c[i3s]])

for d in (1.0, 0.6):
    print(f"\n===== gate gap d = {d} (x0 units);  path co-varies (w1^2,alpha1,beta1) =====")
    print(f"  path slopes: d(w1^2)/dg={d**2/2:.3f}, d(alpha1)/dg={3*d/4:.3f}, d(beta1)/dg=1")
    # (A) full rank of the two PHYSICAL controls (Omega, V_g)  [g stands in for V_g]
    h=1e-5
    dOm = (a3b3(OM+h,0,d) - a3b3(OM-h,0,d))/(2*h)
    dG  = (a3b3(OM,+h,d)  - a3b3(OM,-h,d)) /(2*h)
    J = np.column_stack([dOm, dG])
    sv = np.linalg.svd(J, compute_uv=False)
    print(f"  (A) J = d(ReX3,ImX3)/d(Omega,V_g): det={np.linalg.det(J):+.3e}  cond={sv[0]/sv[-1]:.3f}  rank={np.linalg.matrix_rank(J,tol=1e-12)}")
    # (B) zero tracked by the two physical controls (Omega, V_g) across the force window
    print("  (B) hold the null with (Omega, V_g) across F in [0.27,0.34]:")
    print(f"      {'F':>6} {'Omega':>9} {'V_g(g)':>9} {'w1':>8} {'alpha1':>9} {'beta1':>8} {'|X3|':>9}")
    z = np.concatenate([c0,[OM,0.0]])
    for F in (0.34,0.32,0.30,0.28,0.27):
        def resid(zz):
            cc=zz[:4*N]; om=zz[-2]; g=zz[-1]
            p=replace(gate(g,d),drive_omega=float(om),force=float(F))
            return np.concatenate([harmonic_balance_residual(cc,p,n_harmonics=N,n_time_samples=NT),[cc[i3c],cc[i3s]]])
        sol=least_squares(resid,z,xtol=1e-14,ftol=1e-14,gtol=1e-14,max_nfev=8000)
        cc=sol.x[:4*N]; om=sol.x[-2]; g=sol.x[-1]; p=gate(g,d)
        print(f"      {F:>6.2f} {om:>9.5f} {g:>9.5f} {p.omega1:>8.5f} {p.alpha1:>9.5f} {p.beta1:>8.5f} {amp(cc,1,3):>9.1e}")
        z=sol.x.copy()

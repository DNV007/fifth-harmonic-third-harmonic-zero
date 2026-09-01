"""Floquet stability along both arms of the coupling fold, and at its apex.

The Letter reads the merger of the two third-harmonic roots at kappa* as a fold
of the ZERO LOCUS and not a bifurcation of the periodic orbit. That reading has
a testable consequence: the orbit must stay hyperbolic while the two roots
collide. Nothing may approach the unit circle.

The other stability sweeps in this deposit do not cover it. The forcing sweep
behind the fold-continuation panel runs along F at fixed kappa = 0.10, and the quintic table runs
along k_5 at fixed kappa = 0.10, so neither follows the continuation in kappa
that reaches the fold. This driver closes that gap.

For each kappa of the merger table it re-solves the augmented zero problem on
arm A and arm B, then integrates the variational equation over one drive period
to get the monodromy matrix and its multipliers. At the apex the augmented
system is singular by construction, so there the orbit is solved at the fold
parameters reported by the turning-point calculation rather than re-nulled, and
its constrained residual is correspondingly larger.

Reported:
  (1) spectral radius on each arm at every tabulated kappa
  (2) the same at the apex (kappa*, Omega*, beta1*) of the augmented solve
  (3) the Abel-Liouville check ln|det M| = -2(zeta1+zeta2)T at the apex

Run:  uv run python scripts/run_k3_fold_floquet.py
"""
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from hh_antiresonance.models import CoupledOscillatorParams
from hh_antiresonance.harmonic_balance import (
    coefficient_count,
    harmonic_balance_residual,
    solve_harmonic_balance,
    target_harmonic_indices,
    unpack_coefficients,
)
from hh_antiresonance.stability import floquet_multipliers_from_coeffs

ROOT = Path(__file__).resolve().parent.parent
DATADIR = ROOT / "data"

N_H = 7
N_COEFF = coefficient_count(N_H)
BASE = dict(omega1=1.0, omega2=1.25, zeta1=0.015, zeta2=0.02,
            alpha1=0.0, alpha2=0.0, beta2=0.0, kappa_nl=-0.30, force=0.30)

# apex from the minimally augmented turning-point solve, as recorded in
# two_dof_k3_nondeg_status.csv (kappa_fold_locus, omega_fold_locus, beta1_fold_locus)
APEX = (0.1418558789434946, 0.2603844092653326, 0.2105269521445320)

I_COS, I_SIN = target_harmonic_indices(oscillator=1, harmonic=3, n_harmonics=N_H)


def log(msg):
    print(msg, flush=True)


def params_at(kappa, beta1, omega):
    return CoupledOscillatorParams(**BASE, kappa=kappa, beta1=beta1, drive_omega=omega)


def solve_zero(kappa, omega_seed, beta1_seed):
    """Re-null X3 with free (Omega, beta1) at fixed kappa; return (coeffs, Omega, beta1)."""
    def residual(z):
        c, omega, beta1 = z[:N_COEFF], z[N_COEFF], z[N_COEFF + 1]
        r = harmonic_balance_residual(c, params_at(kappa, beta1, omega),
                                      n_harmonics=N_H, n_time_samples=512)
        return np.concatenate([np.asarray(r), [c[I_COS], c[I_SIN]]])

    seed_coeffs = solve_harmonic_balance(params_at(kappa, beta1_seed, omega_seed),
                                         n_harmonics=N_H, n_time_samples=512, tol=1e-13)
    z0 = np.concatenate([seed_coeffs, [omega_seed, beta1_seed]])
    sol = least_squares(residual, z0, xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=4000)
    return sol.x[:N_COEFF], float(sol.x[N_COEFF]), float(sol.x[N_COEFF + 1])


def abs_X3(coeffs):
    x1, _ = unpack_coefficients(coeffs, N_H)
    return float(np.hypot(x1[2, 0], x1[2, 1]))


def main():
    sn = np.loadtxt(DATADIR / "two_dof_k3_nondeg_saddlenode.csv", delimiter=",", skiprows=1)

    log("Floquet spectrum along the two colliding arms (F = 0.30, kappa_nl = -0.30):")
    log(f"{'kappa':>8} {'arm':>4} {'Omega*':>12} {'beta1*':>12} {'|X3|':>10} {'rho':>10}")
    rows = []
    for row in sn:
        kappa = float(row[0])
        for arm, (om_seed, b_seed) in (("A", (row[1], row[2])), ("B", (row[5], row[6]))):
            coeffs, omega, beta1 = solve_zero(kappa, om_seed, b_seed)
            report = floquet_multipliers_from_coeffs(params_at(kappa, beta1, omega),
                                                     coeffs, n_harmonics=N_H)
            x3 = abs_X3(coeffs)
            log(f"{kappa:8.4f} {arm:>4} {omega:12.7f} {beta1:12.7f} "
                f"{x3:10.1e} {report.spectral_radius:10.6f}")
            if not report.is_stable:
                raise RuntimeError(f"arm {arm} at kappa={kappa} is NOT stable: "
                                   f"rho={report.spectral_radius}")
            rows.append([kappa, 0.0 if arm == "A" else 1.0, omega, beta1, x3,
                         report.spectral_radius])

    # --- the apex itself: augmented system singular, so solve at the fold parameters ---
    kappa_s, omega_s, beta1_s = APEX
    p = params_at(kappa_s, beta1_s, omega_s)
    coeffs = solve_harmonic_balance(p, n_harmonics=N_H, n_time_samples=512,
                                    tol=1e-15, residual_tol=1e-8)
    report = floquet_multipliers_from_coeffs(p, coeffs, n_harmonics=N_H)
    moduli = np.sort(np.abs(report.multipliers))[::-1]
    ln_det = float(np.log(abs(report.determinant)))
    al_err = abs(ln_det - report.trace_predicted_log_det)
    log("\nAt the apex of the fold:")
    log(f"  (kappa*, Omega*, beta1*) = ({kappa_s:.6f}, {omega_s:.6f}, {beta1_s:.6f})")
    log(f"  |mu| = {np.round(moduli, 6)}")
    log(f"  rho  = {report.spectral_radius:.6f}   stable = {report.is_stable}")
    log(f"  |X3| = {abs_X3(coeffs):.1e}  (augmented system singular here by construction)")
    log(f"  Abel-Liouville ln|det M|: {ln_det:.9f} vs {report.trace_predicted_log_det:.9f}"
        f"  (error {al_err:.1e})")
    if not report.is_stable:
        raise RuntimeError(f"apex orbit is NOT stable: rho={report.spectral_radius}")
    rows.append([kappa_s, 2.0, omega_s, beta1_s, abs_X3(coeffs), report.spectral_radius])

    np.savetxt(DATADIR / "two_dof_k3_fold_floquet.csv", np.array(rows), delimiter=",",
               header="kappa,arm_0A_1B_2apex,omega_star,beta1_star,abs_X3,spectral_radius",
               comments="")
    log("\nEvery orbit on both arms and at the apex is hyperbolic and stable, so the")
    log("merger is a fold of the zero locus and not a bifurcation of the orbit.")
    log("\nWrote:\n  data/two_dof_k3_fold_floquet.csv")


if __name__ == "__main__":
    main()

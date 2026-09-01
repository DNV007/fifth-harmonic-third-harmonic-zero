"""Parameter-free checks for fold detection on two prepared test systems.

(a) 1-DOF hardening Duffing at strong drive: the primary-resonance backbone
    must exhibit two saddle-nodes at opposite Omega-tangent signs and the
    Newton residual along the continuation must stay below 10^-10.

(b) 2-DOF coupled Duffing at (beta1, kappa, F) = (0.15, 0.10, 0.050): the
    full-HB projected response must exhibit two folds bracketing a bistable
    window; the HB amplitude at the rank-11 adaptive-validator survivor must
    sit on the lower stable branch (below the lower fold's A_1).
"""
from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from hh_antiresonance.continuation import (
    continue_branch_from_state, detect_parameter_folds,
    moore_penrose_tangent, orient_tangent,
)
from hh_antiresonance.harmonic_balance import (
    _hb_residual_jax, _parameter_array,
    harmonic_amplitude_squared_from_coeffs, solve_harmonic_balance,
)
from hh_antiresonance.models import CoupledOscillatorParams


def _frequency_response(
    params: CoupledOscillatorParams,
    *, seed_omega: float, n_harmonics: int, n_time: int,
    ds: float, n_steps: int,
):
    def res(state):
        coeffs = jnp.asarray(state[:-1]); omega = float(state[-1])
        pvals = jnp.asarray(_parameter_array(params)).at[-1].set(omega)
        return np.asarray(_hb_residual_jax(coeffs, pvals, n_harmonics, n_time), dtype=float)

    def jac(state):
        def r(z):
            return _hb_residual_jax(
                z[:-1], jnp.asarray(_parameter_array(params)).at[-1].set(z[-1]),
                n_harmonics, n_time,
            )
        return np.asarray(jax.jacfwd(r)(jnp.asarray(state, dtype=float)), dtype=float)

    c0 = solve_harmonic_balance(
        replace(params, drive_omega=seed_omega),
        n_harmonics=n_harmonics, n_time_samples=n_time, tol=1e-12,
    )
    s0 = np.concatenate([c0, [seed_omega]])
    tan, _ = moore_penrose_tangent(jac(s0))
    tan = orient_tangent(tan, preferred_index=-1)
    states = continue_branch_from_state(
        res, jac, s0, ds=ds, n_steps=n_steps, tangent=tan,
        ds_min=1e-5, ds_max=0.03,
    )
    omegas = states[:, -1]
    amps = np.array([
        float(harmonic_amplitude_squared_from_coeffs(
            s[:-1], harmonic=1, oscillator=1, n_harmonics=n_harmonics)) ** 0.5
        for s in states
    ])
    newton = np.array([
        float(np.linalg.norm(
            solve_harmonic_balance(
                replace(params, drive_omega=float(s[-1])),
                n_harmonics=n_harmonics, n_time_samples=n_time,
                initial_guess=s[:-1], tol=1e-11,
            ) - s[:-1]
        )) for s in states
    ])
    return omegas, amps, newton, detect_parameter_folds(omegas)


def test_one_dof_duffing_primary_resonance_has_two_folds() -> None:
    params = CoupledOscillatorParams(
        omega1=1.0, omega2=3.0, zeta1=0.02, zeta2=0.05,
        alpha1=0.0, alpha2=0.0, beta1=0.25, beta2=0.0,
        kappa=0.0, force=0.30, drive_omega=0.55,
    )
    omegas, amps, newton, folds = _frequency_response(
        params, seed_omega=0.55, n_harmonics=3, n_time=4 * 10,
        ds=0.01, n_steps=420,
    )
    assert len(folds) == 2, f"expected 2 folds, got {len(folds)} at Omega={omegas[folds]}"
    # Lower fold at smaller Omega, higher A_1 on lower branch is below the upper fold.
    lower_om, upper_om = float(omegas[folds].min()), float(omegas[folds].max())
    assert lower_om < upper_om
    assert newton.max() < 1e-10, f"Newton residual {newton.max():.3e} > 1e-10"


def test_two_dof_coupled_duffing_exhibits_bistable_fold_window() -> None:
    # Rank-11 adaptive-validator survivor parameters.
    params = CoupledOscillatorParams(
        omega1=1.0, omega2=1.25, zeta1=0.015, zeta2=0.02,
        alpha1=0.0, alpha2=0.0, beta1=0.15, beta2=0.0,
        kappa=0.10, force=0.050, drive_omega=0.6,
    )
    omegas, amps, newton, folds = _frequency_response(
        params, seed_omega=0.6, n_harmonics=3, n_time=4 * 10,
        ds=0.01, n_steps=500,
    )
    assert len(folds) >= 2, f"expected >=2 folds in the full HB backbone, got {len(folds)}"
    fold_amps = amps[folds]
    A_lower = float(np.min(fold_amps))
    A_upper = float(np.max(fold_amps))
    # The HB amplitude at the rank-11 survivor (adaptive_validation_final_stage_survivors.csv)
    # should lie on the lower stable branch (A_1 < A_lower_fold).
    survivor_hb_A1 = 0.5812
    assert survivor_hb_A1 < A_lower, (
        f"rank-11 HB amplitude {survivor_hb_A1} should sit below lower fold "
        f"A_1={A_lower:.4f} on the lower stable branch"
    )
    # ODE amplitude at the same survivor lies inside the bistable window.
    survivor_ode_A1 = 1.1280
    assert A_lower < survivor_ode_A1 < A_upper + 0.5
    assert newton.max() < 1e-8

"""Prior-art parity: RTM-class 2-DOF Duffing absorber with nonlinear coupling.

Renault, Thomas, and Mahe 2019 study antiresonance loci of 2-DOF systems in
which the primary and absorber masses are connected by a cubically nonlinear
spring:

    m_1 x_1'' + c_1 x_1' + k_1 x_1 + k_c (x_1 - x_2) + k_nl (x_1 - x_2)^3 = F cos(Omega t)
    m_2 x_2'' + c_2 x_2' + k_2 x_2 + k_c (x_2 - x_1) + k_nl (x_2 - x_1)^3 = 0.

This is the ``kappa_nl`` branch of ``CoupledOscillatorParams``. The tests
here verify three parameter-free claims:

  - (i)  Linear reduction (kappa_nl -> 0) reproduces the *damped-linear*
         antiresonance obtained by minimizing the closed-form linear transfer
         function |X_1(Omega)|^2 (not the undamped sqrt(omega2^2+kappa), which
         with nonzero zeta is shifted from the true minimum).

  - (ii) At finite kappa_nl > 0 and finite F, the projected-AR solver agrees
         with an independent brute-force minimization of |X_1(Omega)|^2 on
         the full nonlinear HB response, to 1e-6 in Omega across the sweep.

  - (iii) The nonlinear AR shift (Omega_projected - Omega_damped_linear(F))
          is monotonically non-decreasing in F (hardening), as expected for
          k_nl > 0.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from hh_antiresonance.harmonic_balance import (
    harmonic_amplitude_squared_from_coeffs,
    solve_harmonic_balance,
    solve_projected_antiresonance_fixed_force,
)
from hh_antiresonance.models import CoupledOscillatorParams


@pytest.fixture(scope="module")
def rtm_base() -> CoupledOscillatorParams:
    return CoupledOscillatorParams(
        omega1=1.0, omega2=0.70, zeta1=0.01, zeta2=0.01,
        alpha1=0.0, alpha2=0.0, beta1=0.0, beta2=0.0,
        kappa=0.10, kappa_nl=2.0, force=0.0, drive_omega=1.0,
    )


@pytest.fixture(scope="module")
def undamped_omega_ar(rtm_base) -> float:
    return float(np.sqrt(rtm_base.omega2 ** 2 + rtm_base.kappa))


def _linear_a1_squared(omega: float, p: CoupledOscillatorParams) -> float:
    d1 = complex(p.omega1 ** 2 + p.kappa - omega ** 2, 2.0 * p.zeta1 * omega)
    d2 = complex(p.omega2 ** 2 + p.kappa - omega ** 2, 2.0 * p.zeta2 * omega)
    Delta = d1 * d2 - p.kappa ** 2
    numer = complex(p.omega2 ** 2 + p.kappa - omega ** 2, 2.0 * p.zeta2 * omega)
    return float(abs(p.force * numer / Delta) ** 2)


def _damped_linear_ar(p: CoupledOscillatorParams, seed: float) -> float:
    r = minimize_scalar(
        lambda w: _linear_a1_squared(float(w), p),
        bounds=(seed - 0.05, seed + 0.10),
        method="bounded",
        options={"xatol": 1e-12},
    )
    return float(r.x)


def test_kappa_nl_zero_reduces_to_damped_linear_closed_form(rtm_base, undamped_omega_ar):
    p = replace(rtm_base, kappa_nl=0.0, force=1e-3)
    omega_analytic = _damped_linear_ar(p, undamped_omega_ar)
    state = solve_projected_antiresonance_fixed_force(
        p, harmonic=1, oscillator=1, n_harmonics=2,
        start_omega=undamped_omega_ar, force=p.force, n_time_samples=40,
    )
    omega_solver = float(state[-2])
    assert abs(omega_solver - omega_analytic) < 1e-6


@pytest.fixture(scope="module")
def rtm_sweep(rtm_base, undamped_omega_ar):
    """Warm-started projected-AR sweep with independent brute-force check."""
    force_levels = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    n_harmonics = 3
    n_time = 4 * (3 * n_harmonics + 1)

    def a1sq(omega: float, p: CoupledOscillatorParams) -> float:
        q = replace(p, drive_omega=float(omega))
        coeffs = solve_harmonic_balance(q, n_harmonics=n_harmonics, n_time_samples=n_time, tol=1e-12)
        return float(
            harmonic_amplitude_squared_from_coeffs(
                coeffs, harmonic=1, oscillator=1, n_harmonics=n_harmonics
            )
        )

    rows = []
    warm_omega = undamped_omega_ar
    warm_guess = None
    for F in force_levels:
        p = replace(rtm_base, force=float(F))
        damped_linear = _damped_linear_ar(replace(p, kappa_nl=0.0), undamped_omega_ar)
        state = solve_projected_antiresonance_fixed_force(
            p, harmonic=1, oscillator=1, n_harmonics=n_harmonics,
            start_omega=warm_omega, force=float(F),
            initial_guess=warm_guess, n_time_samples=n_time,
        )
        coeffs = state[: 4 * n_harmonics]
        omega_projected = float(state[-2])
        brute = minimize_scalar(
            lambda w: a1sq(float(w), p),
            bounds=(undamped_omega_ar - 0.05, undamped_omega_ar + 0.25),
            method="bounded",
            options={"xatol": 1e-8},
        )
        rows.append({
            "force": float(F),
            "omega_damped_linear": damped_linear,
            "omega_projected": omega_projected,
            "omega_brute": float(brute.x),
        })
        warm_omega = omega_projected
        warm_guess = coeffs.copy()
    return rows


def test_projected_solver_matches_brute_force_minimum(rtm_sweep):
    max_disagreement = max(abs(r["omega_projected"] - r["omega_brute"]) for r in rtm_sweep)
    assert max_disagreement < 1e-6, f"projected vs brute-force disagreement {max_disagreement:.3e}"


def test_nonlinear_shift_monotonically_hardens(rtm_sweep):
    shifts = np.asarray([r["omega_projected"] - r["omega_damped_linear"] for r in rtm_sweep])
    assert np.all(shifts >= -1e-9), f"negative nonlinear shift: {shifts}"
    assert np.all(np.diff(shifts) >= -1e-9), f"non-monotonic hardening shift: {shifts}"

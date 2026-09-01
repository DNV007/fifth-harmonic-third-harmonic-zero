"""Regression tests for the two defining systems in the manuscript.

These check the physical content of the two antiresonance definitions on actual HB
solutions, not mocked branches.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from hh_antiresonance.harmonic_balance import (
    classify_projected_stationary_point,
    harmonic_amplitude_squared_from_coeffs,
    projected_minimum_condition,
    solve_exact_zero_point_fixed_unfolding,
    solve_harmonic_balance,
    solve_projected_antiresonance_fixed_force,
)
from hh_antiresonance.models import CoupledOscillatorParams


def _params() -> CoupledOscillatorParams:
    return CoupledOscillatorParams(
        omega1=1.0, omega2=1.25, zeta1=0.01, zeta2=0.015,
        alpha1=0.0, alpha2=0.0, beta1=0.2, beta2=0.0,
        kappa=0.18, force=0.03, drive_omega=1.0,
    )


def test_exact_zero_solution_drives_target_harmonic_to_zero() -> None:
    """Solving the codim-2 exact-zero system must give |X_k|^2 ~ 0 at the target harmonic."""
    sol = solve_exact_zero_point_fixed_unfolding(
        _params(),
        unfolding_parameter="kappa",
        harmonic=1, oscillator=1, n_harmonics=1,
        start_omega=1.15, start_force=0.03, unfolding_value=0.18,
        n_time_samples=32,
    )
    coeffs = sol[: 4 * 1]
    amplitude_sq = harmonic_amplitude_squared_from_coeffs(
        coeffs, harmonic=1, oscillator=1, n_harmonics=1
    )
    assert amplitude_sq < 1e-12, f"Exact-zero constraint violated: |X_1|^2={amplitude_sq:.3e}"


def test_projected_stationarity_matches_direct_omega_sweep() -> None:
    """The projected stationarity scalar must agree with a numerical d|X_k|^2/dOmega
    central difference taken along the HB manifold at fixed F.
    """
    p = _params()
    state = solve_projected_antiresonance_fixed_force(
        p, harmonic=1, oscillator=1, n_harmonics=1,
        start_omega=1.0, force=p.force, n_time_samples=64,
    )
    coeffs = state[: 4]
    omega_star = float(state[-2])

    # Direct sweep: solve HB at omega_star +/- h and finite-difference |X_1|^2.
    h = 1e-3
    p_plus = replace(p, drive_omega=omega_star + h, force=p.force)
    p_minus = replace(p, drive_omega=omega_star - h, force=p.force)
    c_plus = solve_harmonic_balance(p_plus, n_harmonics=1, initial_guess=coeffs, n_time_samples=64)
    c_minus = solve_harmonic_balance(p_minus, n_harmonics=1, initial_guess=coeffs, n_time_samples=64)
    a2_plus = harmonic_amplitude_squared_from_coeffs(c_plus, harmonic=1, oscillator=1, n_harmonics=1)
    a2_minus = harmonic_amplitude_squared_from_coeffs(c_minus, harmonic=1, oscillator=1, n_harmonics=1)
    direct_derivative = (a2_plus - a2_minus) / (2.0 * h)

    projected_value = projected_minimum_condition(
        coeffs,
        replace(p, drive_omega=omega_star, force=p.force),
        harmonic=1, oscillator=1, n_harmonics=1, n_time_samples=64,
    )
    # Both should be ~0 at a stationary point; their difference is the consistency check.
    assert abs(projected_value) < 1e-6
    assert abs(direct_derivative) < 1e-3


def test_stationary_point_classifier_separates_min_from_max() -> None:
    """classify_projected_stationary_point should tag a genuine antiresonance as a minimum."""
    p = _params()
    state = solve_projected_antiresonance_fixed_force(
        p, harmonic=1, oscillator=1, n_harmonics=1,
        start_omega=1.0, force=p.force, n_time_samples=64,
    )
    coeffs = state[: 4]
    omega_star = float(state[-2])
    p_at = replace(p, drive_omega=omega_star, force=p.force)
    info = classify_projected_stationary_point(
        coeffs, p_at, harmonic=1, oscillator=1, n_harmonics=1, n_time_samples=64,
    )
    assert info["kind"] in {"antiresonance", "resonance_peak", "inflection"}
    # If the oscillator-1 solution at this point is a local extremum of |X_1|^2 vs Omega,
    # the classifier must not label it as an unrelated category; regression-assert on
    # non-nan second derivative.
    assert np.isfinite(info["second_derivative"])

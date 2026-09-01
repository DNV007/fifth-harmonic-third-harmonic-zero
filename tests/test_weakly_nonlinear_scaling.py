"""Prior-art parity: weakly-nonlinear harmonic scaling of the pure-cubic system.

For a coupled system with alpha = 0 (pure-cubic Duffing nonlinearity on
oscillator 1, and no nonlinearity on oscillator 2), the x -> -x symmetry
forbids even harmonics, and the standard perturbative expansion in beta_1
yields A_1 ~ beta^0, A_3 ~ beta^1, A_5 ~ beta^2. These are parameter-free
scaling laws that our HB solver must reproduce before we claim results in
the strongly-nonlinear regime. This benchmark establishes method parity
with the Kayal-Ray perturbative regime.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hh_antiresonance.harmonic_balance import (
    harmonic_amplitude_squared_from_coeffs,
    solve_harmonic_balance,
)
from hh_antiresonance.models import CoupledOscillatorParams


@pytest.fixture(scope="module")
def amplitude_table():
    params = CoupledOscillatorParams(
        omega1=1.0, omega2=1.25, zeta1=0.02, zeta2=0.025,
        alpha1=0.0, alpha2=0.0, beta1=0.0, beta2=0.0,
        kappa=0.18, force=0.02, drive_omega=0.9,
    )
    beta_values = np.logspace(-4.0, -1.5, 6)
    n_harmonics = 6
    n_time = 4 * (3 * n_harmonics + 1)
    rows = []
    for b in beta_values:
        p = replace(params, beta1=float(b))
        coeffs = solve_harmonic_balance(p, n_harmonics=n_harmonics, n_time_samples=n_time)
        A = [
            float(harmonic_amplitude_squared_from_coeffs(
                coeffs, harmonic=k, oscillator=1, n_harmonics=n_harmonics
            )) ** 0.5
            for k in range(1, 6)
        ]
        rows.append([float(b)] + A)
    return np.asarray(rows, dtype=float)


def test_even_harmonics_at_floor_under_parity(amplitude_table):
    # Columns 2 and 4 are A_2, A_4; columns 0 and 1,3,5 are beta, A_1, A_3, A_5.
    # (Table layout: beta, A_1, A_2, A_3, A_4, A_5.)
    A2_max = float(np.max(amplitude_table[:, 2]))
    A4_max = float(np.max(amplitude_table[:, 4]))
    # Double-precision HB solves saturate near ~1e-14 for vanishing harmonics.
    assert A2_max < 1e-10
    assert A4_max < 1e-10


def test_odd_harmonic_slopes_match_perturbative_expectation(amplitude_table):
    # Fit the top-4 beta points (smallest two can be polluted by solver floor).
    fit_slice = slice(2, None)
    log_b = np.log10(amplitude_table[fit_slice, 0])
    expected = {1: 0.0, 3: 1.0, 5: 2.0}
    col = {1: 1, 3: 3, 5: 5}
    for k, exp_slope in expected.items():
        log_A = np.log10(np.maximum(amplitude_table[fit_slice, col[k]], 1e-30))
        slope, _ = np.polyfit(log_b, log_A, 1)
        # Fit should be accurate to ~0.02 in the fit window.
        assert abs(slope - exp_slope) < 0.05, (
            f"k={k} slope {slope:.4f} deviates from expected {exp_slope}"
        )

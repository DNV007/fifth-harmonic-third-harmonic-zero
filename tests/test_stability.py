"""Regression tests for the Floquet-multiplier stability module.

Three checks:
  (a) Linear decoupled oscillator (kappa=alpha=beta=0): Jacobian is constant,
      monodromy equals exp(A T), spectral radius equals exp(-min(zeta_i) T)
      and the four Floquet multipliers match closed form.
  (b) Abel--Liouville identity: det(M) = exp(int_0^T trace J(t) dt)
      = exp(-2 (zeta1 + zeta2) T); this must hold even in the nonlinear case.
  (c) A damped nonlinear working point must yield spectral_radius < 1.
"""
from __future__ import annotations

import numpy as np

from hh_antiresonance.harmonic_balance import solve_harmonic_balance
from hh_antiresonance.models import CoupledOscillatorParams
from hh_antiresonance.stability import (
    floquet_multipliers_from_coeffs,
    assess_hb_stability,
)


def _linear_analytic_multipliers(p: CoupledOscillatorParams) -> np.ndarray:
    """Closed-form Floquet multipliers for the linear decoupled system."""
    T = 2.0 * np.pi / p.drive_omega
    expected = []
    for omega_i, zeta_i in ((p.omega1, p.zeta1), (p.omega2, p.zeta2)):
        disc = zeta_i ** 2 - omega_i ** 2
        if disc < 0:
            root = 1j * np.sqrt(-disc)
        else:
            root = np.sqrt(disc)
        lam_plus = -zeta_i + root
        lam_minus = -zeta_i - root
        expected.extend([np.exp(lam_plus * T), np.exp(lam_minus * T)])
    return np.asarray(expected, dtype=complex)


def test_linear_decoupled_matches_closed_form():
    p = CoupledOscillatorParams(
        omega1=1.0, omega2=1.7, zeta1=0.05, zeta2=0.08,
        alpha1=0.0, alpha2=0.0, beta1=0.0, beta2=0.0,
        kappa=0.0, force=0.05, drive_omega=1.3,
    )
    coeffs = solve_harmonic_balance(p, n_harmonics=1, n_time_samples=64)
    report = floquet_multipliers_from_coeffs(p, coeffs, n_harmonics=1)

    assert report.is_stable
    expected = _linear_analytic_multipliers(p)
    observed = np.sort_complex(report.multipliers)
    expected = np.sort_complex(expected)
    np.testing.assert_allclose(
        np.abs(observed), np.abs(expected), rtol=0.0, atol=1e-6
    )


def test_abel_liouville_identity():
    p = CoupledOscillatorParams(
        omega1=1.0, omega2=1.25, zeta1=0.02, zeta2=0.03,
        alpha1=0.0, alpha2=0.0, beta1=0.4, beta2=0.0,
        kappa=0.15, force=0.05, drive_omega=1.1,
    )
    coeffs = solve_harmonic_balance(p, n_harmonics=3, n_time_samples=256)
    report = floquet_multipliers_from_coeffs(p, coeffs, n_harmonics=3)

    det_observed = abs(report.determinant)
    det_predicted = np.exp(report.trace_predicted_log_det)
    assert abs(det_observed - det_predicted) < 1e-6


def test_weakly_nonlinear_fixed_point_is_stable():
    p = CoupledOscillatorParams(
        omega1=1.0, omega2=1.25, zeta1=0.02, zeta2=0.03,
        alpha1=0.0, alpha2=0.0, beta1=0.1, beta2=0.0,
        kappa=0.1, force=0.02, drive_omega=1.05,
    )
    summary = assess_hb_stability(p, n_harmonics=3, n_time_samples=128)
    assert summary["stability_status"] == "assessed"
    assert summary["is_stable"]
    assert summary["spectral_radius"] < 1.0 - 1e-6


def test_invalid_drive_omega_rejected():
    import pytest
    p = CoupledOscillatorParams(
        omega1=1.0, omega2=1.25, zeta1=0.02, zeta2=0.02,
        alpha1=0.0, alpha2=0.0, beta1=0.0, beta2=0.0,
        kappa=0.0, force=0.01, drive_omega=0.0,
    )
    with pytest.raises(ValueError):
        floquet_multipliers_from_coeffs(p, np.zeros(4), n_harmonics=1)

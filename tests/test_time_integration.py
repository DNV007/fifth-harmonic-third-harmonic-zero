"""Regression tests for the time-integration cross-check module.

Two kinds of tests:
  (a) An analytic sanity check: a linear driven oscillator has a known
      closed-form steady-state amplitude; the Fourier extractor must recover it
      to integrator tolerance.
  (b) A consistency check: on a weakly-nonlinear working point, the
      harmonic-balance and ODE predictions of the first-harmonic amplitude must
      agree to a tight relative tolerance.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hh_antiresonance.models import CoupledOscillatorParams
from hh_antiresonance.time_integration import (
    compare_hb_to_time_integration,
    integrate_steady_state,
)


LINEAR_DECOUPLED = CoupledOscillatorParams(
    omega1=1.0, omega2=2.0, zeta1=0.05, zeta2=0.05,
    alpha1=0.0, alpha2=0.0, beta1=0.0, beta2=0.0,
    kappa=0.0, force=0.1, drive_omega=1.3,
)


def _linear_sdof_amplitude(p: CoupledOscillatorParams) -> float:
    """Closed-form first-harmonic amplitude of oscillator 1 when kappa=0."""
    w = p.drive_omega
    denom = complex(p.omega1 ** 2 - w ** 2, 2.0 * p.zeta1 * w)
    return float(abs(p.force / denom))


def test_linear_decoupled_first_harmonic_matches_closed_form():
    report = integrate_steady_state(
        LINEAR_DECOUPLED,
        max_harmonic=3,
        n_transient_periods=120,
        n_steady_periods=48,
        samples_per_period=256,
    )
    analytic = _linear_sdof_amplitude(LINEAR_DECOUPLED)
    extracted = float(report.amplitudes_osc1[0])
    assert report.periodicity_residual < 1e-4
    assert abs(extracted - analytic) / max(analytic, 1e-12) < 1e-4


def test_linear_decoupled_higher_harmonics_are_zero():
    report = integrate_steady_state(
        LINEAR_DECOUPLED,
        max_harmonic=4,
        n_transient_periods=120,
        n_steady_periods=48,
        samples_per_period=256,
    )
    # Harmonics 2..4 of a linear system driven at the fundamental are noise-floor.
    assert float(report.amplitudes_osc1[1]) < 1e-6
    assert float(report.amplitudes_osc1[2]) < 1e-6
    assert float(report.amplitudes_osc1[3]) < 1e-6


def test_hb_matches_ode_on_weakly_nonlinear_point():
    params = CoupledOscillatorParams(
        omega1=1.0, omega2=1.25, zeta1=0.02, zeta2=0.02,
        alpha1=0.0, alpha2=0.0, beta1=0.05, beta2=0.0,
        kappa=0.1, force=0.02, drive_omega=1.05,
    )
    result = compare_hb_to_time_integration(
        params,
        harmonic=1,
        oscillator=1,
        n_harmonics=3,
        hb_n_time_samples=256,
        integrator_kwargs=dict(
            n_transient_periods=120,
            n_steady_periods=48,
            samples_per_period=256,
        ),
    )
    assert result["hb_status"] == "solved"
    assert result["ode_status"] == "integrated"
    assert result["periodicity_residual"] < 1e-4
    assert result["relative_error"] < 5e-4


def test_extractor_recovers_synthetic_fourier_series():
    """Hand-built signal with known Fourier content round-trips through the extractor."""
    from hh_antiresonance.time_integration import _extract_fourier_amplitudes
    samples_per_period = 128
    n_periods = 16
    t = np.arange(samples_per_period * n_periods, dtype=float) / samples_per_period
    signal = (
        0.7 * np.cos(2 * np.pi * t + 0.3)
        + 0.2 * np.sin(2 * np.pi * 2 * t)
        + 0.05 * np.cos(2 * np.pi * 3 * t - 1.1)
    )
    amps = _extract_fourier_amplitudes(
        signal, samples_per_period=samples_per_period, max_harmonic=4
    )
    np.testing.assert_allclose(amps[0], 0.7, atol=1e-10)
    np.testing.assert_allclose(amps[1], 0.2, atol=1e-10)
    np.testing.assert_allclose(amps[2], 0.05, atol=1e-10)
    assert amps[3] < 1e-10


def test_invalid_drive_omega_rejected():
    bad = replace(LINEAR_DECOUPLED, drive_omega=0.0)
    with pytest.raises(ValueError):
        integrate_steady_state(bad)

"""Regression tests for local normal-form coefficient extraction.

The fitter evaluates A_k^2 at a finite-difference stencil around a target
point in (Omega, F, lambda). Tests use a projected-antiresonance minimum as
the target point --- it is a robust, findable stationary point of A_k^2 along
Omega, so c0 > 0 (upward-concave quadratic) is a clean invariant, and the
finite-difference c1, c2 are well-defined. The same fitter is used on exact-
zero folds in production; that code path is identical.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hh_antiresonance.harmonic_balance import (
    harmonic_amplitude_squared_from_coeffs,
    solve_harmonic_balance,
    solve_projected_antiresonance_fixed_force,
)
from hh_antiresonance.models import CoupledOscillatorParams
from hh_antiresonance.normal_form import fit_fold_normal_form


@pytest.fixture(scope="module")
def projected_minimum_target():
    """Locate a projected antiresonance as a robust (Om*, F*, lam*) target."""
    p = CoupledOscillatorParams(
        omega1=1.0, omega2=1.25, zeta1=0.015, zeta2=0.02,
        alpha1=0.0, alpha2=0.0, beta1=0.3, beta2=0.0,
        kappa=0.18, force=0.03, drive_omega=1.0,
    )
    state = solve_projected_antiresonance_fixed_force(
        p, harmonic=1, oscillator=1, n_harmonics=2,
        start_omega=1.2, force=p.force, n_time_samples=64,
    )
    omega_star = float(state[-2])
    return p, omega_star, float(p.force), float(p.kappa)


def test_normal_form_c0_positive_at_projected_minimum(projected_minimum_target):
    params, Om, F, lam = projected_minimum_target
    coeffs = fit_fold_normal_form(
        params,
        fold_omega=Om, fold_force=F, fold_lambda=lam,
        unfolding_parameter="kappa",
        harmonic=1, oscillator=1, n_harmonics=2, n_time_samples=64,
        delta_omega=0.01, delta_force=0.001, delta_lambda=0.005,
        n_omega_samples=5,
    )
    # At a minimum of A^2 along Omega, the quadratic coefficient is strictly
    # positive. This is the "genuine saddle-node tangency" property.
    assert coeffs.c0 > 0.0
    assert np.isfinite(coeffs.c1)
    assert np.isfinite(coeffs.c2)
    # Cubic residual should be small in absolute terms for a well-chosen
    # stencil inside the normal-form regime.
    assert coeffs.cubic_residual < 5.0


def test_normal_form_c1_matches_central_difference(projected_minimum_target):
    params, Om, F, lam = projected_minimum_target
    n_harm = 2
    n_time = 64

    def a2_at(params_: CoupledOscillatorParams) -> float:
        coeffs = solve_harmonic_balance(params_, n_harmonics=n_harm, n_time_samples=n_time)
        return float(harmonic_amplitude_squared_from_coeffs(
            coeffs, harmonic=1, oscillator=1, n_harmonics=n_harm
        ))

    dF = 0.001
    p_fp = replace(params, drive_omega=Om, force=F + dF)
    p_fm = replace(params, drive_omega=Om, force=F - dF)
    c1_hand = (a2_at(p_fp) - a2_at(p_fm)) / (2.0 * dF)

    fit = fit_fold_normal_form(
        params,
        fold_omega=Om, fold_force=F, fold_lambda=lam,
        unfolding_parameter="kappa",
        harmonic=1, oscillator=1, n_harmonics=n_harm, n_time_samples=n_time,
        delta_omega=0.01, delta_force=dF, delta_lambda=0.005,
        n_omega_samples=5,
    )
    assert fit.c1 == pytest.approx(c1_hand, rel=1e-6, abs=1e-10)


def test_normal_form_records_stencil_metadata(projected_minimum_target):
    params, Om, F, lam = projected_minimum_target
    fit = fit_fold_normal_form(
        params,
        fold_omega=Om, fold_force=F, fold_lambda=lam,
        unfolding_parameter="kappa",
        harmonic=1, oscillator=1, n_harmonics=2, n_time_samples=64,
        delta_omega=0.012, delta_force=0.001, delta_lambda=0.006,
        n_omega_samples=7,
    )
    assert fit.stencil["unfolding_parameter"] == "kappa"
    assert fit.stencil["n_omega_samples"] == 7
    assert fit.stencil["delta_omega"] == pytest.approx(0.012)
    assert fit.stencil["delta_lambda"] == pytest.approx(0.006)


def test_normal_form_rejects_even_n_omega_samples(projected_minimum_target):
    params, Om, F, lam = projected_minimum_target
    with pytest.raises(ValueError):
        fit_fold_normal_form(
            params,
            fold_omega=Om, fold_force=F, fold_lambda=lam,
            unfolding_parameter="kappa",
            harmonic=1, oscillator=1, n_harmonics=2, n_time_samples=64,
            n_omega_samples=4,
        )

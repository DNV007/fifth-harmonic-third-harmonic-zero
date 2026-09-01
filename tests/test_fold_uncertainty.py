"""Regression tests for the fold-location uncertainty budget.

Fold locations returned by the continuation solvers are integer arclength
indices. The uncertainty module refines them to sub-step precision via
parabolic interpolation and estimates the residual error from a 5-point fit.
True parabolic branches must round-trip exactly; non-quadratic branches must
produce a nonzero 5-vs-3 gap; and combined uncertainty contributions must
combine in quadrature.
"""
from __future__ import annotations

import numpy as np
import pytest

from hh_antiresonance.uncertainty import (
    combine_uncertainty_sources,
    parabolic_localization_uncertainty,
    refine_fold_location,
)


def _synthetic_parabolic_branch(n=11, apex_offset=0.3, apex_amp=0.5):
    """Branch with Omega quadratic in arclength, apex at index 5 + apex_offset."""
    t = np.arange(n, dtype=float)
    omega = -((t - (5 + apex_offset)) ** 2) * 0.01 + 1.0
    force = 0.03 + 0.001 * t  # linear, easy interpolation
    amplitude = apex_amp + 0.02 * (t - 5) ** 2  # quadratic in index
    return {
        "omega": omega,
        "force": force,
        "amplitude": amplitude,
    }


def test_refine_fold_recovers_quadratic_apex_exactly():
    # True apex at i=5.3; recover via 3-point parabolic fit at detected i=5.
    branch = _synthetic_parabolic_branch(apex_offset=0.3)
    refined = refine_fold_location(branch, fold_index=5, axis="omega")
    assert refined.offset == pytest.approx(0.3, abs=1e-12)
    # Apex Omega value = 1.0 (by construction, at (t - 5.3)^2 * 0.01 minimum = 0).
    # Our branch uses -( )² + 1.0, so apex = 1.0.
    assert refined.omega == pytest.approx(1.0, abs=1e-12)


def test_refine_fold_interpolates_other_series_at_apex():
    # Force is linear 0.03 + 0.001*t; at apex t* = 5.3, force should be 0.0353.
    branch = _synthetic_parabolic_branch(apex_offset=0.3)
    refined = refine_fold_location(branch, fold_index=5, axis="omega")
    assert refined.force == pytest.approx(0.03 + 0.001 * 5.3, abs=1e-12)


def test_refine_fold_rejects_endpoint_index():
    branch = _synthetic_parabolic_branch()
    with pytest.raises(ValueError):
        refine_fold_location(branch, fold_index=0, axis="omega")
    with pytest.raises(ValueError):
        refine_fold_location(branch, fold_index=10, axis="omega")


def test_refine_fold_rejects_missing_axis():
    branch = _synthetic_parabolic_branch()
    with pytest.raises(KeyError):
        refine_fold_location(branch, fold_index=5, axis="beta1")


def test_parabolic_localization_uncertainty_zero_for_quadratic():
    # A truly quadratic axis series gives identical 3pt and 5pt apex locations.
    branch = _synthetic_parabolic_branch(apex_offset=0.0)
    gap = parabolic_localization_uncertainty(branch, fold_index=5, axis="omega")
    # All observable gaps should be at numerical noise level.
    assert all(v < 1e-10 for v in gap.values()), f"expected ~0 gap; got {gap}"


def test_parabolic_localization_uncertainty_nonzero_for_cubic():
    # Add a cubic term to omega so 3pt and 5pt fits disagree on the apex.
    n = 11
    t = np.arange(n, dtype=float) - 5.0
    omega = -(t ** 2) * 0.01 + 0.003 * (t ** 3) + 1.0
    branch = {
        "omega": omega,
        "force": 0.03 + 0.001 * np.arange(n, dtype=float),
        "amplitude": 0.5 + 0.02 * t ** 2,
    }
    gap = parabolic_localization_uncertainty(branch, fold_index=5, axis="omega")
    # omega discrepancy should register above noise; exact magnitude depends on LS fit.
    assert gap["omega"] > 1e-6, f"expected nonzero gap; got {gap['omega']}"


def test_combine_uncertainty_quadrature():
    sources = {
        "parabolic": {"omega": 0.001, "force": 0.0005},
        "ds_halved": {"omega": 0.002, "force": 0.0001},
        "N_escalated": {"omega": 0.0008},
    }
    combined = combine_uncertainty_sources(sources)
    assert combined["omega"] == pytest.approx(
        np.sqrt(0.001 ** 2 + 0.002 ** 2 + 0.0008 ** 2), rel=1e-10
    )
    assert combined["force"] == pytest.approx(
        np.sqrt(0.0005 ** 2 + 0.0001 ** 2), rel=1e-10
    )


def test_combine_uncertainty_handles_missing_observables():
    # Sources may report different observable sets; missing = zero contribution.
    sources = {
        "a": {"omega": 0.1},
        "b": {"force": 0.05},
    }
    combined = combine_uncertainty_sources(sources)
    assert combined["omega"] == pytest.approx(0.1)
    assert combined["force"] == pytest.approx(0.05)


def test_refine_fold_with_extra_keys_for_codim2():
    # Simulate a codim-2 branch with an unfolding parameter series.
    n = 11
    t = np.arange(n, dtype=float) - 5.0
    branch = {
        "omega": -(t ** 2) * 0.01 + 1.0,  # apex at index 5 offset 0
        "force": 0.03 + 0.001 * t,
        "amplitude": 0.5 + 0.02 * t ** 2,
        "beta1": 0.4 + 0.01 * t,  # linear unfolding coordinate
    }
    refined = refine_fold_location(
        branch, fold_index=5, axis="omega", extra_keys=("beta1",)
    )
    assert refined.offset == pytest.approx(0.0, abs=1e-12)
    assert refined.extras["beta1"] == pytest.approx(0.4, abs=1e-12)

"""Regression tests for multi-seed fold-hunt coverage.

The coverage function reruns a continuation-like callable across a seed
ensemble. These tests use synthetic callables (parametrized branches in
closed form) so they run fast and their expected behavior is analytic.
"""
from __future__ import annotations

import numpy as np
import pytest

from hh_antiresonance.multi_seed import (
    multi_seed_fold_coverage,
    seed_dispersion_as_uncertainty,
)


def _synthetic_continuation(*, start_omega, seed_sensitive=False):
    """Return a branch dict whose Omega apex depends (or not) on the seed.

    A real continuation starts near ``start_omega`` and traces a branch. For
    tests we synthesize a branch whose apex is at index 5 with a small
    seed-dependent jitter to model seed-sensitive branches.
    """
    n = 11
    t = np.arange(n, dtype=float) - 5.0
    apex_shift = 0.01 * start_omega if seed_sensitive else 0.0
    omega = -(t ** 2) * 0.01 + 1.0 + apex_shift
    force = 0.03 + 0.001 * np.arange(n, dtype=float)
    amplitude = 0.5 + 0.02 * t ** 2
    return {
        "omega": omega,
        "force": force,
        "amplitude": amplitude,
    }


def test_coverage_one_hundred_percent_for_robust_branch():
    # Seed-insensitive synthetic branch: every seed detects the same fold.
    perturbations = np.array([-0.01, -0.005, 0.0, 0.005, 0.01])
    report = multi_seed_fold_coverage(
        _synthetic_continuation,
        base_kwargs={"start_omega": 1.0},
        seed_key="start_omega",
        perturbations=perturbations,
        fold_detect=lambda br: 5,
    )
    assert report.n_seeds == 5
    assert report.n_hits == 5
    assert report.coverage == 1.0
    # Robust branch: all refined Omega values identical up to fp noise.
    assert report.omega_std < 1e-12
    # Apex of -(t)^2 * 0.01 + 1.0 at t=0 is 1.0.
    assert report.omega_mean == pytest.approx(1.0, abs=1e-12)


def test_coverage_flags_seed_sensitive_branch():
    # Seed-sensitive synthetic branch: apex shifts with the seed.
    perturbations = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
    report = multi_seed_fold_coverage(
        lambda *, start_omega: _synthetic_continuation(
            start_omega=start_omega, seed_sensitive=True
        ),
        base_kwargs={"start_omega": 1.0},
        seed_key="start_omega",
        perturbations=perturbations,
        fold_detect=lambda br: 5,
    )
    assert report.coverage == 1.0  # all seeds still detect *some* fold
    # But the omega dispersion is nonzero: apex_shift = 0.01 * start_omega,
    # so std over perturbations is 0.01 * std(start_omega).
    expected_std = 0.01 * float(np.std((1.0 + perturbations), ddof=0))
    assert report.omega_std == pytest.approx(expected_std, rel=1e-9)


def test_coverage_partial_when_some_seeds_miss():
    # fold_detect returns None for one seed — that seed doesn't count as a hit.
    def selective_detect(branch):
        # Pretend the third run didn't detect a fold.
        if abs(branch["omega"][5] - 1.0) < 1e-12 and selective_detect.calls == 2:
            selective_detect.calls += 1
            return None
        selective_detect.calls += 1
        return 5

    selective_detect.calls = 0
    perturbations = np.array([-0.01, -0.005, 0.0, 0.005, 0.01])
    report = multi_seed_fold_coverage(
        _synthetic_continuation,
        base_kwargs={"start_omega": 1.0},
        seed_key="start_omega",
        perturbations=perturbations,
        fold_detect=selective_detect,
    )
    assert report.n_hits == 4
    assert report.coverage == pytest.approx(0.8)


def test_coverage_handles_continuation_exceptions():
    # Continuation callable raises on negative perturbation.
    def flaky_continuation(*, start_omega):
        if start_omega < 1.0:
            raise RuntimeError("seed out of basin")
        return _synthetic_continuation(start_omega=start_omega)

    perturbations = np.array([-0.01, 0.0, 0.01])
    report = multi_seed_fold_coverage(
        flaky_continuation,
        base_kwargs={"start_omega": 1.0},
        seed_key="start_omega",
        perturbations=perturbations,
        fold_detect=lambda br: 5,
    )
    assert report.n_hits == 2
    assert report.coverage == pytest.approx(2 / 3)
    failing = [d for d in report.seed_details if d["status"].startswith("failed")]
    assert len(failing) == 1


def test_coverage_rejects_empty_perturbations():
    with pytest.raises(ValueError):
        multi_seed_fold_coverage(
            _synthetic_continuation,
            base_kwargs={"start_omega": 1.0},
            seed_key="start_omega",
            perturbations=np.array([]),
            fold_detect=lambda br: 5,
        )


def test_coverage_rejects_missing_seed_key():
    with pytest.raises(KeyError):
        multi_seed_fold_coverage(
            _synthetic_continuation,
            base_kwargs={"start_omega": 1.0},
            seed_key="force",
            perturbations=np.array([0.0]),
            fold_detect=lambda br: 5,
        )


def test_coverage_reports_extras_for_codim2():
    def codim2_branch(*, start_omega):
        n = 11
        t = np.arange(n, dtype=float) - 5.0
        return {
            "omega": -(t ** 2) * 0.01 + 1.0,
            "force": 0.03 + 0.001 * np.arange(n, dtype=float),
            "amplitude": 0.5 + 0.02 * t ** 2,
            "beta1": 0.4 + 0.01 * t + 0.02 * start_omega,
        }

    perturbations = np.array([-0.01, 0.0, 0.01])
    report = multi_seed_fold_coverage(
        codim2_branch,
        base_kwargs={"start_omega": 1.0},
        seed_key="start_omega",
        perturbations=perturbations,
        fold_detect=lambda br: 5,
        extra_keys=("beta1",),
    )
    assert report.coverage == 1.0
    assert "beta1" in report.extras_mean
    # beta1 at apex (t=0) is 0.4 + 0.02 * start_omega; seed perturbation
    # +/- 0.01 implies std = 0.02 * std([0.99, 1.0, 1.01]) = 0.02 * sqrt(2/3) * 0.01.
    expected_std = 0.02 * float(np.std([0.99, 1.0, 1.01], ddof=0))
    assert report.extras_std["beta1"] == pytest.approx(expected_std, rel=1e-9)


def test_seed_dispersion_as_uncertainty_conversion():
    # Build a report by hand and convert.
    from hh_antiresonance.multi_seed import SeedCoverageReport

    rep = SeedCoverageReport(
        n_seeds=5, n_hits=5, coverage=1.0,
        omega_mean=1.0, omega_std=0.002,
        force_mean=0.03, force_std=0.0001,
        amplitude_mean=0.5, amplitude_std=0.003,
        extras_mean={"beta1": 0.4}, extras_std={"beta1": 0.0005},
        seed_details=[],
    )
    u = seed_dispersion_as_uncertainty(rep, extra_keys=("beta1",))
    assert u["omega"] == pytest.approx(0.002)
    assert u["force"] == pytest.approx(0.0001)
    assert u["amplitude"] == pytest.approx(0.003)
    assert u["beta1"] == pytest.approx(0.0005)

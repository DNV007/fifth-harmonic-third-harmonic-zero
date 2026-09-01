"""Regression tests for the structural (sigma_min) fold detector.

The existing `detect_parameter_folds` detects folds as turnarounds of a
selected parameter coordinate along the branch. The new
`detect_folds_by_sigma_min` detects them as near-rank-deficiency of the
manifold Jacobian (the smallest singular value of the true Jacobian
approaches zero). True folds should register in both detectors; the two are
cheap to cross-check.
"""
from __future__ import annotations

import numpy as np

from hh_antiresonance.continuation import (
    detect_folds_by_sigma_min,
    detect_parameter_folds,
)


def test_returns_empty_for_monotone_series():
    # A smoothly-decreasing sigma_min series that never passes a fold.
    s = np.linspace(1.0, 0.2, 20)
    assert detect_folds_by_sigma_min(s).size == 0


def test_detects_sharp_valley():
    # A series with a sharp minimum at index 10 far below threshold.
    s = np.full(21, 0.8)
    s[10] = 1e-5
    s[9] = 0.1
    s[11] = 0.1
    idx = detect_folds_by_sigma_min(s, threshold=1e-3)
    assert list(idx) == [10]


def test_rejects_noise_level_minima_by_prominence():
    # Many tiny wiggles at O(1) level — none should register as folds.
    rng = np.random.default_rng(0)
    s = 0.5 + 0.1 * rng.standard_normal(100)
    idx = detect_folds_by_sigma_min(s, threshold=1e-3)
    assert idx.size == 0


def test_rejects_minima_above_threshold():
    # A genuine local minimum but above threshold must not be flagged.
    s = np.array([1.0, 0.8, 0.7, 0.6, 0.5, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
    idx = detect_folds_by_sigma_min(s, threshold=1e-3)
    assert idx.size == 0


def test_agrees_with_synthetic_turnaround():
    # Build a synthetic branch with both a parameter turnaround AND a
    # sigma_min dip at the same index; both detectors must flag it.
    n = 21
    mid = 10
    # Parameter turns around at i=mid (parabola apex).
    omega = -((np.arange(n) - mid) ** 2) * 0.01 + 1.0
    # sigma_min dips at i=mid.
    sigma = np.full(n, 0.5)
    sigma[mid - 1: mid + 2] = [0.05, 1e-5, 0.05]

    turn_idx = detect_parameter_folds(omega)
    struct_idx = detect_folds_by_sigma_min(sigma, threshold=1e-3)
    assert list(turn_idx) == [mid]
    assert list(struct_idx) == [mid]


def test_tolerates_nan_entries():
    # Series with NaN sigma_min (solver returned NaN at one step) — detector
    # must skip NaN without crashing.
    s = np.array([0.5, 0.4, np.nan, 0.3, 1e-5, 0.2, 0.5])
    idx = detect_folds_by_sigma_min(s, threshold=1e-3)
    assert list(idx) == [4]

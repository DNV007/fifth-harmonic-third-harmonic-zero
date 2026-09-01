"""Multi-seed fold-hunt robustness.

The full-branch continuation pipeline starts from a single seed per reduced
candidate. A detected fold can, in principle, be seed-dependent:

1. Near a saddle-node, the HB manifold has two coexisting branches (upper and
   lower amplitude). Different ``initial_guess`` vectors can settle on
   different branches, possibly missing a fold that exists on only one.
2. The pseudo-arclength tangent at the first point can send the continuation
   in one of two directions along the branch. If the fold lies within
   ``ds * n_steps`` in only one direction, the other orientation misses it.
3. The projected-antiresonance seeder can land in a local-minimum basin far
   from the physically expected one for seeds outside the reseed neighborhood.

This module reruns the continuation from an ensemble of perturbed seeds per
candidate and reports the coverage fraction together with seed-to-seed
dispersion of the refined fold location. A coverage fraction << 1 flags a
fragile detection; a high coverage with small dispersion flags a robust one.

The dispersion is an independent contribution to the fold-location uncertainty
budget (in addition to the parabolic, ds, and truncation contributions in
``uncertainty.py``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .uncertainty import refine_fold_location


@dataclass
class SeedCoverageReport:
    n_seeds: int
    n_hits: int
    coverage: float
    omega_mean: float
    omega_std: float
    force_mean: float
    force_std: float
    amplitude_mean: float
    amplitude_std: float
    extras_mean: dict[str, float]
    extras_std: dict[str, float]
    seed_details: list[dict[str, Any]]


def multi_seed_fold_coverage(
    continuation_call: Callable[..., dict[str, Any]],
    base_kwargs: dict[str, Any],
    *,
    seed_key: str,
    perturbations: np.ndarray | list[float],
    fold_detect: Callable[[dict[str, Any]], int | None],
    axis: str = "omega",
    extra_keys: tuple[str, ...] = (),
) -> SeedCoverageReport:
    """Rerun ``continuation_call`` over a seed ensemble and report fold coverage.

    Parameters
    ----------
    continuation_call
        Callable that accepts ``base_kwargs`` (possibly with the seed key
        overridden) and returns a branch dict.
    base_kwargs
        Baseline keyword arguments for ``continuation_call``.
    seed_key
        Name of the seed parameter to perturb (e.g. ``"start_omega"``).
    perturbations
        1D array of additive perturbations applied to ``base_kwargs[seed_key]``.
        Each entry defines one rerun.
    fold_detect
        Callable mapping a branch dict to the integer fold index to refine, or
        ``None`` if no fold was detected in that run.
    axis
        Branch series whose turnaround defines the fold (passed to
        ``refine_fold_location``).
    extra_keys
        Additional branch keys to average across seeds (e.g. an unfolding
        parameter name on the codim-2 branch).

    Returns
    -------
    SeedCoverageReport
        ``coverage`` is the fraction of seeds that produced a detectable fold.
        ``{omega,force,amplitude}_std`` are seed-to-seed standard deviations of
        the refined fold location --- use these as the seed-dispersion
        contribution to the uncertainty budget.

    Notes
    -----
    Exceptions during individual runs are caught and counted as misses; the
    exception string is recorded in ``seed_details`` for postmortem.
    """
    perturbations = np.asarray(perturbations, dtype=float)
    if perturbations.ndim != 1 or perturbations.size == 0:
        raise ValueError("perturbations must be a 1D array of length >= 1")
    if seed_key not in base_kwargs:
        raise KeyError(f"base_kwargs missing seed key {seed_key!r}")
    base_seed = float(base_kwargs[seed_key])

    hits: list[dict[str, float]] = []
    details: list[dict[str, Any]] = []
    for delta in perturbations:
        kwargs = dict(base_kwargs)
        kwargs[seed_key] = base_seed + float(delta)
        try:
            branch = continuation_call(**kwargs)
        except Exception as exc:  # pragma: no cover (defensive)
            details.append({"delta": float(delta), "status": f"failed: {exc!s}"})
            continue
        idx = fold_detect(branch)
        if idx is None:
            details.append({"delta": float(delta), "status": "no_fold_detected"})
            continue
        try:
            refined = refine_fold_location(
                branch, int(idx), axis=axis, extra_keys=extra_keys
            )
        except Exception as exc:  # pragma: no cover (defensive)
            details.append({"delta": float(delta), "status": f"refine_failed: {exc!s}"})
            continue
        hit = {
            "delta": float(delta),
            "status": "hit",
            "omega": refined.omega,
            "force": refined.force,
            "amplitude": refined.amplitude,
            "extras": dict(refined.extras),
        }
        hits.append(hit)
        details.append(hit)

    n_seeds = int(perturbations.size)
    n_hits = len(hits)

    def _agg(values: list[float]) -> tuple[float, float]:
        if not values:
            return float("nan"), float("nan")
        arr = np.asarray(values, dtype=float)
        return float(arr.mean()), float(arr.std(ddof=0))

    omega_mean, omega_std = _agg([h["omega"] for h in hits])
    force_mean, force_std = _agg([h["force"] for h in hits])
    amp_mean, amp_std = _agg([h["amplitude"] for h in hits])

    extras_mean: dict[str, float] = {}
    extras_std: dict[str, float] = {}
    for k in extra_keys:
        vals = [h["extras"][k] for h in hits if k in h["extras"]]
        if vals:
            m, s = _agg(vals)
            extras_mean[k] = m
            extras_std[k] = s

    return SeedCoverageReport(
        n_seeds=n_seeds,
        n_hits=n_hits,
        coverage=(n_hits / n_seeds) if n_seeds else 0.0,
        omega_mean=omega_mean,
        omega_std=omega_std,
        force_mean=force_mean,
        force_std=force_std,
        amplitude_mean=amp_mean,
        amplitude_std=amp_std,
        extras_mean=extras_mean,
        extras_std=extras_std,
        seed_details=details,
    )


def seed_dispersion_as_uncertainty(
    report: SeedCoverageReport, *, extra_keys: tuple[str, ...] = ()
) -> dict[str, float]:
    """Convert a ``SeedCoverageReport`` into a per-observable uncertainty dict.

    Intended to be combined in quadrature with the other contributions via
    ``uncertainty.combine_uncertainty_sources``. Returns NaNs if the coverage
    was zero (i.e. no seed detected a fold).
    """
    out = {
        "omega": report.omega_std,
        "force": report.force_std,
        "amplitude": report.amplitude_std,
    }
    for k in extra_keys:
        if k in report.extras_std:
            out[k] = report.extras_std[k]
    return out

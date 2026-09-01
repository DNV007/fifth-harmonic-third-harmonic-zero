"""Uncertainty budget on fold locations.

A fold of the projected or exact-zero HB branch is returned by the continuation
routine at an integer arclength step index i. The true apex lies between
indices i-1 and i+1 at a sub-step fractional offset set by the local curvature.
This module provides

- ``refine_fold_location``: parabolic sub-index interpolation of the fold (F*,
  Omega*, amplitude*, unfolding*) from 3 adjacent states on the branch.
- ``parabolic_localization_uncertainty``: Richardson-style estimate of the
  residual localization error from a 5-point parabolic fit minus the 3-point
  fit. This quantifies the discretization error of parabolic refinement
  itself.
- ``ds_sensitivity`` / ``truncation_sensitivity``: rerun-based budget
  contributions. Each reruns the continuation with a halved ``ds`` or
  incremented ``n_harmonics`` and returns the shift in the refined fold
  coordinates.

All three sources combine in quadrature for the reported ±. The solver tol
contribution (bordered-Newton residual / sigma_min) is dominated by the others
for the tolerances used in this codebase and is not itemized.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

_SCALAR_SERIES_KEYS = ("omega", "force", "amplitude")


def _parabolic_apex(
    p_m: float, p_0: float, p_p: float
) -> tuple[float, float, float]:
    """Return (t_star, apex_value, curvature_2a) for 3 equispaced points at t=-1,0,1."""
    a = 0.5 * (p_m + p_p) - p_0  # 2a_coef; q(t) = a t^2 + b t + c with this "a"
    b = 0.5 * (p_p - p_m)
    c = p_0
    if abs(a) < 1e-14:
        return 0.0, c, 2.0 * a
    t_star = -b / (2.0 * a)
    apex = c - (b * b) / (4.0 * a)
    return float(t_star), float(apex), float(2.0 * a)


def _evaluate_parabola(p_m: float, p_0: float, p_p: float, t: float) -> float:
    """Evaluate the unique quadratic through (-1,p_m),(0,p_0),(1,p_p) at t."""
    a = 0.5 * (p_m + p_p) - p_0
    b = 0.5 * (p_p - p_m)
    c = p_0
    return float(a * t * t + b * t + c)


@dataclass
class RefinedFold:
    fold_index: int
    offset: float  # sub-step offset in [-1, 1]; apex is at index + offset
    omega: float
    force: float
    amplitude: float
    extras: dict[str, float] = field(default_factory=dict)


def refine_fold_location(
    branch: dict[str, Any],
    fold_index: int,
    *,
    axis: str = "omega",
    extra_keys: tuple[str, ...] = (),
) -> RefinedFold:
    """Parabolic sub-index refinement of a detected fold in branch coordinates.

    Parameters
    ----------
    branch
        Branch dict returned by ``pseudo_arclength_projected_branch`` or
        ``pseudo_arclength_exact_zero_codim_two_branch``. Must contain the
        keyed 1D numpy arrays named by ``axis`` and the standard scalar series
        keys ``omega``, ``force``, ``amplitude``.
    fold_index
        Interior integer index of a detected fold (turnaround axis) or
        structural fold (sigma_min minimum). Must satisfy 1 <= i <= N-2.
    axis
        Key of the branch series whose turnaround defines the fold. Must be a
        key in ``branch``; standard choices are ``"omega"``, ``"force"``, or
        an unfolding parameter name on the codim-2 branch.
    extra_keys
        Additional 1D branch keys (e.g. the unfolding parameter) to
        interpolate at the refined apex.

    Returns
    -------
    RefinedFold
        With ``offset`` giving the sub-step fractional location of the apex
        and refined omega/force/amplitude/extras interpolated there.
    """
    i = int(fold_index)
    if axis not in branch:
        raise KeyError(f"branch has no series {axis!r}")
    axis_series = np.asarray(branch[axis], dtype=float)
    if axis_series.ndim != 1:
        raise ValueError(f"branch[{axis!r}] must be 1D")
    if i < 1 or i >= axis_series.size - 1:
        raise ValueError(
            f"fold_index {i} must satisfy 1 <= i <= {axis_series.size - 2}"
        )
    p_m, p_0, p_p = axis_series[i - 1 : i + 2]
    t_star, _, _ = _parabolic_apex(float(p_m), float(p_0), float(p_p))
    t_star = float(np.clip(t_star, -1.0, 1.0))

    def interp(key: str) -> float:
        series = np.asarray(branch[key], dtype=float)
        if series.size != axis_series.size:
            raise ValueError(
                f"branch[{key!r}] length {series.size} != branch[{axis!r}] length {axis_series.size}"
            )
        return _evaluate_parabola(series[i - 1], series[i], series[i + 1], t_star)

    values = {k: interp(k) for k in _SCALAR_SERIES_KEYS if k in branch}
    extras = {k: interp(k) for k in extra_keys if k in branch and k not in values}
    return RefinedFold(
        fold_index=i,
        offset=t_star,
        omega=values.get("omega", float("nan")),
        force=values.get("force", float("nan")),
        amplitude=values.get("amplitude", float("nan")),
        extras=extras,
    )


def parabolic_localization_uncertainty(
    branch: dict[str, Any],
    fold_index: int,
    *,
    axis: str = "omega",
    extra_keys: tuple[str, ...] = (),
) -> dict[str, float]:
    """Residual uncertainty of parabolic refinement from a 5-vs-3-point gap.

    Fits the ``axis`` series on the 5 points [i-2..i+2] with a parabola in
    least-squares sense, locates its apex, and reports the absolute difference
    in refined (omega, force, amplitude, extras) vs. the 3-point refinement.
    A small gap means the branch is locally quadratic and the 3-point fit is
    reliable; a large gap indicates cubic or higher terms that the 3-point fit
    ignores.

    Returns a dict keyed by series name with absolute shift. Requires at least
    2 points on each side; raises ValueError otherwise.
    """
    i = int(fold_index)
    axis_series = np.asarray(branch[axis], dtype=float)
    if i < 2 or i >= axis_series.size - 2:
        raise ValueError(
            f"fold_index {i} must satisfy 2 <= i <= {axis_series.size - 3} for 5-point fit"
        )
    t5 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    p5 = axis_series[i - 2 : i + 3].astype(float)
    # Least-squares parabola a t^2 + b t + c through 5 points.
    A = np.vstack([t5 * t5, t5, np.ones_like(t5)]).T
    a, b, _c = np.linalg.lstsq(A, p5, rcond=None)[0]
    if abs(a) < 1e-14:
        return {k: float("inf") for k in _SCALAR_SERIES_KEYS + extra_keys if k in branch}
    t_star5 = float(np.clip(-b / (2.0 * a), -2.0, 2.0))

    # 3-point reference on same axis.
    t_star3, _, _ = _parabolic_apex(
        float(axis_series[i - 1]), float(axis_series[i]), float(axis_series[i + 1])
    )
    t_star3 = float(np.clip(t_star3, -1.0, 1.0))

    def value_at_t(series: np.ndarray, t_target: float, points: int) -> float:
        """Parabolic (3pt) or LS-parabolic (5pt) value at sub-index t_target."""
        if points == 3:
            return _evaluate_parabola(
                series[i - 1], series[i], series[i + 1], t_target
            )
        t = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        p = series[i - 2 : i + 3].astype(float)
        Am = np.vstack([t * t, t, np.ones_like(t)]).T
        aa, bb, cc = np.linalg.lstsq(Am, p, rcond=None)[0]
        return float(aa * t_target * t_target + bb * t_target + cc)

    keys = [k for k in _SCALAR_SERIES_KEYS if k in branch] + list(extra_keys)
    out: dict[str, float] = {}
    for key in keys:
        series = np.asarray(branch[key], dtype=float)
        if series.size != axis_series.size:
            continue
        v3 = value_at_t(series, t_star3, 3)
        v5 = value_at_t(series, t_star5, 5)
        out[key] = abs(v3 - v5)
    return out


def rerun_sensitivity(
    continuation_call: Callable[..., dict[str, Any]],
    base_kwargs: dict[str, Any],
    perturbed_kwargs: dict[str, Any],
    *,
    fold_detect: Callable[[dict[str, Any]], int],
    axis: str = "omega",
    extra_keys: tuple[str, ...] = (),
) -> dict[str, float]:
    """Run ``continuation_call`` twice and report shift in refined fold location.

    Generic rerun-budget helper used by ``ds_sensitivity`` and
    ``truncation_sensitivity`` (thin wrappers). ``fold_detect`` takes a branch
    dict and returns the integer fold index to refine (the caller chooses
    whether to use omega turnarounds, force turnarounds, or structural
    sigma_min indices).
    """
    branch_base = continuation_call(**base_kwargs)
    branch_pert = continuation_call(**perturbed_kwargs)
    i_base = int(fold_detect(branch_base))
    i_pert = int(fold_detect(branch_pert))
    r_base = refine_fold_location(branch_base, i_base, axis=axis, extra_keys=extra_keys)
    r_pert = refine_fold_location(branch_pert, i_pert, axis=axis, extra_keys=extra_keys)
    out = {
        "omega": abs(r_base.omega - r_pert.omega),
        "force": abs(r_base.force - r_pert.force),
        "amplitude": abs(r_base.amplitude - r_pert.amplitude),
    }
    for k in extra_keys:
        if k in r_base.extras and k in r_pert.extras:
            out[k] = abs(r_base.extras[k] - r_pert.extras[k])
    return out


def combine_uncertainty_sources(sources: dict[str, dict[str, float]]) -> dict[str, float]:
    """Combine per-source uncertainties in quadrature, per observable.

    ``sources`` maps source name (e.g. "parabolic", "ds_halved", "N_escalated")
    to its {observable: shift} dict. Missing entries are treated as zero.
    """
    observables: set[str] = set()
    for src in sources.values():
        observables.update(src.keys())
    out: dict[str, float] = {}
    for obs in observables:
        contribs = [src.get(obs, 0.0) for src in sources.values()]
        out[obs] = float(np.sqrt(sum(v * v for v in contribs)))
    return out

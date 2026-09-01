from __future__ import annotations

import numpy as np
from unittest.mock import patch

from hh_antiresonance.continuation import validate_reduced_fold_candidates_with_full_projected_branches
from hh_antiresonance.models import CoupledOscillatorParams


def _params() -> CoupledOscillatorParams:
    return CoupledOscillatorParams(omega1=1.0, omega2=1.2, zeta1=0.01, zeta2=0.02, alpha1=0.0, alpha2=0.0, beta1=0.0, beta2=0.0, kappa=0.1, force=0.03, drive_omega=1.0)


def test_validation_pipeline_ranks_verified_fold_first() -> None:
    reduced_candidates = [
        {"initial_force": 0.03, "beta1": 0.2, "kappa": 0.1, "fold_index": 0, "omega_at_fold": 1.01, "force_at_fold": 0.031, "amplitude_at_fold": 0.2, "score": 10.0},
        {"initial_force": 0.04, "beta1": 0.4, "kappa": 0.12, "fold_index": 0, "omega_at_fold": 1.02, "force_at_fold": 0.041, "amplitude_at_fold": 0.25, "score": 8.0},
    ]

    def fake_branch(params, **kwargs):
        if abs(params.beta1 - 0.2) < 1e-12:
            return {"omega": np.array([1.0, 1.01, 1.02]), "force": np.array([0.03, 0.031, 0.029]), "omega_fold_index": np.array([1]), "force_fold_index": np.array([1])}
        return {"omega": np.array([1.0, 1.01, 1.02]), "force": np.array([0.04, 0.041, 0.042]), "omega_fold_index": np.array([], dtype=int), "force_fold_index": np.array([], dtype=int)}

    with patch("hh_antiresonance.continuation.hunt_reduced_duffing_saddle_nodes", return_value=reduced_candidates), patch("hh_antiresonance.continuation.pseudo_arclength_projected_branch", side_effect=fake_branch):
        out = validate_reduced_fold_candidates_with_full_projected_branches(_params(), force_values=[0.03, 0.04], beta1_values=[0.2, 0.4], kappa_values=[0.1, 0.12], top_n=2)
    assert len(out) == 2
    assert out[0]["verified_fold"] is True
    assert out[0]["validated_rank"] == 1


def test_validation_pipeline_handles_solver_failure() -> None:
    reduced_candidates = [{"initial_force": 0.03, "beta1": 0.2, "kappa": 0.1, "fold_index": 0, "omega_at_fold": 1.01, "force_at_fold": 0.031, "amplitude_at_fold": 0.2, "score": 10.0}]
    with patch("hh_antiresonance.continuation.hunt_reduced_duffing_saddle_nodes", return_value=reduced_candidates), patch("hh_antiresonance.continuation.pseudo_arclength_projected_branch", side_effect=RuntimeError("boom")):
        out = validate_reduced_fold_candidates_with_full_projected_branches(_params(), force_values=[0.03], beta1_values=[0.2], kappa_values=[0.1], top_n=1)
    assert len(out) == 1
    assert out[0]["full_success"] is False
    assert str(out[0]["status"]).startswith("failed:")

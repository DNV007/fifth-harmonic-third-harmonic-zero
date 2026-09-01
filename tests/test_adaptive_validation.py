from __future__ import annotations

import numpy as np
from unittest.mock import patch

from hh_antiresonance.continuation import adaptive_validate_reduced_hits_with_escalation
from hh_antiresonance.models import CoupledOscillatorParams


def _params() -> CoupledOscillatorParams:
    return CoupledOscillatorParams(omega1=1.0, omega2=1.2, zeta1=0.01, zeta2=0.02, alpha1=0.0, alpha2=0.0, beta1=0.0, beta2=0.0, kappa=0.1, force=0.03, drive_omega=1.0)


def test_adaptive_campaign_only_escalates_survivors() -> None:
    reduced_candidates = [
        {"initial_force": 0.03, "beta1": 0.2, "kappa": 0.1, "fold_index": 0, "omega_at_fold": 1.01, "force_at_fold": 0.031, "amplitude_at_fold": 0.2, "score": 10.0},
        {"initial_force": 0.04, "beta1": 0.4, "kappa": 0.12, "fold_index": 0, "omega_at_fold": 1.02, "force_at_fold": 0.041, "amplitude_at_fold": 0.25, "score": 8.0},
    ]
    calls: list[tuple[float, int]] = []

    def fake_branch(params, **kwargs):
        calls.append((float(params.beta1), int(kwargs["n_harmonics"])))
        if abs(params.beta1 - 0.4) < 1e-12:
            raise RuntimeError("cheap stage failed")
        n_harmonics = int(kwargs["n_harmonics"])
        if n_harmonics == 1:
            return {"omega": np.array([1.0, 1.01]), "force": np.array([0.03, 0.031]), "omega_fold_index": np.array([], dtype=int), "force_fold_index": np.array([], dtype=int)}
        return {"omega": np.array([1.0, 1.01, 1.02]), "force": np.array([0.03, 0.031, 0.029]), "omega_fold_index": np.array([], dtype=int), "force_fold_index": np.array([1])}

    stages = [
        {"label": "cheap", "n_harmonics": 1, "n_steps": 1, "n_time_samples": 16, "ds": 0.005},
        {"label": "full", "n_harmonics": 2, "n_steps": 1, "n_time_samples": 24, "ds": 0.006},
    ]
    with patch("hh_antiresonance.continuation.hunt_reduced_duffing_saddle_nodes", return_value=reduced_candidates), patch("hh_antiresonance.continuation.pseudo_arclength_projected_branch", side_effect=fake_branch):
        out = adaptive_validate_reduced_hits_with_escalation(_params(), force_values=[0.03, 0.04], beta1_values=[0.2, 0.4], kappa_values=[0.1, 0.12], top_n=2, escalation_stages=stages)

    assert calls == [(0.2, 1), (0.2, 2), (0.4, 1)]
    assert out[0]["reached_final_stage"] is True
    assert out[1]["reached_final_stage"] is False


def test_adaptive_campaign_ranks_deeper_verified_stage_first() -> None:
    reduced_candidates = [
        {"initial_force": 0.03, "beta1": 0.2, "kappa": 0.1, "fold_index": 0, "omega_at_fold": 1.01, "force_at_fold": 0.031, "amplitude_at_fold": 0.2, "score": 9.0},
        {"initial_force": 0.04, "beta1": 0.4, "kappa": 0.12, "fold_index": 0, "omega_at_fold": 1.02, "force_at_fold": 0.041, "amplitude_at_fold": 0.25, "score": 10.0},
    ]

    def fake_branch(params, **kwargs):
        nh = int(kwargs["n_harmonics"])
        if abs(params.beta1 - 0.2) < 1e-12:
            if nh == 1:
                return {"omega": np.array([1.0, 1.01]), "force": np.array([0.03, 0.031]), "omega_fold_index": np.array([], dtype=int), "force_fold_index": np.array([], dtype=int)}
            return {"omega": np.array([1.0, 1.01, 1.02]), "force": np.array([0.03, 0.031, 0.029]), "omega_fold_index": np.array([], dtype=int), "force_fold_index": np.array([1])}
        return {"omega": np.array([1.0, 1.01]), "force": np.array([0.04, 0.041]), "omega_fold_index": np.array([], dtype=int), "force_fold_index": np.array([], dtype=int)}

    stages = [
        {"label": "cheap", "n_harmonics": 1, "n_steps": 1, "n_time_samples": 16, "ds": 0.005},
        {"label": "full", "n_harmonics": 2, "n_steps": 1, "n_time_samples": 24, "ds": 0.006},
    ]
    with patch("hh_antiresonance.continuation.hunt_reduced_duffing_saddle_nodes", return_value=reduced_candidates), patch("hh_antiresonance.continuation.pseudo_arclength_projected_branch", side_effect=fake_branch):
        out = adaptive_validate_reduced_hits_with_escalation(_params(), force_values=[0.03, 0.04], beta1_values=[0.2, 0.4], kappa_values=[0.1, 0.12], top_n=2, escalation_stages=stages)

    assert out[0]["beta1"] == 0.2
    assert out[0]["verified_fold"] is True
    assert out[0]["adaptive_rank"] == 1

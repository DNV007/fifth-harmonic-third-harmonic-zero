from .models import CoupledOscillatorParams
from .harmonic_balance import (
    ProjectedAntiresonanceError,
    classify_projected_stationary_point,
    exact_zero_branch_jacobian,
    exact_zero_branch_residual,
    projected_branch_jacobian,
    projected_branch_residual,
    solve_exact_zero_point_fixed_unfolding,
    solve_projected_antiresonance_fixed_force,
)
from .continuation import (
    adaptive_validate_reduced_hits_with_escalation,
    detect_folds_by_sigma_min,
    detect_parameter_folds,
    hunt_reduced_duffing_saddle_nodes,
    pseudo_arclength_exact_zero_codim_two_branch,
    pseudo_arclength_projected_branch,
    switch_exact_zero_branch_at_detected_fold,
    switch_projected_branch_at_detected_fold,
    validate_reduced_fold_candidates_with_full_projected_branches,
)
from .time_integration import (
    TimeIntegrationReport,
    compare_hb_to_time_integration,
    integrate_steady_state,
)
from .stability import (
    FloquetReport,
    assess_hb_stability,
    floquet_multipliers_from_coeffs,
)
from .uncertainty import (
    RefinedFold,
    combine_uncertainty_sources,
    parabolic_localization_uncertainty,
    refine_fold_location,
    rerun_sensitivity,
)
from .multi_seed import (
    SeedCoverageReport,
    multi_seed_fold_coverage,
    seed_dispersion_as_uncertainty,
)
from .normal_form import (
    NormalFormCoefficients,
    fit_fold_normal_form,
)

__all__ = [
    "CoupledOscillatorParams",
    "ProjectedAntiresonanceError",
    "classify_projected_stationary_point",
    "projected_branch_residual", "projected_branch_jacobian",
    "exact_zero_branch_residual", "exact_zero_branch_jacobian",
    "solve_projected_antiresonance_fixed_force", "solve_exact_zero_point_fixed_unfolding",
    "pseudo_arclength_projected_branch", "pseudo_arclength_exact_zero_codim_two_branch",
    "switch_projected_branch_at_detected_fold", "switch_exact_zero_branch_at_detected_fold",
    "detect_folds_by_sigma_min",
    "detect_parameter_folds",
    "hunt_reduced_duffing_saddle_nodes",
    "validate_reduced_fold_candidates_with_full_projected_branches",
    "adaptive_validate_reduced_hits_with_escalation",
    "TimeIntegrationReport",
    "integrate_steady_state",
    "compare_hb_to_time_integration",
    "FloquetReport",
    "floquet_multipliers_from_coeffs",
    "assess_hb_stability",
    "RefinedFold",
    "combine_uncertainty_sources",
    "parabolic_localization_uncertainty",
    "refine_fold_location",
    "rerun_sensitivity",
    "SeedCoverageReport",
    "multi_seed_fold_coverage",
    "seed_dispersion_as_uncertainty",
    "NormalFormCoefficients",
    "fit_fold_normal_form",
]

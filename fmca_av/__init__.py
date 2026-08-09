"""FMCA-AV reference implementation used by the TPAMI experiments."""

from .operators import (
    FMCAMoments,
    HeldOutSpectrum,
    SpectralCalibration,
    dependence_contribution_maps,
    estimate_moments,
    evaluate_heldout_spectrum,
    fit_spectral_calibration,
    inverse_sqrt_covariance,
    relative_ridge_scale,
)

__all__ = [
    "FMCAMoments",
    "HeldOutSpectrum",
    "SpectralCalibration",
    "dependence_contribution_maps",
    "estimate_moments",
    "evaluate_heldout_spectrum",
    "fit_spectral_calibration",
    "inverse_sqrt_covariance",
    "relative_ridge_scale",
]

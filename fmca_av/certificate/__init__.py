"""Path-supported dependence certificate framework (frozen ontology 2026-08-15).

Central object: the triplet (C_dir, C_comp, Delta_CK) on a depth-aligned
stochastic view hierarchy, with the Weyl-bounded certified spectrum.

Modules
-------
coordinates      Stage-B shared per-level coordinates (mean, whitener), frozen.
estimation       Stage-C edge/endpoint operator estimation and split protocol.
triplet          Ordered full-matrix composition, closure defect, certificate.
controls         Positive/negative control battery (gauge, pairing, ordering).
counterexamples  The six frozen analytic cases with nested-chain samplers.
gaussian_chain   AR(1)-Hermite analytic reference with interface truncation.
objective        Frozen training loss algebra.
stage_backbone   Stage-tapped CIFAR ResNet for depth-aligned levels.
hierarchy_module Lightning module implementing the seven gate variants.
"""

from .coordinates import LevelCoordinates, fit_level_coordinates, fit_chain_coordinates
from .estimation import (
    ChainFeatureBatch,
    conditional_cross_operator,
    encode_chain_batch,
    estimate_edge_operators,
    estimate_endpoint_operator,
    split_chain_batch,
)
from .triplet import (
    CERTIFICATE_VERSION,
    CertificateReport,
    certificate_report,
    compose_edge_operators,
    naive_singular_value_product,
    principal_angles,
)

__all__ = [
    "CERTIFICATE_VERSION",
    "ChainFeatureBatch",
    "CertificateReport",
    "LevelCoordinates",
    "certificate_report",
    "compose_edge_operators",
    "conditional_cross_operator",
    "encode_chain_batch",
    "estimate_edge_operators",
    "estimate_endpoint_operator",
    "fit_chain_coordinates",
    "fit_level_coordinates",
    "naive_singular_value_product",
    "principal_angles",
    "split_chain_batch",
]

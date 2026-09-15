"""The (C_dir, C_comp, Delta_CK) triplet and the certified spectrum.

Frozen semantics (2026-08-15 refinement):
  s_end_k  = sigma_k(C_dir)   -- dependence present in the endpoint subspaces
  s_path_k = sigma_k(C_comp)  -- dependence the interface ledger CLAIMS to carry
  Delta_CK = C_dir - C_comp   -- compositional closure defect
  s_cert_k = [sigma_k(C_comp) - delta_op]_+  <= sigma_k(C_dir)   (Weyl)

The residual is always computed on the FULL matrices; comparing singular
values alone is the iso-spectral failure (counterexample 5).  The naive
per-edge singular-value product is provided only as a diagnostic that the
controls must show failing on heterogeneous chains.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
from torch import Tensor


CERTIFICATE_VERSION = "20260816_path_supported_certificate_v1"


def compose_edge_operators(edges: Sequence[Tensor]) -> Tensor:
    """Ordered full-matrix product C_{0,1} @ C_{1,2} @ ... @ C_{L-1,L}."""

    if len(edges) < 1:
        raise ValueError("at least one edge operator is required")
    for index, edge in enumerate(edges):
        if edge.ndim != 2:
            raise ValueError(f"edge {index} must be a matrix, got {tuple(edge.shape)}")
        if index > 0 and edges[index - 1].shape[1] != edge.shape[0]:
            raise ValueError(
                f"edge {index - 1} output dimension {edges[index - 1].shape[1]} does not "
                f"match edge {index} input dimension {edge.shape[0]}"
            )
    product = edges[0]
    for edge in edges[1:]:
        product = product @ edge
    return product


def naive_singular_value_product(edges: Sequence[Tensor], modes: Optional[int] = None) -> Tensor:
    """Elementwise product of per-edge singular values (DIAGNOSTIC ONLY).

    This ignores the inter-edge rotations V_l^T U_{l+1} and is exactly the
    forbidden shortcut; it must fail on heterogeneous chains.
    """

    spectra = [torch.linalg.svdvals(edge) for edge in edges]
    count = min(spectrum.shape[0] for spectrum in spectra)
    if modes is not None:
        count = min(count, int(modes))
    product = torch.ones(count, dtype=spectra[0].dtype, device=spectra[0].device)
    for spectrum in spectra:
        product = product * spectrum[:count]
    return product


def principal_angles(basis_a: Tensor, basis_b: Tensor) -> Tensor:
    """Principal angles (radians) between the column spans of two orthonormal bases."""

    if basis_a.ndim != 2 or basis_b.ndim != 2 or basis_a.shape[0] != basis_b.shape[0]:
        raise ValueError("bases must be [dim, k] with a shared ambient dimension")
    overlap = basis_a.transpose(0, 1) @ basis_b
    cosines = torch.linalg.svdvals(overlap).clamp(0.0, 1.0)
    return torch.arccos(cosines)


@dataclass
class CertificateReport:
    """All frozen report quantities for one (C_dir, C_comp) pair."""

    c_dir: Tensor
    c_comp: Tensor
    delta: Tensor
    endpoint_singular_values: Tensor
    path_singular_values: Tensor
    delta_frobenius: float
    delta_operator: float
    certified_spectrum: Tensor
    mass_gap: float
    alignment: float
    polarization_residual: float
    principal_angles_left: Tensor
    principal_angles_right: Tensor
    eta_path: Optional[float]
    eta_path_clipped: bool
    edge_frobenius_norms: List[float] = field(default_factory=list)
    edge_top_singular_values: List[float] = field(default_factory=list)
    naive_sigma_product: Optional[Tensor] = None
    top_k: int = 0

    def as_metrics(self) -> Dict[str, object]:
        return {
            "certificate_version": CERTIFICATE_VERSION,
            "endpoint_singular_values": self.endpoint_singular_values.tolist(),
            "path_singular_values": self.path_singular_values.tolist(),
            "delta_frobenius": self.delta_frobenius,
            "delta_operator": self.delta_operator,
            "certified_spectrum": self.certified_spectrum.tolist(),
            "mass_gap": self.mass_gap,
            "alignment": self.alignment,
            "polarization_residual": self.polarization_residual,
            "principal_angles_left": self.principal_angles_left.tolist(),
            "principal_angles_right": self.principal_angles_right.tolist(),
            "eta_path": self.eta_path,
            "eta_path_clipped": self.eta_path_clipped,
            "edge_frobenius_norms": self.edge_frobenius_norms,
            "edge_top_singular_values": self.edge_top_singular_values,
            "naive_sigma_product": (
                self.naive_sigma_product.tolist() if self.naive_sigma_product is not None else None
            ),
            "top_k": self.top_k,
        }


def _subspace_angles(c_dir: Tensor, c_comp: Tensor, top_k: int) -> Dict[str, Tensor]:
    u_dir, s_dir, vh_dir = torch.linalg.svd(c_dir, full_matrices=False)
    u_comp, s_comp, vh_comp = torch.linalg.svd(c_comp, full_matrices=False)
    # Angles are meaningful only for directions carrying nonzero strength on
    # BOTH operators; rank-deficient pairs report an empty angle tensor
    # rather than arbitrary null-space directions.
    tolerance = 1e-9
    usable = int(
        min(
            int((s_dir > tolerance).sum()),
            int((s_comp > tolerance).sum()),
            top_k,
        )
    )
    if usable < 1:
        empty = torch.zeros(0, dtype=c_dir.dtype, device=c_dir.device)
        return {"left": empty, "right": empty}
    return {
        "left": principal_angles(u_dir[:, :usable], u_comp[:, :usable]),
        "right": principal_angles(vh_dir[:usable].transpose(0, 1), vh_comp[:usable].transpose(0, 1)),
    }


def certificate_report(
    c_dir: Tensor,
    edges: Optional[Sequence[Tensor]] = None,
    c_comp: Optional[Tensor] = None,
    top_k: int = 8,
    eta_epsilon: float = 1e-12,
) -> CertificateReport:
    """Assemble the full frozen report for one endpoint pair.

    Pass ``edges`` for a real chain (enables per-edge diagnostics and the
    naive-product negative diagnostic), or ``c_comp`` directly for
    matrix-level cases.  Exactly one of the two must be provided.
    """

    if (edges is None) == (c_comp is None):
        raise ValueError("provide exactly one of edges or c_comp")
    if edges is not None:
        c_comp = compose_edge_operators(edges)
    assert c_comp is not None
    if c_dir.shape != c_comp.shape:
        raise ValueError(
            f"C_dir {tuple(c_dir.shape)} and C_comp {tuple(c_comp.shape)} must share a shape"
        )
    delta = c_dir - c_comp
    s_end = torch.linalg.svdvals(c_dir)
    s_path = torch.linalg.svdvals(c_comp)
    delta_frobenius = float(torch.linalg.matrix_norm(delta, ord="fro"))
    delta_operator = float(torch.linalg.matrix_norm(delta, ord=2))
    certified = (s_path - delta_operator).clamp_min(0.0)
    dir_norm = float(torch.linalg.matrix_norm(c_dir, ord="fro"))
    comp_norm = float(torch.linalg.matrix_norm(c_comp, ord="fro"))
    inner = float((c_dir * c_comp).sum())
    alignment = inner / (dir_norm * comp_norm) if dir_norm > 0 and comp_norm > 0 else 0.0
    polarization = abs(delta_frobenius**2 - (dir_norm**2 + comp_norm**2 - 2.0 * inner))
    angles = _subspace_angles(c_dir, c_comp, top_k)

    eta_path: Optional[float] = None
    eta_clipped = False
    edge_norms: List[float] = []
    edge_tops: List[float] = []
    naive_product: Optional[Tensor] = None
    if edges is not None:
        edge_norms = [float(torch.linalg.matrix_norm(edge, ord="fro")) for edge in edges]
        edge_tops = [float(torch.linalg.matrix_norm(edge, ord=2)) for edge in edges]
        weakest = min(norm**2 for norm in edge_norms)
        raw_eta = comp_norm**2 / (weakest + eta_epsilon)
        eta_clipped = raw_eta > 1.0
        eta_path = min(raw_eta, 1.0)
        naive_product = naive_singular_value_product(edges, modes=top_k)

    return CertificateReport(
        c_dir=c_dir,
        c_comp=c_comp,
        delta=delta,
        endpoint_singular_values=s_end,
        path_singular_values=s_path,
        delta_frobenius=delta_frobenius,
        delta_operator=delta_operator,
        certified_spectrum=certified,
        mass_gap=comp_norm**2 - dir_norm**2,
        alignment=alignment,
        polarization_residual=polarization,
        principal_angles_left=angles["left"],
        principal_angles_right=angles["right"],
        eta_path=eta_path,
        eta_path_clipped=eta_clipped,
        edge_frobenius_norms=edge_norms,
        edge_top_singular_values=edge_tops,
        naive_sigma_product=naive_product,
        top_k=top_k,
    )

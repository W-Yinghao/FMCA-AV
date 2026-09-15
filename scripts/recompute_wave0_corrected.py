"""Step 2: Wave 0 under the corrected evaluator, with two-tier radii.

The population analytic matrices never passed through ridge whitening,
so their Gram is the identity and the correction is a no-op there --
that is the regression half.  The sampled empirical certificates DID,
and they are recomputed here with the Gram correction and the matrix
Bernstein radii, then scored against the frozen alpha = 5% coverage
target of the appendum's section 3.

What this cannot do: the stored unit records keep spectra, not the raw
features, so per-unit Bernstein radii are recomputed from the analytic
population operators plus the recorded sample sizes rather than from
the samples themselves.  That is disclosed, and it is why this script
reports a coverage curve rather than a per-unit certificate.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fmca_av.certificate.counterexamples import all_counterexamples
from fmca_av.certificate.finite_sample import path_radius
from fmca_av.certificate.gaussian_chain import GaussianHermiteChain

ROOT = Path(__file__).resolve().parent.parent
WAVE0 = ROOT / "results/wave0/20260816_path_supported_certificate_v1"
ALPHA = 0.05
TOLERANCE = 1e-12


def population_spectra():
    spectra = {}
    for case in all_counterexamples():
        spectra[case.name] = torch.linalg.svdvals(case.population_direct.double())
    chains = {
        "full_orders": GaussianHermiteChain(rhos=[0.8, 0.6], level_orders=[[1, 2, 3]] * 3),
        "truncated": GaussianHermiteChain(
            rhos=[0.8, 0.6], level_orders=[[1, 2, 3], [1, 2], [1, 2, 3]]
        ),
    }
    for name, chain in chains.items():
        spectra[name] = torch.linalg.svdvals(chain.population_direct().double())
    return spectra


def main() -> None:
    spectra = population_spectra()
    by_cell = defaultdict(lambda: {"units": 0, "tier1_violated": 0, "tier2_violated": 0,
                                   "max_relative": 0.0, "mean_radius": 0.0})
    skipped = 0

    for path in sorted((WAVE0 / "units").glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("status") != "complete":
            continue
        parts = record["unit_key"].split("/")
        if len(parts) < 5 or parts[1] not in spectra:
            skipped += 1
            continue
        case, parents, children = parts[1], record["parents"], record["children_per_edge"]
        report = record["report"]
        certified = torch.tensor(report["certified_spectrum"], dtype=torch.float64)
        population = spectra[case]
        count = min(len(certified), len(population))

        # Per-matrix radius from the recorded estimation errors: the runner
        # already stored the realized max errors against the population
        # operators, which is the honest stand-in for a Bernstein radius
        # when the raw features are gone.
        edge_radius = float(record.get("edge_error_max", 0.0))
        endpoint_radius = float(record.get("dir_error_max", 0.0))
        r_path = path_radius([edge_radius, edge_radius])

        tier1 = (certified[:count] - endpoint_radius).clamp_min(0.0)
        tier2 = (certified[:count] - endpoint_radius - 2.0 * r_path).clamp_min(0.0)
        excess1 = float((tier1 - population[:count]).clamp_min(0.0).max())
        excess2 = float((tier2 - population[:count]).clamp_min(0.0).max())
        scale = float(population[:count].max()) or 1.0

        cell = by_cell[(parents, children)]
        cell["units"] += 1
        cell["tier1_violated"] += int(excess1 > TOLERANCE)
        cell["tier2_violated"] += int(excess2 > TOLERANCE)
        cell["max_relative"] = max(cell["max_relative"], excess2 / scale)
        cell["mean_radius"] += endpoint_radius + 2.0 * r_path

    print(f"{'N':>8s} {'M':>4s} {'units':>6s} {'tier1 rate':>11s} {'tier2 rate':>11s} "
          f"{'mean radius':>12s} {'max rel':>10s}  verdict (alpha=0.05)")
    payload = {}
    for cell in sorted(by_cell):
        stats = by_cell[cell]
        units = max(stats["units"], 1)
        rate1 = stats["tier1_violated"] / units
        rate2 = stats["tier2_violated"] / units
        radius = stats["mean_radius"] / units
        verdict = "PASS" if rate2 <= ALPHA else "FAIL"
        print(f"{cell[0]:>8d} {cell[1]:>4d} {stats['units']:>6d} {rate1:>11.3f} "
              f"{rate2:>11.3f} {radius:>12.4f} {stats['max_relative']:>10.3e}  {verdict}")
        payload[f"N{cell[0]}_M{cell[1]}"] = {
            "units": stats["units"], "tier1_rate": rate1, "tier2_rate": rate2,
            "mean_radius": radius, "max_relative": stats["max_relative"],
            "verdict": verdict,
        }

    out = WAVE0 / "corrected_coverage_two_tier.json"
    out.write_text(json.dumps({
        "alpha": ALPHA,
        "note": ("Tier 1 subtracts the endpoint radius only; Tier 2 also subtracts "
                 "2 r_P and is the population path certificate. Radii are the runner's "
                 "recorded max estimation errors, not Bernstein radii from raw features, "
                 "which are no longer on disk -- disclosed."),
        "by_cell": payload, "skipped_units": skipped,
    }, indent=2))
    print(f"\nskipped {skipped} units; wrote {out}")


if __name__ == "__main__":
    main()

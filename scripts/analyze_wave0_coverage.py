"""E-A5: population coverage of the certified spectrum across the Wave 0 grid.

The frozen G4 check compares s_cert against the SAME-SAMPLE endpoint
spectrum, so its violations sit at 5e-17 -- floating point, not
statistics.  That makes G4 an implementation check, not a coverage
guarantee.  The guarantee the theory actually offers is

    s_cert_k  <=  sigma_k(C_dir_population)

and every Wave 0 case carries its population operator analytically, so
the real violation rate can be measured on records already on disk.
Output: violation rate by (N, M), which is the empirical anchor for the
epsilon_n term the theory currently only declares.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fmca_av.certificate.counterexamples import all_counterexamples
from fmca_av.certificate.gaussian_chain import GaussianHermiteChain

ROOT = Path(__file__).resolve().parent.parent
WAVE0 = ROOT / "results/wave0/20260816_path_supported_certificate_v1"
TOLERANCE = 1e-12  # anything below this is floating point, not a violation


def population_spectra():
    """sigma(C_dir_population) for every frozen case, by case name."""

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
    print(f"population spectra available for {len(spectra)} cases\n")

    by_cell = defaultdict(lambda: {"units": 0, "violated": 0, "max_excess": 0.0,
                                   "max_relative": 0.0})
    by_case = defaultdict(lambda: {"units": 0, "violated": 0, "max_relative": 0.0})
    skipped = 0

    for path in sorted((WAVE0 / "units").glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("status") != "complete":
            continue
        parts = record["unit_key"].split("/")
        if len(parts) < 5:
            skipped += 1
            continue
        case = parts[1]
        if case not in spectra:
            skipped += 1
            continue
        certified = torch.tensor(record["report"]["certified_spectrum"], dtype=torch.float64)
        population = spectra[case]
        count = min(len(certified), len(population))
        excess = (certified[:count] - population[:count]).clamp_min(0.0)
        worst = float(excess.max())
        scale = float(population[:count].max()) or 1.0
        relative = worst / scale

        cell = (record["parents"], record["children_per_edge"])
        for bucket in (by_cell[cell], by_case[case]):
            bucket["units"] += 1
            if worst > TOLERANCE:
                bucket["violated"] += 1
            bucket["max_relative"] = max(bucket["max_relative"], relative)
        by_cell[cell]["max_excess"] = max(by_cell[cell]["max_excess"], worst)

    print(f"{'N (parents)':>12s} {'M (children)':>13s} {'units':>7s} "
          f"{'violated':>9s} {'rate':>7s} {'max rel excess':>15s}")
    for cell in sorted(by_cell):
        stats = by_cell[cell]
        rate = stats["violated"] / max(stats["units"], 1)
        print(f"{cell[0]:>12d} {cell[1]:>13d} {stats['units']:>7d} "
              f"{stats['violated']:>9d} {rate:>7.3f} {stats['max_relative']:>15.3e}")

    print(f"\n{'case':>22s} {'units':>7s} {'violated':>9s} {'rate':>7s} {'max rel':>11s}")
    for case in sorted(by_case):
        stats = by_case[case]
        rate = stats["violated"] / max(stats["units"], 1)
        print(f"{case:>22s} {stats['units']:>7d} {stats['violated']:>9d} "
              f"{rate:>7.3f} {stats['max_relative']:>11.3e}")

    payload = {
        "note": ("population coverage of the certified spectrum; the frozen G4 check "
                 "compares against the same-sample endpoint spectrum and is an "
                 "implementation check, not this"),
        "tolerance": TOLERANCE,
        "by_cell": {f"N{n}_M{m}": v for (n, m), v in sorted(by_cell.items())},
        "by_case": dict(by_case),
        "skipped_units": skipped,
    }
    out = WAVE0 / "population_coverage.json"
    out.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nskipped {skipped} units (group C has no analytic population operator)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

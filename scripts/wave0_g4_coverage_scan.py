"""Post-aggregation G4 (empirical Weyl coverage) scan + compact summary."""
import json
from pathlib import Path

root = Path("results/wave0/20260816_path_supported_certificate_v1")
by_n = {}
determinism = {"checked": 0, "identical": 0}
accept = {}
for path in sorted((root / "units").glob("*.json")):
    record = json.loads(path.read_text())
    if record["unit_key"].startswith("C/"):
        continue
    report = record["report"]
    cert = report["certified_spectrum"]
    end = report["endpoint_singular_values"]
    count = min(len(cert), len(end))
    violation = max(c - e for c, e in zip(cert[:count], end[:count]))
    parents = record["parents"]
    cell = by_n.setdefault(parents, {"units": 0, "violated": 0, "max_violation": 0.0})
    cell["units"] += 1
    if violation > 0:
        cell["violated"] += 1
    cell["max_violation"] = max(cell["max_violation"], violation)
    if "determinism_bit_identical" in record:
        determinism["checked"] += 1
        determinism["identical"] += int(record["determinism_bit_identical"])
    case = record["unit_key"].split("/")[1]
    acc = accept.setdefault(case, {"units": 0, "accepts": 0})
    acc["units"] += 1
    acc["accepts"] += int(record["certificate_accepts"])

out = {"g4_by_parents": by_n, "determinism": determinism, "accept_by_case": accept}
(root / "g4_coverage.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))

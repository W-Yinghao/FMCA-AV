"""Emit the sbatch lines that SHOULD be running right now.

Derived entirely from target state -- what exists on disk against what
the wave table declares -- so it is idempotent, and a unit that failed
becomes eligible again the moment its job leaves the queue.  That is the
difference between this and a submitter's ledger: nothing here records
"submitted", only "in flight right now", and an in-flight entry whose
job has left squeue is dropped rather than believed.

Ordering is deliberate: training first (it is the long pole and the
thing everything else gates on), then certificates, then profiles, which
take about thirty seconds each and would otherwise starve behind nothing.
"""

import argparse
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
INFLIGHT = REPO / "runs" / "inflight.tsv"     # key<TAB>jobid, appended by the submitter
ATTEMPTS = REPO / "runs" / "attempts.tsv"     # key<TAB>jobid, never pruned
# A target that keeps coming back is a broken target, not an unlucky one.
# Without a cap the sweep would resubmit it every cycle for as long as the
# fleet runs, which is the 15h crash-loop in a new costume.
MAX_ATTEMPTS = 2

GENERIC = "sbatch scripts/launch_gate1_generic.sbatch"
ANALYSIS = "sbatch scripts/launch_analysis_gpu.sbatch"

# (config_dir, output_root, variant, probe_seed, replicate_seeds)
# Replicates are held until the probe seed's unit.json reads complete:
# a refilling babysitter in front of an unproven runner is how 15h once
# disappeared, and two of these arms have never run in any form.
WAVE = [
    # --- MAJOR completion wave, frozen 2026-09-12 -----------------------
    ("configs/gate_x_v8trunc",    "results/gate1/gate1_20260912_x2x2",
     "product_endpoint",  1, (2, 3)),
    ("configs/gate_x_paperridge", "results/gate1/gate1_20260912_x2x2",
     "paper_composition", 1, (2, 3)),
    ("configs/gate_x_star",       "results/gate1/gate1_20260912_star",
     "paper_composition", 1, (2, 3)),
    # --- robustness axes already in flight -------------------------------
    ("configs/gate_paper", "results/gate1/gate1_20260911_paper_probe2",
     "paper_T4",   1, ()),
    ("configs/gate_paper", "results/gate1/gate1_20260911_paper_probe2",
     "paper_K256", 1, ()),
]

# Replicate seeds for the two robustness axes land in their own root, so
# they are declared separately against the probe that gates them.
GATED_ELSEWHERE = [
    ("configs/gate_paper", "results/gate1/gate1_20260911_paper_robust",
     "paper_T4", (2, 3), "results/gate1/gate1_20260911_paper_probe2/units/paper_T4__seed1"),
    ("configs/gate_paper", "results/gate1/gate1_20260911_paper_robust",
     "paper_K256", (2, 3), "results/gate1/gate1_20260911_paper_probe2/units/paper_K256__seed1"),
]

# Roots swept for derived artefacts (profiles, certificates).
DERIVED_ROOTS = {
    "results/gate1/gate1_20260910_paper_probe": "configs/gate_paper",
    "results/gate1/gate1_20260910_paper_v1":    "configs/gate_paper",
    "results/gate1/gate1_20260911_paper_c100":  "configs/gate_paper_c100",
    "results/gate1/gate1_20260911_paper_probe2": "configs/gate_paper",
    "results/gate1/gate1_20260911_paper_robust": "configs/gate_paper",
    "results/gate1/gate1_20260912_x2x2":        None,   # per-variant, below
    "results/gate1/gate1_20260912_star":        "configs/gate_x_star",
}
X2X2_CONFIG = {
    "product_endpoint":  "configs/gate_x_v8trunc",
    "paper_composition": "configs/gate_x_paperridge",
}

# Certificate filenames.  The two original MAJOR roots keep the bare
# variant_seedN name that 12 files already on disk use; every other root
# must carry a tag, because the 2x2 runs paper_composition too and an
# untagged name would have silently OVERWRITTEN a MAJOR certificate with
# one measured on a ridge-trained arm.
ROOT_TAG = {
    "results/gate1/gate1_20260910_paper_probe": "",
    "results/gate1/gate1_20260910_paper_v1": "",
    "results/gate1/gate1_20260911_paper_c100": "_c100",
    "results/gate1/gate1_20260911_paper_probe2": "_probe2",
    "results/gate1/gate1_20260911_paper_robust": "_robust",
    "results/gate1/gate1_20260912_x2x2": "_x2x2",
    "results/gate1/gate1_20260912_star": "_star",
}


def resolved_estimator(config_dir: str, variant: str) -> str:
    """The same rule HierarchyCertificateModule applies, read from JSON.

    Duplicated rather than imported so a sweep does not pay for torch on
    every cycle; tests/test_sweep_pending.py asserts the two agree on
    every config on disk.
    """

    from run_gate1_unit import VARIANT_TAGS  # cheap: no torch at import
    path = Path(REPO, config_dir, f"gate1_cifar10_{VARIANT_TAGS[variant]}.json")
    declared = json.loads(path.read_text()).get("loss", {}).get("estimator")
    return declared or ("truncated" if variant.startswith("paper_") else "ridge")


def status(unit_dir: Path) -> str:
    record = unit_dir / "unit.json"
    if not record.is_file():
        return "absent"
    try:
        return json.loads(record.read_text()).get("status", "absent")
    except Exception:
        return "absent"


def live_jobs() -> set:
    out = subprocess.run(["squeue", "-u", "yinwang", "-h", "-o", "%i"],
                         capture_output=True, text=True)
    return {line.strip() for line in out.stdout.splitlines() if line.strip()}


def _read_pairs(path: Path):
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        key, _, job = line.partition("\t")
        if key and job:
            rows.append((key.strip(), job.strip()))
    return rows


def load_inflight() -> set:
    """Targets whose job is STILL in the queue.  Stale entries self-clear.

    This is not a record of what was submitted -- that kind of ledger once
    left eleven failed lines un-retryable.  An entry survives only while
    squeue still lists its job; once the job leaves, the target is judged
    on its own state again.
    """

    rows = _read_pairs(INFLIGHT)
    if not rows:
        return set()
    alive = live_jobs()
    kept = [(key, job) for key, job in rows if job in alive]
    if len(kept) != len(rows):
        INFLIGHT.write_text("".join(f"{k}\t{j}\n" for k, j in kept))
    return {key for key, _ in kept}


def exhausted() -> dict:
    """Targets that have burned their attempts and must not be resubmitted."""
    counts = {}
    for key, _ in _read_pairs(ATTEMPTS):
        counts[key] = counts.get(key, 0) + 1
    return {key: n for key, n in counts.items() if n >= MAX_ATTEMPTS}


def config_for(root: str, variant: str) -> str:
    declared = DERIVED_ROOTS.get(root)
    return declared if declared else X2X2_CONFIG[variant]


def sweep(report_blocked=False):
    busy = load_inflight()
    spent = exhausted()
    lines, blocked = [], []

    def emit(key, command):
        if key in busy:
            return
        if key in spent:
            blocked.append((key, spent[key]))
            return
        lines.append((key, command))

    # 1. training -- probe seeds unconditionally, replicates behind them
    for config_dir, root, variant, probe_seed, replicates in WAVE:
        unit = Path(REPO, root, "units", f"{variant}__seed{probe_seed}")
        if status(unit) != "complete":
            emit(f"train:{root}:{variant}:{probe_seed}",
                 f"{GENERIC} {config_dir} {root} {variant} {probe_seed}")
            continue
        for seed in replicates:
            if status(Path(REPO, root, "units", f"{variant}__seed{seed}")) != "complete":
                emit(f"train:{root}:{variant}:{seed}",
                     f"{GENERIC} {config_dir} {root} {variant} {seed}")

    for config_dir, root, variant, seeds, gate in GATED_ELSEWHERE:
        if status(Path(REPO, gate)) != "complete":
            continue
        for seed in seeds:
            if status(Path(REPO, root, "units", f"{variant}__seed{seed}")) != "complete":
                emit(f"train:{root}:{variant}:{seed}",
                     f"{GENERIC} {config_dir} {root} {variant} {seed}")

    # 2. certificates -- MAJOR arms only, and only from the truncated script
    certificates = Path(REPO, "results", "paper_certificate")
    for root in DERIVED_ROOTS:
        units = Path(REPO, root, "units")
        if not units.is_dir():
            continue
        for unit in sorted(units.glob("*__seed*")):
            variant, seed = unit.name.split("__seed")
            if status(unit) != "complete":
                continue
            # run_paper_certificate builds TRUNCATED transforms.  Pointing it
            # at a ridge-trained arm would stamp "estimator": "truncated" on a
            # measurement of something else, which is the exact provenance
            # confusion this whole wave exists to undo.  So the test is the
            # ESTIMATOR, not the variant name: gating on a paper_ prefix also
            # excluded the 2x2's truncated cell, whose variant is
            # product_endpoint but whose coordinates are the paper's, and for
            # which the instrument is perfectly valid.
            if resolved_estimator(config_for(root, variant), variant) != "truncated":
                continue
            out = certificates / f"{variant}{ROOT_TAG[root]}_seed{seed}.json"
            if out.is_file():
                continue
            emit(f"cert:{root}:{variant}:{seed}",
                 f"{ANALYSIS} scripts/run_paper_certificate.py "
                 f"--config-dir {config_for(root, variant)} --output-root {root} "
                 f"--variant {variant} --seed {seed} --out {out.relative_to(REPO)}")

    # 3. block profiles -- cheap, so they go last and never block a training
    for root in DERIVED_ROOTS:
        units = Path(REPO, root, "units")
        if not units.is_dir():
            continue
        for unit in sorted(units.glob("*__seed*")):
            variant, seed = unit.name.split("__seed")
            if status(unit) != "complete":
                continue
            if (unit / "blockwise_profile.json").is_file():
                continue
            if not (unit / "checkpoints" / "last.ckpt").is_file():
                continue
            emit(f"prof:{root}:{variant}:{seed}",
                 f"{ANALYSIS} scripts/run_layerwise_profile.py "
                 f"--config-dir {config_for(root, variant)} --output-root {root} "
                 f"--variant {variant} --seed {seed} --granularity block")
    return (lines, blocked) if report_blocked else lines


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-keys", action="store_true",
                        help="print KEY\\tCOMMAND so a submitter can record in-flight")
    parser.add_argument("--blocked", action="store_true",
                        help="instead list targets that exhausted their attempts")
    arguments = parser.parse_args()
    lines, blocked = sweep(report_blocked=True)
    if arguments.blocked:
        for key, count in blocked:
            print(f"{key}\tattempts={count}")
        return
    for key, command in lines:
        print(f"{key}\t{command}" if arguments.with_keys else command)


if __name__ == "__main__":
    main()

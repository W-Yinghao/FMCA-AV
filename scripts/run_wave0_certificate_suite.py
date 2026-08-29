"""Wave 0 runner: counterexample battery + analytic chains (preregistered).

Prereg: prereg/WAVE0_CERTIFICATE_PREREG_FROZEN_20260816.md
Units are addressed by REAL string keys (never a bench index), written
append-only under the output root, and skipped when already complete.
Failures are recorded loud with a reason code; nothing degrades to None.

Usage:
  run_wave0_certificate_suite.py --group all            # every unit
  run_wave0_certificate_suite.py --probe                # one probe unit
  run_wave0_certificate_suite.py --unit <unit_key>      # one named unit
"""

import argparse
import json
import sys
import zlib
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fmca_av.certificate.controls import (
    endpoint_pairing_shuffle,
    pairing_noise_floor,
    random_orthogonal,
    rotate_interface,
    shuffle_edge_order,
    uncentered_coordinates,
)
from fmca_av.certificate.coordinates import fit_level_coordinates
from fmca_av.certificate.counterexamples import all_counterexamples
from fmca_av.certificate.estimation import (
    ChainFeatureBatch,
    encode_chain_batch,
    estimate_edge_operators,
    estimate_endpoint_operator,
    level_calibration_features,
)
from fmca_av.certificate.gaussian_chain import GaussianHermiteChain
from fmca_av.certificate.triplet import (
    CERTIFICATE_VERSION,
    certificate_report,
    compose_edge_operators,
    naive_singular_value_product,
)
from fmca_av.markov import nonnormal_chain, stationary_distribution

CERTIFY_THRESHOLD = 0.05
PARENTS_GRID = (1000, 10000, 100000)
CHILDREN_GRID = (1, 4, 16)
SEED_GRID = tuple(range(1, 21))
BASE_SEED = 20260816


def _unit_seed(unit_key: str, role: str) -> int:
    return (BASE_SEED + zlib.crc32(f"{unit_key}/{role}".encode())) % (2**31 - 1)


def _fit_pipeline(source, unit_key: str, parents: int, children: int):
    calibration = source.sample(
        max(parents // 4, 500),
        children_per_edge=children,
        endpoint_descendants=children,
        generator=torch.Generator().manual_seed(_unit_seed(unit_key, "calibration")),
    )
    coordinates = [
        fit_level_coordinates(level_calibration_features(calibration, level))
        for level in range(calibration.num_levels)
    ]
    batch = source.sample(
        parents,
        children_per_edge=children,
        endpoint_descendants=children,
        generator=torch.Generator().manual_seed(_unit_seed(unit_key, "estimation")),
    )
    encoded = encode_chain_batch(batch, coordinates)
    return calibration, batch, encoded


def _controls(unit_key: str, calibration, batch, encoded, edges, offset: float = 3.0) -> dict:
    rotation = random_orthogonal(
        edges[0].shape[1], torch.Generator().manual_seed(_unit_seed(unit_key, "rotation"))
    )
    base_comp = compose_edge_operators(edges)
    both = compose_edge_operators(rotate_interface(edges, 1, rotation, "both"))
    left = compose_edge_operators(rotate_interface(edges, 1, rotation, "left"))
    gauge_invariance = float((both - base_comp).abs().max())
    one_sided_change = float((left - base_comp).abs().max())

    floor = pairing_noise_floor(
        encoded.chain[0],
        encoded.children[0],
        repeats=10,
        generator=torch.Generator().manual_seed(_unit_seed(unit_key, "pairing")),
    )
    true_edge_norm = float(torch.linalg.matrix_norm(edges[0], ord="fro"))
    endpoint_shuffled = endpoint_pairing_shuffle(
        encoded, torch.Generator().manual_seed(_unit_seed(unit_key, "endpoint_pairing"))
    )

    def shift(chain_batch: ChainFeatureBatch) -> ChainFeatureBatch:
        return ChainFeatureBatch(
            chain=[states + offset for states in chain_batch.chain],
            children=[values + offset for values in chain_batch.children],
            endpoint_descendants=(
                chain_batch.endpoint_descendants + offset
                if chain_batch.endpoint_descendants is not None
                else None
            ),
        )

    shifted_calibration, shifted_batch = shift(calibration), shift(batch)
    centered_coordinates = [
        fit_level_coordinates(level_calibration_features(shifted_calibration, level))
        for level in range(shifted_calibration.num_levels)
    ]
    centered_comp = compose_edge_operators(
        estimate_edge_operators(encode_chain_batch(shifted_batch, centered_coordinates))
    )
    uncentered_comp = compose_edge_operators(
        estimate_edge_operators(
            encode_chain_batch(shifted_batch, uncentered_coordinates(shifted_calibration))
        )
    )

    interface_dims = {edge.shape[0] for edge in edges} | {edges[-1].shape[1]}
    order_shuffle_change = None
    if len(interface_dims) == 1 and len(edges) >= 2:
        shuffled = compose_edge_operators(shuffle_edge_order(edges, list(range(len(edges)))[::-1]))
        order_shuffle_change = float((shuffled - base_comp).abs().max())

    naive = naive_singular_value_product(edges)
    true_spectrum = torch.linalg.svdvals(base_comp)[: naive.shape[0]]
    return {
        "gauge_invariance_max_change": gauge_invariance,
        "one_sided_rotation_max_change": one_sided_change,
        "pairing_floor_max": float(floor.max()),
        "true_first_edge_frobenius": true_edge_norm,
        "endpoint_shuffle_frobenius": float(
            torch.linalg.matrix_norm(endpoint_shuffled, ord="fro")
        ),
        "centering_off_top_singular": float(torch.linalg.matrix_norm(uncentered_comp, ord=2)),
        "centering_on_top_singular": float(torch.linalg.matrix_norm(centered_comp, ord=2)),
        "layer_order_shuffle_max_change": order_shuffle_change,
        "naive_sigma_product_max_error": float((naive - true_spectrum).abs().max()),
    }


def _sampled_unit(source, unit_key: str, parents: int, children: int, population_edges, population_direct, check_determinism: bool) -> dict:
    calibration, batch, encoded = _fit_pipeline(source, unit_key, parents, children)
    edges = estimate_edge_operators(encoded)
    c_dir = estimate_endpoint_operator(encoded)
    report = certificate_report(c_dir, edges=edges)
    edge_error = max(
        float((edge - target).abs().max()) for edge, target in zip(edges, population_edges)
    )
    record = {
        "unit_key": unit_key,
        "certificate_version": CERTIFICATE_VERSION,
        "parents": parents,
        "children_per_edge": children,
        "edge_error_max": edge_error,
        "dir_error_max": float((c_dir - population_direct).abs().max()),
        "certificate_accepts": bool(float(report.certified_spectrum.max()) > CERTIFY_THRESHOLD),
        "report": report.as_metrics(),
        "controls": _controls(unit_key, calibration, batch, encoded, edges),
    }
    if check_determinism:
        _, _, replay = _fit_pipeline(source, unit_key, parents, children)
        replay_dir = estimate_endpoint_operator(replay)
        record["determinism_bit_identical"] = bool(torch.equal(replay_dir, c_dir))
    return record


def group_a_units():
    for case in all_counterexamples():
        for parents in PARENTS_GRID:
            for children in CHILDREN_GRID:
                for seed in SEED_GRID:
                    yield f"A/{case.name}/N{parents}/M{children}/seed{seed}", case
    return


def group_b_units():
    configs = {
        "full_orders": GaussianHermiteChain(rhos=[0.8, 0.6], level_orders=[[1, 2, 3]] * 3),
        "truncated": GaussianHermiteChain(
            rhos=[0.8, 0.6], level_orders=[[1, 2, 3], [1, 2], [1, 2, 3]]
        ),
    }
    for name, chain in configs.items():
        for parents in PARENTS_GRID:
            for children in CHILDREN_GRID:
                for seed in SEED_GRID:
                    yield f"B/{name}/N{parents}/M{children}/seed{seed}", chain


def _deflated_markov_operator(transition: torch.Tensor, mu_in: torch.Tensor):
    mu_out = mu_in @ transition
    root_in, root_out = mu_in.sqrt(), mu_out.sqrt()
    operator = root_in[:, None] * transition / root_out[None, :]
    return operator - root_in[:, None] @ root_out[None, :], mu_out


def group_c_records() -> dict:
    records = {}
    diag_edges = [
        torch.diag(torch.tensor([0.9, 0.4], dtype=torch.float64)),
        torch.diag(torch.tensor([0.4, 0.9], dtype=torch.float64)),
    ]
    rotation = random_orthogonal(2, torch.Generator().manual_seed(_unit_seed("C/rotated", "rotation")))
    for name, edges in {
        "C/misaligned_diag": diag_edges,
        "C/rotated_misaligned": rotate_interface(diag_edges, 1, rotation, "both"),
    }.items():
        comp = compose_edge_operators(edges)
        naive = naive_singular_value_product(edges)
        true_spectrum = torch.linalg.svdvals(comp)[: naive.shape[0]]
        records[name] = {
            "unit_key": name,
            "composed_singular_values": true_spectrum.tolist(),
            "naive_sigma_product": naive.tolist(),
            "naive_sigma_product_max_error": float((naive - true_spectrum).abs().max()),
        }
    generator = torch.Generator().manual_seed(_unit_seed("C/nonnormal", "chains"))
    first = nonnormal_chain(8, generator)
    second = nonnormal_chain(8, generator)
    mu_0 = stationary_distribution(first)
    edge_01, mu_1 = _deflated_markov_operator(first, mu_0)
    edge_12, _ = _deflated_markov_operator(second, mu_1)
    direct, _ = _deflated_markov_operator(first @ second, mu_0)
    comp = compose_edge_operators([edge_01, edge_12])
    naive = naive_singular_value_product([edge_01, edge_12])
    true_spectrum = torch.linalg.svdvals(comp)[: naive.shape[0]]
    records["C/nonnormal_markov"] = {
        "unit_key": "C/nonnormal_markov",
        "closure_identity_max_error": float((comp - direct).abs().max()),
        "composed_singular_values": true_spectrum.tolist(),
        "naive_sigma_product": naive.tolist(),
        "naive_sigma_product_max_error": float((naive - true_spectrum).abs().max()),
    }
    return records


def _unit_path(output_root: Path, unit_key: str) -> Path:
    return output_root / "units" / (unit_key.replace("/", "__") + ".json")


def run_unit(output_root: Path, unit_key: str, source) -> str:
    path = _unit_path(output_root, unit_key)
    if path.is_file():
        payload = json.loads(path.read_text())
        if payload.get("status") == "complete":
            return "skipped"
    path.parent.mkdir(parents=True, exist_ok=True)
    parts = unit_key.split("/")
    parents = int(parts[2][1:])
    children = int(parts[3][1:])
    check_determinism = parents == min(PARENTS_GRID)
    if hasattr(source, "population_edges") and callable(source.population_edges):
        population_edges = source.population_edges()
        population_direct = source.population_direct()
    else:
        population_edges = source.population_edges
        population_direct = source.population_direct
    try:
        record = _sampled_unit(
            source, unit_key, parents, children, population_edges, population_direct,
            check_determinism,
        )
        record["status"] = "complete"
    except Exception as error:  # loud failure record; never a silent None
        record = {"unit_key": unit_key, "status": "failed", "reason": repr(error)}
    path.write_text(json.dumps(record, indent=2))
    return record["status"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="results/wave0/" + CERTIFICATE_VERSION)
    parser.add_argument("--group", choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--unit", default=None)
    parser.add_argument(
        "--unit-prefix",
        default=None,
        help="run only units whose real key starts with this prefix (sharding across "
        "SLURM jobs; execution parallelism only, no preregistered quantity changes)",
    )
    arguments = parser.parse_args()
    output_root = Path(arguments.output_root)

    units = []
    if arguments.group in {"A", "all"}:
        units.extend(group_a_units())
    if arguments.group in {"B", "all"}:
        units.extend(group_b_units())
    if arguments.probe:
        units = [unit for unit in units if unit[0] == "A/closed_chain/N1000/M4/seed1"]
    if arguments.unit_prefix is not None:
        units = [unit for unit in units if unit[0].startswith(arguments.unit_prefix)]
        if not units:
            raise SystemExit(f"no units match prefix {arguments.unit_prefix!r}")
    if arguments.unit is not None:
        units = [unit for unit in units if unit[0] == arguments.unit]
        if not units:
            raise SystemExit(f"unknown unit key {arguments.unit!r}")

    statuses = {"complete": 0, "skipped": 0, "failed": 0}
    for unit_key, source in units:
        statuses[run_unit(output_root, unit_key, source)] += 1
        if statuses["failed"]:
            print(f"FAILED at {unit_key}; stopping loud.", flush=True)
            break
        total_done = statuses["complete"] + statuses["skipped"]
        if total_done % 50 == 0:
            print(f"progress: {statuses}", flush=True)

    if arguments.group in {"C", "all"} and not arguments.probe and arguments.unit is None:
        for unit_key, record in group_c_records().items():
            path = _unit_path(output_root, unit_key)
            path.parent.mkdir(parents=True, exist_ok=True)
            record["status"] = "complete"
            path.write_text(json.dumps(record, indent=2))
            statuses["complete"] += 1

    print(json.dumps({"statuses": statuses, "output_root": str(output_root)}, indent=2))
    if statuses["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Task 25.4 Phase 3, Track A: rigorous project-eigenvalue certification.

For each fixed (n,c), one directed enclosure of the largest project matrix is
reused for all eight leading principal blocks.  This is the computational form
of the project's exact fixed-c nesting advantage.  The finite Ritz enclosures
are combined separately with the Phase 2 exact-rational tail and with the now
certified hybrid finite-head/tail formula.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np
from flint import arb, ctx

from task_12_1_assemble_matrix import assemble, assemble_routeA
from task_25_4_phase3_gate0_interval import REFERENCE_DPS, _ball_record
from task_25_4_phase3_interval_ritz import (
    BALL_BITS,
    _matrix_contains,
    _ordinary_matrix_contained,
    assemble_interval_matrix,
    certified_bessel_zeros,
    certified_weyl_eigen_intervals,
    intersect_interval_matrices,
    interval_riemann_ID,
)


BUFFERS = (5, 8, 12, 20, 32, 48, 64, 96)
BASE_COARSE_CELLS = {10: 512, 20: 1024, 40: 2048, 80: 4096}
SHARP_CASE = {"n": 0, "c": 10, "P": 8, "coarse_cells": 32768}


def explicit_tail_fraction(n: int, c: int, P: int) -> Fraction:
    """Exact rational Phase 2 missing-trace upper bound."""
    B = 12 * P + 9
    D = B * B - 16 * c * c
    if D <= 0:
        raise ValueError("explicit tail requires 3(P+3/4)>c")
    if n == 0:
        numerator = 16 * c * B * (12 * B + D)
        denominator = 9 * D * D
    else:
        numerator = 2 * (4 * c + 4 * n * n - 1) * B * (12 * B + D)
        denominator = 3 * D * D
    return Fraction(numerator, denominator)


def arb_from_fraction(value: Fraction) -> arb:
    return arb(f"{value.numerator}/{value.denominator}")


def compact_matrix_records(matrix: list[list[arb]]) -> list[list[dict[str, str]]]:
    return [
        [
            {
                "lower": value.lower().str(65, radius=False),
                "upper": value.upper().str(65, radius=False),
            }
            for value in row
        ]
        for row in matrix
    ]


def interval_endpoints(ritz: arb, tail_upper: arb) -> tuple[arb, arb]:
    lower = ritz.lower().max(arb(0))
    upper = (ritz.upper() + tail_upper).min(arb(1))
    if upper < lower:
        raise RuntimeError("empty continuous eigenvalue enclosure")
    return lower, upper


def selected_certificates(
    selected_modes: list[dict[str, object]],
    ritz_intervals: list[arb],
    explicit_tail: Fraction,
    hybrid_tail: arb,
) -> list[dict[str, object]]:
    explicit_ball = arb_from_fraction(explicit_tail)
    rows = []
    for mode in selected_modes:
        if not mode["available"]:
            rows.append({"category": mode["category"], "available": False, "m": None})
            continue
        m = int(mode["m"])
        ritz = ritz_intervals[m]
        explicit_lower, explicit_upper = interval_endpoints(ritz, explicit_ball.upper())
        hybrid_lower, hybrid_upper = interval_endpoints(ritz, hybrid_tail.upper())
        reference = float(mode["lambda_ref"])
        u_ref = float(mode["u_ref"])
        explicit_contains = bool(
            explicit_lower <= arb(reference) and arb(reference) <= explicit_upper
        )
        hybrid_contains = bool(
            hybrid_lower <= arb(reference) and arb(reference) <= hybrid_upper
        )
        finite_mid = float(ritz.mid())
        observed_error = abs(finite_mid - reference)
        explicit_width = float(explicit_upper - explicit_lower)
        hybrid_width = float(hybrid_upper - hybrid_lower)
        rows.append(
            {
                "category": mode["category"],
                "available": True,
                "m": m,
                "lambda_ref": reference,
                "u_ref": u_ref,
                "finite_Ritz_interval": _ball_record(ritz),
                "explicit_continuous_interval": {
                    "lower": explicit_lower.str(65, radius=False),
                    "upper": explicit_upper.str(65, radius=False),
                    "width": explicit_width,
                    "contains_ordinary_reference": explicit_contains,
                    "conservatism_ratio_chi": explicit_width / max(observed_error, u_ref),
                },
                "hybrid_continuous_interval": {
                    "lower": hybrid_lower.str(65, radius=False),
                    "upper": hybrid_upper.str(65, radius=False),
                    "width": hybrid_width,
                    "contains_ordinary_reference": hybrid_contains,
                    "conservatism_ratio_chi": hybrid_width / max(observed_error, u_ref),
                },
                "ordinary_observed_finite_Ritz_error": observed_error,
            }
        )
    return rows


def build_enclosed_matrix(
    n: int, c: int, P: int, coarse_cells: int
) -> tuple[list[list[arb]], dict[str, object], list[dict[str, object]]]:
    roots, root_records = certified_bessel_zeros(n, P)
    I0, D0, direct0, coarse_meta = interval_riemann_ID(n, c, roots, coarse_cells)
    I1, D1, direct1, refined_meta = interval_riemann_ID(n, c, roots, 2 * coarse_cells)
    partial0 = assemble_interval_matrix(roots, I0, D0)
    partial1 = assemble_interval_matrix(roots, I1, D1)
    matrix0, overlap0 = intersect_interval_matrices(direct0, partial0)
    matrix1, overlap1 = intersect_interval_matrices(direct1, partial1)

    ordinary_B, cache = assemble(n, float(c), P=P, G=48, return_cache=True)
    ordinary_A = assemble_routeA(n, float(c), P, cache)
    contains_B, distance_B = _ordinary_matrix_contained(matrix1, ordinary_B)
    contains_A, distance_A = _ordinary_matrix_contained(matrix1, ordinary_A)
    checks = {
        "all_bessel_zeros_uniquely_enclosed": all(
            bool(record["passed"]) for record in root_records
        ),
        "refined_I_D_contained_in_coarse": all(
            I0[k].contains(I1[k]) and D0[k].contains(D1[k]) for k in range(P)
        ),
        "refined_direct_Gram_contained_in_coarse": _matrix_contains(direct1, direct0),
        "refined_partial_fraction_contained_in_coarse": _matrix_contains(partial1, partial0),
        "refined_intersection_contained_in_coarse": _matrix_contains(matrix1, matrix0),
        "coarse_routes_overlap": overlap0,
        "refined_routes_overlap": overlap1,
        "ordinary_route_A_contained": contains_A,
        "ordinary_route_B_contained": contains_B,
        "ordinary_route_A_worst_outside_distance": distance_A,
        "ordinary_route_B_worst_outside_distance": distance_B,
    }
    checks["all_passed"] = all(
        bool(value) for key, value in checks.items() if "distance" not in key
    )
    metadata = {
        "coarse": coarse_meta,
        "refined": refined_meta,
        "checks": checks,
    }
    return matrix1, metadata, root_records


def certify_case(case: dict[str, object], artifact_dir: Path) -> dict[str, object]:
    n, c, M = int(case["n"]), int(case["c"]), int(case["M_n_c"])
    dimensions = [M + buffer for buffer in BUFFERS]
    H = dimensions[-1]
    started = time.perf_counter()
    full_matrix, assembly_meta, root_records = build_enclosed_matrix(
        n, c, H, BASE_COARSE_CELLS[c]
    )

    sharp = None
    if n == SHARP_CASE["n"] and c == SHARP_CASE["c"]:
        sharp_matrix, sharp_meta, sharp_roots = build_enclosed_matrix(
            n,
            c,
            SHARP_CASE["P"],
            SHARP_CASE["coarse_cells"],
        )
        sharp = {
            "P": SHARP_CASE["P"],
            "coarse_cells": SHARP_CASE["coarse_cells"],
            "matrix": sharp_matrix,
            "metadata": sharp_meta,
            "root_records": sharp_roots,
        }

    block_rows = []
    for P in dimensions:
        matrix = [row[:P] for row in full_matrix[:P]]
        source = "largest certified matrix leading block"
        if sharp is not None and P == sharp["P"]:
            matrix = sharp["matrix"]
            source = "separate sharp nested certification run"

        ritz_intervals, eigen_meta = certified_weyl_eigen_intervals(matrix)
        explicit = explicit_tail_fraction(n, c, P)
        remainder_H = arb_from_fraction(explicit_tail_fraction(n, c, H))
        diagonal_head = arb(0)
        for k in range(P, H):
            diagonal_head += full_matrix[k][k].upper()
        hybrid = diagonal_head + remainder_H
        certificates = selected_certificates(
            case["selected_modes"], ritz_intervals, explicit, hybrid
        )
        block_rows.append(
            {
                "P": P,
                "matrix_source": source,
                "eigenvalue_method": eigen_meta,
                "explicit_tail": {
                    "numerator": explicit.numerator,
                    "denominator": explicit.denominator,
                    "decimal_upper": float(explicit),
                },
                "hybrid_tail": {
                    "H": H,
                    "validated_diagonal_head": _ball_record(diagonal_head),
                    "analytic_remainder_at_H": _ball_record(remainder_H),
                    "certified_upper_ball": _ball_record(hybrid),
                },
                "selected_mode_certificates": certificates,
            }
        )

    all_reference_contained = all(
        row[tail_name]["contains_ordinary_reference"]
        for block in block_rows
        for row in block["selected_mode_certificates"]
        if row["available"]
        for tail_name in ("explicit_continuous_interval", "hybrid_continuous_interval")
    )
    nontrivial_width_count = sum(
        row[tail_name]["width"] < 1
        for block in block_rows
        for row in block["selected_mode_certificates"]
        if row["available"]
        for tail_name in ("explicit_continuous_interval", "hybrid_continuous_interval")
    )

    artifact = {
        "task": "25.4 Phase 3 Track A case",
        "evidence_label": "certified numerical",
        "parameter_path": "fixed continuous c",
        "n": n,
        "c": c,
        "M_n_c": M,
        "H": H,
        "ball_precision_bits": BALL_BITS,
        "largest_matrix_assembly": assembly_meta,
        "bessel_zero_certificates": root_records,
        "largest_interval_matrix": compact_matrix_records(full_matrix),
        "sharp_run": (
            None
            if sharp is None
            else {
                "P": sharp["P"],
                "metadata": sharp["metadata"],
                "bessel_zero_certificates": sharp["root_records"],
                "interval_matrix": compact_matrix_records(sharp["matrix"]),
            }
        ),
        "blocks": block_rows,
        "all_reference_contained": all_reference_contained,
        "nontrivial_width_count": nontrivial_width_count,
        "elapsed_seconds": time.perf_counter() - started,
    }
    rendered = json.dumps(artifact, indent=2)
    artifact_path = artifact_dir / f"track_A_n{n}_c{c}.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(rendered + "\n", encoding="utf-8")
    digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    return {
        "n": n,
        "c": c,
        "M_n_c": M,
        "H": H,
        "artifact": str(artifact_path),
        "sha256": digest,
        "all_assembly_checks_passed": bool(assembly_meta["checks"]["all_passed"]),
        "sharp_checks_passed": bool(
            sharp is None or sharp["metadata"]["checks"]["all_passed"]
        ),
        "all_reference_contained": all_reference_contained,
        "nontrivial_width_count": nontrivial_width_count,
        "elapsed_seconds": artifact["elapsed_seconds"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/task_25_4_phase3/reference_mode_manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/task_25_4_phase3/track_A_certification.json"),
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("data/task_25_4_phase3/track_A_cases"),
    )
    args = parser.parse_args()

    ctx.prec = BALL_BITS
    mp.mp.dps = REFERENCE_DPS
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    started = time.perf_counter()
    cases = []
    for case in manifest["cases"]:
        summary = certify_case(case, args.artifact_dir)
        cases.append(summary)
        print(
            f"n={summary['n']} c={summary['c']}: assembly={summary['all_assembly_checks_passed']} "
            f"reference={summary['all_reference_contained']} nontrivial={summary['nontrivial_width_count']}",
            flush=True,
        )

    gate_A = all(
        row["all_assembly_checks_passed"]
        and row["sharp_checks_passed"]
        and row["all_reference_contained"]
        for row in cases
    )
    nontrivial_count = sum(row["nontrivial_width_count"] for row in cases)
    gate_B = nontrivial_count >= 1
    result = {
        "task": "25.4 Phase 3 Track A certification summary",
        "evidence_label": "certified numerical",
        "parameter_path": "fixed continuous c",
        "largest_matrix_reuse": True,
        "base_coarse_cells": BASE_COARSE_CELLS,
        "sharp_case": SHARP_CASE,
        "cases": cases,
        "gate_A_passed": gate_A,
        "gate_B_nontrivial_width_passed": gate_B,
        "nontrivial_continuous_interval_count": nontrivial_count,
        "elapsed_seconds_total": time.perf_counter() - started,
    }
    rendered = json.dumps(result, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({"output": str(args.output), "sha256": digest, **result}, indent=2))
    return 0 if gate_A and gate_B else 1


if __name__ == "__main__":
    raise SystemExit(main())

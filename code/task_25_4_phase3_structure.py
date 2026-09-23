"""Task 25.4 Phase 3, Track B: structure, nesting, monotonicity, and reuse.

This program intentionally does not load reference eigenvalues and does not
compute project/Boulsane accuracy.  It runs before Track C, as preregistered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.special import jn_zeros, jv, jvp

from task_12_1_assemble_matrix import assemble, assemble_routeA


BUFFERS = (5, 8, 12, 20, 32, 48, 64, 96)
CONTROL_DIMS = (16, 24, 32, 48, 64)
EPS = np.finfo(float).eps
AUDIT_FACTOR = 1e-11


def _G_n(n: int, x: np.ndarray, y: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    same = np.abs(x - y) < tol
    with np.errstate(divide="ignore", invalid="ignore"):
        off = (
            np.sqrt(np.abs(x * y))
            / (x * x - y * y + 1e-300)
            * (
                x * jv(n + 1, x) * jv(n, y)
                - y * jv(n + 1, y) * jv(n, x)
            )
        )
    diagonal = 0.5 * (
        (jv(n + 1, x) + x * jvp(n + 1, x, 1)) * jv(n, x)
        - x * jv(n + 1, x) * jvp(n, x, 1)
    )
    return np.where(same, diagonal, off)


def boulsane_matrix(n: int, N: int, zeros: np.ndarray, omega: float) -> np.ndarray:
    s = zeros[:N]
    X, Y = np.meshgrid(s, s, indexing="ij")
    kernel = omega * _G_n(n, omega * X, omega * Y)
    denominator = (
        np.sqrt(np.outer(s, s))
        * np.outer(np.abs(jv(n + 1, s)), np.abs(jv(n + 1, s)))
    )
    matrix = 2 * kernel / denominator
    return 0.5 * (matrix + matrix.T)


@lru_cache(maxsize=None)
def gauss_rule(Q: int) -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(Q)


def boulsane_direct_gram(
    n: int, N: int, zeros: np.ndarray, omega: float, Q: int
) -> np.ndarray:
    t, w = gauss_rule(Q)
    r = 0.5 * omega * (t + 1)
    weights = 0.5 * omega * w
    s = zeros[:N]
    basis = (
        np.sqrt(2 * r)[:, None]
        * jv(n, np.outer(r, s))
        / np.abs(jv(n + 1, s))[None, :]
    )
    matrix = basis.T @ (weights[:, None] * basis)
    return 0.5 * (matrix + matrix.T)


def descending_eigenvalues(matrix: np.ndarray) -> np.ndarray:
    return np.linalg.eigvalsh(0.5 * (matrix + matrix.T))[::-1]


def matrix_audit(
    primary: np.ndarray,
    alternatives: dict[str, np.ndarray],
) -> dict[str, object]:
    norm = float(np.linalg.norm(primary, 2))
    tolerance = AUDIT_FACTOR * max(1.0, norm)
    symmetry = float(np.max(np.abs(primary - primary.T)))
    disagreements = {
        name: float(np.linalg.norm(primary - alternate, 2))
        for name, alternate in alternatives.items()
    }
    maximum = max([symmetry, *disagreements.values()], default=symmetry)
    return {
        "matrix_norm_2": norm,
        "tolerance": tolerance,
        "symmetry_defect_max_abs": symmetry,
        "disagreements_norm_2": disagreements,
        "maximum_audit_disagreement": maximum,
        "passed": bool(maximum <= tolerance),
    }


def transition_record(
    previous: dict[str, object],
    current: dict[str, object],
    dimension_name: str,
) -> dict[str, object]:
    old_matrix = previous["matrix"]
    new_matrix = current["matrix"]
    old_dimension = int(previous[dimension_name])
    new_dimension = int(current[dimension_name])
    block = new_matrix[:old_dimension, :old_dimension]
    defect = float(np.linalg.norm(old_matrix - block, 2))
    relative = defect / max(1.0, float(np.linalg.norm(old_matrix, 2)))

    old_values = previous["eigenvalues"]
    new_values = current["eigenvalues"]
    increments = new_values[:old_dimension] - old_values
    assembly_old = float(previous["audit"]["maximum_audit_disagreement"])
    assembly_new = float(current["audit"]["maximum_audit_disagreement"])
    tau = 10 * (assembly_old + assembly_new + new_dimension * EPS)
    return {
        f"{dimension_name}_old": old_dimension,
        f"{dimension_name}_new": new_dimension,
        "block_defect_norm_2": defect,
        "relative_block_defect": relative,
        "tau": tau,
        "eigenvalue_increments": increments.tolist(),
        "raw_negative_increment_count": int(np.sum(increments < 0)),
        "significant_negative_increment_count": int(np.sum(increments < -tau)),
        "minimum_increment": float(np.min(increments)),
    }


def strip_arrays(row: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in row.items() if key not in {"matrix", "eigenvalues"}}


def project_case(n: int, c: int, M: int) -> dict[str, object]:
    dimensions = [M + buffer for buffer in BUFFERS]
    largest_P = dimensions[-1]
    largest, largest_cache = assemble(n, float(c), P=largest_P, G=48, return_cache=True)
    rows = []
    for P in dimensions:
        B32 = assemble(n, float(c), P=P, G=32)
        B48, cache = assemble(n, float(c), P=P, G=48, return_cache=True)
        route_A = assemble_routeA(n, float(c), P, cache)
        common_block = largest[:P, :P]
        audit = matrix_audit(
            B48,
            {"G32": B32, "route_A_G48": route_A, "largest_common_block": common_block},
        )
        rows.append(
            {
                "P": P,
                "matrix": B48,
                "eigenvalues": descending_eigenvalues(B48),
                "audit": audit,
            }
        )
    transitions = [transition_record(rows[i - 1], rows[i], "P") for i in range(1, len(rows))]
    return {
        "n": n,
        "c": c,
        "M_n_c": M,
        "parameter_path": "fixed continuous c",
        "rows": [strip_arrays(row) for row in rows],
        "transitions": transitions,
        "all_rows_audited": all(bool(row["audit"]["passed"]) for row in rows),
        "significant_negative_increment_count": sum(
            int(row["significant_negative_increment_count"]) for row in transitions
        ),
        "maximum_block_defect_norm_2": max(
            (float(row["block_defect_norm_2"]) for row in transitions), default=0.0
        ),
    }


def large_N_epsilon(n: int, N: int, zeros: np.ndarray) -> float:
    return float((N + n / 2 + 1 / 4) * np.pi - zeros[N - 1])


def boulsane_fixed_c_case(n: int, c: int, M: int) -> dict[str, object]:
    dimensions = [M + buffer for buffer in BUFFERS]
    zeros = jn_zeros(n, max(dimensions) + 1)
    rows = []
    for N in dimensions:
        epsilon = large_N_epsilon(n, N, zeros)
        omega = float(c / (zeros[N - 1] + epsilon))
        closed = boulsane_matrix(n, N, zeros, omega)
        gram2000 = boulsane_direct_gram(n, N, zeros, omega, 2000)
        gram3000 = boulsane_direct_gram(n, N, zeros, omega, 3000)
        audit = matrix_audit(
            closed,
            {"direct_Gram_Q2000": gram2000, "direct_Gram_Q3000": gram3000},
        )
        audit["direct_Gram_Q2000_vs_Q3000_norm_2"] = float(
            np.linalg.norm(gram2000 - gram3000, 2)
        )
        audit["passed"] = bool(
            audit["passed"]
            and audit["direct_Gram_Q2000_vs_Q3000_norm_2"] <= audit["tolerance"]
        )
        rows.append(
            {
                "N": N,
                "epsilon": epsilon,
                "omega": omega,
                "matrix": closed,
                "eigenvalues": descending_eigenvalues(closed),
                "audit": audit,
            }
        )
    transitions = [transition_record(rows[i - 1], rows[i], "N") for i in range(1, len(rows))]
    return {
        "n": n,
        "c": c,
        "M_n_c": M,
        "parameter_path": "fixed continuous c; Boulsane large-N offset",
        "rows": [strip_arrays(row) for row in rows],
        "transitions": transitions,
        "all_rows_audited": all(bool(row["audit"]["passed"]) for row in rows),
        "significant_negative_increment_count": sum(
            int(row["significant_negative_increment_count"]) for row in transitions
        ),
        "maximum_block_defect_norm_2": max(
            (float(row["block_defect_norm_2"]) for row in transitions), default=0.0
        ),
    }


def fixed_native_boulsane_controls() -> list[dict[str, object]]:
    controls = []
    for n in (0, 2):
        zeros = jn_zeros(n, max(CONTROL_DIMS) + 1)
        rows = []
        for N in CONTROL_DIMS:
            matrix = boulsane_matrix(n, N, zeros, 0.5)
            rows.append(
                {
                    "N": N,
                    "omega": 0.5,
                    "effective_c": float(0.5 * zeros[N]),
                    "matrix": matrix,
                    "eigenvalues": descending_eigenvalues(matrix),
                    "audit": {
                        "maximum_audit_disagreement": 0.0,
                        "passed": True,
                    },
                }
            )
        transitions = [transition_record(rows[i - 1], rows[i], "N") for i in range(1, len(rows))]
        controls.append(
            {
                "n": n,
                "parameter_path": "fixed native omega=1/2; effective c changes with N",
                "rows": [strip_arrays(row) for row in rows],
                "transitions": transitions,
                "maximum_block_defect_norm_2": max(
                    float(row["block_defect_norm_2"]) for row in transitions
                ),
            }
        )
    return controls


def dpss_matrix(N: int, W: float) -> np.ndarray:
    indices = np.arange(N)
    difference = indices[:, None] - indices[None, :]
    matrix = np.empty((N, N), dtype=float)
    diagonal = difference == 0
    matrix[diagonal] = 2 * W
    matrix[~diagonal] = np.sin(2 * np.pi * W * difference[~diagonal]) / (
        np.pi * difference[~diagonal]
    )
    return matrix


def dpss_control(path: str) -> dict[str, object]:
    rows = []
    for N in CONTROL_DIMS:
        W = 0.1 if path == "fixed W" else 20 / (np.pi * N)
        matrix = dpss_matrix(N, W)
        rows.append(
            {
                "N": N,
                "W": float(W),
                "effective_c_pi_N_W": float(np.pi * N * W),
                "matrix": matrix,
                "eigenvalues": descending_eigenvalues(matrix),
                "audit": {"maximum_audit_disagreement": 0.0, "passed": True},
            }
        )
    transitions = [transition_record(rows[i - 1], rows[i], "N") for i in range(1, len(rows))]
    return {
        "parameter_path": (
            "classical DPSS fixed W=0.1" if path == "fixed W" else "classical DPSS fixed c=20"
        ),
        "rows": [strip_arrays(row) for row in rows],
        "transitions": transitions,
        "maximum_block_defect_norm_2": max(
            float(row["block_defect_norm_2"]) for row in transitions
        ),
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
        default=Path("data/task_25_4_phase3/track_B_structure.json"),
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    started = time.perf_counter()
    project_cases = []
    boulsane_cases = []
    for case in manifest["cases"]:
        n, c, M = int(case["n"]), int(case["c"]), int(case["M_n_c"])
        project = project_case(n, c, M)
        boulsane = boulsane_fixed_c_case(n, c, M)
        project_cases.append(project)
        boulsane_cases.append(boulsane)
        print(
            f"n={n} c={c}: project block max={project['maximum_block_defect_norm_2']:.3e}; "
            f"Boulsane block max={boulsane['maximum_block_defect_norm_2']:.3e}",
            flush=True,
        )

    native_controls = fixed_native_boulsane_controls()
    dpss_controls = [dpss_control("fixed W"), dpss_control("fixed c")]
    result = {
        "task": "25.4 Phase 3 Track B structure",
        "evidence_label": "ordinary numerical structural diagnostics",
        "reference_eigenvalues_loaded": False,
        "buffers": list(BUFFERS),
        "assembly_audit_relative_factor": AUDIT_FACTOR,
        "project_fixed_c": project_cases,
        "boulsane_fixed_c_large_N_offset": boulsane_cases,
        "boulsane_fixed_native_controls": native_controls,
        "classical_DPSS_controls": dpss_controls,
        "summary": {
            "all_project_rows_audited": all(case["all_rows_audited"] for case in project_cases),
            "all_boulsane_rows_audited": all(case["all_rows_audited"] for case in boulsane_cases),
            "project_significant_negative_increment_count": sum(
                case["significant_negative_increment_count"] for case in project_cases
            ),
            "boulsane_significant_negative_increment_count": sum(
                case["significant_negative_increment_count"] for case in boulsane_cases
            ),
            "maximum_project_block_defect_norm_2": max(
                case["maximum_block_defect_norm_2"] for case in project_cases
            ),
            "minimum_boulsane_fixed_c_block_defect_norm_2": min(
                transition["block_defect_norm_2"]
                for case in boulsane_cases
                for transition in case["transitions"]
            ),
            "maximum_boulsane_fixed_native_block_defect_norm_2": max(
                case["maximum_block_defect_norm_2"] for case in native_controls
            ),
            "DPSS_fixed_W_maximum_block_defect_norm_2": dpss_controls[0][
                "maximum_block_defect_norm_2"
            ],
            "DPSS_fixed_c_minimum_block_defect_norm_2": min(
                row["block_defect_norm_2"] for row in dpss_controls[1]["transitions"]
            ),
        },
        "elapsed_seconds_total": time.perf_counter() - started,
    }
    rendered = json.dumps(result, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({"output": str(args.output), "sha256": digest, **result["summary"]}, indent=2))
    return 0 if result["summary"]["all_project_rows_audited"] and result["summary"]["all_boulsane_rows_audited"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

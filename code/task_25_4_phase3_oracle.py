"""Task 25.4 Phase 3, Track C oracle-offset diagnostic.

This script is deliberately separate from the preregistered fixed-offset run.
It searches only after that artifact has been saved and hashed.  The oracle is
diagnostic and is never presented as a practical construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import jn_zeros

from task_25_4_phase3_mechanism import BUFFERS, audit_boulsane
from task_25_4_phase3_structure import boulsane_matrix


GRID_RATIOS = np.linspace(0.25, 1.25, 65)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class OracleEvaluator:
    def __init__(
        self,
        n: int,
        c: int,
        N: int,
        m: int,
        lambda_ref: float,
        zeros: np.ndarray,
    ) -> None:
        self.n = n
        self.c = c
        self.N = N
        self.m = m
        self.lambda_ref = lambda_ref
        self.zeros = zeros
        self.cache: dict[float, tuple[float, float]] = {}

    def __call__(self, epsilon: float) -> float:
        key = float(epsilon)
        if key not in self.cache:
            D = float(self.zeros[self.N - 1] + key)
            omega = float(self.c / D)
            matrix = boulsane_matrix(self.n, self.N, self.zeros, omega)
            value = float(np.linalg.eigvalsh(matrix)[::-1][self.m])
            self.cache[key] = (self.lambda_ref - value, value)
        return self.cache[key][0]

    def record(self, epsilon: float) -> dict[str, float]:
        error = float(self(epsilon))
        value = self.cache[float(epsilon)][1]
        return {
            "epsilon": float(epsilon),
            "epsilon_over_pi": float(epsilon / np.pi),
            "signed_eigenvalue_error": error,
            "absolute_eigenvalue_error": abs(error),
            "lambda_Boulsane": value,
        }


def unique_records(records: list[dict[str, float]], tolerance: float = 1e-10) -> list[dict[str, float]]:
    ordered = sorted(records, key=lambda row: row["epsilon"])
    unique: list[dict[str, float]] = []
    for row in ordered:
        if not unique or abs(row["epsilon"] - unique[-1]["epsilon"]) > tolerance:
            unique.append(row)
        elif row["absolute_eigenvalue_error"] < unique[-1]["absolute_eigenvalue_error"]:
            unique[-1] = row
    return unique


def search_dimension(
    n: int,
    c: int,
    N: int,
    m: int,
    lambda_ref: float,
    zeros: np.ndarray,
) -> dict[str, object]:
    started = time.perf_counter()
    evaluator = OracleEvaluator(n, c, N, m, lambda_ref, zeros)
    grid_epsilon = GRID_RATIOS * np.pi
    grid_error = np.asarray([evaluator(value) for value in grid_epsilon])

    zeros_found: list[dict[str, float]] = []
    for left, right, f_left, f_right in zip(
        grid_epsilon[:-1], grid_epsilon[1:], grid_error[:-1], grid_error[1:]
    ):
        if f_left == 0:
            zeros_found.append(evaluator.record(float(left)))
        if f_left * f_right < 0:
            root = brentq(evaluator, float(left), float(right), xtol=1e-13, rtol=1e-14)
            zeros_found.append(evaluator.record(float(root)))
    if grid_error[-1] == 0:
        zeros_found.append(evaluator.record(float(grid_epsilon[-1])))
    zeros_found = unique_records(zeros_found)

    minima: list[dict[str, float]] = []
    absolute = np.abs(grid_error)
    for index in range(1, len(grid_epsilon) - 1):
        if absolute[index] <= absolute[index - 1] and absolute[index] <= absolute[index + 1]:
            result = minimize_scalar(
                lambda epsilon: abs(evaluator(float(epsilon))),
                bounds=(float(grid_epsilon[index - 1]), float(grid_epsilon[index + 1])),
                method="bounded",
                options={"xatol": 1e-13, "maxiter": 200},
            )
            minima.append(evaluator.record(float(result.x)))
    minima = unique_records(minima)

    candidates = [*zeros_found, *minima]
    candidates.extend(
        [evaluator.record(float(grid_epsilon[0])), evaluator.record(float(grid_epsilon[-1]))]
    )
    optimum = min(
        candidates,
        key=lambda row: (row["absolute_eigenvalue_error"], row["epsilon"]),
    )
    epsilon = float(optimum["epsilon"])
    D = float(zeros[N - 1] + epsilon)
    omega = float(c / D)
    matrix = boulsane_matrix(n, N, zeros, omega)
    audit = audit_boulsane(n, N, zeros, omega, matrix)
    h = float(np.pi / D)
    deviation = epsilon - np.pi / 2
    return {
        "N": N,
        "grid": [
            {
                "epsilon_over_pi": float(ratio),
                "signed_eigenvalue_error": float(error),
            }
            for ratio, error in zip(GRID_RATIOS, grid_error)
        ],
        "detected_signed_error_zeros": zeros_found,
        "refined_local_minima": minima,
        "optimum": optimum,
        "D": D,
        "omega": omega,
        "h": h,
        "deviation_from_half_mesh": float(deviation),
        "absolute_deviation_from_half_mesh": abs(float(deviation)),
        "absolute_deviation_over_h": abs(float(deviation)) / h,
        "audit": audit,
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_case(case: dict[str, object]) -> dict[str, object]:
    n, c, M = int(case["n"]), int(case["c"]), int(case["M_n_c"])
    knee = next(row for row in case["selected_modes"] if row["category"] == "knee")
    m = int(knee["m"])
    lambda_ref = float(knee["lambda_ref"])
    dimensions = [M + buffer for buffer in BUFFERS]
    zeros = jn_zeros(n, max(dimensions) + 1)
    rows = [search_dimension(n, c, N, m, lambda_ref, zeros) for N in dimensions]
    last = rows[-4:]
    deviations = [row["absolute_deviation_from_half_mesh"] for row in last]
    ratios = [row["absolute_deviation_over_h"] for row in last]
    decreasing = all(right < left for left, right in zip(deviations[:-1], deviations[1:]))
    positive_ratios = [ratio for ratio in ratios if ratio > 0]
    factor = (
        max(positive_ratios) / min(positive_ratios)
        if len(positive_ratios) == len(ratios)
        else math.inf
    )
    supported = bool(decreasing and deviations[-1] <= 0.1 * np.pi and factor <= 4)
    return {
        "n": n,
        "c": c,
        "M_n_c": M,
        "m_knee": m,
        "lambda_ref": lambda_ref,
        "parameter_path": "fixed continuous c",
        "rows": rows,
        "last_four_assessment": {
            "dimensions": [row["N"] for row in last],
            "absolute_deviations": deviations,
            "absolute_deviation_over_h": ratios,
            "deviations_strictly_decrease": bool(decreasing),
            "last_deviation_at_most_0.1_pi": bool(deviations[-1] <= 0.1 * np.pi),
            "ratio_variation_factor": float(factor),
            "ratio_varies_by_at_most_factor_four": bool(factor <= 4),
            "O_h_support_criteria_passed": supported,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/task_25_4_phase3/reference_mode_manifest.json"),
    )
    parser.add_argument(
        "--fixed",
        type=Path,
        default=Path("data/task_25_4_phase3/track_C_fixed_offsets.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/task_25_4_phase3/track_C_oracle.json"),
    )
    args = parser.parse_args()
    if not args.fixed.exists():
        raise FileNotFoundError("The fixed-offset artifact must exist before the oracle run")
    fixed_hash = sha256(args.fixed)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    started = time.perf_counter()
    cases = []
    for case in manifest["cases"]:
        result = run_case(case)
        cases.append(result)
        passed = result["last_four_assessment"]["O_h_support_criteria_passed"]
        print(f"n={result['n']} c={result['c']}: oracle O(h) support={passed}", flush=True)
    payload = {
        "task": "25.4 Phase 3 Track C oracle offset",
        "evidence_label": "diagnostic oracle; not a practical construction",
        "parameter_path": "fixed continuous c",
        "fixed_offset_artifact": str(args.fixed),
        "fixed_offset_artifact_sha256_before_oracle": fixed_hash,
        "grid_epsilon_over_pi": {
            "minimum": 0.25,
            "maximum": 1.25,
            "spacing": 1 / 64,
            "point_count": 65,
        },
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "O_h_support_case_count": sum(
                bool(case["last_four_assessment"]["O_h_support_criteria_passed"])
                for case in cases
            ),
            "all_selected_matrices_audited": all(
                row["audit"]["passed"] for case in cases for row in case["rows"]
            ),
        },
        "elapsed_seconds_total": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    digest = sha256(args.output)
    print(json.dumps({"output": str(args.output), "sha256": digest, **payload["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

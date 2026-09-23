"""Task 25.4 Phase 3, Track E refinement and timing campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import scipy
from scipy.special import jn_zeros, jv
from task_12_1_assemble_matrix import _panels_nodes, assemble
from task_25_4_phase3_mechanism import BUFFERS, offset_value
from task_25_4_phase3_structure import boulsane_matrix


REPETITIONS = 7
G = 48


def distribution(values: list[float]) -> dict[str, object]:
    return {
        "values_seconds": values,
        "median_seconds": float(statistics.median(values)),
        "Q1_seconds": float(np.quantile(values, 0.25)),
        "Q3_seconds": float(np.quantile(values, 0.75)),
        "IQR_seconds": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
    }


def benchmark(function: Callable[[], object]) -> tuple[dict[str, object], object]:
    function()
    wall: list[float] = []
    cpu: list[float] = []
    result: object = None
    for _ in range(REPETITIONS):
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        result = function()
        cpu.append(time.process_time() - cpu_start)
        wall.append(time.perf_counter() - wall_start)
    return {
        "warmup_count": 1,
        "repetition_count": REPETITIONS,
        "wall": distribution(wall),
        "process_CPU": distribution(cpu),
    }, result


def all_integrals(
    n: int, c: int, zeros: np.ndarray, G: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u, weights = _panels_nodes(float(c), zeros[zeros < c], G)
    common = u * jv(n, u) ** 2 * weights
    squares = zeros * zeros
    denominator = u[:, None] ** 2 - squares[None, :]
    I = np.sum(common[:, None] / denominator, axis=0)
    D = np.sum(common[:, None] / denominator**2, axis=0)
    return I, D, u, weights, common


def extend_project_block(
    old: np.ndarray,
    zeros: np.ndarray,
    I_old: np.ndarray,
    u: np.ndarray,
    common: np.ndarray,
) -> np.ndarray:
    old_P = old.shape[0]
    P = len(zeros)
    matrix = np.empty((P, P))
    matrix[:old_P, :old_P] = old
    squares = zeros * zeros
    denominator = u[:, None] ** 2 - squares[None, old_P:]
    I_new = np.sum(common[:, None] / denominator, axis=0)
    D_new = np.sum(common[:, None] / denominator**2, axis=0)
    I = np.concatenate((I_old, I_new))
    for m in range(old_P, P):
        matrix[m, m] = 2 * squares[m] * D_new[m - old_P]
        for k in range(m):
            value = (
                2
                * zeros[m]
                * zeros[k]
                * (I[m] - I[k])
                / (squares[m] - squares[k])
            )
            matrix[m, k] = matrix[k, m] = value
    return matrix


def matrix_storage_bytes(dimension: int) -> int:
    return int(dimension * dimension * np.dtype(float).itemsize)


def case_timings(case: dict[str, object]) -> dict[str, object]:
    n, c, M = int(case["n"]), int(case["c"]), int(case["M_n_c"])
    dimensions = [M + buffer for buffer in BUFFERS]
    rows = []
    for index in range(4, len(dimensions)):
        P_previous = dimensions[index - 1]
        P = dimensions[index]
        N = P + 1
        zeros_project = jn_zeros(n, P)
        old_matrix = assemble(n, float(c), P=P_previous, G=G)
        I, _, u, _, common = all_integrals(n, c, zeros_project, G)
        I_old = I[:P_previous]

        full_timing, full_matrix = benchmark(
            lambda: assemble(n, float(c), P=P, G=G)
        )
        extension_timing, extended_matrix = benchmark(
            lambda: extend_project_block(
                old_matrix, zeros_project, I_old, u, common
            )
        )
        disagreement = float(np.linalg.norm(full_matrix - extended_matrix, 2))
        tolerance = 1e-11 * max(1.0, float(np.linalg.norm(full_matrix, 2)))
        project_eigensolve_timing, _ = benchmark(
            lambda: np.linalg.eigvalsh(full_matrix)
        )

        zeros_boulsane = jn_zeros(n, N + 1)
        paths = {}
        for path in (
            "large_N",
            "half_mesh",
            "next_zero",
            "left_control",
            "right_control",
        ):
            epsilon = offset_value(path, n, N, zeros_boulsane)
            D = float(zeros_boulsane[N - 1] + epsilon)
            omega = float(c / D)
            assembly_timing, matrix = benchmark(
                lambda n=n, N=N, z=zeros_boulsane, omega=omega: boulsane_matrix(
                    n, N, z, omega
                )
            )
            eigensolve_timing, _ = benchmark(lambda matrix=matrix: np.linalg.eigvalsh(matrix))
            paths[path] = {
                "epsilon": epsilon,
                "epsilon_over_pi": float(epsilon / np.pi),
                "omega": omega,
                "assembly": assembly_timing,
                "eigensolve": eigensolve_timing,
                "cache_status": "Bessel zeros cached; all matrix entries rebuilt",
                "reused_entry_count": 0,
                "output_matrix_storage_bytes": matrix_storage_bytes(N),
            }

        rows.append(
            {
                "P_previous": P_previous,
                "P": P,
                "Boulsane_N": N,
                "project_full_rebuild": {
                    **full_timing,
                    "cache_status": "no matrix entries reused; assembler constructs quadrature and zeros",
                    "reused_entry_count": 0,
                    "output_matrix_storage_bytes": matrix_storage_bytes(P),
                },
                "project_block_extension": {
                    **extension_timing,
                    "cache_status": "old block, old integrals, Bessel zeros, and quadrature cached",
                    "reused_entry_count": P_previous * P_previous,
                    "new_entry_count": P * P - P_previous * P_previous,
                    "output_matrix_storage_bytes": matrix_storage_bytes(P),
                    "approximate_peak_matrix_storage_bytes": (
                        matrix_storage_bytes(P_previous) + matrix_storage_bytes(P)
                    ),
                },
                "project_full_vs_extension_norm_2": disagreement,
                "project_extension_audit_tolerance": tolerance,
                "project_extension_audit_passed": bool(disagreement <= tolerance),
                "project_eigensolve": project_eigensolve_timing,
                "Boulsane_fixed_c_paths": paths,
            }
        )
    return {
        "n": n,
        "c": c,
        "M_n_c": M,
        "parameter_path": "fixed continuous c",
        "rows": rows,
    }


def environment_record() -> dict[str, object]:
    thread_variables = {
        name: os.environ.get(name)
        for name in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_CPU_count": os.cpu_count(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "thread_count_environment": thread_variables,
        "thread_count_record": (
            "No thread-count override is set; BLAS uses its runtime default. "
            "Logical CPU count is recorded above."
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
        default=Path("data/task_25_4_phase3/track_E_timing.json"),
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    started = time.perf_counter()
    cases = []
    for case in manifest["cases"]:
        result = case_timings(case)
        cases.append(result)
        speedups = [
            row["project_full_rebuild"]["wall"]["median_seconds"]
            / row["project_block_extension"]["wall"]["median_seconds"]
            for row in result["rows"]
        ]
        print(
            f"n={result['n']} c={result['c']}: extension speedups="
            + ",".join(f"{value:.2f}" for value in speedups),
            flush=True,
        )
    payload = {
        "task": "25.4 Phase 3 Track E refinement and timing",
        "evidence_label": "machine-specific ordinary timings",
        "parameter_path": "fixed continuous c",
        "environment": environment_record(),
        "method": {
            "warmups": 1,
            "repetitions": REPETITIONS,
            "project_quadrature_order_per_panel": G,
            "assembly_and_eigensolve_timed_separately": True,
            "Boulsane_closed_form_entry_advantage_acknowledged": True,
        },
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "timed_dimension_count": sum(len(case["rows"]) for case in cases),
            "all_project_extension_audits_passed": all(
                row["project_extension_audit_passed"]
                for case in cases
                for row in case["rows"]
            ),
        },
        "elapsed_seconds_total": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({"output": str(args.output), "sha256": digest, **payload["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

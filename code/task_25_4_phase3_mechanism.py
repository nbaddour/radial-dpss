"""Task 25.4 Phase 3, Track C fixed-offset mechanism campaign.

The five preregistered offsets are evaluated before any oracle search.  Each
matrix is audited against direct Gram quadrature at Q=2000 and Q=3000.  The
frozen knee mode is used at every dimension.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import scipy.stats as stats
from scipy.special import jn_zeros, jv, jvp

from task_25_4_phase3_freeze_manifest import nystrom_A
from task_25_4_phase3_structure import boulsane_direct_gram, boulsane_matrix


BUFFERS = (5, 8, 12, 20, 32, 48, 64, 96)
EPS = np.finfo(float).eps


def kernel_cross(n: int, c: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    X = x[:, None]
    Y = y[None, :]
    cX = c * X
    cY = c * Y
    denominator = Y * Y - X * X
    with np.errstate(divide="ignore", invalid="ignore"):
        kernel = c * (
            X * jv(n, cY) * jvp(n, cX)
            - Y * jv(n, cX) * jvp(n, cY)
        ) / denominator
    near = np.abs(denominator) < 1e-14
    if np.any(near):
        midpoint = 0.5 * (X + Y)
        cmid = c * midpoint
        diagonal = (c * c / 2) * (
            jvp(n, cmid) ** 2
            + (1 - (n / cmid) ** 2) * jv(n, cmid) ** 2
        )
        kernel = np.where(near, diagonal, kernel)
    return kernel


def reference_values_at_nodes(
    n: int,
    c: int,
    eigenvalue: float,
    reference_x: np.ndarray,
    reference_w: np.ndarray,
    reference_function: np.ndarray,
    nodes: np.ndarray,
) -> np.ndarray:
    kernel = kernel_cross(n, float(c), nodes, reference_x)
    return kernel @ (reference_function * reference_x * reference_w) / eigenvalue


def offset_value(path: str, n: int, N: int, zeros: np.ndarray) -> float:
    if path == "large_N":
        return float((N + n / 2 + 1 / 4) * np.pi - zeros[N - 1])
    if path == "half_mesh":
        return float(np.pi / 2)
    if path == "next_zero":
        return float(zeros[N] - zeros[N - 1])
    if path == "left_control":
        return float(np.pi / 4)
    if path == "right_control":
        return float(3 * np.pi / 4)
    raise ValueError(path)


def audit_boulsane(
    n: int, N: int, zeros: np.ndarray, omega: float, matrix: np.ndarray
) -> dict[str, object]:
    gram2000 = boulsane_direct_gram(n, N, zeros, omega, 2000)
    gram3000 = boulsane_direct_gram(n, N, zeros, omega, 3000)
    norm = float(np.linalg.norm(matrix, 2))
    tolerance = 1e-11 * max(1.0, norm)
    symmetry = float(np.max(np.abs(matrix - matrix.T)))
    closed_2000 = float(np.linalg.norm(matrix - gram2000, 2))
    closed_3000 = float(np.linalg.norm(matrix - gram3000, 2))
    refinement = float(np.linalg.norm(gram2000 - gram3000, 2))
    maximum = max(symmetry, closed_2000, closed_3000, refinement)
    return {
        "matrix_norm_2": norm,
        "tolerance": tolerance,
        "symmetry_defect": symmetry,
        "closed_vs_Q2000_norm_2": closed_2000,
        "closed_vs_Q3000_norm_2": closed_3000,
        "Q2000_vs_Q3000_norm_2": refinement,
        "maximum_disagreement": maximum,
        "passed": bool(maximum <= tolerance),
    }


def regression(rows: list[dict[str, object]], error_key: str, path: str, M: int, u_ref: float) -> dict[str, object]:
    eligible = [
        row
        for row in rows
        if row["N"] >= M + 16
        and abs(float(row[error_key])) >= 10 * u_ref
        and row["audit"]["passed"]
    ]
    result: dict[str, object] = {
        "error": error_key,
        "eligible_dimensions": [row["N"] for row in eligible],
        "eligible_count": len(eligible),
        "reported": False,
    }
    if len(eligible) < 5:
        result["reason"] = "fewer than five eligible points"
        return result

    x = np.log([row["h"] for row in eligible])
    y = np.log([abs(row[error_key]) for row in eligible])
    fit = stats.linregress(x, y)
    critical = stats.t.ppf(0.975, len(eligible) - 2)
    slope_ci = [float(fit.slope - critical * fit.stderr), float(fit.slope + critical * fit.stderr)]
    fit_drop = stats.linregress(x[1:], y[1:])
    stable = bool(abs(float(fit.slope - fit_drop.slope)) <= 0.35)
    if path == "next_zero":
        compatible = bool(0.75 <= fit.slope <= 1.25)
    elif path in {"half_mesh", "large_N"}:
        compatible = bool(fit.slope >= 1.6)
    else:
        compatible = None
    result.update(
        {
            "slope": float(fit.slope),
            "slope_95_percent_CI": slope_ci,
            "R_squared": float(fit.rvalue * fit.rvalue),
            "slope_after_deleting_first": float(fit_drop.slope),
            "stable_under_first_point_deletion": stable,
            "compatible_with_window": compatible if stable else False,
            "reported": stable,
            "reason": None if stable else "slope changes by more than 0.35 after deleting first point",
        }
    )
    return result


def case_run(case: dict[str, object]) -> dict[str, object]:
    n, c, M = int(case["n"]), int(case["c"]), int(case["M_n_c"])
    knee = next(row for row in case["selected_modes"] if row["category"] == "knee")
    m = int(knee["m"])
    lambda_ref = float(knee["lambda_ref"])
    u_ref = float(knee["u_ref"])
    endpoint = float(case["knee_endpoint"]["absolute_value_Q2"])
    C = lambda_ref * endpoint * endpoint
    Q2 = int(case["quadrature_orders"]["Q2"])
    count = M + 6
    values, x_ref, w_ref, psi_ref = nystrom_A(n, float(c), Q2, count)
    psi_m = psi_ref[:, m]

    dimensions = [M + buffer for buffer in BUFFERS]
    zeros = jn_zeros(n, max(dimensions) + 1)
    paths = {}
    for path in ("large_N", "half_mesh", "next_zero", "left_control", "right_control"):
        rows = []
        for N in dimensions:
            started = time.perf_counter()
            epsilon = offset_value(path, n, N, zeros)
            D = float(zeros[N - 1] + epsilon)
            omega = float(c / D)
            eligible = 0 < omega < 1
            if not eligible:
                rows.append(
                    {
                        "N": N,
                        "epsilon": epsilon,
                        "omega": omega,
                        "eligible": False,
                    }
                )
                continue
            nodes = zeros[:N] / D
            weights = 2 / (D * D * jv(n + 1, zeros[:N]) ** 2)
            matrix = boulsane_matrix(n, N, zeros, omega)
            audit = audit_boulsane(n, N, zeros, omega, matrix)
            eigenvalues = np.linalg.eigvalsh(matrix)[::-1]
            lambda_b = float(eigenvalues[m])
            psi_nodes = reference_values_at_nodes(
                n, c, lambda_ref, x_ref, w_ref, psi_m, nodes
            )
            vector = np.sqrt(weights) * psi_nodes
            denominator = float(vector @ vector)
            rayleigh = float(vector @ matrix @ vector / denominator)
            e_lambda = lambda_ref - lambda_b
            e_rayleigh = lambda_ref - rayleigh
            h = float(np.pi / D)
            a = float(epsilon / np.pi - 0.5)
            gamma_denominator = a * h * C
            resolved_rayleigh = abs(e_rayleigh) >= 10 * u_ref
            row = {
                "N": N,
                "epsilon": epsilon,
                "epsilon_over_pi": float(epsilon / np.pi),
                "omega": omega,
                "eligible": True,
                "h": h,
                "a": a,
                "lambda_Boulsane": lambda_b,
                "e_lambda": e_lambda,
                "sampled_Rayleigh": rayleigh,
                "e_Rayleigh": e_rayleigh,
                "reference_floor_resolved_eigenvalue": abs(e_lambda) >= 10 * u_ref,
                "reference_floor_resolved_Rayleigh": resolved_rayleigh,
                "Gamma": (
                    e_rayleigh / gamma_denominator
                    if resolved_rayleigh and gamma_denominator != 0
                    else None
                ),
                "next_zero_normalized": (
                    e_rayleigh / (h * C)
                    if path == "next_zero" and resolved_rayleigh
                    else None
                ),
                "half_mesh_scaled": (
                    abs(e_rayleigh) / (h * h)
                    if path in {"half_mesh", "large_N"} and resolved_rayleigh
                    else None
                ),
                "audit": audit,
                "elapsed_seconds": time.perf_counter() - started,
            }
            rows.append(row)

        rayleigh_fit = regression(rows, "e_Rayleigh", path, M, u_ref)
        eigenvalue_fit = regression(rows, "e_lambda", path, M, u_ref)
        next_last_three = []
        if path == "next_zero":
            next_last_three = [
                row["next_zero_normalized"]
                for row in rows
                if row.get("next_zero_normalized") is not None
            ][-3:]
        paths[path] = {
            "rows": rows,
            "Rayleigh_rate_fit": rayleigh_fit,
            "eigenvalue_rate_fit": eigenvalue_fit,
            "next_zero_last_three_normalized": next_last_three,
            "next_zero_coefficient_window_passed": (
                len(next_last_three) == 3
                and all(0.25 <= value <= 0.75 for value in next_last_three)
            ) if path == "next_zero" else None,
        }

    return {
        "n": n,
        "c": c,
        "M_n_c": M,
        "m_knee": m,
        "lambda_ref": lambda_ref,
        "u_ref": u_ref,
        "endpoint_abs": endpoint,
        "C_m": C,
        "parameter_path": "fixed continuous c",
        "paths": paths,
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
        default=Path("data/task_25_4_phase3/track_C_fixed_offsets.json"),
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    started = time.perf_counter()
    cases = []
    for case in manifest["cases"]:
        result = case_run(case)
        cases.append(result)
        fits = {
            path: result["paths"][path]["Rayleigh_rate_fit"].get("slope")
            for path in ("large_N", "half_mesh", "next_zero")
        }
        print(f"n={result['n']} c={result['c']}: Rayleigh slopes={fits}", flush=True)

    eligible_cases = len(cases)
    path_pass_counts = {
        path: sum(
            bool(case["paths"][path]["Rayleigh_rate_fit"].get("compatible_with_window"))
            for case in cases
        )
        for path in ("large_N", "half_mesh", "next_zero")
    }
    coefficient_pass_count = sum(
        bool(case["paths"]["next_zero"]["next_zero_coefficient_window_passed"])
        for case in cases
    )
    c80_contradictions = [
        {"n": case["n"], "path": path}
        for case in cases
        if case["c"] == 80
        for path in ("large_N", "half_mesh", "next_zero")
        if case["paths"][path]["Rayleigh_rate_fit"].get("reported")
        and not case["paths"][path]["Rayleigh_rate_fit"].get("compatible_with_window")
    ]
    gate_C = bool(
        all(path_pass_counts[path] >= 0.75 * eligible_cases for path in path_pass_counts)
        and coefficient_pass_count >= 0.75 * eligible_cases
        and not c80_contradictions
    )
    eigen_eligible = {
        path: [
            case
            for case in cases
            if case["paths"][path]["eigenvalue_rate_fit"].get("reported")
        ]
        for path in ("large_N", "half_mesh", "next_zero")
    }
    eigen_pass = {
        path: (
            len(rows) > 0
            and sum(
                bool(case["paths"][path]["eigenvalue_rate_fit"].get("compatible_with_window"))
                for case in rows
            )
            >= 0.75 * len(rows)
        )
        for path, rows in eigen_eligible.items()
    }
    result = {
        "task": "25.4 Phase 3 Track C fixed offsets",
        "evidence_label": "ordinary numerical mechanism diagnostics",
        "oracle_included": False,
        "cases": cases,
        "summary": {
            "eligible_case_count": eligible_cases,
            "Rayleigh_compatible_case_counts": path_pass_counts,
            "next_zero_coefficient_window_case_count": coefficient_pass_count,
            "c80_contradictions": c80_contradictions,
            "gate_C_mechanism_corroboration_passed": gate_C,
            "eigenvalue_rate_eligible_case_counts": {
                path: len(rows) for path, rows in eigen_eligible.items()
            },
            "eigenvalue_rate_75_percent_passed": eigen_pass,
            "all_matrices_audited": all(
                row["audit"]["passed"]
                for case in cases
                for path in case["paths"].values()
                for row in path["rows"]
                if row.get("eligible")
            ),
        },
        "elapsed_seconds_total": time.perf_counter() - started,
    }
    rendered = json.dumps(result, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(json.dumps({"output": str(args.output), "sha256": digest, **result["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
